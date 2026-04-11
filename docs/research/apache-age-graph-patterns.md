# Apache AGE — graph query patterns for Z3rno memory relationships

**Status:** Draft research, informs Week 1 Friday graph schema implementation
**Date:** 2026-04-11
**Authors:** F2 (AI/ML), F1 (Full-Stack), F3 (DevOps)
**Tags:** graph, age, cypher, memory-relationships, recall

> This is a research artifact, not a specification. Patterns documented here are *candidates* for the Week 1 Friday graph schema implementation. The final selected patterns will be implemented in `z3rno_core.engine.recall()` and `z3rno_core.graph.cypher_builder` and documented in `docs/SCHEMA.md`.

---

## 1. Why graph at all for agent memory?

Z3rno's memory engine combines three retrieval modalities in a single query: vector similarity (pgvector), graph traversal (Apache AGE), and temporal filtering (SCD Type 2). The first two are well-understood; the question this doc answers is **what graph structure best models the relationships an AI agent's memories actually have**, and **what Cypher patterns we need to make those relationships queryable**.

Three things only a graph can do well:

1. **Multi-hop entity reasoning.** "What does the agent know about the user's company's competitors?" needs `User → WORKS_AT → Company → COMPETES_WITH → Company` — three hops. Doing this with recursive SQL CTEs gets ugly past two hops; doing it with Cypher is natural.
2. **Memory provenance chains.** "This semantic memory was derived from these episodic memories, which in turn came from these working memories." A `DERIVED_FROM` chain with arbitrary depth is the cleanest model. Useful for GDPR audits, compliance reporting, and intelligent forgetting (you can't delete a semantic memory without acknowledging the episodic memories it summarizes).
3. **Contradiction detection across the memory store.** "Does this new memory contradict any existing memory?" is a `CONTRADICTS` edge between two memory vertices. Detecting it cheaply requires graph indexes, not full-table SQL scans.

Things a graph is **not** the right tool for:

- **Plain metadata filtering.** "All memories tagged `customer-support` and `urgent`" is a JSONB GIN index query, not a graph query.
- **Full-text search.** That's `tsvector`/`tsquery` in PostgreSQL core.
- **Pure vector similarity.** That's pgvector HNSW.
- **Single-hop foreign-key joins.** A `memory_relationships` relational table with a `source_memory_id` FK and a `target_memory_id` FK does this faster than AGE for one-hop queries.

The **hybrid query path** documented in Doc 08 §4.4.3 (vector + graph + temporal in a single SQL query) is where Z3rno's value lives. AGE is one of three legs of that stool — not a replacement for the other two.

---

## 2. Proposed graph schema

### 2.1 Vertex labels

| Label | Represents | Key properties | Cardinality (per tenant) |
|---|---|---|---|
| `Memory` | Any of the four memory types (working / episodic / semantic / procedural). Mirrors a row in the relational `memories` table. | `memory_id` (UUID, FK back to `memories.id`), `memory_type`, `org_id`, `agent_id` | High — one per memory row. Could be 100K-10M+. |
| `Agent` | An AI agent. Mirrors a row in `agents`. | `agent_id` (UUID), `org_id`, `name` | Low — typically 1-100 per tenant. |
| `User` | An end-user that an agent interacts with. **Not the same as a Z3rno tenant user** — this is the user the agent is serving. | `user_id` (UUID), `org_id` | Medium — one per end-user the tenant tracks. |
| `Concept` | An extracted entity (a person, company, place, topic, preference, tool). Created via LLM-based entity extraction during `store()`. | `concept_id` (UUID), `org_id`, `name`, `type` (one of `person`, `company`, `place`, `topic`, `preference`, `tool`, ...) | Medium-high — one per unique concept across all memories. Could be tens of thousands. |
| `Session` | A conversation or task session that grouped working memories before they were promoted. **Optional** — sessions are Redis-only by primary decision; this vertex exists only if a tenant explicitly chooses to materialize sessions in the graph for analytics. | `session_id` (UUID), `org_id`, `agent_id` | Low-medium. |

Every vertex has an `org_id` property — this is how we enforce tenant isolation in graph queries (since AGE doesn't natively support PostgreSQL RLS on graph vertices, see §5.2).

### 2.2 Edge labels

| Label | Direction | Source → Target | When created | Used by |
|---|---|---|---|---|
| `RELATES_TO` | undirected (logically) | Memory ↔ Memory | Generic association — caller-supplied via `store(relationships=...)` | Recall with `include_graph_context=True`, default depth 1 |
| `DERIVED_FROM` | directed | Memory → Memory | When summarisation runs (`episodic → semantic`) or transition (`working → episodic`). The new memory points back at its sources. | Provenance audits, GDPR cascades, compliance reporting |
| `CONTRADICTS` | symmetric | Memory ↔ Memory | When poisoning detection or contradiction detection identifies a conflict | Recall fusion (down-weight contradicted memories), admin review |
| `SUPPORTS` | symmetric | Memory ↔ Memory | When two memories reinforce each other (extracted via LLM consistency check) | Recall fusion (boost mutually-supported memories) |
| `SUPERSEDES` | directed | Memory → Memory | When a memory replaces an older one (e.g. user's address changed). The newer memory points at the older one. | Recall default-filter (exclude superseded memories from current results) |
| `CAUSED_BY` | directed | Memory → Memory | When a memory describes a causal consequence of another (LLM-extracted) | Causal reasoning queries, root-cause analysis |
| `PARTICIPATED_IN` | directed | Memory → Session | Links a working memory to its session vertex (only if Session is materialized) | Session subgraph queries |
| `BELONGS_TO` | directed | Memory → Agent | Every memory belongs to exactly one agent (also enforced relationally) | Multi-agent coordination, agent-scoped recall |
| `MENTIONS` | directed | Memory → Concept | When a memory mentions an extracted concept (entity extraction during `store()`) | Concept-driven recall ("find all memories about Acme Corp") |
| `WORKS_AT` / `LIVES_IN` / `PREFERS` / etc. | directed | Concept → Concept | Entity-to-entity relationships extracted by the LLM during `store()` | Multi-hop entity reasoning |

> **Note:** the `WORKS_AT` / `LIVES_IN` / `PREFERS` family is open-ended. We don't try to enumerate every possible relationship at schema-design time — instead, we let the LLM-based entity extractor propose relationship types and store them as edge labels. AGE supports arbitrary edge labels, so this is fine.

### 2.3 Why this schema and not "just edges between memories"

A simpler design is: every memory is a vertex, every relationship is an edge between two memories, no separate `Concept` vertices. We rejected this because:

- **Concept reuse.** If 100 memories all mention "Acme Corp", we want one `Concept(Acme Corp)` vertex with 100 `MENTIONS` edges, not 100 disconnected mentions of the string "Acme Corp" inside memory `metadata` JSONB.
- **Multi-hop entity queries.** "What does the agent know about Acme Corp's competitors?" requires a `Concept(Acme Corp) → COMPETES_WITH → Concept(Beta Corp)` edge. There's no place to put that edge in a memory-only schema.
- **Concept-driven recall.** "Find all memories that mention any subsidiary of Acme Corp" is `MATCH (c:Concept {name: 'Acme Corp'})-[:HAS_SUBSIDIARY*]->(sub:Concept)<-[:MENTIONS]-(m:Memory) RETURN m`. Trivial in graph, painful in pure vector or pure relational.

The trade-off: entity extraction is now a required part of `store()`, which means every write costs an LLM call. Mitigation: entity extraction is deferred to a Celery worker, so synchronous writes return immediately and the graph is enriched asynchronously. The eventual-consistency window is acceptable because recall already filters by `valid_to IS NULL`.

---

## 3. Core query patterns

Each pattern below has a Cypher example that should run against an AGE-enabled PostgreSQL with the `memory_graph` graph created (which the `00-z3rno-preload.sh` initdb hook handles in dev — see `docker/postgres/`).

> **Every Cypher query in Z3rno wraps a `cypher()` function call.** The full SQL form is:
> ```sql
> SELECT * FROM cypher('memory_graph', $$
>   <Cypher query>
> $$) AS (col1 agtype, col2 agtype, ...);
> ```
> The pattern docs below show only the Cypher inside `$$...$$` for readability. The wrapper is added by `z3rno_core.graph.cypher_builder` at query time.

### 3.1 One-hop neighbours of a memory

**Use case:** `recall(include_graph_context=True, graph_depth=1)` — for each vector-search result, fetch its directly-related memories.

```cypher
MATCH (m:Memory {memory_id: '<uuid>', org_id: '<org-uuid>'})-[r]->(neighbour:Memory)
WHERE neighbour.org_id = '<org-uuid>'
RETURN neighbour.memory_id AS id, type(r) AS relationship, neighbour.memory_type AS type
LIMIT 10;
```

**Notes:**
- The `org_id` filter on both the source and the target is the tenant isolation guard. We can't rely on RLS here (see §5.2).
- `type(r)` returns the edge label as a string.
- Default `LIMIT 10` to avoid runaway results on densely-connected memories.

### 3.2 Multi-hop derivation chain

**Use case:** GDPR audit — "show me the full provenance chain of this semantic memory back to its working-memory roots."

```cypher
MATCH path = (m:Memory {memory_id: '<uuid>', org_id: '<org-uuid>'})-[:DERIVED_FROM*1..10]->(ancestor:Memory)
WHERE ancestor.org_id = '<org-uuid>'
RETURN path;
```

**Notes:**
- `[:DERIVED_FROM*1..10]` is a variable-length path, 1 to 10 hops. Cap at 10 to prevent unbounded traversal on accidentally-cyclical graphs.
- `RETURN path` returns the full sequence of vertices and edges. The Python side unpacks this into a tree structure.
- If the chain is empty (no `DERIVED_FROM` ancestors), the result set is empty — caller handles that.

### 3.3 Contradiction detection on store

**Use case:** when storing a new memory, before committing, check if any existing memory contradicts it via embedding similarity + LLM consistency check. If contradicted, create a `CONTRADICTS` edge.

```cypher
MATCH (a:Memory {memory_id: '<new-uuid>', org_id: '<org-uuid>'}),
      (b:Memory {memory_id: '<existing-uuid>', org_id: '<org-uuid>'})
CREATE (a)-[r:CONTRADICTS {detected_at: timestamp(), confidence: 0.85}]->(b),
       (b)-[:CONTRADICTS {detected_at: timestamp(), confidence: 0.85}]->(a)
RETURN r;
```

**Notes:**
- AGE doesn't have native undirected edges, so symmetric relationships (`CONTRADICTS`, `SUPPORTS`) are stored as **two directed edges**, one in each direction. Queries can match either direction with `-[r:CONTRADICTS]-` (the lack of arrow makes it undirected at query time).
- Edge properties (`detected_at`, `confidence`) live on both copies. They must be kept in sync if updated.

### 3.4 Find contradicted memories

**Use case:** when ranking recall results, downweight any memory that has at least one `CONTRADICTS` edge above a confidence threshold.

```cypher
MATCH (m:Memory {org_id: '<org-uuid>'})-[r:CONTRADICTS]-(other:Memory)
WHERE r.confidence > 0.7
RETURN m.memory_id AS id, count(other) AS contradiction_count, max(r.confidence) AS max_confidence;
```

**Notes:**
- `-[r:CONTRADICTS]-` (no arrow) matches in either direction, leveraging the symmetric pair we created in §3.3.
- `count(other)` and `max(r.confidence)` are aggregations — AGE supports the standard Cypher aggregate functions.

### 3.5 Session subgraph

**Use case:** when a session ends, find all working memories that participated in it so they can be promoted to episodic.

```cypher
MATCH (s:Session {session_id: '<uuid>', org_id: '<org-uuid>'})<-[:PARTICIPATED_IN]-(m:Memory)
WHERE m.memory_type = 'working' AND m.org_id = '<org-uuid>'
RETURN m.memory_id AS id, m.memory_type AS type;
```

**Notes:**
- Only relevant if Session vertices are materialized in the graph (per §2.1, this is opt-in).
- For tenants that don't materialize sessions, the equivalent query is a JSONB filter on `memories.metadata->>'session_id'` — faster, no graph traversal needed.

### 3.6 Concept-driven recall

**Use case:** "Find all memories that mention any subsidiary of Acme Corp."

```cypher
MATCH (root:Concept {name: 'Acme Corp', type: 'company', org_id: '<org-uuid>'})-[:HAS_SUBSIDIARY*0..3]->(sub:Concept)<-[:MENTIONS]-(m:Memory)
WHERE m.org_id = '<org-uuid>' AND sub.org_id = '<org-uuid>'
RETURN DISTINCT m.memory_id AS id;
```

**Notes:**
- `[:HAS_SUBSIDIARY*0..3]` matches the root concept itself (depth 0) and up to 3 levels of subsidiaries.
- `DISTINCT` because a memory can mention multiple concepts in the subtree.

### 3.7 Shortest path between two memories

**Use case:** "How does the agent connect this user complaint to that engineering decision?" Useful for explainability.

```cypher
MATCH (a:Memory {memory_id: '<uuid-a>', org_id: '<org-uuid>'}),
      (b:Memory {memory_id: '<uuid-b>', org_id: '<org-uuid>'}),
      path = shortestPath((a)-[*..6]-(b))
RETURN path, length(path) AS hops;
```

**Notes:**
- `shortestPath` is a built-in Cypher function. AGE supports it.
- The `*..6` cap prevents unbounded traversal — adjust based on graph density.
- If no path exists, the result set is empty.

### 3.8 Hybrid vector + graph query

**Use case:** the canonical Z3rno recall — vector similarity finds candidate memories, then graph traversal enriches each with related context. This is the query in Doc 08 §2 Principle 2.

```sql
WITH vector_candidates AS (
    SELECT id, content, embedding <=> $1 AS similarity
    FROM memories
    WHERE org_id = current_setting('app.current_org_id')::UUID
      AND agent_id = $2
      AND deleted_at IS NULL
      AND valid_to IS NULL
    ORDER BY embedding <=> $1
    LIMIT 20
)
SELECT
    vc.id,
    vc.content,
    vc.similarity,
    graph_data.related_memories
FROM vector_candidates vc
LEFT JOIN LATERAL (
    SELECT array_agg(neighbour_id) AS related_memories
    FROM cypher('memory_graph', $$
        MATCH (m:Memory {memory_id: $memory_id})-[r]->(n:Memory)
        WHERE n.org_id = $org_id
        RETURN n.memory_id AS neighbour_id
        LIMIT 5
    $$, jsonb_build_object('memory_id', vc.id::text, 'org_id', current_setting('app.current_org_id'))) AS (neighbour_id agtype)
) graph_data ON true
ORDER BY vc.similarity
LIMIT 10;
```

**Notes:**
- The `LEFT JOIN LATERAL` is the magic — for each row from `vector_candidates`, run a Cypher subquery in the graph to fetch related memories. `LATERAL` is what lets the subquery reference the outer row's `vc.id`.
- The Cypher query uses **parameter binding** via `jsonb_build_object` — never string-concatenate user input into a Cypher query (Cypher injection is real).
- This is the pattern Doc 08 §4.4.3 describes for hybrid search. Implemented in `z3rno_core.engine.recall()` Week 2.

### 3.9 Point-in-time graph snapshot

**Use case:** "What did the agent know about Acme Corp on March 15?"

```cypher
MATCH (root:Concept {name: 'Acme Corp', org_id: '<org-uuid>'})-[r:MENTIONS]-(m:Memory)
WHERE m.org_id = '<org-uuid>'
  AND m.valid_from <= '2026-03-15T00:00:00Z'
  AND (m.valid_to IS NULL OR m.valid_to > '2026-03-15T00:00:00Z')
RETURN m.memory_id, m.content;
```

**Notes:**
- Temporal filtering is applied via the **vertex properties** (`valid_from`, `valid_to`) that mirror the relational `memories` table. We sync vertex properties on every memory update — see §5.4.
- Edges don't currently have temporal versioning. If a `WORKS_AT` relationship changes ("user used to work at Acme, now works at Beta"), we delete the old edge and create a new one. **Open question:** do we need edge-level temporal versioning? Probably yes for compliance, but it's a Phase 2 question.

### 3.10 Detecting orphan memories

**Use case:** find memories with no graph connections at all — useful for intelligent forgetting (orphans are candidates for deletion if also low-importance).

```cypher
MATCH (m:Memory {org_id: '<org-uuid>'})
WHERE NOT (m)-[]-()
RETURN m.memory_id, m.memory_type;
```

**Notes:**
- `NOT (m)-[]-()` matches memories with no edges in any direction. AGE supports this pattern.
- Combine with the relational `importance_score` filter in the application layer; running this in a single SQL would require another LATERAL join.

---

## 4. AGE-specific gotchas (learned during the smoke test)

### 4.1 `LOAD 'age'` per session

Every psql session that runs Cypher queries must first call `LOAD 'age';`. The library is preloaded via `shared_preload_libraries`, but the session-level path setup needs an explicit `LOAD`. The Z3rno server middleware handles this once per connection; ad-hoc psql sessions need to remember.

```sql
LOAD 'age';
SET search_path = ag_catalog, public;
-- now Cypher queries work
```

### 4.2 `search_path` must include `ag_catalog`

`SET search_path = ag_catalog, public;` is required so that AGE's functions (`cypher`, `create_graph`, `drop_graph`, `agtype` operators) resolve without schema-prefixing. The server middleware also handles this.

### 4.3 Cypher queries return `agtype` — cast or unpack on the Python side

AGE returns Cypher results wrapped in a custom `agtype` data type. Example:

```sql
SELECT * FROM cypher('memory_graph', $$
  MATCH (m:Memory) RETURN m.memory_id
$$) AS (memory_id agtype);
```

The `memory_id` column is `agtype` even though the underlying value is a UUID string. To get a Python `uuid.UUID`, the `z3rno_core.graph.agtype_decode()` helper (to be written Week 1 Friday) parses the wrapped JSON and extracts the value. Until that helper exists, callers do it inline:

```python
import json
raw = row['memory_id']           # something like '"1234-5678-...".'
unwrapped = json.loads(raw.rstrip('::vertex'))  # depends on the wrapper format
```

> **Open question:** the cleanest extraction approach. AGE has multiple agtype wrapper formats depending on whether the value is a vertex, edge, scalar, or path. We need a generic decoder.

### 4.4 No native foreign-key constraints between vertices and the relational `memories` table

AGE vertices live in their own internal storage, separate from the regular PostgreSQL tables. We can't `FOREIGN KEY (memory_id) REFERENCES memories(id)` from the graph side. **Consequence:** if a memory is hard-deleted from the relational table (GDPR), its corresponding vertex must be explicitly dropped via Cypher. The `forget(hard_delete=True)` path handles this — see Week 2 Wednesday.

### 4.5 Indexing limitations

AGE creates a B-tree index on vertex labels by default. To get fast property lookups (e.g. `MATCH (m:Memory {memory_id: '<uuid>'})`), we need to **manually create indexes on the JSONB property storage**:

```sql
SELECT create_property_index('memory_graph', 'Memory', 'memory_id');
SELECT create_property_index('memory_graph', 'Memory', 'org_id');
SELECT create_property_index('memory_graph', 'Concept', 'name');
SELECT create_property_index('memory_graph', 'Concept', 'org_id');
```

These are issued in Alembic migration `009_create_graph_schema.py` (per Doc 02 Week 1 Wednesday).

### 4.6 `CASE` and `WITH` work; some advanced Cypher does not

AGE 1.6+/PG17 supports most of openCypher 9, including `WITH`, `UNWIND`, `CASE`, `OPTIONAL MATCH`, aggregation functions, and `shortestPath`. It does **not** yet support:
- `CALL` to subqueries (Cypher 25)
- APOC procedures (Neo4j-specific)
- Some advanced path patterns (`(a)-[*..]-(b) WHERE all(...)`)

When in doubt, write a small test against the live container before committing the pattern.

### 4.7 `gen_random_uuid()` doesn't work inside Cypher

Cypher has its own `randomUUID()` function but AGE 1.6 doesn't ship it. If you need a UUID inside a Cypher query, generate it on the SQL side and pass it as a parameter:

```sql
SELECT * FROM cypher('memory_graph', $$
  CREATE (c:Concept {concept_id: $id, name: $name})
  RETURN c
$$, jsonb_build_object('id', gen_random_uuid()::text, 'name', 'Acme Corp')) AS (c agtype);
```

### 4.8 Tenant isolation cannot use PostgreSQL RLS

AGE vertices and edges live in tables created by `create_graph()` under a private schema (something like `ag_label_vertex` and `ag_label_edge` inside the `memory_graph` namespace). These tables don't have `org_id` columns we can attach RLS policies to.

**Mitigation:** every Cypher query in Z3rno **must** filter by `org_id` as a vertex property. The `z3rno_core.graph.cypher_builder` module is responsible for injecting `WHERE n.org_id = $org_id` into every query. We will NOT trust callers to remember this — the builder is the only sanctioned interface to AGE from application code, and it always adds the filter.

This is documented in `docs/MULTI_TENANCY.md` as a Week 1 Thursday deliverable.

---

## 5. Performance trade-offs

### 5.1 When graph beats metadata filtering

- **Multi-hop queries (3+ hops):** graph wins by orders of magnitude. Recursive CTEs are O(N^depth) in the worst case; graph indexes prune early.
- **Set membership across relationships:** "memories that mention any concept that is a subsidiary of Acme Corp" — graph in milliseconds, SQL in seconds.
- **Contradiction lookups:** "is there any memory that contradicts this one?" — graph index on `CONTRADICTS` edges is fast; SQL would scan the relationships table.

### 5.2 When metadata filtering beats graph

- **Single-hop FK joins.** "Find all memories for agent X" is `WHERE agent_id = X` in SQL, no graph traversal needed.
- **JSONB tag filters.** "Find all memories tagged `urgent`" is a GIN index lookup on `metadata`, not a graph query.
- **Pure attribute filtering.** "Find memories with importance_score > 0.7" — no graph involvement at all.

The rule of thumb: **if the question can be answered by looking at one row at a time in the relational table, don't use the graph.** Graph queries cost more per call than relational queries because of the agtype wrapping, the schema indirection, and the lack of RLS. Save them for the cases that genuinely need traversal.

### 5.3 Caching frequently-traversed paths

For hot queries like "all concepts mentioned by this user's recent memories", we should consider materialising the result in Redis with a 5-minute TTL. The hybrid query in §3.8 is the natural place to add this — wrap the LATERAL graph subquery with a Redis cache lookup keyed on `(org_id, memory_id, depth)`.

This is **Phase 2 work** (Week 13+ in the cloud phase). For MVP, every recall hits the graph fresh. Document the cache opportunity in `docs/PERFORMANCE.md` so we don't forget.

### 5.4 Keeping graph in sync with relational

Every `store()` writes both:
1. A row in `memories` (relational, ACID)
2. A vertex in `memory_graph` with the same `memory_id` and key properties (`memory_type`, `org_id`, `agent_id`, `valid_from`, `valid_to`)

Plus zero or more edges if relationships are provided.

Both writes must happen in the **same transaction**. If the graph write fails, the relational write rolls back. This is a Z3rno commitment — there is no "eventually consistent" graph state. Documented in Doc 08 §2 Principle 2.

When a memory is **updated** (SCD Type 2: new row inserted, old row's `valid_to` set), the graph vertex's `valid_from`/`valid_to` properties must be updated to match. There are two strategies here, and we should pick one in Week 1 Friday:

- **Strategy A: vertex per memory version.** Every SCD row gets its own vertex. Graph queries naturally see version history. Higher graph storage, but cleanest.
- **Strategy B: single vertex per memory_id.** Vertex represents the *current* state; old versions live only in the relational table. Graph queries always show "current" state. Smaller graph, but loses temporal queries on the graph side.

> **Open question for Week 1 Friday:** which SCD strategy do we use for graph vertices? Most likely **Strategy B** for MVP (smaller graph, faster), with a note that Strategy A may be needed for compliance use cases later.

---

## 6. Open questions for Week 1 Friday and Week 2

1. **agtype decoder.** Need a robust Python helper that takes any AGE result column and returns a usable Python type (`uuid.UUID`, `datetime`, `dict`, `list`, etc.). Probably 50 lines, 20 test cases. Owner: F2.
2. **`cypher_builder` API surface.** What does the Python interface to AGE look like? Options:
   - Raw string templating with parameter binding (simple, error-prone)
   - Method chaining (`builder.match('Memory').where(org_id=...).return_('memory_id')`) — pretty, more code
   - DSL with classes — overkill for MVP
   Decision in Week 1 Friday.
3. **SCD strategy for graph vertices.** A or B per §5.4. Decision in Week 1 Friday.
4. **Edge versioning.** Do edges need `valid_from`/`valid_to` for compliance? Suspected yes for `WORKS_AT`-style relationships, no for `RELATES_TO` / `MENTIONS`. Phase 2 problem; document the gap and move on.
5. **Property index strategy.** Which vertex/edge properties get indexed at graph creation time? See §4.5 — there's a small set, but tuning these affects every query.
6. **Concept extraction.** Which LLM and prompt do we use for entity extraction during `store()`? Cohere has a dedicated NER model; OpenAI requires prompt engineering with `response_format=json`. Trade-off: cost vs accuracy. Phase 2 unless a design partner blocks on it.
7. **Cypher injection prevention.** Verify `cypher_builder` never string-concatenates untrusted input. Add a unit test: pass a malicious string with `'); DROP GRAPH memory_graph; --` and confirm it errors instead of executing.
8. **AGE memory leak under high write load.** Anecdotal reports of AGE leaking memory in long-running connections. We should benchmark in Week 7 (load testing) and either restart workers periodically or report upstream.

---

## 7. References

- **Apache AGE docs:** <https://age.apache.org/age-manual/master/index.html>
- **AGE PG17 release notes:** `PG17/v1.7.0-rc0` — <https://github.com/apache/age/releases/tag/PG17%2Fv1.7.0-rc0>
- **openCypher 9 reference:** <https://opencypher.org/resources/>
- **Doc 08 §4.4 Vector Search Architecture:** `z3rno-process-docs/08-Architecture-Document.md` (the vector + graph + temporal hybrid query)
- **Doc 08 §4.5 Graph Memory Architecture:** `z3rno-process-docs/08-Architecture-Document.md` (the graph model overview)
- **Tech Stack Doc §2.4 Apache AGE:** `z3rno-process-docs/09-Tech-Stack-Document.md` (why we picked AGE over Neo4j)
- **`docker/postgres/Dockerfile`:** AGE compiled from `PG17` branch, source-of-truth for the version we ship.
- **`docs/adr/` (planned):** ADR-002 will lock the SCD-vs-graph strategy from §5.4 once decided in Week 1 Friday.

---

## 8. Next steps

After this doc is reviewed in the Week 1 Friday demo:
1. Implement `z3rno_core.graph` package with the patterns in §3
2. Write unit tests for `cypher_builder` covering each pattern
3. Write the `agtype_decode` helper
4. Decide SCD strategy (§5.4), write ADR-002
5. Add `create_property_index` calls to Alembic migration `009_create_graph_schema.py`
6. Document the agreed `cypher_builder` API in `docs/SCHEMA.md`

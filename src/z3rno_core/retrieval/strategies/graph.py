"""GRAPH retrieval — vector seed → AGE subgraph expansion → LLM synthesis.

Pipeline:
  1. Embed the query (via ``embedding_provider`` from ``**extra``).
  2. Run a vector kNN on ``memories`` to find ``seed_count`` seed Memos.
  3. For each seed, traverse the AGE graph up to ``hops`` (default 1).
  4. Re-fetch the full memo rows for every node in the expanded subgraph.
  5. (Optional) Ask ``llm_gateway`` to synthesize an answer using the
     subgraph as context. The answer lives in
     ``results[0].metadata["graph_answer"]``.
  6. Return seed Memos in vector-rank order with their N-hop neighbours
     in ``graph_context``.

Graceful degradation:
  * ``llm_gateway=None`` → skip step 5; still return seeds + subgraph.
  * No ``embedding_provider`` OR empty embedding → return ``[]``
    (GRAPH can't seed without vector signal).
  * AGE extension not loaded → seed Memos returned with empty
    ``graph_context``; warning logged. Mirrors the Phase A graph_writer
    posture.

RLS: GRAPH writes were done with ``org_id`` baked into AGE node
properties at the Phase A graph_writer. The Cypher MATCH includes
``{org_id: '<uuid>'}`` so cross-tenant nodes are invisible even if
the AGE schema bypasses Postgres RLS.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Connection, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import LLMGateway, LLMGatewayError
from z3rno_core.engine.embedding import EmbeddingProvider
from z3rno_core.graph.queries import _validate_uuid_for_cypher
from z3rno_core.retrieval._filters import build_where_clause
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)

logger = logging.getLogger(__name__)

_GRAPH_NAME = "memory_graph"
_DEFAULT_SEED_COUNT = 3
_DEFAULT_HOPS = 1


@register_strategy
class GraphStrategy(RetrievalStrategy):
    name = "GRAPH"
    requires_query_embedding = True
    # LLM is optional — GRAPH degrades to "seeds + subgraph, no synthesis"
    # when no gateway is configured. requires_llm reflects best-case use,
    # not strict requirement, so AUTO can prefer GRAPH when LLM is present
    # without rejecting it outright when absent.
    requires_llm = False

    async def retrieve(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        agent_id: UUID,
        query: str,
        top_k: int,
        memory_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        embedding_provider: EmbeddingProvider | None = extra.get("embedding_provider")
        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        seed_count: int = int(extra.get("seed_count", _DEFAULT_SEED_COUNT))
        hops: int = int(extra.get("hops", _DEFAULT_HOPS))
        time_range = extra.get("time_range")
        as_of = extra.get("as_of")
        include_deleted = bool(extra.get("include_deleted", False))

        if not query or embedding_provider is None:
            # GRAPH requires a query to vector-seed from. Without an
            # embedding provider we can't run kNN; return empty rather
            # than fall back to non-graph behaviour silently.
            return []

        query_embedding = await embedding_provider.embed_text(query)
        if not query_embedding:
            return []

        # --- 1+2. Vector-seed the search ---
        where_clause, params = build_where_clause(
            org_id=org_id,
            agent_id=agent_id,
            memory_type=memory_type,
            user_id=extra.get("user_id"),
            metadata_filter=metadata_filter,
            conversation_id=extra.get("conversation_id"),
            time_range=time_range,
            as_of=as_of,
            include_deleted=include_deleted,
        )
        params["query_vec"] = "[" + ",".join(str(x) for x in query_embedding) + "]"
        params["seed_count"] = seed_count
        seed_sql = f"""
            SELECT id, content, summary, memory_type, importance_score,
                   recall_count, created_at, valid_from, metadata,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM public.memories
            WHERE {where_clause}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :seed_count
        """
        seeds = list((await conn.execute(text(seed_sql), params)).fetchall())

        if not seeds:
            return []

        # --- 3. AGE expansion ---
        # Run under a savepoint so an AGE failure (extension not loaded
        # in the testcontainer, syntax error, etc.) doesn't poison the
        # surrounding transaction. Same posture as the Phase A
        # graph_writer.
        seed_ids = [UUID(str(row[0])) for row in seeds]
        subgraphs = await self._expand_subgraphs(
            conn, org_id=org_id, seed_ids=seed_ids, hops=hops
        )

        # --- 4. Re-fetch any non-seed nodes that appeared in the subgraph
        # so we have full memo content to hand to the LLM synthesis step
        # and to populate graph_context with rich data.
        all_node_ids: set[UUID] = set(seed_ids)
        for sg in subgraphs.values():
            for node in sg["nodes"]:
                if isinstance(node.get("id"), str):
                    try:
                        all_node_ids.add(UUID(node["id"]))
                    except ValueError:
                        # AGE may return ids with extra quoting; best-effort.
                        continue

        # --- 5. Optional LLM synthesis ---
        graph_answer: str | None = None
        if llm_gateway is not None:
            try:
                graph_answer = await self._synthesize_answer(
                    llm_gateway, query=query, seeds=seeds, subgraphs=subgraphs
                )
            except LLMGatewayError:
                logger.warning("graph.synthesis.llm_failed", exc_info=True)
            except Exception:
                logger.warning("graph.synthesis.unexpected_error", exc_info=True)

        # --- 6. Build the response ---
        now = datetime.now().astimezone()
        results: list[StrategyResult] = []
        for i, row in enumerate(seeds[:top_k]):
            similarity = float(row[9]) if row[9] is not None else 0.0
            if similarity < similarity_threshold:
                continue

            importance = float(row[4])
            recency_days = max((now - row[6]).total_seconds() / 86400, 0.01)
            recency_score = min(1.0, 1.0 / recency_days)

            # 50% graph-seed similarity, 30% subgraph richness, 20% importance.
            seed_id = UUID(str(row[0]))
            sg = subgraphs.get(seed_id, {"nodes": [], "edges": []})
            richness = min(1.0, (len(sg["nodes"]) + len(sg["edges"])) / 10.0)
            relevance = 0.50 * similarity + 0.30 * richness + 0.20 * importance

            metadata = dict(row[8] or {})
            # The synthesized answer rides on the top result so consumers
            # who just take results[0] still see it. Empty when LLM
            # wasn't available or synthesis failed.
            if i == 0 and graph_answer:
                metadata = {**metadata, "graph_answer": graph_answer}

            results.append(
                StrategyResult(
                    memory_id=seed_id,
                    content=row[1],
                    summary=row[2],
                    memory_type=row[3],
                    importance_score=importance,
                    relevance_score=round(relevance, 4),
                    recall_count=row[5],
                    created_at=row[6],
                    valid_from=row[7],
                    metadata=metadata,
                    score_components={
                        "vector": round(similarity, 4),
                        "graph_richness": round(richness, 4),
                        "importance": round(importance, 4),
                        "recency": round(recency_score, 4),
                    },
                    graph_context=sg["nodes"] + sg["edges"],
                )
            )

        return results

    async def _expand_subgraphs(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        seed_ids: list[UUID],
        hops: int,
    ) -> dict[UUID, dict[str, list[dict[str, Any]]]]:
        """Return ``{seed_id: {nodes: [...], edges: [...]}}`` per seed.

        Empty dicts are returned for seeds whose subgraph query fails —
        e.g. AGE extension isn't loaded in this database, or the seed
        has no neighbours.
        """
        out: dict[UUID, dict[str, list[dict[str, Any]]]] = {}

        def _runner(
            sync_conn: Connection, sid: UUID
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            return _fetch_subgraph_sync(
                sync_conn, org_id=org_id, seed_id=sid, hops=hops
            )

        for seed in seed_ids:
            try:
                nodes, edges = await conn.run_sync(_runner, seed)
                out[seed] = {"nodes": nodes, "edges": edges}
            except (DBAPIError, ValueError) as exc:
                logger.warning(
                    "graph.subgraph.expand_failed",
                    extra={"seed": str(seed), "error": str(exc)[:200]},
                )
                out[seed] = {"nodes": [], "edges": []}
        return out

    async def _synthesize_answer(
        self,
        gateway: LLMGateway,
        *,
        query: str,
        seeds: list[Any],
        subgraphs: dict[UUID, dict[str, list[dict[str, Any]]]],
    ) -> str:
        """Ask the LLM to answer the query using the seeds + subgraph context."""
        context_lines: list[str] = []
        for row in seeds:
            mid = str(row[0])
            content = (row[1] or "").strip()[:400]
            context_lines.append(f"Memory {mid[:8]}: {content}")
            sg = subgraphs.get(UUID(mid), {"nodes": [], "edges": []})
            for edge in sg["edges"][:5]:
                context_lines.append(
                    f"  -- {edge.get('type', '?')} --> "
                    f"{str(edge.get('target', '?'))[:8]}"
                )

        system = (
            "You are a memory-graph summarizer. Use ONLY the memories and "
            "relationships shown below to answer the user's query. If the "
            "graph doesn't contain enough information, say so explicitly. "
            "Keep the answer under 200 words."
        )
        user = (
            f"Query: {query}\n\n"
            f"Graph context:\n" + "\n".join(context_lines)
        )
        return await gateway.complete(
            system=system, user=user, max_tokens=300, temperature=0.0
        )


def _fetch_subgraph_sync(
    conn: Connection,
    *,
    org_id: UUID,
    seed_id: UUID,
    hops: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the AGE Cypher under a savepoint; return (nodes, edges) lists.

    Wrapped in ``conn.begin_nested()`` so an AGE failure (extension not
    loaded, syntax error) doesn't kill the outer transaction.
    """
    safe_seed = _validate_uuid_for_cypher(seed_id)
    safe_org = _validate_uuid_for_cypher(org_id)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    with conn.begin_nested():
        # LOAD + SET search_path + cypher() in one statement isn't legal
        # under asyncpg's prepared-statement protocol — sync psycopg
        # connections (which is what run_sync hands us once unwrapped)
        # accept multi-statement strings. Use the same preamble shape as
        # graph/queries.py.
        preamble = "LOAD 'age'; SET search_path = ag_catalog, \"$user\", public;"

        # Subgraph as nodes + edges. Cap nodes per seed to keep response
        # size bounded — operators on huge graphs can tune via the
        # subgraph_node_limit kwarg, exposed in C.3 if needed.
        node_sql = (
            f"{preamble} "
            f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
            f"MATCH (m:Memory {{id: '{safe_seed}', org_id: '{safe_org}'}})"
            f"-[r*1..{hops}]-(n:Memory) "
            f"WHERE n.org_id = '{safe_org}' "
            f"RETURN DISTINCT n.id AS id, n.memory_type AS memory_type "
            f"LIMIT 50 "
            f"$$) AS (id agtype, memory_type agtype)"
        )
        # AGE not loaded → DBAPIError propagates and the outer savepoint
        # rolls back; caller catches and returns empty subgraph.
        for row in conn.execute(text(node_sql)).fetchall():
            nodes.append({
                "id": _agtype_to_str(row[0]),
                "memory_type": _agtype_to_str(row[1]),
            })

        edge_sql = (
            f"{preamble} "
            f"SELECT * FROM cypher('{_GRAPH_NAME}', $$ "
            f"MATCH (m:Memory {{id: '{safe_seed}', org_id: '{safe_org}'}})"
            f"-[r]-(n:Memory {{org_id: '{safe_org}'}}) "
            f"RETURN DISTINCT type(r) AS rel_type, n.id AS target "
            f"LIMIT 50 "
            f"$$) AS (rel_type agtype, target agtype)"
        )
        for row in conn.execute(text(edge_sql)).fetchall():
            edges.append({
                "type": _agtype_to_str(row[0]),
                "target": _agtype_to_str(row[1]),
            })

    return nodes, edges


def _agtype_to_str(value: Any) -> str:
    """AGE returns agtype values with quoting; strip it for plain strings."""
    if value is None:
        return ""
    s = str(value)
    # ``agtype`` strings come back as `"abc"` (with surrounding quotes).
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s

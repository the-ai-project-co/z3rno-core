# 001: Embedding model — OpenAI `text-embedding-3-small` (1536 dims), hardcoded for MVP

**Status:** Accepted
**Date:** 2026-04-11
**Deciders:** F1, F2, F3
**Tags:** embeddings, vector-search, openai, mvp-scope

## Context

The Z3rno memory engine stores text content as both raw strings and dense vector embeddings. Recall works by embedding the query text and running a cosine similarity search against the stored embeddings via pgvector's HNSW index. Every memory write goes through an embedding API call. Every memory read involves at least one embedding API call (for the query text).

The embedding model choice cascades into several locked-in decisions across the stack:

1. **Vector dimension** — pgvector's `vector(N)` column type fixes `N` at the column level. Different models produce embeddings of different dimensions (OpenAI `text-embedding-3-small` is 1536, OpenAI `text-embedding-3-large` is 3072, Cohere `embed-english-v3.0` is 1024, `nomic-embed-text` is 768, etc.). Mixing dimensions in one column is not possible. Supporting multiple dimensions in one Z3rno tenant requires either per-tenant tables, multiple embedding columns, or a configurable column dimension at tenant-create time.

2. **Re-embedding cost** — if we ever switch models, every memory in every tenant must be re-embedded. At 100M memories with API-based embedding costing $0.02 per 1M tokens, a tenant migration is on the order of $200-2000 of API spend, plus 24-72 hours of re-indexing. This is not catastrophic but it's not free.

3. **Distance metric** — different models are tuned for different distance metrics. OpenAI's `text-embedding-3-*` family is normalized to unit length, so cosine, dot product, and L2 distance are equivalent up to a constant. Models that aren't normalized (some local sentence-transformers) require cosine specifically.

4. **Provider lock-in** — calling OpenAI's API directly couples Z3rno to OpenAI's pricing, availability, and rate limits. Doc 09 §4.6 documents the strategy of routing all embedding/LLM calls through LiteLLM as an abstraction layer, so we can swap providers per tenant without code changes.

5. **Self-host story** — open-source users running `docker compose up` should not need an OpenAI API key just to evaluate Z3rno. The embedding pipeline must support a local fallback (sentence-transformers running in the worker container) for offline development.

The MVP timeline is 8 weeks. We need to pick *one* default and ship.

## Decision

For Phase 1 MVP we will use **OpenAI `text-embedding-3-small`** as the sole embedding model, with **vector dimension hardcoded to 1536** in the `memories.embedding` column.

LiteLLM is the only API surface we call into for embeddings. The embedding model name is configurable via `EMBEDDING_MODEL` env var, but **the database column dimension is not** — switching to a different model in MVP requires either choosing a 1536-dim model (`embed-multilingual-light-v3.0`, some local models) or accepting that the schema will reject the embeddings at insert time.

Multi-dimension support — per-tenant configurable dimension, or multiple `embedding_*` columns — is **explicitly out of scope** until post-v1.

For local-development self-hosting without an OpenAI key, the dev environment falls back to a **random-vector stub** (1536 floats sampled from a normal distribution, normalized to unit length). This is documented behaviour, not a bug — recall results from a random-stub embedding are meaningless, but every other code path (store, RLS, audit, lifecycle, AGE graph) works identically. Real embeddings only matter when you start measuring recall quality.

## Consequences

### Positive

- **Single dimension simplifies the schema.** `embedding vector(1536)` is one line in the migration. No per-tenant DDL, no schema branching, no migration hell at upgrade time.
- **OpenAI is the path of least resistance for AI engineers.** Our target users (LangChain / CrewAI / OpenAI Agents builders) already have OpenAI keys. Asking them to add Cohere or to deploy a local embedding model is friction we don't need at MVP.
- **`text-embedding-3-small` is OpenAI's best price/performance model as of 2026-04.** 1536 dims, $0.02 / 1M tokens, MTEB score above the previous-generation `text-embedding-ada-002` and within ~1% of `text-embedding-3-large` for ~5x less cost.
- **LiteLLM gives us provider-agnostic API surface for free.** Even though MVP only calls OpenAI, the abstraction is in place from day one. When we add a Cohere or a local provider in Phase 2, no business logic changes.
- **HNSW index tuning is easier when dimension is fixed.** `m=16, ef_construction=200` are tuned per dimension; locking dimension means we can publish "production-ready HNSW config for Z3rno" without qualifications.
- **Random-stub fallback removes the OpenAI key as a hard dependency** for local development, lowering the bar for OSS contributors who want to test schema changes without spending real money.

### Negative

- **Single point of failure on the OpenAI side.** If OpenAI has an outage or rate-limits us, every Z3rno tenant that depends on managed cloud writes is affected. Mitigation: LiteLLM can be configured with fallback chains (OpenAI → Anthropic → local) post-MVP.
- **Cost is variable per write.** `text-embedding-3-small` is cheap but not free. A high-throughput tenant generating 10M memories per month incurs ~$200/month in embedding costs alone, which we either pass through or absorb on the Pro/Team tiers. (For reference: Pro is $29/mo with 100K memory cap = $2/mo embedding cost at scale, easily absorbed. Team is $199/mo with 1M memory cap = $20/mo embedding cost.)
- **Hardcoded 1536 means we cannot accept Cohere `embed-english-v3.0` (1024) or `text-embedding-3-large` (3072) in MVP.** Users who specifically want one of those models cannot use Z3rno until we ship multi-dimension support.
- **Re-embedding migration debt.** When we eventually decide to support multiple dimensions or to upgrade to a newer OpenAI model, every existing tenant needs re-embedding. We will need a documented migration playbook (ADR-NNN, post-v1).
- **The random-stub fallback is a footgun for users who don't read the docs.** Someone running `docker compose up` without an `OPENAI_API_KEY` set will get a working stack with meaningless recall results. We must surface a loud warning at server startup and in `make dev-up` output.

### Neutral

- The choice does **not** lock us out of self-hosting, on-premise deployments, or air-gapped environments. Sentence-transformers `all-MiniLM-L6-v2` (384 dims) and `nomic-embed-text` (768 dims) can run inside the worker container with no external API. Both produce dimensions that don't match 1536, so they need a `vector(384)` or `vector(768)` column — which is a Phase 2 decision.

## Alternatives Considered

### A. OpenAI `text-embedding-3-large` (3072 dims)

The next size up. Marginal MTEB improvement (~1%) over `-small`, but **5x more expensive per token** and the 3072 dimensions double the storage cost and roughly halve the HNSW query throughput at the same RAM budget.

**Why rejected:** The MTEB delta is not worth the cost or the throughput hit at MVP scale. If a customer specifically needs the marginal accuracy, we can offer it as a per-tenant configurable model in Phase 2 (when multi-dimension support lands).

**When to reconsider:** Once we ship per-tenant model selection, expose `text-embedding-3-large` as a Pro/Team-tier option with explicit cost disclosure.

### B. Cohere `embed-english-v3.0` (1024 dims)

Cohere's flagship embedding model. Strong MTEB scores, slightly cheaper than OpenAI at high volume, **better English-language retrieval quality** in some benchmarks. Different dimension (1024) means it can't share the column with OpenAI embeddings.

**Why rejected:** AI agent builders are predominantly OpenAI-first today. Adding Cohere as the default forces every user to set up a Cohere account before they can try Z3rno. We picked the path of least resistance.

**When to reconsider:** If a design partner specifically requests Cohere, ship it as a per-tenant option (Phase 2). The LiteLLM abstraction means the code change is ~5 lines once multi-dimension support is in place.

### C. `nomic-embed-text` (768 dims, local)

A high-quality open-source embedding model that can run inside the worker container with no external API. Smaller dimension (768) means cheaper storage and faster HNSW queries. Cost: zero (per-call) at the price of GPU/CPU compute on the worker host.

**Why rejected for MVP:** Running a local embedding model adds non-trivial complexity to the worker container — model weights to bundle (~100MB), tokenizer setup, optional GPU support, cold-start latency on the first call. Self-hosters who want zero-API-cost should be able to opt into this in Phase 2, but it's not the default path because the average MVP user already has an OpenAI key.

**When to reconsider:** Phase 2 self-hosting story. Add as `EMBEDDING_PROVIDER=local` env var, ship as a separate Docker image variant `z3rno-server:local-embed-17`.

### D. Per-tenant configurable dimension at MVP

Add an `embedding_dimension` column to `tenants` and create separate per-dimension `memories_<N>` tables, joined via a view.

**Why rejected:** The complexity is enormous for an MVP. Cross-tenant queries become impossible, the SQLAlchemy models multiply, the migration tooling has to handle dynamic table creation, and the ORM layer can't statically type the embedding column. This is a Phase 2 problem that requires its own ADR.

**When to reconsider:** After we have at least one design partner who blocks on it. Until then, hardcoding 1536 is the right call.

### E. Skip embeddings entirely; use full-text search (`tsvector`) for MVP

PostgreSQL has built-in full-text search via `tsvector`/`tsquery`. It's free, fast, and requires no external API. We could ship MVP without embeddings and add them in Phase 2.

**Why rejected:** Vector similarity search is a *defining* feature of Z3rno. The whole point of "memory database for AI agents" is that you can recall semantically-similar past memories, not just lexically-matching ones. Shipping without vectors would defeat the value prop. Full-text search is a complement (we use it in the hybrid query path documented in Doc 08 §4.4.3), not a replacement.

**When to reconsider:** Never. This was a non-starter, included only for completeness.

## References

- Architecture Doc §4.4 (Vector Search Architecture): `z3rno-process-docs/08-Architecture-Document.md`
- Tech Stack Doc §2.2 (pgvector) and §4.6 (LiteLLM): `z3rno-process-docs/09-Tech-Stack-Document.md`
- Task Breakdown Week 1 Day 1 task F2-1: `z3rno-process-docs/02-Detailed-Task-Breakdown.md`
- OpenAI embeddings docs: <https://platform.openai.com/docs/guides/embeddings>
- LiteLLM embedding API: <https://docs.litellm.ai/docs/embedding/supported_embedding>
- pgvector dimension type docs: <https://github.com/pgvector/pgvector#vector-type>
- MTEB leaderboard (for ongoing model comparison): <https://huggingface.co/spaces/mteb/leaderboard>

## Open follow-ups (post-Accepted)

- **Phase 2 ADR** — multi-dimension support. Requires schema decision (per-tenant column? separate tables? view-based union?).
- **Phase 2 ADR** — local embedding fallback for self-hosters. Requires worker container variant decision.
- **Loud-warning surface** — when the server starts without `OPENAI_API_KEY` and is using the random-stub fallback, emit a clearly-coloured warning in stdout, in `/v1/health`, and in the dashboard. Ship in Week 3 when the server `/v1/health` endpoint is built.

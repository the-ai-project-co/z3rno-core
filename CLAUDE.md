# CLAUDE.md

## Project

z3rno-core is the core memory engine library for Z3rno. It contains PostgreSQL schema definitions (SQLAlchemy 2.0 models), Alembic migrations, and the engine functions (store, recall, forget, audit, lifecycle management). All other Z3rno components depend on this library.

## Quick Reference

```bash
uv sync --dev                    # Install dependencies
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run mypy .                    # Type check (strict mode)
uv run pytest                    # Run unit tests
uv run pytest -m integration     # Run integration tests (needs DATABASE_URL)
make migrate                     # Apply Alembic migrations
make seed                        # Load dev seed data
```

## Architecture

- `src/z3rno_core/models/` — 7 SQLAlchemy 2.0 declarative models (Tenant, Agent, ApiKey, LifecyclePolicy, Memory, MemoryRelationship, AuditLog)
- `src/z3rno_core/engine/` — Core operations: store(), recall(), forget(), audit(), lifecycle (sweep, decay, retention, transitions)
- `src/z3rno_core/security/` — RLS helpers: set_org_context(), clear_org_context()
- `src/z3rno_core/temporal/` — SCD Type 2 query helpers: get_memory_at_time(), get_memory_history()
- `src/z3rno_core/graph/` — Apache AGE graph sync and Cypher traversal queries
- `src/z3rno_core/distill/` — **Phase A — Forge distillation:** LLM Gateway (LiteLLM + Instructor), entity/relationship extraction, summarization, graph writer (writes Memos + provenance + AGE edges)
- `src/z3rno_core/chunking/` — **Phase A:** token-aware (tiktoken) and paragraph-boundary chunkers; pure functions
- `src/z3rno_core/forge/` — **Phase A:** ForgePipeline orchestrator (parse → distill → retain) with idempotency, bounded concurrency, distill_jobs lifecycle
- `src/z3rno_core/loaders/` — **Phase B.1+B.2 — Loaders:** Loader ABC + LoaderRegistry; PDF (pypdf), DOCX (python-docx), CSV (stdlib + dialect sniff), Markdown, plain text, code (24 languages), URL (httpx + BeautifulSoup HTML extraction + opt-in Playwright fallback). **Phase B.2:** ImageLoader + AudioLoader with magic-byte sniffing for JPEG/PNG/GIF/WebP/MP3/WAV/FLAC/OGG.
- `src/z3rno_core/storage/` — **Phase B.1+B.2:** StorageBackend ABC + LocalStorageBackend (filesystem) + **S3StorageBackend (Phase B.2)** with prefix sandbox + cross-bucket block.
- `src/z3rno_core/ingest/` — **Phase B.1:** IngestPipeline orchestrator (parse → dedupe → load → store → optional auto-distill); `ingest_jobs` state helpers; one Memo per ingest in B.1.
- `src/z3rno_core/multimodal/` — **Phase B.2:** MultimodalProvider ABC + LiteLLMMultimodalProvider (vision via gpt-4o; audio via Whisper) + StubMultimodalProvider for tests.
- `src/z3rno_core/scrapers/` — **Phase B.2:** SearchProvider ABC + TavilyScraper for Tavily-driven web discovery.
- `src/z3rno_core/refine/` — **Phase D:** RefinePipeline orchestrator (dedupe → infer → summarize → reweight → prune); `refine_jobs` state helpers; `record_feedback` + edge-weight EMA.
- `src/z3rno_core/ontology/` — **Phase D:** OWL/TTL loader (rdflib, lazy) + OntologyResolver (exact + fuzzy via rapidfuzz). Optional `[ontology]` extra.
- `src/z3rno_core/codegraph/` — **Phase D:** tree-sitter parser (Python + TypeScript), AST → CodeMemo/CodeEdge extractor, writer to memories + memory_relationships. Optional `[codegraph]` extra.
- `migrations/versions/` — 25 Alembic migrations (001–025; **016** = Phase B.1 datasets/ingest_jobs; **023** = Phase D memo_type/ontology_uri columns + feedback table; **024** = refine_jobs table; **025** = Phase F memories.distill_provenance + 'distill' audit op)
- `seeds/dev_seed.py` — Dev seed data (2 tenants, 500 memories, 1000 audit entries)
- `docs/` — SCHEMA.md, MULTI_TENANCY.md, ADR-001
- `../z3rno-process-docs/improvements/PHASE-A-IMPLEMENTATION.md` — full operator reference for the Forge pipeline
- `../z3rno-process-docs/improvements/PHASE-B1-IMPLEMENTATION.md` — full operator reference for the Ingestion surface
- `../z3rno-process-docs/improvements/PHASE-B2-IMPLEMENTATION.md` — full operator reference for Multimodal + Search + S3 + Playwright
- `../z3rno-process-docs/improvements/PHASE-C-IMPLEMENTATION.md` — full operator reference for the retrieval strategies
- `../z3rno-process-docs/improvements/PHASE-D-IMPLEMENTATION.md` — full operator reference for refine + ontology + codegraph
- `../z3rno-docs/concepts/verbs.mdx` — canonical Z3rno verb table (store / recall / forget / audit / ingest / distill / refine)

## Phase A — Forge (opt-in)

The Forge pipeline is dormant until the operator sets `DISTILL_ENABLED=true` in the server tier. With the flag off:
- The `/v1/distill` route is not registered (OpenAPI byte-identical to pre-Phase-A).
- The `z3rno.forge_distill` Celery task self-rejects with `{status: "rejected", reason: "distill_disabled"}` and zero DB I/O.
- `distill_jobs` and `entity_provenance` tables (Migration 015) sit empty.

**Z3rno lexicon** (Z3rno-native, no externally-coined words): **Forge** (pipeline), **distill** (verb), **refine** (Phase D verb), **Memo** (graph node base class — Phase D), **AUTO** (Phase C retrieval router).

**Public API (z3rno_core.distill):**
- `LLMGateway` / `LiteLLMGateway` / `StubLLMGateway` + 5 typed exceptions (`LLMTimeoutError`, `LLMRateLimitError`, `LLMProviderError`, `LLMValidationError`, `LLMGatewayError`)
- `Entity`, `Relationship`, `Triplet`, `DistillResult` (frozen, mergeable, case-insensitive dedupe)
- `extract_from_chunk` / `extract_from_chunks` (concurrent, partial-failure tolerant)
- `summarize_text` / `rolling_summarize` (concise/bullet/abstractive styles, map-reduce)
- `write_distill_result` + lifecycle helpers (`insert_distill_job`, `update_distill_job`, `already_distilled`)

**Public API (z3rno_core.forge):**
- `ForgePipeline(gateway=..., embedding_provider=..., options=ForgeOptions(...))`
- `await pipeline.run(engine, *, org_id, agent_id, memory_ids, job_id?)` → `ForgeRunSummary`

**Provenance.** Every Memo the Forge writes carries `prompt_hash` (SHA-256 of `(system, user)` prompts), `model`, `chunk_index`, `char_start/end`. **Phase F slice 1** (Migration 025) added the denormalized `memories.distill_provenance` JSONB column + a new `distill` audit-chain operation. The blob carries a `correlation_id` that links the Memo row, the `entity_provenance` row, and the audit-log entry. Set `DISTILL_PROVENANCE_REQUIRED=true` on the server tier to abort the Forge transaction if any of those three writes fails. The `verify_chain(conn, memo_id=…)` helper in `z3rno_core.distill.provenance` walks the chain and returns a `ChainVerdict`.

**AGE writes are best-effort.** Apache AGE not being loaded (testcontainer, etc.) logs a warning and skips graph mirroring; the relational state stays consistent. AGE writes are also wrapped in `conn.begin_nested()` savepoints (since v0.3.1) so a failure doesn't poison the surrounding transaction.

## Phase B.1 — Ingestion (opt-in)

The ingestion surface is dormant until the operator sets `INGEST_ENABLED=true` in the server tier. With the flag off:
- `/v1/ingest`, `/v1/ingest/file`, and `/v1/datasets` routes are not registered (OpenAPI byte-identical to pre-Phase-B-1).
- The `z3rno.ingest_run` Celery task self-rejects with `{status: "rejected", reason: "ingest_disabled"}` and zero DB I/O.
- `datasets`, `ingest_jobs` tables (Migration 016) and `dataset_id` columns sit empty/NULL.

**Public API (z3rno_core.loaders):**
- `Loader` ABC + `LoaderResult` schema; `LoaderRegistry` with magic-byte sniffing
- `PdfLoader`, `DocxLoader`, `CsvLoader`, `MarkdownLoader`, `PlainTextLoader`, `CodeLoader`, `UrlLoader`
- `fetch_url(...)` — async HTTP fetch with scheme allowlist, timeout, response-size cap; returns `FetchResult`
- `get_default_registry()` — process-wide singleton with all 7 loaders pre-registered

**Public API (z3rno_core.storage):**
- `StorageBackend` ABC + `LocalStorageBackend` (Phase B.1) — `store_artifact / read_artifact / delete_artifact`

**Public API (z3rno_core.ingest):**
- `IngestPipeline(registry=..., storage=..., embedding_provider=..., url_*=...)`
- `await pipeline.run(engine, *, org_id, agent_id, ingest_input, dataset_id?, options?, post_ingest?, ...)` → `IngestRunSummary`
- `IngestInput(kind="text|url|file", ...)`, `IngestOptions(auto_distill=, chunk_size=, ...)`
- State helpers: `insert_ingest_job`, `update_ingest_job`, `find_memory_by_source_uri`

**`engine.store.store()` extended.** Now accepts `dataset_id: UUID | None` so the FK is set during INSERT (not via UPDATE) — sidesteps the SCD-2 trigger's recursion guard. Existing callers don't pass it; default `None` preserves all pre-Phase-B-1 behavior.

**Idempotency.** URL re-ingests dedupe on `(org_id, dataset_id, source_uri)`. Text and file ingests intentionally do not — every call creates a new Memo. Phase D's `refine()` will add content-hash dedupe for files.

## Phase B.2 — Multimodal + Search + S3 + Playwright (opt-in)

Four independently-gated capabilities. Each is dormant by default:

- **`MULTIMODAL_ENABLED=true`** activates ImageLoader + AudioLoader. Vision routes through `litellm.acompletion`, audio through `litellm.atranscription` (Whisper). Worker registers them with the loader registry on startup via `register_multimodal_loaders()`.
- **`STORAGE_BACKEND=s3`** activates `S3StorageBackend` for managed-cloud deployments. Same `StorageBackend` interface as `LocalStorageBackend` — IngestPipeline is backend-agnostic. Returns `s3://...` URIs.
- **`TAVILY_API_KEY` set** activates `POST /v1/ingest/search`. Asks Tavily for top-N URLs, enqueues one ingest_run per hit.
- **`URL_PLAYWRIGHT_ENABLED=true`** + `pip install 'z3rno-core[playwright]'` activates JS-rendered fallback in the URL loader.

**Public API additions (z3rno_core.multimodal):**
- `MultimodalProvider` ABC + `LiteLLMMultimodalProvider` + `StubMultimodalProvider` + factory
- `ImageDescription`, `AudioTranscript` schemas
- `MultimodalError` and 3 typed subclasses

**Public API additions (z3rno_core.scrapers):**
- `SearchProvider` ABC + `TavilyScraper` + `SearchResult` schema

**Public API additions (z3rno_core.storage):**
- `S3StorageBackend` — aioboto3-backed, prefix sandbox, cross-bucket block

**Public API additions (z3rno_core.loaders):**
- `ImageLoader`, `AudioLoader` (each requires a `MultimodalProvider` on construction)
- `register_multimodal_loaders(registry, image_loader=, audio_loader=)` helper
- `render_with_playwright(url)` — lazy-imports Playwright, raises clear error if extra not installed
- `sniff_mime_type()` extended for image/* + audio/* magic bytes

## Phase D — Graph Intelligence (opt-in)

Refine + ontology grounding + code-graph extraction. All capabilities are dormant by default — each one flips on independently. Migration 023 + 024 ship the storage; everything else stays inert until flags are flipped server-side.

**Z3rno lexicon**: **refine** (verb), **Memo** (typed graph node — Pydantic projection in `z3rno_core.graph.primitives`, distinct from the SA `Memory` row), **CODE** (Phase D retrieval strategy).

**Public API (z3rno_core.refine):**
- `RefinePipeline(options=RefineOptions(...), gateway=LLMGateway | None)` — orchestrates dedupe → infer → summarize → reweight → prune.
- `await pipeline.run(conn, *, org_id, dataset_id?, job_id?)` → `RefineRunSummary` (counters + per-stage results).
- `record_feedback(conn, *, org_id, agent_id, signal, memory_id?, edge_id?, reason?)` → UUID. Exactly-one-of `memory_id`/`edge_id` enforced at Python and DB CHECK layers.
- `run_dedupe / run_infer / run_summarize / run_reweight / run_prune` — individual stage callables for tests + ad-hoc use.
- State helpers: `insert_refine_job`, `update_refine_job` (mirror ingest/distill state helpers).

**Public API (z3rno_core.ontology):** — optional, requires `[ontology]` extra (rdflib + rapidfuzz)
- `OntologyResolver(index, strategy="exact|fuzzy", fuzzy_threshold=0.80)`
- `load_ontology(path)` → `OntologyIndex` (lru_cached per process).
- `OntologyEntry`, `ResolveMatch`.

**Public API (z3rno_core.codegraph):** — optional, requires `[codegraph]` extra (tree-sitter + grammars)
- `parse_source(src, *, language="python|typescript")` → `ParsedSource`.
- `extract(parsed, *, module_name)` → `ExtractResult` of `CodeMemo` (MODULE/CLASS/FUNCTION/IMPORT) + `CodeEdge` (DEFINES/IMPORTS/CALLS/INHERITS).
- `await write_extraction(conn, *, org_id, agent_id, extraction, dataset_id?)` → `CodegraphWriteResult`. Edges land in `memory_relationships` with `metadata.codegraph_kind` discriminator (no enum migration needed).

**Public API additions (z3rno_core.graph):**
- `Memo`, `Edge` — frozen Pydantic graph-node projection (distinct from SA `Memory` row).
- `Triplet` — re-exported from `z3rno_core.distill`.

**Forge integration.** `ForgePipeline(..., ontology_resolver=resolver)` threads the resolver into `write_distill_result` so every distilled entity Memo gets its `memo_type` + `ontology_uri` populated transactionally. Resolver failures log + skip; the Memo is preserved.

**Ingest integration.** `IngestOptions(codegraph_enabled=True)` runs the tree-sitter extractor after the text Memo lands, when the loader's `metadata.language` is `python` / `typescript`. Failures attach as `codegraph_failed` warnings; the text Memo is preserved.

**Retrieval integration.** `CODE` strategy joins Phase C's nine. Recursive-CTE one-hop neighborhood over `memory_relationships` filtered to codegraph edges — no LLM, no embedding, no AGE required.

**Feedback / refine lifecycle:**
- Feedback rows are **never deleted by reweight** — every cycle recomputes from scratch. Idempotent.
- Dedupe uses SCD-2 supersede (`valid_to = now()` + `deleted_at = now()`), never hard-merge. Lineage stamped on the primary's `metadata.refine.merged_from`.
- Summarize cluster cache lives in the SUMMARY Memo's `metadata.refine.cluster_hash`; reruns skip already-summarized clusters.

## Key Conventions

- Python 3.11+, src/ layout, hatchling build backend
- Ruff for lint+format (line length 100), mypy strict mode with pydantic plugin
- All models inherit from Base and use OrgScopedMixin for tenant isolation
- Tenant key is `org_id` (not tenant_id) everywhere
- Audit column is `operation` (not action)
- Memory uses `recall_count`/`last_recalled_at` (not access_count)
- Soft delete via `deleted_at` timestamp (no is_deleted boolean)
- Temporal versioning via `valid_from`/`valid_to` (NULL = current, no 9999-12-31 sentinel)
- Embedding dimension hardcoded at 1536 (ADR-001)
- RLS session variable is `app.current_org_id`
- Conventional commits (feat:, fix:, docs:, test:, ci:)

## Database

- PostgreSQL 17 with extensions: pgvector, pgvectorscale, Apache AGE, pg_cron, pgaudit, pgcrypto
- Custom Docker image: `ghcr.io/the-ai-project-co/z3rno-postgres:17`
- Alembic uses psycopg (sync driver). Engine functions use asyncpg (async).
- DATABASE_URL env var overrides alembic.ini default

## Testing

- Unit tests: `uv run pytest` (no DB needed, **882 passed / 3 skipped** as of Phase D / v0.16.0)
- Integration tests: `DATABASE_URL=postgresql+psycopg://... uv run pytest -m integration` (~21 tests against testcontainer; 7 Phase A + 6 Phase B.1 + 8 Phase D schema)
- Optional extras for full test coverage: `uv pip install rdflib rapidfuzz tree-sitter tree-sitter-python tree-sitter-typescript` (Phase D slices 4 + 5)
- Coverage threshold: 95% (unit tests with mocked DB connections cover async engine paths)
- Seeds and lifecycle tests excluded from per-file ignores (S608, PLR2004, etc.)

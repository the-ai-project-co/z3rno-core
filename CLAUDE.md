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
- `migrations/versions/` — 15 Alembic migrations (001-014 + extensions)
- `seeds/dev_seed.py` — Dev seed data (2 tenants, 500 memories, 1000 audit entries)
- `docs/` — SCHEMA.md, MULTI_TENANCY.md, ADR-001

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

- Unit tests: `uv run pytest` (no DB needed, 143 tests)
- Integration tests: `DATABASE_URL=postgresql+psycopg://... uv run pytest -m integration`
- Coverage threshold: 95% (unit tests with mocked DB connections cover async engine paths)
- Seeds and lifecycle tests excluded from per-file ignores (S608, PLR2004, etc.)

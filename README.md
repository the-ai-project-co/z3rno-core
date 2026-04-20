# z3rno-core

[![PyPI](https://img.shields.io/pypi/v/z3rno-core)](https://pypi.org/project/z3rno-core/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/the-ai-project-co/z3rno-core/actions/workflows/ci.yml/badge.svg)](https://github.com/the-ai-project-co/z3rno-core/actions/workflows/ci.yml)

Core memory engine for Z3rno -- PostgreSQL schema, SQLAlchemy models, Alembic migrations, and the store/recall/forget/audit library.

## Features

- **7 relational tables** -- tenants, API keys, agents, memories, memory relationships, lifecycle policies, audit log
- **15 Alembic migrations** -- fully versioned schema evolution
- **Row-Level Security (RLS)** -- multi-tenant isolation at the database level
- **SCD Type 2 versioning** -- temporal history for every memory mutation
- **pgvector HNSW indexing** -- fast approximate nearest-neighbor vector search
- **Apache AGE graph layer** -- entity and relationship traversal via Cypher queries
- **Hash-chained audit log** -- tamper-evident record of every store, recall, forget, and update

## Quickstart

### Run migrations

```bash
# Set your database URL
export DATABASE_URL="postgresql://z3rno:password@localhost:5432/z3rno"

# Install
pip install -e .

# Apply all migrations
alembic upgrade head
```

### Use the engine

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from z3rno_core.engine import store, recall, forget, audit

engine = create_async_engine("postgresql+asyncpg://z3rno:password@localhost/z3rno")
Session = async_sessionmaker(engine)

async with Session() as db:
    # Store a memory
    memory = await store(db, org_id=org_id, agent_id="agent-1", content="User prefers dark mode")

    # Recall by semantic similarity
    results = await recall(db, org_id=org_id, agent_id="agent-1", query="user preferences", top_k=5)

    # Soft-delete a memory
    await forget(db, org_id=org_id, memory_id=memory.id)

    # Query the audit trail
    entries = await audit(db, org_id=org_id, agent_id="agent-1")
```

For a detailed step-by-step setup, see [QUICKSTART.md](QUICKSTART.md).

Full documentation: [astron-bb4261fd.mintlify.app](https://astron-bb4261fd.mintlify.app)

## Architecture

The schema follows a strict dependency order:

```
tenants -> api_keys -> agents -> memories -> memory_relationships
                                    |
                                    +-> lifecycle_policies
                                    +-> audit_log
```

All engine functions operate within a single async SQLAlchemy session and respect RLS via `SET LOCAL` on the org context.

## Documentation

- [SCHEMA.md](docs/SCHEMA.md) -- full table and column reference
- [MULTI_TENANCY.md](docs/MULTI_TENANCY.md) -- RLS design and tenant isolation
- [Architecture Decision Records](docs/adr/) -- design rationale

## Development

```bash
uv sync --dev
uv run ruff check .
uv run mypy .
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Apache 2.0 -- see [LICENSE](LICENSE).

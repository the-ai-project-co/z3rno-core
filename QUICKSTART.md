# Quickstart: z3rno-core

A detailed getting-started guide for the Z3rno core memory engine library.

## Prerequisites

- Python 3.11+
- PostgreSQL 17 with pgvector and Apache AGE extensions
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

If you do not have PostgreSQL set up locally, the easiest path is to use the z3rno-server Docker Compose stack which bundles a pre-configured Postgres image:

```bash
# From the z3rno-server repo
docker compose -f docker-compose.dev.yml up postgres
```

## Step-by-step Installation

### 1. Clone the repository

```bash
git clone https://github.com/the-ai-project-co/z3rno-core.git
cd z3rno-core
```

### 2. Install dependencies

```bash
# With uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### 3. Configure your database

```bash
export DATABASE_URL="postgresql://z3rno:z3rno_dev_password@localhost:5432/z3rno"
```

### 4. Run migrations

```bash
alembic upgrade head
```

This applies all 15 migrations, creating the full Z3rno schema (tenants, agents, memories, etc.).

## Running Locally

Once migrations are applied, you can use z3rno-core as a library in any Python project:

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from z3rno_core.engine import store, recall, forget, audit

DATABASE_URL = "postgresql+asyncpg://z3rno:z3rno_dev_password@localhost:5432/z3rno"

engine = create_async_engine(DATABASE_URL)
Session = async_sessionmaker(engine)

async def main():
    async with Session() as db:
        # Store a memory
        memory = await store(
            db,
            org_id="org-1",
            agent_id="agent-1",
            content="User prefers dark mode",
        )
        print(f"Stored memory: {memory.id}")

        # Recall by semantic similarity
        results = await recall(
            db,
            org_id="org-1",
            agent_id="agent-1",
            query="user preferences",
            top_k=5,
        )
        print(f"Recalled {len(results)} memories")

        # Soft-delete a memory
        await forget(db, org_id="org-1", memory_id=memory.id)
        print("Memory forgotten")

        # Query the audit trail
        entries = await audit(db, org_id="org-1", agent_id="agent-1")
        print(f"Audit entries: {len(entries)}")

asyncio.run(main())
```

## First Working Example

Save the script above as `example.py` and run it:

```bash
python example.py
# Output:
# Stored memory: 550e8400-e29b-41d4-a716-446655440000
# Recalled 1 memories
# Memory forgotten
# Audit entries: 2
```

## Running Tests

```bash
uv run pytest
```

For linting and type checking:

```bash
uv run ruff check .
uv run mypy .
```

## Common Issues / Troubleshooting

### 1. "connection refused" when running migrations

PostgreSQL is not running or not listening on port 5432. If using Docker Compose from z3rno-server, ensure the postgres container is healthy:

```bash
docker compose -f docker-compose.dev.yml ps
```

### 2. "extension pgvector does not exist"

You are using a vanilla PostgreSQL image. Z3rno requires the pre-built `ghcr.io/the-ai-project-co/z3rno-postgres:17` image or a PostgreSQL instance with pgvector, Apache AGE, and pg_cron installed.

### 3. "role z3rno does not exist"

Create the role manually or use the Docker image which handles this automatically:

```sql
CREATE ROLE z3rno WITH LOGIN PASSWORD 'z3rno_dev_password';
CREATE DATABASE z3rno OWNER z3rno;
```

### 4. Alembic "Target database is not up to date"

You have unapplied migrations. Run:

```bash
alembic upgrade head
```

### 5. Import errors after installation

Make sure you installed in editable mode (`pip install -e .`) or that the package is on your Python path. If using uv, ensure you are running inside the project environment:

```bash
uv run python example.py
```

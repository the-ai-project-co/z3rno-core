# z3rno-postgres

> Pre-built PostgreSQL 17 Docker image with every extension Z3rno needs, ready to `docker pull` and run.

## What's inside

| Extension | Version | Source | Purpose |
|---|---|---|---|
| `pgvector` | 0.8+ | apt (`postgresql-17-pgvector`) | Vector similarity search (HNSW) |
| `pgvectorscale` | 0.5.1 | Timescale .deb release | DiskANN index for large vector sets |
| `age` (Apache AGE) | master | compiled from source | openCypher graph queries |
| `pg_cron` | 1.6+ | apt (`postgresql-17-cron`) | In-database job scheduler |
| `pgaudit` | latest | apt (`postgresql-17-pgaudit`) | Session-level audit logging |
| `pgcrypto` | built-in | `postgresql-17-contrib` | Crypto functions (uuid, hashing) |

## Pulling

```bash
docker pull ghcr.io/the-ai-project-co/z3rno-postgres:17
```

## Running locally (for dev)

```bash
docker run --rm -it \
    -e POSTGRES_DB=z3rno \
    -e POSTGRES_USER=z3rno \
    -e POSTGRES_PASSWORD=z3rno \
    -p 5432:5432 \
    ghcr.io/the-ai-project-co/z3rno-postgres:17
```

Then connect with `psql` and verify:

```sql
\dx                       -- should list: vector, vectorscale, age, pg_cron, pgaudit, pgcrypto
SELECT * FROM cypher('memory_graph', $$
    MATCH (n) RETURN count(n)
$$) AS (count agtype);    -- should return 0
```

## Building locally

```bash
cd docker/postgres
docker build -t ghcr.io/the-ai-project-co/z3rno-postgres:17 .
```

Optional build args:

| Arg | Default | Description |
|---|---|---|
| `AGE_REF` | `master` | Git ref for Apache AGE source (tag or branch) |
| `PGVECTORSCALE_VERSION` | `0.5.1` | Release version of pgvectorscale |

Example override:

```bash
docker build \
    --build-arg AGE_REF=release/PG17/1.5.0 \
    --build-arg PGVECTORSCALE_VERSION=0.5.1 \
    -t ghcr.io/the-ai-project-co/z3rno-postgres:17 \
    .
```

## Published from CI

This image is built and pushed automatically by `.github/workflows/postgres-image.yml` on every push to `main` in `z3rno-core`. Image tags produced:

- `ghcr.io/the-ai-project-co/z3rno-postgres:17` — latest PG17 build
- `ghcr.io/the-ai-project-co/z3rno-postgres:17-{short_sha}` — pinned by commit
- `ghcr.io/the-ai-project-co/z3rno-postgres:17-{date}` — pinned by day

## How the extension setup works

```
first container boot
  │
  │  docker-entrypoint.sh runs initdb on empty data dir
  │    │
  │    └── docker-entrypoint-initdb.d/00-z3rno-preload.sh runs:
  │          ALTER SYSTEM SET shared_preload_libraries = 'vectorscale,age,pg_cron,pgaudit'
  │          ALTER SYSTEM SET cron.database_name = '${POSTGRES_DB}'
  │          ALTER SYSTEM SET pgaudit.log = 'ddl'
  │
  │  docker-entrypoint.sh stops postgres, then restarts it
  │    │
  │    └── new postmaster reads postgresql.auto.conf,
  │        applies the shared_preload_libraries and loads all four extensions
  │
  │  container now serves 5432 with preload libs active
  │
z3rno-server boots, runs Alembic migration 001_create_extensions.py:
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
  CREATE EXTENSION IF NOT EXISTS age;
  CREATE EXTENSION IF NOT EXISTS pg_cron;
  CREATE EXTENSION IF NOT EXISTS pgaudit;
  CREATE EXTENSION IF NOT EXISTS pgcrypto;
  → extensions are now available for queries
```

This separation — image ships the preload config, application ships the CREATE EXTENSION — keeps concerns clean. The image knows nothing about which schemas the application wants; the application knows nothing about postgres startup config.

## Known warnings

- **Apache AGE does not yet ship a tagged PG17 release.** This image compiles AGE from the master branch, which generally works but is not a stable release. When Apache AGE tags a PG17 release (expected 2026), pin `AGE_REF` to that tag and rebuild.
- **`shared_preload_libraries`** is set to `'vectorscale,age,pg_cron,pgaudit'` via `ALTER SYSTEM` on first boot. If you mount a custom `postgresql.conf` that doesn't include these, pg_cron, AGE, pgvectorscale, and pgaudit will silently stop working. Always append to the list, never replace.
- **`cron.database_name` tracks `POSTGRES_DB`.** pg_cron is single-database; the preload script points it at whatever `POSTGRES_DB` you launched the container with. If you need jobs in multiple databases, you'll need to run multiple pg_cron-enabled clusters.

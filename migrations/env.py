"""Alembic environment configuration for z3rno-core.

Migrations run **synchronously** using psycopg (not asyncpg). The DATABASE_URL
env var overrides the alembic.ini placeholder so that no credentials are
committed to source control.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ---------------------------------------------------------------------------
# Alembic Config object — gives access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# Override sqlalchemy.url from the environment if present.
database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# Python logging from the .ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import our models so Base.metadata knows about every table
# ---------------------------------------------------------------------------
from z3rno_core.models import Base  # noqa: E402

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL scripts without a live database connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    from sqlalchemy import event  # noqa: PLC0415

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # v0.20.2 — force public-first search_path on every new connection.
    # Apache AGE (loaded by migration 010) sets
    # ``search_path = ag_catalog, "$user", public`` at the session
    # level; without overriding it on connect, every subsequent
    # unqualified CREATE TABLE lands in ag_catalog instead of public,
    # and the matching ``ALTER TABLE public.x ENABLE ROW LEVEL
    # SECURITY`` lines later in the same migration blow up with
    # ``UndefinedTable``. Migration 015 (distill_jobs) is the famous
    # case. The event fires before any migration runs, so the default
    # is set before alembic's per-migration transactions begin.
    @event.listens_for(connectable, "connect")
    def _set_search_path(dbapi_conn, _conn_record):  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        cur.execute("SET search_path = public, ag_catalog")
        cur.close()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # One transaction per migration so any migration-local
            # ``SET LOCAL search_path = ...`` (e.g. migration 010's
            # AGE setup) doesn't leak forward into the next migration.
            transaction_per_migration=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

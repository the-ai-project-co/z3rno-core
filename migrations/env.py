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
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            # Force search_path to start with public so unqualified DDL
            # lands in the right schema. The z3rno-postgres image
            # pre-installs Apache AGE which prepends `ag_catalog` to
            # the role's search_path; without this override every
            # CREATE TABLE in a fresh DB silently lands in ag_catalog
            # and breaks subsequent ALTER TABLE statements that *do*
            # qualify ``public``. Issued inside the transaction so it
            # piggy-backs the same commit as the migrations themselves.
            from sqlalchemy import text as _sa_text  # noqa: PLC0415

            connection.execute(_sa_text("SET LOCAL search_path = public"))
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

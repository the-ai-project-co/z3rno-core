"""019 - Move misplaced Phase A + B.1 tables from ag_catalog to public.

Apache AGE puts ``ag_catalog`` first on the database's ``search_path``.
Migrations 015 (Phase A) and 016 (Phase B.1) issued unqualified
``CREATE TABLE`` statements, so the resulting tables landed in
``ag_catalog`` rather than ``public``. The role ``z3rno_app`` has no
``USAGE`` on ``ag_catalog`` and thus cannot query its own data when
RLS is exercised under that role — surfaced as
``relation "distill_jobs" does not exist`` in two integration tests.

Production CRUD via the superuser ``z3rno`` happened to work because
the search_path lookup also finds these tables in ``ag_catalog``. But
the moment any code-path runs as ``z3rno_app`` (which is the correct
production posture for RLS enforcement), it fails.

This migration moves the four misplaced tables to ``public`` via
``ALTER TABLE … SET SCHEMA``. Data, indexes, FKs, RLS policies, and
GRANTs are all preserved by the move.

Idempotent: only acts on tables that exist in ``ag_catalog``.
"""

from __future__ import annotations

from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None

# Tables created by Migrations 015 + 016 that landed in the wrong schema.
_MISPLACED = (
    "distill_jobs",
    "entity_provenance",
    "datasets",
    "ingest_jobs",
)


def upgrade() -> None:
    for table in _MISPLACED:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_tables
                    WHERE schemaname = 'ag_catalog' AND tablename = '{table}'
                ) AND NOT EXISTS (
                    SELECT 1 FROM pg_tables
                    WHERE schemaname = 'public' AND tablename = '{table}'
                ) THEN
                    EXECUTE 'ALTER TABLE ag_catalog.{table} SET SCHEMA public';
                END IF;
            END
            $$
        """)


def downgrade() -> None:
    # Reverse: move them back to ag_catalog. Same idempotent guard.
    for table in _MISPLACED:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_tables
                    WHERE schemaname = 'public' AND tablename = '{table}'
                ) AND NOT EXISTS (
                    SELECT 1 FROM pg_tables
                    WHERE schemaname = 'ag_catalog' AND tablename = '{table}'
                ) THEN
                    EXECUTE 'ALTER TABLE public.{table} SET SCHEMA ag_catalog';
                END IF;
            END
            $$
        """)

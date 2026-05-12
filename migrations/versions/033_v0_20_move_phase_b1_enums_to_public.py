"""033 - v0.20.7: Move Phase A/B.1 enum types from ag_catalog to public.

Migration 016 created the ``ingest_job_kind``, ``ingest_job_status``,
and ``distill_job_status`` enum types via unqualified ``CREATE TYPE``
statements. Because Apache AGE puts ``ag_catalog`` first on the
database's ``search_path``, the enums landed in ``ag_catalog`` rather
than ``public``.

Migration 019 moved the *tables* (``ingest_jobs``, ``distill_jobs``,
etc.) back to ``public`` but skipped the enum types — so today's
``INSERT INTO public.ingest_jobs (... CAST(:k AS ingest_job_kind))``
fails with ``UndefinedObjectError: type "ingest_job_kind" does not
exist`` because the unqualified enum lookup hits ``public`` first
(after migration 019 ran a ``SET search_path TO public, ag_catalog``
or after operators removed ag_catalog from search_path).

Surfaced during the v0.20 starter-kit smoke (2026-05-12). See
``z3rno-process-docs/improvements/operator-notes/
V0-20-STARTER-KIT-SMOKE-2026-05-12.md`` Bug C for the full trace.

Idempotent: only moves a type when it's currently in ``ag_catalog``.

Revision ID: 033
Revises: 032
Create Date: 2026-05-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None

_ENUMS = (
    "ingest_job_kind",
    "ingest_job_status",
    "distill_job_status",
)


def upgrade() -> None:
    conn = op.get_bind()
    for enum_name in _ENUMS:
        # Only act if the enum currently lives in ag_catalog. Idempotent
        # across reruns and safe on deploys where 019 created them in
        # the right schema from the start.
        result = conn.execute(
            sa.text(
                """
                SELECT n.nspname
                FROM pg_type t
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE t.typname = :name
                """
            ),
            {"name": enum_name},
        ).fetchone()
        if result is None:
            # Enum doesn't exist anywhere — earlier migration must have
            # been rolled back or never reached. Skip.
            continue
        if result[0] == "ag_catalog":
            conn.execute(sa.text(f"ALTER TYPE ag_catalog.{enum_name} SET SCHEMA public"))


def downgrade() -> None:
    conn = op.get_bind()
    for enum_name in _ENUMS:
        result = conn.execute(
            sa.text(
                """
                SELECT n.nspname
                FROM pg_type t
                JOIN pg_namespace n ON n.oid = t.typnamespace
                WHERE t.typname = :name
                """
            ),
            {"name": enum_name},
        ).fetchone()
        if result is None:
            continue
        if result[0] == "public":
            conn.execute(sa.text(f"ALTER TYPE public.{enum_name} SET SCHEMA ag_catalog"))

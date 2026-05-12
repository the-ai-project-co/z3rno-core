"""034 - v0.21.1: partial index on (org_id, agent_id, user_id) for recall.

Closes slice 21.1 from V0-21-PLAN.md. ``memories.user_id`` is the
per-end-user column populated at store time; until v0.21 it wasn't
exposed as a recall predicate (the field was plumbed through the
RecallRequest contract but ``build_where_clause`` ignored it). Now
that v0.21 wires user_id into the WHERE clause, multi-user agents
will hit this column on every recall they scope to a single user.

A partial index over ``(org_id, agent_id, user_id)`` restricted to
the live row set (``deleted_at IS NULL AND valid_to IS NULL``)
keeps the predicate selective without paying for ancient SCD-2
versions or soft-deleted rows. ``user_id`` last so org+agent stay
the cheapest prefix matches for queries that omit user scoping
entirely.

Idempotent via ``CREATE INDEX IF NOT EXISTS``.

Revision ID: 034
Revises: 033
Create Date: 2026-05-12
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None

_INDEX_NAME = "ix_memories_org_agent_user_live"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_INDEX_NAME}
        ON public.memories (org_id, agent_id, user_id)
        WHERE user_id IS NOT NULL
          AND deleted_at IS NULL
          AND valid_to IS NULL
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS public.{_INDEX_NAME}")

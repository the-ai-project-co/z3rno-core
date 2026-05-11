"""025 - Phase F slice 1: distill_provenance column + 'distill' audit op.

Adds the denormalized provenance blob on ``memories`` so any Memo
born of a Forge run carries its source + model + prompt_hash + audit
correlation id in one JSONB cell. Also extends the
``audit_operation_enum`` with ``distill`` so the audit-chain row that
covers each distilled Memo is distinguishable from the ``store`` op
that wrote the underlying Memory row.

Why a denormalized JSONB instead of relying on the ``entity_provenance``
table (Migration 015): join-free Memo lookups + clean ``WHERE
distill_provenance IS NOT NULL`` queries for the provenance-required
validator. ``entity_provenance`` continues to hold the full row-level
history; this column mirrors the most-recent entry per Memo.

Phase F is **opt-in**: setting ``DISTILL_PROVENANCE_REQUIRED=true``
on the server makes any Forge write without a complete chain raise.
Until then this migration is purely additive — old behavior is
preserved.

Schema-qualified (``public.*``) per the v0.7.x migration CI guard.

Revision ID: 025
Revises: 024
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add the 'distill' value to audit_operation_enum.
    #    Postgres 12+ allows ADD VALUE inside a transaction as long as
    #    the new value isn't *used* in the same transaction — Alembic
    #    runs each migration in its own transaction, so by the time
    #    application code wants to insert audit rows with operation
    #    'distill', the type is already extended + committed.
    op.execute("ALTER TYPE audit_operation_enum ADD VALUE IF NOT EXISTS 'distill'")

    # 2. Denormalized provenance JSONB on memories. NULL means "pre-
    #    Phase-F Memo" — the validator treats NULL as a broken chain
    #    when DISTILL_PROVENANCE_REQUIRED=true.
    op.add_column(
        "memories",
        sa.Column("distill_provenance", sa.dialects.postgresql.JSONB(), nullable=True),
    )

    # 3. Partial index for the validator's hot path: "find Memos
    #    *without* provenance in this org" is a cheap O(log n) lookup
    #    on grounded chains; broken-chain rows are the rare case.
    op.execute(
        "CREATE INDEX ix_memories_org_provenance_missing "
        "ON public.memories (org_id) "
        "WHERE distill_provenance IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.ix_memories_org_provenance_missing")
    op.drop_column("memories", "distill_provenance")
    # Postgres doesn't support removing an enum value cleanly. Leave
    # 'distill' in the enum — it's harmless on the way down and matches
    # the existing convention for enum-extending migrations.

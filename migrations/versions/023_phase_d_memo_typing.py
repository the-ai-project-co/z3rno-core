"""023 - Phase D slice 1: memo_type + ontology_uri columns, feedback table.

Lays the storage foundation for Phase D (Graph Intelligence). Nothing
in this migration is wired into engine paths yet — `refine()` (slice
3), the ontology resolver (slice 4), and the `POST /v1/feedback`
endpoint (slice 2) all consume what this migration creates.

What's added:

  1. ``memories.memo_type TEXT`` — graph-node subtype label. Free-form
     by design: ontologies are open-world, so an enum would fight the
     model. NULL on every existing row; populated by slice 3's
     ``refine`` pipeline and slice 4's ontology grounding.

  2. ``memories.ontology_uri TEXT`` — canonical entity URI when the
     resolver maps a Memo to an OWL concept. Partial index on
     ``(org_id, ontology_uri) WHERE ontology_uri IS NOT NULL`` keeps the
     index narrow — only grounded Memos pay the cost.

  3. ``feedback`` table — captures up/down signals from agents (or
     downstream eval harnesses) on Memos and AGE edges. Slice 3's
     reweight stage drains it. RLS-isolated per the project convention.

Schema-qualified (``public.*``) per the v0.7.x migration CI guard.

Revision ID: 023
Revises: 022
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. memories.memo_type + memories.ontology_uri
    # ------------------------------------------------------------------
    op.add_column(
        "memories",
        sa.Column("memo_type", sa.Text(), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("ontology_uri", sa.Text(), nullable=True),
    )

    # Partial index: only grounded rows are indexed. Keeps the b-tree
    # narrow on tenants that haven't enabled the resolver yet.
    op.execute(
        "CREATE INDEX ix_memories_org_ontology_uri "
        "ON public.memories (org_id, ontology_uri) "
        "WHERE ontology_uri IS NOT NULL"
    )

    # ------------------------------------------------------------------
    # 2. feedback table
    # ------------------------------------------------------------------
    # Exactly one of memory_id / edge_id is required — enforced at the
    # API layer (slice 2) AND with a CHECK constraint here so direct
    # SQL inserts cannot violate the invariant.
    op.create_table(
        "feedback",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        # AGE edges are not relational rows; we identify them by a stable
        # string id (composed by the graph writer). TEXT, not UUID.
        sa.Column("edge_id", sa.Text(), nullable=True),
        sa.Column("signal", sa.SmallInteger(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_feedback_org", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "signal IN (-1, 0, 1)",
            name="ck_feedback_signal_range",
        ),
        sa.CheckConstraint(
            "(memory_id IS NOT NULL)::int + (edge_id IS NOT NULL)::int = 1",
            name="ck_feedback_target_exactly_one",
        ),
    )

    op.create_index(
        "ix_feedback_org_memory",
        "feedback",
        ["org_id", "memory_id"],
        postgresql_where=sa.text("memory_id IS NOT NULL"),
    )
    op.create_index(
        "ix_feedback_org_edge",
        "feedback",
        ["org_id", "edge_id"],
        postgresql_where=sa.text("edge_id IS NOT NULL"),
    )
    op.create_index(
        "ix_feedback_org_created",
        "feedback",
        ["org_id", sa.text("created_at DESC")],
    )

    # ------------------------------------------------------------------
    # 3. RLS — same pattern as Migration 012 / 015 / 016
    # ------------------------------------------------------------------
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.feedback FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON feedback
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON feedback TO z3rno_app';
            END IF;
        END
        $$
    """)


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON feedback")
    op.execute("ALTER TABLE IF EXISTS public.feedback DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.feedback NO FORCE ROW LEVEL SECURITY")

    op.drop_index("ix_feedback_org_created", table_name="feedback")
    op.drop_index("ix_feedback_org_edge", table_name="feedback")
    op.drop_index("ix_feedback_org_memory", table_name="feedback")
    op.drop_table("feedback")

    op.execute("DROP INDEX IF EXISTS public.ix_memories_org_ontology_uri")
    op.drop_column("memories", "ontology_uri")
    op.drop_column("memories", "memo_type")

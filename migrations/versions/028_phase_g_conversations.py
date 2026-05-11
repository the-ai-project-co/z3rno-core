"""028 - Phase G slice 2: conversations + turn-aware Memos.

A ``conversation`` is the framework-agnostic shape of a chat session
between an agent and a user (or another agent). Every Memo created
during the conversation gets two extra columns — ``conversation_id``
and ``turn_index`` — so retrieval can be scoped to the session and
ordered by turn.

Why direct FK columns on ``memories`` instead of a join table: the
expected cardinality is 1:1 (a Memo belongs to exactly one
conversation). A join table would force every recall to JOIN; a
nullable FK + partial index over ``conversation_id IS NOT NULL``
keeps the existing query plans intact for non-conversation Memos.

The ``conversations`` table itself stores session metadata:
``summary_cadence`` (every N turns, the engine emits a SUMMARY Memo),
``turn_count`` (cached counter — the Memo column is authoritative for
ordering), ``last_summary_turn`` (high-water mark for the most recent
summary).

RLS-scoped by ``org_id`` so sessions never leak across tenants.

Revision ID: 028
Revises: 027
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "summary_cadence",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("10"),
        ),
        sa.Column(
            "turn_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "last_summary_turn",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_conversations_org", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "summary_cadence >= 1",
            name="ck_conversations_summary_cadence_positive",
        ),
        sa.CheckConstraint(
            "turn_count >= 0 AND last_summary_turn >= 0",
            name="ck_conversations_counters_nonneg",
        ),
        sa.CheckConstraint(
            "last_summary_turn <= turn_count",
            name="ck_conversations_summary_high_water",
        ),
    )

    op.create_index(
        "ix_conversations_org_agent",
        "conversations",
        ["org_id", "agent_id", "created_at"],
    )
    op.create_index(
        "ix_conversations_org_user",
        "conversations",
        ["org_id", "user_id"],
        postgresql_where=sa.text("user_id IS NOT NULL"),
    )

    # Memos belong to (at most) one conversation. Direct FK columns
    # keep non-conversation Memos zero-cost.
    op.add_column(
        "memories",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("turn_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "memories",
        sa.Column("turn_role", sa.Text(), nullable=True),
    )
    op.create_foreign_key(
        "fk_memories_conversation",
        "memories",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_memories_turn_role",
        "memories",
        "turn_role IS NULL OR turn_role IN ('user','assistant','system','tool','summary')",
    )
    # Partial index — only conversation-bound Memos pay the index cost.
    op.execute(
        "CREATE INDEX ix_memories_conversation_turn "
        "ON public.memories (conversation_id, turn_index) "
        "WHERE conversation_id IS NOT NULL"
    )

    # RLS on conversations
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.conversations FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON conversations
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON conversations TO z3rno_app';
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.ix_memories_conversation_turn")
    op.drop_constraint("ck_memories_turn_role", "memories", type_="check")
    op.drop_constraint("fk_memories_conversation", "memories", type_="foreignkey")
    op.drop_column("memories", "turn_role")
    op.drop_column("memories", "turn_index")
    op.drop_column("memories", "conversation_id")

    op.execute("DROP POLICY IF EXISTS tenant_isolation ON conversations")
    op.execute("ALTER TABLE IF EXISTS public.conversations DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.conversations NO FORCE ROW LEVEL SECURITY")
    op.drop_index("ix_conversations_org_user", table_name="conversations")
    op.drop_index("ix_conversations_org_agent", table_name="conversations")
    op.drop_table("conversations")

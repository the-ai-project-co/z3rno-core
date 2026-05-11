"""029 - Phase G slice 6: usage_counters table.

Per-org daily counters for token spend, embedding calls, LLM calls,
and storage bytes. Aggregating monthly is a SUM over the date range.

Schema:
  * (org_id, period_day, kind) — composite PK so increments are
    one INSERT ... ON CONFLICT DO UPDATE.
  * ``kind`` ∈ {tokens, embeddings, llm_calls, storage_bytes}.
  * ``count`` is monotonic per (org_id, period_day, kind); the engine
    pre-flight check against budgets reads it directly. No SCD-2 here
    — counters are stateless accumulators, not history.

RLS-scoped by ``org_id``.

Revision ID: 029
Revises: 028
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_counters",
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("period_day", sa.Date(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("org_id", "period_day", "kind"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_usage_counters_org", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "kind IN ('tokens','embeddings','llm_calls','storage_bytes')",
            name="ck_usage_counters_kind",
        ),
        sa.CheckConstraint("count >= 0", name="ck_usage_counters_count_nonneg"),
    )
    op.create_index(
        "ix_usage_counters_org_period",
        "usage_counters",
        ["org_id", "period_day"],
    )

    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.usage_counters ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.usage_counters FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON usage_counters
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT, UPDATE ON usage_counters TO z3rno_app';
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON usage_counters")
    op.execute("ALTER TABLE IF EXISTS public.usage_counters DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.usage_counters NO FORCE ROW LEVEL SECURITY")
    op.drop_index("ix_usage_counters_org_period", table_name="usage_counters")
    op.drop_table("usage_counters")

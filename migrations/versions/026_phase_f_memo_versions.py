"""026 - Phase F slice 3: memo_versions table.

SCD-2 on **graph node properties** — distinct from the existing
row-level SCD-2 on ``memories``. Every time a Memo's graph-visible
properties change (memo_type, ontology_uri, content, metadata,
distill_provenance), the writer:

  * closes the prior version (``valid_to = now()``)
  * inserts a fresh row with ``version = prev_version + 1``,
    ``properties = <jsonb snapshot>``, ``valid_from = now()``.

Why a shadow table and not Apache AGE's built-in properties: AGE
nodes don't support SCD-2 natively and the AGE Cypher view is a
mirror. The ``TEMPORAL`` retrieval strategy (Phase C.4) walks this
shadow table to answer ``as_of=…`` queries on graph nodes.

RLS-scoped by ``org_id`` — graph history never leaks across tenants.

Schema-qualified (``public.*``) per the v0.7.x migration CI guard.

Revision ID: 026
Revises: 025
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memo_versions",
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("properties", postgresql.JSONB(), nullable=False),
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("memo_id", "version"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_memo_versions_org", ondelete="CASCADE"
        ),
        sa.CheckConstraint("version >= 1", name="ck_memo_versions_version_positive"),
    )

    # Hot path: "what was this Memo's state at T?" — look up by
    # (memo_id, valid_from <= T < valid_to OR valid_to IS NULL).
    op.create_index(
        "ix_memo_versions_memo_validity",
        "memo_versions",
        ["memo_id", "valid_from", "valid_to"],
    )
    # Tenant-scoped scans (audit / history exports).
    op.create_index(
        "ix_memo_versions_org_memo",
        "memo_versions",
        ["org_id", "memo_id"],
    )
    # Current-version-only index (the common write path: find the
    # currently-open row to close before inserting a new one).
    op.execute(
        "CREATE INDEX ix_memo_versions_org_current "
        "ON public.memo_versions (org_id, memo_id) "
        "WHERE valid_to IS NULL"
    )

    # RLS — mirrors the pattern used by every other tenant-scoped
    # table since Migration 012.
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.memo_versions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.memo_versions FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON memo_versions
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON memo_versions TO z3rno_app';
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON memo_versions")
    op.execute("ALTER TABLE IF EXISTS public.memo_versions DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.memo_versions NO FORCE ROW LEVEL SECURITY")
    op.execute("DROP INDEX IF EXISTS public.ix_memo_versions_org_current")
    op.drop_index("ix_memo_versions_org_memo", table_name="memo_versions")
    op.drop_index("ix_memo_versions_memo_validity", table_name="memo_versions")
    op.drop_table("memo_versions")

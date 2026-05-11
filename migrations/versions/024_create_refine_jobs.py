"""024 - Phase D slice 3: refine_jobs table.

Lifecycle table for the ``refine()`` pipeline. Mirrors the shape of
``distill_jobs`` (Migration 015) and ``ingest_jobs`` (Migration 016)
so operators see one consistent job surface across Forge / Ingest /
Refine.

Why a separate table:
  * The refine pipeline runs both on Celery beat AND on-demand via
    ``POST /v1/refine``. Both paths need to surface state to the
    operator.
  * Refine is the only pipeline that *mutates* existing Memos
    (dedupe). The lifecycle row records what changed for audit.

Counters tracked: ``memos_deduped`` (loser Memos SCD-2-superseded),
``edges_reweighted``, ``edges_pruned``. Slice 4 will add
``edges_inferred`` and ``summaries_written`` without a migration —
the JSONB ``metadata`` column carries the open-ended counters.

Schema-qualified (``public.*``) per the v0.7.x migration CI guard.

Revision ID: 024
Revises: 023
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. refine_job_status enum
    # ------------------------------------------------------------------
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'refine_job_status') THEN
                CREATE TYPE refine_job_status AS ENUM (
                    'queued',
                    'running',
                    'completed',
                    'failed',
                    'cancelled',
                    'rejected'
                );
            END IF;
        END
        $$
    """)

    # ------------------------------------------------------------------
    # 2. refine_jobs table
    # ------------------------------------------------------------------
    op.create_table(
        "refine_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        # NULL ⇒ refine the entire org. Non-NULL ⇒ scope to one dataset.
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Lifecycle
        sa.Column(
            "status",
            postgresql.ENUM(
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
                "rejected",
                name="refine_job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="queued",
        ),
        # Trigger source: 'beat' (Celery beat schedule) or 'api' (manual POST).
        sa.Column("trigger", sa.String(16), nullable=False, server_default="api"),
        # Counters — populated by the pipeline as each stage runs.
        sa.Column("memos_scanned", sa.Integer, nullable=False, server_default="0"),
        sa.Column("memos_deduped", sa.Integer, nullable=False, server_default="0"),
        sa.Column("edges_reweighted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("edges_pruned", sa.Integer, nullable=False, server_default="0"),
        sa.Column("feedback_drained", sa.Integer, nullable=False, server_default="0"),
        # Open-ended counters / future stage metadata (infer, summarize).
        sa.Column(
            "job_metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_refine_jobs_org", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_refine_jobs_dataset",
            ondelete="SET NULL",
        ),
    )

    op.create_index(
        "ix_refine_jobs_org_status_created",
        "refine_jobs",
        ["org_id", "status", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_refine_jobs_org_dataset",
        "refine_jobs",
        ["org_id", "dataset_id"],
    )

    # ------------------------------------------------------------------
    # 3. RLS — same shape as Migration 015 / 016 / 023
    # ------------------------------------------------------------------
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    op.execute("ALTER TABLE public.refine_jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE public.refine_jobs FORCE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON refine_jobs
            FOR ALL
            USING ({policy_expr})
            WITH CHECK ({policy_expr})
    """)
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON refine_jobs TO z3rno_app';
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON refine_jobs")
    op.execute("ALTER TABLE IF EXISTS public.refine_jobs DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE IF EXISTS public.refine_jobs NO FORCE ROW LEVEL SECURITY")

    op.drop_index("ix_refine_jobs_org_dataset", table_name="refine_jobs")
    op.drop_index("ix_refine_jobs_org_status_created", table_name="refine_jobs")
    op.drop_table("refine_jobs")
    op.execute("DROP TYPE IF EXISTS refine_job_status")

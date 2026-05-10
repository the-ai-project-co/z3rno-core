"""016 - Phase B.1 schema: datasets, dataset_id columns, ingest_jobs.

Lands the full Phase B.1 schema in a single migration so the feature
ships atomically.

What's added:

  1. **datasets** — project-level container for memories. Replaces the
     "everything under agent_id" model with org -> dataset -> memory.
     Soft-delete via ``deleted_at``. Unique ``(org_id, name)``.

  2. **dataset_id** column on ``memories``, ``distill_jobs``, and
     ``entity_provenance``. Nullable: ``NULL`` means "no dataset specified"
     and lets every pre-Phase-B-1 row continue to work unchanged. Indexes
     on ``(org_id, dataset_id)`` for fast tenant-scoped dataset lookups.

  3. **ingest_jobs** — one row per ``ingest()`` invocation. Mirrors the
     shape of ``distill_jobs`` (Migration 015) so the operator surface is
     consistent. Records ``kind`` (text / url / file), ``source_uri``,
     ``content_type``, ``file_size``, ``mime_type``, ``memory_ids[]``,
     ``status``, ``error``, lifecycle timestamps. Optional FK to
     ``distill_jobs`` when ``INGEST_AUTO_DISTILL=true`` chained the run.

All three tables follow the existing project conventions:
  * ``org_id`` NOT NULL with ``tenant_isolation`` RLS policy
  * grants to ``z3rno_app`` role (created in Migration 012)
  * indexes on the columns the engine queries by

Phase B.1 is **opt-in**: with ``INGEST_ENABLED=false`` (default), the API
routes are not registered and the worker self-rejects, so this migration
only creates infrastructure — no data flows through the new surface
until an operator explicitly turns it on.

Revision ID: 016
Revises: 015
Create Date: 2026-05-10
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. datasets table
    # ------------------------------------------------------------------
    op.create_table(
        "datasets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
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
        # Soft delete: detaching memories (preserves historical lineage)
        # rather than hard-deleting Memo rows.
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_datasets_org", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("org_id", "name", name="uq_datasets_org_name"),
    )

    op.create_index(
        "ix_datasets_org_created",
        "datasets",
        ["org_id", sa.text("created_at DESC")],
    )

    # ------------------------------------------------------------------
    # 2. dataset_id columns on existing tables (nullable; no backfill)
    # ------------------------------------------------------------------
    for table in ("memories", "distill_jobs", "entity_provenance"):
        op.add_column(
            table,
            sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        )
        op.create_foreign_key(
            f"fk_{table}_dataset",
            table,
            "datasets",
            ["dataset_id"],
            ["id"],
            ondelete="SET NULL",  # detach on dataset hard-delete; never cascade-drop memories
        )
        op.create_index(
            f"ix_{table}_org_dataset",
            table,
            ["org_id", "dataset_id"],
        )

    # ------------------------------------------------------------------
    # 3. ingest_job_kind + ingest_job_status enums
    # ------------------------------------------------------------------
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ingest_job_kind') THEN
                CREATE TYPE ingest_job_kind AS ENUM ('text', 'url', 'file');
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ingest_job_status') THEN
                CREATE TYPE ingest_job_status AS ENUM (
                    'queued',
                    'running',
                    'completed',
                    'failed',
                    'cancelled'
                );
            END IF;
        END
        $$
    """)

    # ------------------------------------------------------------------
    # 4. ingest_jobs table
    # ------------------------------------------------------------------
    op.create_table(
        "ingest_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Input descriptor
        sa.Column(
            "kind",
            postgresql.ENUM("text", "url", "file", name="ingest_job_kind", create_type=False),
            nullable=False,
        ),
        sa.Column("source_uri", sa.Text, nullable=True),
        sa.Column("content_type", sa.String(200), nullable=True),
        sa.Column("file_size", sa.BigInteger, nullable=True),
        sa.Column("filename", sa.String(500), nullable=True),
        # Lifecycle
        sa.Column(
            "status",
            postgresql.ENUM(
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
                name="ingest_job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="queued",
        ),
        # Output — memory_ids written by the IngestPipeline.
        sa.Column(
            "memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        # If INGEST_AUTO_DISTILL=true the pipeline enqueues a Forge run and
        # records the resulting distill_job_id here for downstream tracing.
        sa.Column("distill_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Counters
        sa.Column("memos_written", sa.Integer, nullable=False, server_default="0"),
        # Outcome
        sa.Column("error", sa.Text, nullable=True),
        # Timing
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
            ["org_id"], ["tenants.org_id"], name="fk_ingest_jobs_org", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_ingest_jobs_dataset",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["distill_job_id"],
            ["distill_jobs.id"],
            name="fk_ingest_jobs_distill",
            ondelete="SET NULL",
        ),
    )

    op.create_index(
        "ix_ingest_jobs_org_status_created",
        "ingest_jobs",
        ["org_id", "status", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_ingest_jobs_org_agent_created",
        "ingest_jobs",
        ["org_id", "agent_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_ingest_jobs_org_dataset",
        "ingest_jobs",
        ["org_id", "dataset_id"],
    )

    # ------------------------------------------------------------------
    # 5. RLS — same pattern as Migration 012 / 015
    # ------------------------------------------------------------------
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    for table in ("datasets", "ingest_jobs"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL
                USING ({policy_expr})
                WITH CHECK ({policy_expr})
        """)
        # Grant to z3rno_app iff the role exists (Migration 012).
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'z3rno_app') THEN
                    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO z3rno_app';
                END IF;
            END
            $$
        """)


# ---------------------------------------------------------------------------
# Reverse
# ---------------------------------------------------------------------------


def downgrade() -> None:
    # Drop policies + RLS
    for table in ("ingest_jobs", "datasets"):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE IF EXISTS {table} DISABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE IF EXISTS {table} NO FORCE ROW LEVEL SECURITY")

    # ingest_jobs indexes + table
    op.drop_index("ix_ingest_jobs_org_dataset", table_name="ingest_jobs")
    op.drop_index("ix_ingest_jobs_org_agent_created", table_name="ingest_jobs")
    op.drop_index("ix_ingest_jobs_org_status_created", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")
    op.execute("DROP TYPE IF EXISTS ingest_job_status")
    op.execute("DROP TYPE IF EXISTS ingest_job_kind")

    # dataset_id columns + their indexes/FKs (reverse order of upgrade())
    for table in ("entity_provenance", "distill_jobs", "memories"):
        op.drop_index(f"ix_{table}_org_dataset", table_name=table)
        op.drop_constraint(f"fk_{table}_dataset", table, type_="foreignkey")
        op.drop_column(table, "dataset_id")

    # datasets
    op.drop_index("ix_datasets_org_created", table_name="datasets")
    op.drop_table("datasets")

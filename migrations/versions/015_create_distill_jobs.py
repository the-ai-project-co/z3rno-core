"""015 - Create Forge state tables: distill_jobs + entity_provenance.

Phase A migration. Adds two tenant-scoped tables that record Forge pipeline
state and provenance:

  * **distill_jobs** — one row per ``distill()`` call. Tracks status,
    model, tuning, counters, and final outcome. Surfaces job state to
    ``GET /v1/distill/{job_id}`` and to operators.

  * **entity_provenance** — link table that stamps every distilled Memo
    with the source memory_id, the model that produced it, the prompt
    hash, the chunk char-span, and (eventually) the audit-log chain id.
    Phase A populates these rows from the graph-writer; Phase F will
    enforce ``EXTRACTION_PROVENANCE_REQUIRED``.

Both tables follow the project conventions:
  - org_id NOT NULL (RLS enforcement)
  - RLS enabled with tenant_isolation policy
  - granted to z3rno_app role
  - indexed on the columns the engine queries by

Phase A is **opt-in**: this migration creates infrastructure but no data.
With ``DISTILL_ENABLED=false`` the tables remain empty and existing
behavior is unchanged.

Revision ID: 015
Revises: 014
Create Date: 2026-05-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. distill_job_status enum
    # ------------------------------------------------------------------
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'distill_job_status') THEN
                CREATE TYPE distill_job_status AS ENUM (
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
    # 2. distill_jobs table
    # ------------------------------------------------------------------
    op.create_table(
        "distill_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Inputs — array of memory_ids the orchestrator was asked to distill.
        sa.Column(
            "memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        # Lifecycle
        sa.Column(
            "status",
            postgresql.ENUM(
                "queued",
                "running",
                "completed",
                "failed",
                "cancelled",
                name="distill_job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="queued",
        ),
        # Model + tuning recorded for reproducibility / audit.
        sa.Column("model", sa.String(200), nullable=False, server_default=""),
        sa.Column("chunk_size", sa.Integer, nullable=False, server_default="1024"),
        sa.Column("chunk_overlap", sa.Integer, nullable=False, server_default="128"),
        sa.Column("max_concurrency", sa.Integer, nullable=False, server_default="4"),
        # Counters populated as the pipeline runs.
        sa.Column("chunks_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("chunks_failed", sa.Integer, nullable=False, server_default="0"),
        sa.Column("entities_extracted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("relationships_extracted", sa.Integer, nullable=False, server_default="0"),
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
            ["org_id"], ["tenants.org_id"], name="fk_distill_jobs_org", ondelete="CASCADE"
        ),
    )

    op.create_index(
        "ix_distill_jobs_org_status_created",
        "distill_jobs",
        ["org_id", "status", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_distill_jobs_org_agent_created",
        "distill_jobs",
        ["org_id", "agent_id", sa.text("created_at DESC")],
    )

    # ------------------------------------------------------------------
    # 3. entity_provenance table
    # ------------------------------------------------------------------
    op.create_table(
        "entity_provenance",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        # The new Memo this provenance row describes (the entity / relationship Memo).
        sa.Column("memo_id", postgresql.UUID(as_uuid=True), nullable=False),
        # The source memory the Forge distilled from.
        sa.Column("source_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Which distillation run produced this Memo.
        sa.Column("distill_job_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Model + prompt hash — Phase F will enforce non-null + chain-validate.
        sa.Column("model", sa.String(200), nullable=False, server_default=""),
        sa.Column("prompt_hash", sa.String(64), nullable=False, server_default=""),
        # Chunk-level provenance.
        sa.Column("chunk_index", sa.Integer, nullable=True),
        sa.Column("char_start", sa.Integer, nullable=True),
        sa.Column("char_end", sa.Integer, nullable=True),
        # Optional link into the audit log (populated by graph_writer when set).
        sa.Column("audit_chain_id", sa.BigInteger, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_entity_provenance_org", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["distill_job_id"],
            ["distill_jobs.id"],
            name="fk_entity_provenance_job",
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "ix_entity_provenance_org_memo",
        "entity_provenance",
        ["org_id", "memo_id"],
    )
    op.create_index(
        "ix_entity_provenance_org_source",
        "entity_provenance",
        ["org_id", "source_memory_id"],
    )
    op.create_index(
        "ix_entity_provenance_org_job",
        "entity_provenance",
        ["org_id", "distill_job_id"],
    )

    # ------------------------------------------------------------------
    # 4. RLS — same shape as migration 012
    # ------------------------------------------------------------------
    policy_expr = "org_id = current_setting('app.current_org_id')::uuid"
    for table in ("distill_jobs", "entity_provenance"):
        op.execute(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE public.{table} FORCE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                FOR ALL
                USING ({policy_expr})
                WITH CHECK ({policy_expr})
        """)
        # Grant to the application role iff it exists (created by migration 012).
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
    # Drop policies + RLS first so the tables are unlocked.
    for table in ("entity_provenance", "distill_jobs"):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE IF EXISTS public.{table} DISABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE IF EXISTS public.{table} NO FORCE ROW LEVEL SECURITY")

    # Drop indexes then tables (CASCADE cleans up FKs).
    op.drop_index("ix_entity_provenance_org_job", table_name="entity_provenance")
    op.drop_index("ix_entity_provenance_org_source", table_name="entity_provenance")
    op.drop_index("ix_entity_provenance_org_memo", table_name="entity_provenance")
    op.drop_table("entity_provenance")

    op.drop_index("ix_distill_jobs_org_agent_created", table_name="distill_jobs")
    op.drop_index("ix_distill_jobs_org_status_created", table_name="distill_jobs")
    op.drop_table("distill_jobs")

    op.execute("DROP TYPE IF EXISTS distill_job_status")

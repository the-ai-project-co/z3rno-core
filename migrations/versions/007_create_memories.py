"""007 - Create memories table.

The core table. Uses vector(1536) for embeddings and SCD Type 2 temporal
versioning with valid_from/valid_to.

Revision ID: 007
Revises: 006
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        # agent_id is NOT a FK - memories survive agent deletion
        sa.Column("agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "memory_type",
            postgresql.ENUM(
                "working",
                "episodic",
                "semantic",
                "procedural",
                name="memory_type_enum",
                create_type=False,
            ),
            server_default="episodic",
            nullable=False,
        ),
        # Content
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
        # Vector embedding - raw SQL for pgvector type
        sa.Column("embedding_model", sa.String(100), nullable=True),
        # Intelligence
        sa.Column("importance_score", sa.Float, server_default="0.5", nullable=False),
        sa.Column("recall_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("last_recalled_at", sa.DateTime(timezone=True), nullable=True),
        # Temporal versioning (SCD Type 2)
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        # Lifecycle
        sa.Column("pinned", sa.Boolean, server_default="false", nullable=False),
        sa.Column("ttl_expires_at", sa.DateTime(timezone=True), nullable=True),
        # Soft delete
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        # Poisoning detection (Phase 2, columns shipped now)
        sa.Column("quarantined", sa.Boolean, server_default="false", nullable=False),
        sa.Column("anomaly_score", sa.Float, server_default="0.0", nullable=False),
        # Timestamps
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
        # Constraints
        sa.PrimaryKeyConstraint("id", name="pk_memories"),
        sa.ForeignKeyConstraint(
            ["org_id"], ["tenants.org_id"], name="fk_memories_org_id_tenants", ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "importance_score >= 0 AND importance_score <= 1", name="importance_score_range"
        ),
        sa.CheckConstraint("anomaly_score >= 0 AND anomaly_score <= 1", name="anomaly_score_range"),
        sa.CheckConstraint("recall_count >= 0", name="recall_count_non_negative"),
    )

    # Add the vector column via raw SQL (pgvector type not natively supported by Alembic)
    op.execute(f"ALTER TABLE memories ADD COLUMN embedding vector({EMBEDDING_DIM})")

    # Single-column indexes
    op.create_index("ix_memories_org_id", "memories", ["org_id"])
    op.create_index("ix_memories_agent_id", "memories", ["agent_id"])
    op.create_index("ix_memories_user_id", "memories", ["user_id"])


def downgrade() -> None:
    op.drop_table("memories")

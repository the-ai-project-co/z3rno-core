"""008 - Create memory_relationships table.

Relational mirror of the AGE graph edges for RLS and analytics.

Revision ID: 008
Revises: 007
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_relationships",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "relationship_type",
            postgresql.ENUM(
                "derived_from",
                "contradicts",
                "supports",
                "supersedes",
                "related_to",
                "caused_by",
                name="relationship_type_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("weight", sa.Float, server_default="1.0", nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
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
        sa.PrimaryKeyConstraint("id", name="pk_memory_relationships"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["tenants.org_id"],
            name="fk_memory_relationships_org_id_tenants",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_memory_id"],
            ["memories.id"],
            name="fk_memory_relationships_source_memory_id_memories",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_memory_id"],
            ["memories.id"],
            name="fk_memory_relationships_target_memory_id_memories",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("weight >= 0 AND weight <= 1", name="weight_range"),
        sa.CheckConstraint("source_memory_id != target_memory_id", name="no_self_relationship"),
    )
    op.create_index("ix_memory_relationships_org_id", "memory_relationships", ["org_id"])
    op.create_index(
        "ix_memory_relationships_source_memory_id", "memory_relationships", ["source_memory_id"]
    )
    op.create_index(
        "ix_memory_relationships_target_memory_id", "memory_relationships", ["target_memory_id"]
    )
    op.create_index(
        "ix_memory_relationships_relationship_type", "memory_relationships", ["relationship_type"]
    )


def downgrade() -> None:
    op.drop_table("memory_relationships")

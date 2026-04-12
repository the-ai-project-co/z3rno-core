"""006 - Create lifecycle_policies table.

Per-tenant, per-memory-type retention and decay configuration.

Revision ID: 006
Revises: 005
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lifecycle_policies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
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
            nullable=False,
        ),
        # Retention rules
        sa.Column("max_ttl_days", sa.Integer, nullable=True),
        sa.Column("max_count", sa.Integer, nullable=True),
        sa.Column("min_importance", sa.Float, server_default="0.0", nullable=False),
        # Decay configuration
        sa.Column("decay_enabled", sa.Boolean, server_default="true", nullable=False),
        sa.Column("decay_rate", sa.Float, server_default="0.01", nullable=False),
        sa.Column(
            "decay_curve",
            postgresql.ENUM(
                "exponential", "logarithmic", "step", name="decay_curve_enum", create_type=False
            ),
            server_default="exponential",
            nullable=False,
        ),
        sa.Column("decay_floor", sa.Float, server_default="0.1", nullable=False),
        # Compliance
        sa.Column(
            "retention_policy",
            postgresql.ENUM(
                "standard",
                "gdpr_strict",
                "hipaa",
                "financial",
                name="retention_policy_enum",
                create_type=False,
            ),
            server_default="standard",
            nullable=False,
        ),
        sa.Column("gdpr_auto_delete_days", sa.Integer, nullable=True),
        sa.Column("summarize_before_delete", sa.Boolean, server_default="true", nullable=False),
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
        sa.PrimaryKeyConstraint("id", name="pk_lifecycle_policies"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["tenants.org_id"],
            name="fk_lifecycle_policies_org_id_tenants",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "org_id", "memory_type", name="uq_lifecycle_policies_org_id_memory_type"
        ),
        sa.CheckConstraint(
            "min_importance >= 0 AND min_importance <= 1", name="min_importance_range"
        ),
        sa.CheckConstraint("decay_floor >= 0 AND decay_floor <= 1", name="decay_floor_range"),
        sa.CheckConstraint("decay_rate >= 0", name="decay_rate_non_negative"),
    )
    op.create_index("ix_lifecycle_policies_org_id", "lifecycle_policies", ["org_id"])


def downgrade() -> None:
    op.drop_table("lifecycle_policies")

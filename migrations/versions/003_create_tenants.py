"""003 - Create tenants table.

Root table - no foreign keys. All other tenant-scoped tables FK to
tenants.org_id.

Revision ID: 003
Revises: 002
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column(
            "org_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "plan_tier",
            postgresql.ENUM(
                "community", "pro", "team", "enterprise", name="plan_tier_enum", create_type=False
            ),
            server_default="community",
            nullable=False,
        ),
        sa.Column("settings", postgresql.JSONB, server_default="{}", nullable=False),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("org_id", name="pk_tenants"),
    )


def downgrade() -> None:
    op.drop_table("tenants")

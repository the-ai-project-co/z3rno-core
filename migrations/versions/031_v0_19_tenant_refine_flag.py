"""031 - v0.19.4: per-tenant refine opt-in + last-run-at tracker.

Closes the Phase D deferred B item. The beat scheduler reads
``tenants.refine_enabled`` and rotates work across opted-in tenants
via ``tenants.refine_last_run_at`` (oldest first → fair round-robin).

Revision ID: 031
Revises: 030
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "refine_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "refine_last_run_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    # Cron query — "next N tenants to refine" — uses this index.
    op.execute(
        "CREATE INDEX ix_tenants_refine_due "
        "ON public.tenants (refine_last_run_at NULLS FIRST) "
        "WHERE refine_enabled IS TRUE"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.ix_tenants_refine_due")
    op.drop_column("tenants", "refine_last_run_at")
    op.drop_column("tenants", "refine_enabled")

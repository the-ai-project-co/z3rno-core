"""030 - v0.19.2: per-tenant usage budget overrides.

Adds ``tenants.usage_budget JSONB`` so operators can override the
server-global budget caps per tenant (e.g. give a flagship account
a higher monthly token cap without bumping the whole deploy).

Shape:

  {
    "daily_tokens": 100000,
    "daily_llm_calls": 0,
    "daily_embeddings": 0,
    "monthly_tokens": 2000000,
    "monthly_llm_calls": 0,
    "monthly_embeddings": 0
  }

Zero in any slot means "fall through to the server-global default".
NULL means "no overrides at all" (same fallback path). The engine's
``check_budget`` site reads this row first, then merges with the
``Budgets`` instance built from server env.

Revision ID: 030
Revises: 029
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "usage_budget",
            postgresql.JSONB(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("tenants", "usage_budget")

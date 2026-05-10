"""020 - Add ``warnings`` JSONB array to ``ingest_jobs``.

Non-fatal anomalies during an ingest run — CSV row-cap truncation,
Playwright fallback bailing to static, image OCR partial failure, etc. —
should be surfaced to the operator polling ``GET /v1/ingest/{job_id}``.
``error`` is reserved for fatal terminal failures.

The column defaults to ``'[]'::jsonb``. Each warning is a small object:
``{"code": "csv_truncated", "detail": "stopped at 10000 of N rows"}``.
"""

from __future__ import annotations

from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE public.ingest_jobs "
        "ADD COLUMN warnings JSONB NOT NULL DEFAULT '[]'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE public.ingest_jobs DROP COLUMN IF EXISTS warnings")

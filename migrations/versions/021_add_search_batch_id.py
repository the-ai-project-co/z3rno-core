"""021 - Add ``search_batch_id`` to ``ingest_jobs`` for batch tracking.

``POST /v1/ingest/search`` returns N child ``job_id``s — one per
discovered URL. Without a batch identifier the client has to poll
every child to learn "is the batch done?". This column joins them so
``GET /v1/ingest/search/{batch_id}`` can return aggregate status in
one round-trip.

``search_batch_id`` is nullable: it's only set for jobs spawned by the
search endpoint. Direct ``/v1/ingest`` and ``/v1/ingest/file`` calls
leave it NULL.

Indexed on ``(org_id, search_batch_id)`` so the aggregate query is fast
even on a tenant with many concurrent searches.
"""

from __future__ import annotations

from alembic import op

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE public.ingest_jobs "
        "ADD COLUMN search_batch_id UUID"
    )
    op.execute(
        "CREATE INDEX ix_ingest_jobs_search_batch "
        "ON public.ingest_jobs (org_id, search_batch_id) "
        "WHERE search_batch_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.ix_ingest_jobs_search_batch")
    op.execute("ALTER TABLE public.ingest_jobs DROP COLUMN IF EXISTS search_batch_id")

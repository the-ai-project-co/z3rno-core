"""017 - Add 'awaiting_upload' to ingest_job_status (Phase B.2.1).

Lets the API issue a presigned S3 PUT URL and create the
``ingest_jobs`` row in a pre-queued state. The client uploads to S3,
then calls ``POST /v1/ingest/finalize/{job_id}`` which transitions the
status to ``queued`` and enqueues the worker. Without this status the
direct-to-S3 upload flow can't track the "URL issued, waiting for
client upload" interval.

PostgreSQL ``ALTER TYPE ... ADD VALUE`` cannot run inside a transaction
when there's already a value in flight, so we issue the statement
inside an autocommit block.

Revision ID: 017
Revises: 016
Create Date: 2026-05-10
"""

from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE must run outside a transaction.
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE ingest_job_status ADD VALUE IF NOT EXISTS 'awaiting_upload' BEFORE 'queued'"
        )


def downgrade() -> None:
    # PostgreSQL has no `DROP VALUE` for enums. Real reversal would
    # require recreating the enum, repointing the column, and dropping
    # the old type — overkill for an additive label that does no harm
    # when unused. Leave the value in place.
    pass

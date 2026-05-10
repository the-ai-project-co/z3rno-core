"""022 - Add ``memories.content_tsv`` generated tsvector + GIN index.

Phase C.1 introduces the LEXICAL retrieval strategy — Postgres full-
text search via ``plainto_tsquery`` + ``ts_rank``. That needs a
``tsvector`` representation of each Memo's content; computing it
on-the-fly per query would be linear in corpus size. A
``GENERATED ALWAYS AS … STORED`` column gives us the tsvector once
per row (at insert / SCD-2 version) and the index makes lookup O(log
n) on big tenants.

Schema-qualified (``public.*``) per the v0.7.x migration CI guard.

The column is non-nullable but Postgres handles NULL ``content``
gracefully (``to_tsvector('english', NULL)`` returns an empty
tsvector). Existing rows back-fill automatically when the migration
runs because Postgres recomputes the GENERATED expression for every
row at ADD COLUMN time.
"""

from __future__ import annotations

from alembic import op

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # GENERATED ALWAYS AS ... STORED — recomputed automatically whenever
    # the source column (``content``) changes. The 'english' dictionary
    # gives reasonable stemming + stop-word handling; operators serving
    # other languages can override per-query via plainto_tsquery's
    # config argument at the application layer (no schema change needed).
    op.execute(
        "ALTER TABLE public.memories "
        "ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS "
        "(to_tsvector('english', coalesce(content, ''))) STORED"
    )

    # GIN is the standard index choice for tsvector. CONCURRENTLY isn't
    # available inside the migration's transaction (Alembic wraps each
    # revision in a TX). For the current row counts (~M Memos per
    # tenant) the lock is brief; if production scale demands it,
    # operators can recreate the index CONCURRENTLY out-of-band.
    op.execute(
        "CREATE INDEX ix_memories_content_tsv "
        "ON public.memories USING GIN (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.ix_memories_content_tsv")
    op.execute("ALTER TABLE public.memories DROP COLUMN IF EXISTS content_tsv")

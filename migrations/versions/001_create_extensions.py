"""001 - Enable required PostgreSQL extensions.

Revision ID: 001
Revises:
Create Date: 2026-04-12
"""

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # vectorscale depends on vector - CASCADE ensures it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
    op.execute("CREATE EXTENSION IF NOT EXISTS age")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_cron")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgaudit")
    # AGE requires its catalog on the search_path for Cypher queries.
    # CURRENT_DATABASE() can't be used directly in ALTER DATABASE, so use DO block.
    op.execute("""
        DO $$
        BEGIN
            EXECUTE format(
                'ALTER DATABASE %I SET search_path = ag_catalog, "$user", public',
                current_database()
            );
        END
        $$
    """)


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pgaudit")
    op.execute("DROP EXTENSION IF EXISTS pg_cron")
    op.execute("DROP EXTENSION IF EXISTS age CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vectorscale")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")

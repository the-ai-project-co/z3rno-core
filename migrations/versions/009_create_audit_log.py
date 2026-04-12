"""009 - Create audit_log table.

Append-only, hash-chained operation history. Partitioned monthly on
created_at using PostgreSQL native declarative partitioning.

Note: memory_id is NOT a FK - audit rows must survive memory deletion.

Revision ID: 009
Revises: 008
Create Date: 2026-04-12
"""

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the parent partitioned table via raw SQL because SQLAlchemy/Alembic
    # doesn't natively support PARTITION BY.
    op.execute("""
        CREATE TABLE audit_log (
            id          BIGSERIAL       NOT NULL,
            org_id      UUID            NOT NULL REFERENCES tenants(org_id) ON DELETE CASCADE,
            agent_id    UUID,
            user_id     UUID,
            operation   audit_operation_enum NOT NULL,
            memory_id   UUID,
            memory_type memory_type_enum,
            details     JSONB           NOT NULL DEFAULT '{}',
            prev_hash   BYTEA,
            row_hash    BYTEA           NOT NULL,
            api_key_id  UUID,
            ip_address  INET,
            user_agent  TEXT,
            request_id  VARCHAR(64),
            created_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at)
    """)

    # Create initial partitions: current month + 3 months ahead
    op.execute("""
        CREATE TABLE audit_log_y2026m04 PARTITION OF audit_log
            FOR VALUES FROM ('2026-04-01') TO ('2026-05-01')
    """)
    op.execute("""
        CREATE TABLE audit_log_y2026m05 PARTITION OF audit_log
            FOR VALUES FROM ('2026-05-01') TO ('2026-06-01')
    """)
    op.execute("""
        CREATE TABLE audit_log_y2026m06 PARTITION OF audit_log
            FOR VALUES FROM ('2026-06-01') TO ('2026-07-01')
    """)
    op.execute("""
        CREATE TABLE audit_log_y2026m07 PARTITION OF audit_log
            FOR VALUES FROM ('2026-07-01') TO ('2026-08-01')
    """)

    # Default partition for anything outside the defined ranges
    op.execute("CREATE TABLE audit_log_default PARTITION OF audit_log DEFAULT")

    # Indexes
    op.create_index("ix_audit_log_org_id", "audit_log", ["org_id"])
    op.create_index("ix_audit_log_agent_id", "audit_log", ["agent_id"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_operation", "audit_log", ["operation"])
    op.create_index("ix_audit_log_memory_id", "audit_log", ["memory_id"])
    op.create_index("ix_audit_log_api_key_id", "audit_log", ["api_key_id"])
    # Composite indexes
    op.create_index("ix_audit_log_org_created", "audit_log", ["org_id", "created_at"])
    op.create_index("ix_audit_log_org_memory", "audit_log", ["org_id", "memory_id"])


def downgrade() -> None:
    # Drop partitions first, then parent
    op.execute("DROP TABLE IF EXISTS audit_log_default")
    op.execute("DROP TABLE IF EXISTS audit_log_y2026m07")
    op.execute("DROP TABLE IF EXISTS audit_log_y2026m06")
    op.execute("DROP TABLE IF EXISTS audit_log_y2026m05")
    op.execute("DROP TABLE IF EXISTS audit_log_y2026m04")
    op.execute("DROP TABLE IF EXISTS audit_log CASCADE")

"""018 - Create audit_log_pending: async drain queue for the audit chain.

The audit_log table chains row_hash from prev_hash per-org. Computing the
chain in the request critical path serializes every operation for a given
tenant and was the dominant bottleneck identified in v0.6.0 stress testing
(pool exhaustion at >25 concurrent users on a single org).

This migration adds an unchained staging table. Engine operations now
write to audit_log_pending in parallel; a single-consumer drain task
(`z3rno.audit_drain` in z3rno-server) reads pending rows in id-order,
chains them, and inserts into audit_log under a per-org advisory lock.
Tamper-evidence on audit_log is preserved at rest.

The table is intentionally NOT partitioned and NOT immutable — rows are
deleted from it after they're chained into audit_log.
"""

from __future__ import annotations

from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Explicit schema qualification: Apache AGE puts ``ag_catalog`` first on
    # search_path, so an unqualified CREATE TABLE would land there.
    op.execute("""
        CREATE TABLE public.audit_log_pending (
            id            BIGSERIAL       PRIMARY KEY,
            org_id        UUID            NOT NULL REFERENCES public.tenants(org_id) ON DELETE CASCADE,
            agent_id      UUID,
            user_id       UUID,
            operation     audit_operation_enum NOT NULL,
            memory_id     UUID,
            memory_type   memory_type_enum,
            details       JSONB           NOT NULL DEFAULT '{}',
            api_key_id    UUID,
            ip_address    INET,
            user_agent    TEXT,
            request_id    VARCHAR(64),
            created_at    TIMESTAMPTZ     NOT NULL DEFAULT now()
        )
    """)

    # The drainer scans rows for one org in id-order, so a composite index
    # on (org_id, id) is exactly what it needs.
    op.execute(
        "CREATE INDEX ix_audit_log_pending_org_id "
        "ON public.audit_log_pending (org_id, id)"
    )

    # Tenant isolation: same RLS shape as the rest of the schema.
    op.execute("ALTER TABLE public.audit_log_pending ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON public.audit_log_pending
            USING (org_id::text = current_setting('app.current_org_id', true))
    """)

    # Grants — the application role inserts pending rows; the drainer (also
    # z3rno_app for now) reads, deletes, and writes the chained row into
    # public.audit_log. Sequence USAGE is required so BIGSERIAL works.
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON public.audit_log_pending TO z3rno_app"
    )
    op.execute(
        "GRANT USAGE, SELECT ON SEQUENCE public.audit_log_pending_id_seq TO z3rno_app"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS public.audit_log_pending")

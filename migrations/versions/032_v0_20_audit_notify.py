"""032 - v0.20.2: NOTIFY on audit_log_pending insert.

Replaces the periodic-poll drain trigger with a Postgres NOTIFY so
the drain can react in ~50ms instead of waiting for the next beat
tick. Closes Phase 3 item 6's server-side prereq.

Trigger fires AFTER INSERT on audit_log_pending and dispatches:

    NOTIFY z3rno_audit_pending, '<org_id>'

The payload is the org_id so a multi-org listener can wake the
right drain partition immediately. Empty/unparsable payloads are
ignored by the listener — falls back to a full sweep.

The existing periodic poll stays in place (now at a longer interval,
60s default in the chart) as a fallback for the case where the
LISTEN connection drops silently. Drain semantics are unchanged.

Revision ID: 032
Revises: 031
Create Date: 2026-05-12
"""

from __future__ import annotations

from alembic import op

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None

TRIGGER_FN = """
CREATE OR REPLACE FUNCTION notify_audit_pending()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('z3rno_audit_pending', NEW.org_id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

TRIGGER_BIND = """
DROP TRIGGER IF EXISTS trg_notify_audit_pending ON public.audit_log_pending;
CREATE TRIGGER trg_notify_audit_pending
AFTER INSERT ON public.audit_log_pending
FOR EACH ROW
EXECUTE FUNCTION notify_audit_pending();
"""


def upgrade() -> None:
    op.execute(TRIGGER_FN)
    op.execute(TRIGGER_BIND)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_notify_audit_pending ON public.audit_log_pending")
    op.execute("DROP FUNCTION IF EXISTS notify_audit_pending()")

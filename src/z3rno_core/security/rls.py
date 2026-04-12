"""Row-Level Security context management.

The single enforcement point for tenant isolation. Every database operation
in z3rno-server must call ``set_org_context()`` at the start of a transaction
to activate RLS policies.

Usage (sync, e.g. in tests or seeds)::

    with engine.connect() as conn:
        set_org_context(conn, org_id)
        # ... all queries now filtered to org_id ...

Usage (async, e.g. in z3rno-server middleware)::

    async with async_session() as session:
        conn = await session.connection()
        await conn.run_sync(set_org_context, org_id)
        # ... all queries now filtered to org_id ...

The session variable is ``app.current_org_id`` (locked across all docs).
``SET LOCAL`` scopes the value to the current transaction only, so it
automatically clears on COMMIT/ROLLBACK.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, text


def set_org_context(conn: Connection, org_id: UUID | str) -> None:
    """Set the RLS tenant context for the current transaction.

    Issues ``SET LOCAL app.current_org_id = '<uuid>'`` which scopes
    the value to the current transaction. All RLS policies on
    tenant-scoped tables filter on this variable.

    Args:
        conn: An active SQLAlchemy connection (must be inside a transaction).
        org_id: The tenant's org_id (UUID or string).
    """
    conn.execute(text(f"SET LOCAL app.current_org_id = '{org_id}'"))


def clear_org_context(conn: Connection) -> None:
    """Reset the RLS tenant context.

    Normally unnecessary because ``SET LOCAL`` auto-clears on
    COMMIT/ROLLBACK, but useful for explicit cleanup in tests.
    """
    conn.execute(text("RESET app.current_org_id"))

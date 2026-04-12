"""Multi-tenant security helpers.

Provides:

- ``set_org_context(conn, org_id)`` -- issues ``SET LOCAL app.current_org_id``
  on a database connection so that PostgreSQL Row-Level Security policies
  activate. The single enforcement point for tenant isolation.
- ``clear_org_context(conn)`` -- resets the session variable.
"""

from __future__ import annotations

from z3rno_core.security.rls import clear_org_context, set_org_context

__all__ = [
    "clear_org_context",
    "set_org_context",
]

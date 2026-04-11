"""Multi-tenant security helpers.

Provides:

- ``set_org_context(conn, org_id)`` — issues ``SET LOCAL app.current_org_id``
  on a database connection so that PostgreSQL Row-Level Security policies
  activate. The single enforcement point for tenant isolation.
- API key hashing utilities (BCrypt) used by the ``api_keys`` table.

Populated in Week 1 Thursday.
"""

from __future__ import annotations

__all__: list[str] = []

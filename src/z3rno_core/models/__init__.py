"""SQLAlchemy 2.0 declarative models for the Z3rno memory database.

Schema is the authoritative source documented in ``docs/SCHEMA.md`` and the
Architecture Document §4.2. Every model has an ``org_id`` foreign key for
Row-Level Security tenant isolation.

Models live in this package and are exported below for ergonomic imports::

    from z3rno_core.models import Tenant, Memory, AuditLog

Populated by the Tuesday/Wednesday Week 1 schema work.
"""

from __future__ import annotations

__all__: list[str] = [
    # Filled in as each model lands. Ordered by dependency:
    # "Base",
    # "Tenant",
    # "Agent",
    # "Memory",
    # "MemoryRelationship",
    # "AuditLog",
    # "LifecyclePolicy",
    # "ApiKey",
]

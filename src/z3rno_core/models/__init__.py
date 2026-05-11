"""SQLAlchemy 2.0 declarative models for the Z3rno memory database.

Schema is the authoritative source documented in ``docs/SCHEMA.md`` and the
Architecture Document §4.2. Every model has an ``org_id`` foreign key for
Row-Level Security tenant isolation.

Import patterns::

    from z3rno_core.models import Base, Tenant, Memory, AuditLog

For Alembic env.py::

    from z3rno_core.models import Base

    target_metadata = Base.metadata

Dependency order (matters for table creation):
  1. Tenant            (root, no FKs)
  2. Agent             (FK -> tenants.org_id)
  3. ApiKey            (FK -> tenants.org_id)
  4. LifecyclePolicy   (FK -> tenants.org_id)
  5. Memory            (FK -> tenants.org_id, NO FK to agents per Doc 08)
  6. MemoryRelationship (FK -> tenants.org_id, FK -> memories.id x 2)
  7. AuditLog          (FK -> tenants.org_id, NO FK to memories — must survive
                        memory deletion)
"""

from __future__ import annotations

from z3rno_core.models.agent import Agent
from z3rno_core.models.api_key import ApiKey
from z3rno_core.models.audit_log import AuditLog
from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin
from z3rno_core.models.enums import (
    AuditOperation,
    DecayCurve,
    MemoryType,
    PlanTier,
    RelationshipType,
    RetentionPolicy,
)
from z3rno_core.models.feedback import Feedback
from z3rno_core.models.lifecycle_policy import LifecyclePolicy
from z3rno_core.models.memory import EMBEDDING_DIMENSION, Memory
from z3rno_core.models.memory_relationship import MemoryRelationship
from z3rno_core.models.tenant import Tenant

__all__ = [
    # Constants
    "EMBEDDING_DIMENSION",
    "Agent",
    "ApiKey",
    "AuditLog",
    # Enums
    "AuditOperation",
    # Base + mixins
    "Base",
    "DecayCurve",
    "Feedback",
    "LifecyclePolicy",
    "Memory",
    "MemoryRelationship",
    "MemoryType",
    "OrgScopedMixin",
    "PlanTier",
    "RelationshipType",
    "RetentionPolicy",
    # Models, in dependency order
    "Tenant",
    "TimestampMixin",
]

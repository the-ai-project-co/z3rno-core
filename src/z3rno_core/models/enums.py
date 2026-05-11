"""Python-side enum types matched 1:1 with PostgreSQL native enums.

When SQLAlchemy creates an Enum column with a Python ``Enum`` class as its
type, it generates a matching PostgreSQL ``CREATE TYPE ... AS ENUM`` on
table creation. Adding new values requires an Alembic migration with
``op.execute("ALTER TYPE ... ADD VALUE ...")`` — Python enum changes alone
do NOT migrate the database.

Every enum here is locked to the values documented in Doc 08 §4.2.
"""

from __future__ import annotations

from enum import StrEnum


class MemoryType(StrEnum):
    """The four memory tiers documented in Doc 08 §6.1.

    Each tier has different retention, decay, and retrieval semantics. See
    ``z3rno-process-docs/08-Architecture-Document.md`` §6.1 for details.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class PlanTier(StrEnum):
    """Tenant subscription tier — controls rate limits and feature gates.

    Values match the canonical pricing in Doc 01 §1.9 and Doc 08 §11.4.
    """

    COMMUNITY = "community"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class AuditOperation(StrEnum):
    """Operation type recorded in the append-only audit log.

    The set is extended (compared to a naive store/recall/forget) to support
    poisoning detection workflows (quarantine, pin, unpin) and GDPR
    compliance (export, gdpr_delete).
    """

    STORE = "store"
    RECALL = "recall"
    FORGET = "forget"
    UPDATE = "update"
    QUARANTINE = "quarantine"
    PIN = "pin"
    UNPIN = "unpin"
    EXPORT = "export"
    GDPR_DELETE = "gdpr_delete"
    # Phase F — every Forge-distilled Memo writes a 'distill' audit row
    # carrying its provenance correlation id.
    DISTILL = "distill"


class RelationshipType(StrEnum):
    """Edge label for ``memory_relationships`` (relational mirror of AGE edges).

    Documented in ``docs/research/apache-age-graph-patterns.md`` §2.2.
    """

    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"


class DecayCurve(StrEnum):
    """Importance score decay function family.

    Used by ``lifecycle_policies.decay_curve`` to control how quickly an
    unrecalled memory's importance score decreases over time.
    """

    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEP = "step"


class RetentionPolicy(StrEnum):
    """Compliance-aware retention preset for ``lifecycle_policies``.

    ``standard`` is the default; the others enforce stricter retention
    rules required by their respective regulations.
    """

    STANDARD = "standard"
    GDPR_STRICT = "gdpr_strict"
    HIPAA = "hipaa"
    FINANCIAL = "financial"

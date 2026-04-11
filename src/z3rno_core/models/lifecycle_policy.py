"""``lifecycle_policies`` table — per-tenant memory retention/decay configuration.

Every tenant has zero or more lifecycle policies, scoped per ``memory_type``.
A policy controls TTL expiry, retention caps (max count per agent), decay
curves for the importance score, and compliance retention rules (GDPR/HIPAA/
financial).

Schema reference: Doc 08 §4.2 (``lifecycle_policies`` table).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Enum as SAEnum,
    Float,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin
from z3rno_core.models.enums import DecayCurve, MemoryType, RetentionPolicy


class LifecyclePolicy(Base, OrgScopedMixin, TimestampMixin):
    """Per-tenant, per-memory-type retention and decay configuration."""

    __tablename__ = "lifecycle_policies"
    __table_args__ = (
        # Each tenant can have at most one policy per memory type.
        UniqueConstraint("org_id", "memory_type", name="uq_lifecycle_policies_org_id_memory_type"),
        # Importance score thresholds must be in [0, 1].
        CheckConstraint(
            "min_importance >= 0 AND min_importance <= 1",
            name="min_importance_range",
        ),
        CheckConstraint(
            "decay_floor >= 0 AND decay_floor <= 1",
            name="decay_floor_range",
        ),
        CheckConstraint(
            "decay_rate >= 0",
            name="decay_rate_non_negative",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(
            MemoryType,
            name="memory_type_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Retention rules
    # ------------------------------------------------------------------
    max_ttl_days: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Auto-expire after N days. NULL = no TTL.",
    )

    max_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Maximum memories of this type per agent. NULL = unlimited.",
    )

    min_importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0",
        comment="Auto-forget memories below this importance threshold.",
    )

    # ------------------------------------------------------------------
    # Decay configuration
    # ------------------------------------------------------------------
    decay_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    decay_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.01,
        server_default="0.01",
        comment="Importance decrease per day.",
    )

    decay_curve: Mapped[DecayCurve] = mapped_column(
        SAEnum(
            DecayCurve,
            name="decay_curve_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=DecayCurve.EXPONENTIAL,
        server_default=DecayCurve.EXPONENTIAL.value,
    )

    decay_floor: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.1,
        server_default="0.1",
        comment="Minimum importance score before forgetting.",
    )

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------
    retention_policy: Mapped[RetentionPolicy] = mapped_column(
        SAEnum(
            RetentionPolicy,
            name="retention_policy_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=RetentionPolicy.STANDARD,
        server_default=RetentionPolicy.STANDARD.value,
    )

    gdpr_auto_delete_days: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Auto-delete user data after N days of inactivity (GDPR).",
    )

    summarize_before_delete: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="Summarise into a semantic memory before deletion.",
    )

    def __repr__(self) -> str:
        return (
            f"<LifecyclePolicy org_id={self.org_id} type={self.memory_type.value} "
            f"max_count={self.max_count} decay_rate={self.decay_rate}>"
        )

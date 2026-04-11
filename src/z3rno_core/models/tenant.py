"""``tenants`` table — root of the multi-tenant hierarchy.

Every other tenant-scoped table foreign-keys back to ``tenants.org_id``.
RLS policies on the other tables filter on ``current_setting('app.current_org_id')``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum as SAEnum, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, TimestampMixin
from z3rno_core.models.enums import PlanTier


class Tenant(Base, TimestampMixin):
    """An organization (tenant) of the Z3rno service.

    Schema reference: Doc 08 §4.2 (``tenants`` table).

    The ``settings`` JSONB column holds tenant-specific configuration like
    embedding model overrides and rate limit overrides. The default empty
    dict means "use the application defaults from config.py".
    """

    __tablename__ = "tenants"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    org_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------
    plan_tier: Mapped[PlanTier] = mapped_column(
        SAEnum(
            PlanTier,
            name="plan_tier_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=PlanTier.COMMUNITY,
        server_default=PlanTier.COMMUNITY.value,
    )

    # ------------------------------------------------------------------
    # Configuration — JSONB so per-tenant overrides don't need DDL changes.
    # Documented allowed keys (informational, not enforced at DB layer):
    #   max_memories         (int)        override the plan default
    #   max_agents           (int)        override the plan default
    #   embedding_model      (str)        override the global default
    #   embedding_dimension  (int)        must match the column type (1536 in MVP)
    #   rate_limit_per_min   (int)        override the plan default
    #   enable_graph         (bool)       allow AGE graph operations
    #   enable_temporal      (bool)       allow SCD Type 2 temporal queries
    # ------------------------------------------------------------------
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # ------------------------------------------------------------------
    # Suspension — soft state, not the same as deletion.
    # ------------------------------------------------------------------
    suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<Tenant org_id={self.org_id} name={self.name!r} plan={self.plan_tier.value}>"

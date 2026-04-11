"""``memories`` table — the core table holding every agent memory.

This is the most important table in z3rno-core. Every column matches
Doc 08 §4.2.1 exactly.

Key design points (locked):
  - ``valid_to`` is NULLable; NULL means "currently valid". No 9999-12-31
    sentinel.
  - ``deleted_at`` is a TIMESTAMPTZ for soft delete. No is_deleted /
    is_current boolean columns (currency derived from valid_to IS NULL,
    deletion derived from deleted_at IS NOT NULL).
  - ``recall_count`` and ``last_recalled_at`` (NOT access_count /
    last_accessed_at).
  - ``embedding`` is hardcoded ``vector(1536)`` for OpenAI text-embedding-3-small.
    Multi-dimension support is post-v1 per ADR-001.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin
from z3rno_core.models.enums import MemoryType

# ----------------------------------------------------------------------------
# Embedding dimension is hardcoded per ADR-001 (docs/adr/001-embedding-model.md).
# Bumping this value requires a new ADR + migration to re-embed every memory.
# ----------------------------------------------------------------------------
EMBEDDING_DIMENSION = 1536


class Memory(Base, OrgScopedMixin, TimestampMixin):
    """An agent memory — text content + vector embedding + temporal versioning.

    Schema reference: Doc 08 §4.2.1.
    """

    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint(
            "importance_score >= 0 AND importance_score <= 1",
            name="importance_score_range",
        ),
        CheckConstraint(
            "anomaly_score >= 0 AND anomaly_score <= 1",
            name="anomaly_score_range",
        ),
        CheckConstraint(
            "recall_count >= 0",
            name="recall_count_non_negative",
        ),
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    # ``agent_id`` is a UUID column but NOT a foreign key — see Doc 08 §4.2.1.
    # Reasoning: when a tenant hard-deletes an agent, we want the memories to
    # remain available for compliance/audit until the tenant explicitly forgets
    # them. A FK with ON DELETE CASCADE would force the wrong behaviour.
    agent_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # ``user_id`` is the END USER the agent is serving (not a Z3rno tenant
    # user). Optional because agent-global memories don't belong to a user.
    user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------
    # Memory classification
    # ------------------------------------------------------------------
    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(
            MemoryType,
            name="memory_type_enum",
            create_type=False,  # already created by lifecycle_policy.py
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=MemoryType.EPISODIC,
        server_default=MemoryType.EPISODIC.value,
    )

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional auto-generated summary (populated by the summarisation worker
    # for episodic→semantic transitions). Free-form, no length limit.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tenant-defined tags, source, conversation_id, confidence, etc.
    memory_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # actual column name in the database
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # ------------------------------------------------------------------
    # Vector embedding
    # ------------------------------------------------------------------
    # Hardcoded 1536 dims (ADR-001). Nullable because the embedding is
    # generated asynchronously by the worker for high-throughput writes.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="The model used to generate this embedding (e.g. 'text-embedding-3-small').",
    )

    # ------------------------------------------------------------------
    # Intelligence
    # ------------------------------------------------------------------
    importance_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        server_default="0.5",
    )

    recall_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    last_recalled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Temporal versioning (SCD Type 2)
    # ------------------------------------------------------------------
    # ``valid_from`` defaults to now() on insert. ``valid_to`` is NULL while
    # the row is the currently-valid version. When the memory is updated,
    # the trigger sets ``valid_to = now()`` on the old row and INSERTs a new
    # row with ``valid_to = NULL``.
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Lifecycle controls
    # ------------------------------------------------------------------
    pinned: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Pinned memories never auto-decay or auto-forget.",
    )

    # Explicit per-row TTL — separate from lifecycle_policies.max_ttl_days.
    # Use this for per-memory expiry; use lifecycle_policies for tenant-wide.
    ttl_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Soft delete
    # ------------------------------------------------------------------
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Poisoning detection (Phase 2 features, columns shipped now to avoid
    # a later migration)
    # ------------------------------------------------------------------
    quarantined: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    anomaly_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0",
    )

    def __repr__(self) -> str:
        marker = " [DELETED]" if self.deleted_at else ""
        return f"<Memory id={self.id} type={self.memory_type.value} agent={self.agent_id}{marker}>"

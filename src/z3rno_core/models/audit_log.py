"""``audit_log`` table — append-only, hash-chained operation history.

Tamper-evidence guarantees:
  - Database trigger blocks UPDATE and DELETE (created by Alembic migration
    011, scheduled for Week 1 Thursday alongside RLS).
  - Each row's ``row_hash`` is computed from ``prev_hash || canonical(row)``,
    forming a chain. Verification walks the chain in order and recomputes
    each hash.
  - Hard-delete via GDPR is the only legitimate exception, and even then we
    leave a stub row with ``operation = 'gdpr_delete'`` and the content
    fields nulled out.

The table is partitioned monthly on ``created_at`` for performance — see the
Alembic migration for the partitioning strategy. Partition management runs as
a Celery task that pre-creates partitions 3 months ahead.

NOTE: ``memory_id`` is a UUID column but NOT a foreign key. Audit rows must
survive memory deletion (otherwise we couldn't audit GDPR deletes).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, Enum as SAEnum, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin
from z3rno_core.models.enums import AuditOperation, MemoryType


class AuditLog(Base, OrgScopedMixin, TimestampMixin):
    """Append-only audit trail for every memory operation."""

    __tablename__ = "audit_log"

    # ------------------------------------------------------------------
    # Identity — BIGSERIAL because we expect millions of rows per tenant
    # and UUIDs would balloon storage and slow index scans.
    # ------------------------------------------------------------------
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------
    # Operation context
    # ------------------------------------------------------------------
    # ``agent_id`` and ``user_id`` are NOT FKs — audit rows must survive
    # the deletion of the agent or user they reference.
    agent_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    user_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    operation: Mapped[AuditOperation] = mapped_column(
        SAEnum(
            AuditOperation,
            name="audit_operation_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        index=True,
    )

    # Optional reference to the affected memory. Not a FK — see module docstring.
    memory_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    memory_type: Mapped[MemoryType | None] = mapped_column(
        SAEnum(
            MemoryType,
            name="memory_type_enum",
            create_type=False,  # already created by lifecycle_policy.py
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Operation details
    # ------------------------------------------------------------------
    # ``details`` is the full structured payload for the operation:
    # for store, the input metadata; for recall, the query and result IDs;
    # for forget, the reason; for gdpr_delete, the requested-by user.
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # ------------------------------------------------------------------
    # Hash chain for tamper-evidence
    # ------------------------------------------------------------------
    # ``prev_hash`` is the row_hash of the previous audit_log row in the
    # chain (per org_id). NULL only for the very first row in a tenant.
    # ``row_hash`` = SHA-256(prev_hash || canonical_jsonb(this_row_minus_hashes))
    # Computed application-side in z3rno_core.engine.audit.compute_hash().
    prev_hash: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
    )

    row_hash: Mapped[bytes] = mapped_column(
        LargeBinary,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Request metadata (for security forensics)
    # ------------------------------------------------------------------
    api_key_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        INET,
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    request_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} org_id={self.org_id} op={self.operation.value} "
            f"memory_id={self.memory_id}>"
        )

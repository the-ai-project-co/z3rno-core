"""``feedback`` table — Phase D slice 1.

Captures up / neutral / down signals an agent (or eval harness) leaves
on a Memo or AGE edge. Slice 3's ``refine`` reweight stage drains this
table to update edge weights and Memo importance.

Targets exactly one of ``memory_id`` (a row in ``memories``) or
``edge_id`` (a stable string identifier for an AGE edge). The CHECK
constraint is mirrored at the DB level by Migration 023.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, SmallInteger, Text, func, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin


class Feedback(Base, OrgScopedMixin):
    """A single feedback signal from an agent on a Memo or an edge."""

    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(
            "signal IN (-1, 0, 1)",
            name="ck_feedback_signal_range",
        ),
        CheckConstraint(
            "(memory_id IS NOT NULL)::int + (edge_id IS NOT NULL)::int = 1",
            name="ck_feedback_target_exactly_one",
        ),
        Index(
            "ix_feedback_org_memory",
            "org_id",
            "memory_id",
            postgresql_where=text("memory_id IS NOT NULL"),
        ),
        Index(
            "ix_feedback_org_edge",
            "org_id",
            "edge_id",
            postgresql_where=text("edge_id IS NOT NULL"),
        ),
        Index(
            "ix_feedback_org_created",
            "org_id",
            text("created_at DESC"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    agent_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
    )

    memory_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
    )

    # AGE edges aren't relational; we identify them by a stable string id.
    edge_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    signal: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
    )

    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        target = f"memory={self.memory_id}" if self.memory_id else f"edge={self.edge_id}"
        return f"<Feedback id={self.id} {target} signal={self.signal}>"

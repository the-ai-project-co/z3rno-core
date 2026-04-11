"""``memory_relationships`` table — relational mirror of Apache AGE graph edges.

Why mirror the graph relationally?
  - RLS (Apache AGE vertices have no org_id RLS support — see
    docs/research/apache-age-graph-patterns.md §4.8).
  - Analytics (count edges, group by relationship_type).
  - JOINs with the memories table for hybrid queries.
  - Backup integrity (a single pg_dump captures both representations).

The graph and the relational table are kept in sync inside a single
transaction by ``z3rno_core.engine.store()``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Enum as SAEnum, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin
from z3rno_core.models.enums import RelationshipType


class MemoryRelationship(Base, OrgScopedMixin, TimestampMixin):
    """A typed edge between two memories.

    Mirrors an Apache AGE edge. The graph and relational copies are kept
    in sync transactionally; tests live in ``tests/integration/test_graph_sync.py``.
    """

    __tablename__ = "memory_relationships"
    __table_args__ = (
        CheckConstraint(
            "weight >= 0 AND weight <= 1",
            name="weight_range",
        ),
        CheckConstraint(
            "source_memory_id != target_memory_id",
            name="no_self_relationship",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    source_memory_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    target_memory_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    relationship_type: Mapped[RelationshipType] = mapped_column(
        SAEnum(
            RelationshipType,
            name="relationship_type_enum",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        index=True,
    )

    weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        server_default="1.0",
    )

    relationship_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # actual column name in the database
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryRelationship {self.source_memory_id} "
            f"-[{self.relationship_type.value}]-> {self.target_memory_id}>"
        )

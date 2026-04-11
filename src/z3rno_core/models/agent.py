"""``agents`` table — an AI agent operated by a tenant.

An agent is the entity that stores and recalls memories. Multiple agents
can belong to one tenant and they are isolated from each other only by
the application-layer ``agent_id`` filter, not by RLS (which is at the
``org_id`` level).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from z3rno_core.models.base import Base, OrgScopedMixin, TimestampMixin


class Agent(Base, OrgScopedMixin, TimestampMixin):
    """An AI agent that stores and recalls memories.

    Schema reference: Doc 02 Week 1 Tuesday F1 task (``agents`` table).
    Loose mirror of Doc 08's implicit agent concept — agents don't have
    a dedicated section in §4.2 because they're conceptually a labelling
    dimension on memories, but we materialize them as their own table for
    foreign-key integrity, per-agent statistics, and metadata storage.
    """

    __tablename__ = "agents"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )

    # External-system identifier (e.g. LangChain agent name) so callers can
    # look up Z3rno agents by their own identifier without storing the UUID.
    external_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Free-form metadata: framework, version, description, tags, retention
    # overrides. Tenant-defined; not enforced by the DB.
    agent_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # actual column name in the database
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    def __repr__(self) -> str:
        return f"<Agent id={self.id} org_id={self.org_id} name={self.name!r}>"

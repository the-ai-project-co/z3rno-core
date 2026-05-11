"""Feedback ingestion + decay helpers (Phase D slice 2).

The ``feedback`` table (Migration 023) captures up / neutral / down
signals an agent or eval harness leaves on a Memo or AGE edge. This
module exposes the *write* side used by the ``POST /v1/feedback``
endpoint, plus a stub for the decay computation that slice 3's
``reweight`` stage will drive.

RLS is enforced at the database tier; callers must have set
``app.current_org_id`` on the connection before invoking these
helpers. Every helper is SQL-level — no SA ORM session needed —
matching the shape of :mod:`z3rno_core.ingest.state` and
:mod:`z3rno_core.distill.graph_writer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


async def record_feedback(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    signal: int,
    memory_id: UUID | None = None,
    edge_id: str | None = None,
    reason: str | None = None,
    feedback_id: UUID | None = None,
) -> UUID:
    """Insert one feedback row. Returns the assigned id.

    The exactly-one-of (memory_id XOR edge_id) and signal-range
    invariants are enforced by CHECK constraints in Migration 023;
    this helper performs the same validation at the Python layer so
    callers get a clear ``ValueError`` instead of a Postgres
    ``IntegrityError`` for the easy mistakes.
    """
    if signal not in (-1, 0, 1):
        raise ValueError(f"signal must be -1, 0, or 1; got {signal!r}")
    if (memory_id is None) == (edge_id is None):
        raise ValueError("exactly one of memory_id or edge_id must be provided")

    fid = feedback_id or uuid4()
    await conn.execute(
        text("""
            INSERT INTO public.feedback (
                id, org_id, agent_id, memory_id, edge_id, signal, reason, created_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:memory_id AS uuid),
                :edge_id,
                :signal,
                :reason,
                now()
            )
        """),
        {
            "id": str(fid),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "memory_id": str(memory_id) if memory_id else None,
            "edge_id": edge_id,
            "signal": signal,
            "reason": reason,
        },
    )
    return fid


async def decay_weights(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    decay_factor: float,
) -> int:
    """Apply exponential decay to edge weights. Slice 3 fills this in.

    Returns the number of edges whose weight was updated. For now this
    is a stub so slice 2 can wire imports and tests can assert the
    callable exists without dragging in the AGE-aware reweight code.
    """
    _ = (conn, org_id, decay_factor)  # silence unused-arg until slice 3
    return 0

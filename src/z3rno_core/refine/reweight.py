"""Reweight stage of the Refine pipeline (Phase D slice 3).

Drains the ``feedback`` table, aggregates signals per ``edge_id``, and
updates ``memory_relationships.weight`` accordingly. The relational
``memory_relationships`` row is the source of truth for edge weight
(per ``models/memory_relationship.py`` — AGE is a mirror); slice 5 /
graph-sync will propagate the change to AGE.

Weight update model
-------------------
Per edge, given the aggregated signal mean ``s ∈ [-1, +1]``:

    new = clamp01(old * decay + (1 - decay) * ((s + 1) / 2))

That is: blend the existing weight toward the [0, 1] projection of the
feedback mean, weighted by ``decay``. ``decay`` close to 1 → slow,
stable updates; close to 0 → snap to feedback. The default 0.95 (the
``FEEDBACK_WEIGHT_DECAY`` config knob) gives ~10-cycle half-life.

Idempotent
----------
Feedback rows are **not** deleted by reweight. Subsequent cycles
recompute from scratch, so a missed run never permanently loses
signal. (The decay still progresses each cycle the edge weight is
touched — a stale-but-eventually-running scheduler doesn't drift.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class ReweightResult:
    edges_reweighted: int
    feedback_drained: int


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_new_weight(*, old: float, signal_mean: float, decay: float) -> float:
    """Pure helper — exposed for unit tests.

    ``signal_mean`` lives in [-1, +1]; we project to [0, 1] and EMA-blend.
    """
    target = (signal_mean + 1.0) / 2.0
    return _clamp01(old * decay + (1.0 - decay) * target)


async def run_reweight(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    decay: float,
) -> ReweightResult:
    """Aggregate feedback per edge_id, update ``memory_relationships.weight``."""
    rows = (
        await conn.execute(
            text("""
                SELECT edge_id, AVG(signal)::float, COUNT(*)::int
                FROM public.feedback
                WHERE org_id = CAST(:org_id AS uuid)
                  AND edge_id IS NOT NULL
                GROUP BY edge_id
            """),
            {"org_id": str(org_id)},
        )
    ).fetchall()

    edges_reweighted = 0
    feedback_drained = 0
    for edge_id, signal_mean, count in rows:
        feedback_drained += int(count)
        try:
            edge_uuid = UUID(edge_id)
        except (TypeError, ValueError):
            # Non-UUID edge_id — slice 5 (codegraph) introduces string-id
            # edges that don't map to memory_relationships. Skip silently.
            continue

        result = await conn.execute(
            text("""
                UPDATE public.memory_relationships
                SET weight = LEAST(1.0, GREATEST(0.0,
                        weight * :decay + (1.0 - :decay) * ((:s + 1.0) / 2.0)
                    )),
                    updated_at = now()
                WHERE id = CAST(:edge_id AS uuid)
                  AND org_id = CAST(:org_id AS uuid)
            """),
            {
                "decay": decay,
                "s": float(signal_mean),
                "edge_id": str(edge_uuid),
                "org_id": str(org_id),
            },
        )
        if result.rowcount and result.rowcount > 0:
            edges_reweighted += 1

    return ReweightResult(
        edges_reweighted=edges_reweighted,
        feedback_drained=feedback_drained,
    )

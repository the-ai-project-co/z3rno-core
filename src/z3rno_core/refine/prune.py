"""Prune stage of the Refine pipeline (Phase D slice 3).

Drops ``memory_relationships`` rows whose weight has decayed below a
threshold. The corresponding AGE edges are removed by the graph-sync
layer (Phase A pattern: relational is source of truth, AGE mirrors).

Conservative defaults
---------------------
Pruning user-created edges would be alarming. Slice 3 defaults the
threshold low enough that only edges *explicitly driven down* by
negative feedback get pruned — newly-inferred edges (slice 4) start
at 1.0 and won't trip the bar until they've eaten persistent negative
signal across multiple cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class PruneResult:
    edges_pruned: int


async def run_prune(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    threshold: float,
) -> PruneResult:
    """Delete ``memory_relationships`` rows with weight below threshold."""
    result = await conn.execute(
        text("""
            DELETE FROM public.memory_relationships
            WHERE org_id = CAST(:org_id AS uuid)
              AND weight < :threshold
        """),
        {"org_id": str(org_id), "threshold": threshold},
    )
    return PruneResult(edges_pruned=int(result.rowcount or 0))

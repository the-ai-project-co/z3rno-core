"""z3rno_core.memory_tiers — Phase F slice 4 tier routing.

The four-tier memory model (working / episodic / semantic / procedural)
already exists at the schema level — every Memo carries a ``memory_type``
column from Phase A. This package adds an LLM-augmented router that
picks one or more tiers per query so :func:`recall()` can fan out
across the right tiers instead of treating ``memories`` as a flat bag.

Public surface:

  * :class:`MemoryTierRouter` — heuristic + LLM classifier
  * :class:`TierRouteDecision` — what the router decided + why
  * :func:`route_tiers` — convenience wrapper around the default router
"""

from __future__ import annotations

from z3rno_core.memory_tiers.router import (
    MemoryTierRouter,
    TierRouteDecision,
    route_tiers,
)

__all__ = ["MemoryTierRouter", "TierRouteDecision", "route_tiers"]

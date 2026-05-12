"""Retrieval strategy framework — the Phase C foundation.

Defines the narrow async protocol every strategy implements, plus the
result + response dataclasses and the registry that lets the engine
layer dispatch by name.

Architectural choices are spelled out in
``z3rno-process-docs/improvements/plans/PHASE-C-PLAN.md``. Highlights:

* Strategies own *retrieval*, not *recording*. The engine layer writes
  audit_log and bumps recall_count after fusion / re-rank — strategies
  return candidate results and stay stateless.
* Strategies inherit RLS from the caller's connection. They MUST NOT
  switch DB roles or open new connections.
* The ``**extra`` escape hatch on ``retrieve()`` lets strategy-specific
  kwargs (``max_steps`` for TRACE, ``cypher_dry_run`` for ASK, ``hops``
  for GRAPH) ride through without bloating the ABC.
* ``StrategyResult.score_components`` carries the per-source signals
  (vector, lexical, graph distance, …) so fusion + re-rank stay
  explainable. Consumers that just want a flat list can ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

# ---------------------------------------------------------------------------
# Result + response shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyResult:
    """One candidate from a single retrieval strategy.

    Mirrors the existing ``RecallResult`` shape so the engine layer
    can hand the response straight back to consumers. ``relevance_score``
    is the strategy's own 0..1 normalised score; ``score_components``
    carries the raw per-source signals for fusion / re-rank.
    """

    memory_id: UUID
    content: str
    summary: str | None
    memory_type: str
    importance_score: float
    relevance_score: float
    recall_count: int
    created_at: datetime
    valid_from: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, float] = field(default_factory=dict)
    # Only populated by graph-flavoured strategies (GRAPH, TRIPLET, ASK).
    # Each entry is loosely-typed JSON — node properties, edge metadata,
    # or a Cypher row. VECTOR / LEXICAL leave this empty.
    graph_context: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RecallResponse:
    """Wrapper around the strategy results with provenance metadata.

    Backwards-compat with the pre-Phase-C list return is preserved via
    ``__iter__`` and ``__len__`` — callers that did
    ``for r in recall(...)`` keep working.
    """

    results: list[StrategyResult]
    strategy_used: str
    strategies_considered: list[str]
    reranked: bool
    elapsed_ms: float

    def __iter__(self) -> Iterator[StrategyResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> StrategyResult:
        return self.results[index]


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class RetrievalStrategy(ABC):
    """Narrow async interface for one recall strategy."""

    # Canonical enum name (e.g. ``"VECTOR"``). Used by the registry +
    # response provenance. MUST match the value in the public API enum.
    name: ClassVar[str]

    # Class-level capability flags so the AUTO router + engine can gate
    # without inspecting strategy internals.
    requires_query_embedding: ClassVar[bool] = False
    requires_llm: ClassVar[bool] = False

    @abstractmethod
    async def retrieve(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        agent_id: UUID,
        query: str,
        top_k: int,
        memory_type: str | None = None,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        """Return up to ``top_k`` candidates for ``query``.

        Strategies must:
          * Issue all DB queries through ``conn`` — never open new
            connections, never switch DB roles. RLS context is the
            caller's responsibility.
          * Honour ``memory_type`` and ``filters`` if relevant; strategies
            for which the filter doesn't apply silently ignore it.
          * Normalise ``relevance_score`` to the [0, 1] range. Raw
            per-source signals go in ``score_components``.

        Strategies must NOT:
          * Write to audit_log or update recall_count. The engine layer
            does that once after fusion + re-rank, so the audit row
            records the strategy that actually ran (after AUTO + re-rank).
          * Cache results across calls. Each ``retrieve()`` is fresh.

        Strategy-specific kwargs (e.g. ``hops``, ``max_steps``,
        ``llm_gateway``, ``embedding_provider``) flow through ``**extra``.
        Each strategy extracts what it needs — the ABC stays stable
        as new strategies arrive in later slices.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class UnknownStrategyError(ValueError):
    """Raised when a strategy name isn't registered."""


_REGISTRY: dict[str, type[RetrievalStrategy]] = {}


def register_strategy(cls: type[RetrievalStrategy]) -> type[RetrievalStrategy]:
    """Class decorator that registers a strategy under its ``name``.

    Strategies are discovered eagerly when ``z3rno_core.retrieval`` is
    imported — every concrete strategy lives in
    ``z3rno_core/retrieval/strategies/`` and gets imported by the
    package's ``__init__.py``.
    """
    if not getattr(cls, "name", None):
        raise ValueError(
            f"{cls.__name__} has no 'name' class attribute — set "
            "RetrievalStrategy.name to the canonical enum value."
        )
    key = cls.name.upper()
    _REGISTRY[key] = cls
    return cls


def get_strategy(name: str) -> type[RetrievalStrategy]:
    """Look up a strategy class by canonical name (case-insensitive)."""
    key = name.upper()
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY.keys())) or "(none registered)"
        raise UnknownStrategyError(
            f"unknown strategy: {name!r}; known: {known}"
        )
    return cls


def registered_strategies() -> list[str]:
    """Return canonical names of all registered strategies, sorted."""
    return sorted(_REGISTRY.keys())


def _reset_registry_for_tests(replacements: Iterable[type[RetrievalStrategy]] | None = None) -> None:
    """Test-only: wipe the registry and optionally seed it.

    Tests that exercise the registry's "unknown strategy" path can
    reset it cleanly. Production code never touches this.
    """
    _REGISTRY.clear()
    if replacements:
        for cls in replacements:
            register_strategy(cls)

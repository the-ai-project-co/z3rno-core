"""Phase C — Retrieval intelligence.

Public API for the strategy framework. Strategy *implementations* live
in ``z3rno_core.retrieval.strategies`` and self-register via the
``@register_strategy`` decorator. We deliberately do NOT import the
strategies package from here — that would create a circular import
via ``engine.recall`` (which depends on this package, and is itself
imported by ``engine/__init__.py``).

Callers that want to dispatch by name should either:

  * Use ``z3rno_core.engine.recall`` — its module-level side-effect
    import loads the strategies for you.
  * Or explicitly import ``z3rno_core.retrieval.strategies`` first.

See ``z3rno-process-docs/improvements/PHASE-C-PLAN.md`` for the
phase-wide architecture.
"""

from z3rno_core.retrieval.base import (
    RecallResponse,
    RetrievalStrategy,
    StrategyResult,
    UnknownStrategyError,
    get_strategy,
    register_strategy,
    registered_strategies,
)

__all__ = [
    "RecallResponse",
    "RetrievalStrategy",
    "StrategyResult",
    "UnknownStrategyError",
    "get_strategy",
    "register_strategy",
    "registered_strategies",
]

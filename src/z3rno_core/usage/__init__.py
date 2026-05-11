"""Phase G slice 6 — usage telemetry + budget enforcement.

Public surface:

  * ``record_usage(conn, org_id, kind, count)`` — atomic increment.
  * ``get_usage(conn, org_id, since, until)`` — aggregated counters
    over a date range, returned as ``UsageWindow``.
  * ``check_budget(conn, org_id, budgets)`` — pre-flight: raises
    ``BudgetExceededError`` if any counter is already past its cap.

The Forge + refine pipelines call ``check_budget`` at job start; if
it raises, the job is rejected before any LLM/embedding spend lands.
Hard-stop at the job boundary — ≤5% overrun on the in-flight job is
acceptable per the slice design.
"""

from __future__ import annotations

from z3rno_core.usage.counters import (
    BudgetExceededError,
    Budgets,
    UsageKind,
    UsageWindow,
    check_budget,
    get_usage,
    record_usage,
)

__all__ = [
    "BudgetExceededError",
    "Budgets",
    "UsageKind",
    "UsageWindow",
    "check_budget",
    "get_usage",
    "record_usage",
]

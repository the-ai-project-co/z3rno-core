"""Usage-counter storage + budget pre-flight."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class UsageKind(StrEnum):
    TOKENS = "tokens"
    EMBEDDINGS = "embeddings"
    LLM_CALLS = "llm_calls"
    STORAGE_BYTES = "storage_bytes"


@dataclass(frozen=True)
class Budgets:
    """Per-org caps. Zero means no limit (the default).

    Daily caps reset at UTC midnight; monthly caps reset on the first
    of the month UTC. Operators set these as server-side env or via a
    tenants table extension in v0.19; in v0.18 the server forwards a
    single global config that applies to every org.
    """

    daily_tokens: int = 0
    daily_llm_calls: int = 0
    daily_embeddings: int = 0
    monthly_tokens: int = 0
    monthly_llm_calls: int = 0
    monthly_embeddings: int = 0

    def is_empty(self) -> bool:
        return all(v == 0 for v in self.__dict__.values())


@dataclass(frozen=True)
class UsageWindow:
    """Aggregated counts over a window."""

    org_id: UUID
    since: date
    until: date
    tokens: int = 0
    embeddings: int = 0
    llm_calls: int = 0
    storage_bytes: int = 0
    by_day: dict[date, dict[str, int]] = field(default_factory=dict)


class BudgetExceededError(Exception):
    """Raised by ``check_budget`` when any cap is already past.

    Attributes:
      kind: the counter that tripped (``tokens`` / ``embeddings`` / …)
      window: ``"daily"`` or ``"monthly"``
      used: current count
      cap: the configured cap that was breached
    """

    def __init__(self, *, kind: str, window: str, used: int, cap: int) -> None:
        self.kind = kind
        self.window = window
        self.used = used
        self.cap = cap
        super().__init__(f"{window} {kind} budget exceeded: used={used} cap={cap}")


# ---------------------------------------------------------------------------
# Increment
# ---------------------------------------------------------------------------


async def record_usage(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    kind: UsageKind | str,
    count: int,
    period_day: date | None = None,
) -> None:
    """Atomically increment ``(org_id, period_day, kind).count``.

    No-op for ``count <= 0`` — keeps callers from polluting the table
    with idempotent retries that resolved to zero spend.
    """
    if count <= 0:
        return
    kind_str = kind.value if isinstance(kind, UsageKind) else str(kind)
    day = period_day or datetime.now(UTC).date()
    await conn.execute(
        text("""
            INSERT INTO usage_counters (org_id, period_day, kind, count, updated_at)
            VALUES (
                CAST(:org_id AS uuid),
                CAST(:period_day AS date),
                :kind, :count, now()
            )
            ON CONFLICT (org_id, period_day, kind)
            DO UPDATE SET
                count = usage_counters.count + EXCLUDED.count,
                updated_at = now()
        """),
        {
            "org_id": str(org_id),
            "period_day": day.isoformat(),
            "kind": kind_str,
            "count": int(count),
        },
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_usage(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    since: date,
    until: date,
) -> UsageWindow:
    """Aggregated counts over ``[since, until]`` inclusive.

    Returns a ``UsageWindow`` with both total-per-kind and per-day
    breakdown. Empty days simply have no entry in ``by_day``.
    """
    if until < since:
        raise ValueError("until must be >= since")
    result = await conn.execute(
        text("""
            SELECT period_day, kind, count
            FROM usage_counters
            WHERE org_id = CAST(:org_id AS uuid)
              AND period_day BETWEEN CAST(:since AS date)
                                 AND CAST(:until AS date)
            ORDER BY period_day, kind
        """),
        {
            "org_id": str(org_id),
            "since": since.isoformat(),
            "until": until.isoformat(),
        },
    )
    totals = {"tokens": 0, "embeddings": 0, "llm_calls": 0, "storage_bytes": 0}
    by_day: dict[date, dict[str, int]] = {}
    for row in result.fetchall():
        day = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
        kind_str = str(row[1])
        cnt = int(row[2])
        if kind_str in totals:
            totals[kind_str] += cnt
        by_day.setdefault(day, {})[kind_str] = cnt
    return UsageWindow(
        org_id=org_id,
        since=since,
        until=until,
        tokens=totals["tokens"],
        embeddings=totals["embeddings"],
        llm_calls=totals["llm_calls"],
        storage_bytes=totals["storage_bytes"],
        by_day=by_day,
    )


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _month_start(day: date) -> date:
    return day.replace(day=1)


async def resolve_budgets(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    defaults: Budgets,
) -> Budgets:
    """Merge per-tenant overrides on top of server defaults.

    Reads ``tenants.usage_budget`` (Migration 030). Any non-zero
    field on the tenant row wins over the default; zero / missing
    falls through. NULL row → defaults returned unchanged.

    Cheap by design: one indexed PK fetch. Callers cache the result
    per-request if they need to call check_budget multiple times.
    """
    result = await conn.execute(
        text(
            "SELECT usage_budget FROM tenants "
            "WHERE org_id = CAST(:org_id AS uuid) "
            "LIMIT 1"
        ),
        {"org_id": str(org_id)},
    )
    row = result.fetchone()
    if row is None or row[0] is None:
        return defaults
    override = dict(row[0])

    def _pick(name: str) -> int:
        v = int(override.get(name, 0) or 0)
        return v if v > 0 else getattr(defaults, name, 0)

    return Budgets(
        daily_tokens=_pick("daily_tokens"),
        daily_llm_calls=_pick("daily_llm_calls"),
        daily_embeddings=_pick("daily_embeddings"),
        monthly_tokens=_pick("monthly_tokens"),
        monthly_llm_calls=_pick("monthly_llm_calls"),
        monthly_embeddings=_pick("monthly_embeddings"),
    )


async def check_budget(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    budgets: Budgets,
    today: date | None = None,
) -> None:
    """Pre-flight: raise ``BudgetExceededError`` if any cap is past.

    Both daily and monthly windows are checked. Empty budgets (all
    zeros) is a fast-path no-op — operators with no caps configured
    pay zero per-call overhead.
    """
    if budgets.is_empty():
        return
    day = today or datetime.now(UTC).date()
    month_start = _month_start(day)

    daily = await get_usage(conn, org_id=org_id, since=day, until=day)
    monthly = await get_usage(conn, org_id=org_id, since=month_start, until=day)

    pairs: list[tuple[str, str, int, int]] = [
        ("tokens", "daily", daily.tokens, budgets.daily_tokens),
        ("tokens", "monthly", monthly.tokens, budgets.monthly_tokens),
        ("llm_calls", "daily", daily.llm_calls, budgets.daily_llm_calls),
        ("llm_calls", "monthly", monthly.llm_calls, budgets.monthly_llm_calls),
        ("embeddings", "daily", daily.embeddings, budgets.daily_embeddings),
        ("embeddings", "monthly", monthly.embeddings, budgets.monthly_embeddings),
    ]
    for kind, window, used, cap in pairs:
        if cap > 0 and used >= cap:
            raise BudgetExceededError(kind=kind, window=window, used=used, cap=cap)


__all__ = [
    "BudgetExceededError",
    "Budgets",
    "UsageKind",
    "UsageWindow",
    "check_budget",
    "get_usage",
    "record_usage",
    "resolve_budgets",
    "timedelta",
]

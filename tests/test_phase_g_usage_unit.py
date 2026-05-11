"""Phase G slice 6 — usage counters + budget pre-flight tests."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.usage import (
    BudgetExceededError,
    Budgets,
    UsageKind,
    check_budget,
    get_usage,
    record_usage,
)


def _mock_conn() -> MagicMock:
    c = MagicMock()
    c.execute = AsyncMock()
    return c


def _fake_result(rows: list[tuple[object, ...]] | None = None) -> MagicMock:
    r = MagicMock()
    r.fetchall = MagicMock(return_value=rows or [])
    return r


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_usage_skips_zero_and_negative() -> None:
    conn = _mock_conn()
    await record_usage(conn, org_id=uuid4(), kind=UsageKind.TOKENS, count=0)
    await record_usage(conn, org_id=uuid4(), kind="tokens", count=-5)
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_record_usage_upserts_with_increment() -> None:
    conn = _mock_conn()
    await record_usage(conn, org_id=uuid4(), kind=UsageKind.LLM_CALLS, count=3)
    conn.execute.assert_called_once()
    args, _ = conn.execute.call_args
    sql_text = args[0].text if hasattr(args[0], "text") else str(args[0])
    assert "INSERT INTO usage_counters" in sql_text
    assert "ON CONFLICT" in sql_text
    assert "count + EXCLUDED.count" in sql_text


@pytest.mark.asyncio
async def test_record_usage_accepts_string_kind() -> None:
    """The engine call sites pass raw strings ("tokens") sometimes."""
    conn = _mock_conn()
    await record_usage(conn, org_id=uuid4(), kind="embeddings", count=10)
    conn.execute.assert_called_once()


# ---------------------------------------------------------------------------
# get_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_usage_aggregates_totals_per_kind() -> None:
    conn = _mock_conn()
    d1 = date(2026, 5, 1)
    d2 = date(2026, 5, 2)
    conn.execute.return_value = _fake_result(
        rows=[
            (d1, "tokens", 100),
            (d1, "llm_calls", 5),
            (d2, "tokens", 250),
            (d2, "embeddings", 10),
        ]
    )
    window = await get_usage(conn, org_id=uuid4(), since=d1, until=d2)
    assert window.tokens == 350
    assert window.llm_calls == 5
    assert window.embeddings == 10
    assert window.storage_bytes == 0
    # Per-day shape preserved.
    assert window.by_day[d1]["tokens"] == 100
    assert window.by_day[d2]["embeddings"] == 10


@pytest.mark.asyncio
async def test_get_usage_rejects_inverted_window() -> None:
    conn = _mock_conn()
    with pytest.raises(ValueError, match="until must be"):
        await get_usage(conn, org_id=uuid4(), since=date(2026, 5, 5), until=date(2026, 5, 1))


# ---------------------------------------------------------------------------
# check_budget — pre-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_budget_no_caps_is_fast_path() -> None:
    conn = _mock_conn()
    await check_budget(conn, org_id=uuid4(), budgets=Budgets())
    # No SQL executed when every cap is zero.
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_check_budget_passes_when_under_cap() -> None:
    conn = _mock_conn()
    # Daily lookup + monthly lookup; both return small usage.
    today = date(2026, 5, 15)
    conn.execute.return_value = _fake_result(rows=[(today, "tokens", 100)])
    await check_budget(
        conn,
        org_id=uuid4(),
        budgets=Budgets(daily_tokens=1000, monthly_tokens=10_000),
        today=today,
    )
    # Two queries — one for daily, one for monthly.
    assert conn.execute.await_count == 2


@pytest.mark.asyncio
async def test_check_budget_raises_when_daily_cap_hit() -> None:
    conn = _mock_conn()
    today = date(2026, 5, 15)
    conn.execute.return_value = _fake_result(rows=[(today, "tokens", 5000)])
    with pytest.raises(BudgetExceededError) as exc:
        await check_budget(
            conn,
            org_id=uuid4(),
            budgets=Budgets(daily_tokens=1000),
            today=today,
        )
    assert exc.value.kind == "tokens"
    assert exc.value.window == "daily"
    assert exc.value.used == 5000
    assert exc.value.cap == 1000


@pytest.mark.asyncio
async def test_check_budget_raises_when_monthly_llm_calls_hit() -> None:
    conn = _mock_conn()
    today = date(2026, 5, 15)
    # Daily call returns under cap; monthly returns over cap.
    daily_result = _fake_result(rows=[(today, "llm_calls", 5)])
    monthly_result = _fake_result(rows=[(date(2026, 5, 1), "llm_calls", 1100)])

    conn.execute.side_effect = [daily_result, monthly_result]
    with pytest.raises(BudgetExceededError) as exc:
        await check_budget(
            conn,
            org_id=uuid4(),
            budgets=Budgets(daily_llm_calls=100, monthly_llm_calls=1000),
            today=today,
        )
    assert exc.value.window == "monthly"
    assert exc.value.used == 1100


@pytest.mark.asyncio
async def test_budgets_is_empty_predicate() -> None:
    assert Budgets().is_empty() is True
    assert Budgets(daily_tokens=1).is_empty() is False

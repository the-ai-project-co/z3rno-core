"""v0.19.4 — refine scheduler picker tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.refine.scheduler import (
    mark_refine_dispatched,
    pick_refine_tenants,
)


def _mock_conn(rows: list[tuple[object, ...]] | None = None) -> MagicMock:
    conn = MagicMock()
    result = MagicMock()
    result.fetchall = MagicMock(return_value=rows or [])
    conn.execute = AsyncMock(return_value=result)
    return conn


@pytest.mark.asyncio
async def test_pick_returns_empty_when_no_tenants() -> None:
    conn = _mock_conn(rows=[])
    out = await pick_refine_tenants(conn, limit=10)
    assert out == []


@pytest.mark.asyncio
async def test_pick_returns_limit_zero_skips_query() -> None:
    conn = _mock_conn()
    out = await pick_refine_tenants(conn, limit=0)
    assert out == []
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pick_uses_nulls_first_for_round_robin() -> None:
    """The SQL must order NULLS FIRST so brand-new tenants get
    picked before old ones — fair round-robin baseline."""
    conn = _mock_conn()
    await pick_refine_tenants(conn, limit=5)
    args, _ = conn.execute.call_args
    sql_str = args[0].text if hasattr(args[0], "text") else str(args[0])
    assert "ORDER BY refine_last_run_at NULLS FIRST" in sql_str
    assert "FOR UPDATE SKIP LOCKED" in sql_str
    assert "refine_enabled IS TRUE" in sql_str


@pytest.mark.asyncio
async def test_pick_parses_rows() -> None:
    org_a = uuid4()
    org_b = uuid4()
    now = datetime.now(UTC)
    conn = _mock_conn(
        rows=[
            (org_a, None),
            (org_b, now),
        ]
    )
    out = await pick_refine_tenants(conn, limit=5)
    assert len(out) == 2
    assert out[0].org_id == org_a
    assert out[0].last_run_at is None
    assert out[1].last_run_at == now


@pytest.mark.asyncio
async def test_mark_refine_dispatched_runs_update() -> None:
    conn = _mock_conn()
    await mark_refine_dispatched(conn, org_id=uuid4())
    conn.execute.assert_called_once()
    args, _ = conn.execute.call_args
    sql_str = args[0].text if hasattr(args[0], "text") else str(args[0])
    assert "UPDATE tenants SET refine_last_run_at" in sql_str

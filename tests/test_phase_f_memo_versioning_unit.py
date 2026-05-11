"""Unit tests for Phase F slice 3 — memo_versions read/write helpers.

DB is mocked. Live RLS isolation + actual SCD-2 walk is exercised by
the integration suite when DATABASE_URL is set.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.temporal.memo_versioning import (
    MemoVersion,
    get_memo_at,
    list_memo_versions,
    record_memo_version,
)


class _ConnFixture:
    """Helper to build a MagicMock AsyncConnection that returns the
    supplied sequence of (fetchone, fetchall) results."""

    def __init__(self, *fetchone_values: object) -> None:
        self._fetchone_values = list(fetchone_values)
        self.conn = MagicMock()

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            v = self._fetchone_values.pop(0) if self._fetchone_values else None
            result = MagicMock()
            result.fetchone = lambda: v
            return result

        self.conn.execute = AsyncMock(side_effect=_execute)


# ---------------------------------------------------------------------------
# record_memo_version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_first_version_returns_1() -> None:
    """No prior row → new version is 1."""
    fx = _ConnFixture(None)  # SELECT current → None
    next_version = await record_memo_version(
        fx.conn,
        org_id=uuid4(),
        memo_id=uuid4(),
        properties={"memory_type": "semantic"},
    )
    assert next_version == 1
    # One SELECT + one INSERT (no UPDATE since prior row didn't exist).
    assert fx.conn.execute.await_count == 2


@pytest.mark.asyncio
async def test_record_subsequent_version_increments_and_closes_prior() -> None:
    fx = _ConnFixture((3,))  # SELECT current → version 3
    next_version = await record_memo_version(
        fx.conn,
        org_id=uuid4(),
        memo_id=uuid4(),
        properties={"refine_event": "dedupe"},
    )
    assert next_version == 4
    # SELECT + UPDATE-close + INSERT-new.
    assert fx.conn.execute.await_count == 3


@pytest.mark.asyncio
async def test_record_serialises_properties_as_jsonb() -> None:
    fx = _ConnFixture(None)
    payload = {"nested": {"a": 1}, "list": [1, 2, 3]}
    await record_memo_version(
        fx.conn,
        org_id=uuid4(),
        memo_id=uuid4(),
        properties=payload,
    )
    insert_call = fx.conn.execute.await_args_list[-1]
    params = insert_call.args[1]
    assert json.loads(params["properties"]) == payload


# ---------------------------------------------------------------------------
# get_memo_at
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memo_at_returns_none_when_no_row() -> None:
    fx = _ConnFixture(None)
    out = await get_memo_at(fx.conn, memo_id=uuid4())
    assert out is None


@pytest.mark.asyncio
async def test_get_memo_at_returns_current_when_as_of_omitted() -> None:
    mid = uuid4()
    now = datetime.now(UTC)
    fx = _ConnFixture((mid, 2, {"memory_type": "semantic"}, now, None))
    out = await get_memo_at(fx.conn, memo_id=mid)
    assert isinstance(out, MemoVersion)
    assert out.version == 2
    assert out.properties["memory_type"] == "semantic"
    assert out.valid_to is None


@pytest.mark.asyncio
async def test_get_memo_at_passes_as_of_into_query() -> None:
    mid = uuid4()
    target = datetime.now(UTC) - timedelta(days=7)
    fx = _ConnFixture((mid, 1, {"snapshot": True}, datetime.now(UTC) - timedelta(days=14), None))
    out = await get_memo_at(fx.conn, memo_id=mid, as_of=target)
    assert out is not None
    assert out.version == 1
    call = fx.conn.execute.await_args_list[-1]
    assert call.args[1]["as_of"] == target


# ---------------------------------------------------------------------------
# list_memo_versions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_memo_versions_returns_empty_on_missing_memo() -> None:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
    out = await list_memo_versions(conn, memo_id=uuid4())
    assert out == []


@pytest.mark.asyncio
async def test_list_memo_versions_parses_rows() -> None:
    mid = uuid4()
    now = datetime.now(UTC)
    rows = [
        (mid, 2, {"x": 2}, now, None),
        (mid, 1, {"x": 1}, now - timedelta(hours=1), now),
    ]
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: rows))
    out = await list_memo_versions(conn, memo_id=mid)
    assert [v.version for v in out] == [2, 1]
    assert out[0].properties == {"x": 2}
    assert out[1].valid_to == now

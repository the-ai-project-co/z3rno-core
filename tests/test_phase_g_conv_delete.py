"""v0.19.3 — conversation soft-delete unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.conversations import delete_conversation


def _mock_conn(rowcount: int = 1) -> MagicMock:
    conn = MagicMock()
    result = MagicMock()
    result.rowcount = rowcount
    conn.execute = AsyncMock(return_value=result)
    return conn


@pytest.mark.asyncio
async def test_delete_returns_true_when_row_updated() -> None:
    conn = _mock_conn(rowcount=1)
    ok = await delete_conversation(conn, org_id=uuid4(), conversation_id=uuid4())
    assert ok is True
    args, _ = conn.execute.call_args
    sql = args[0].text if hasattr(args[0], "text") else str(args[0])
    assert "UPDATE conversations" in sql
    assert "deleted_at = now()" in sql
    # Idempotency guard: don't double-delete.
    assert "deleted_at IS NULL" in sql


@pytest.mark.asyncio
async def test_delete_returns_false_when_missing_or_already_deleted() -> None:
    conn = _mock_conn(rowcount=0)
    ok = await delete_conversation(conn, org_id=uuid4(), conversation_id=uuid4())
    assert ok is False

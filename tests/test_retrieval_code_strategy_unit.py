"""CODE retrieval strategy registration + smoke tests (Phase D slice 5).

DB-level traversal is covered by the existing retrieval integration
suite when run with DATABASE_URL. This file asserts:

  * The strategy registers under the name "CODE".
  * It declares the right capability flags (no LLM, no embedding).
  * Empty query returns an empty list without touching the DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.retrieval import registered_strategies
from z3rno_core.retrieval.strategies.code import CodeStrategy


def test_code_strategy_is_registered() -> None:
    assert "CODE" in registered_strategies()


def test_code_strategy_capability_flags() -> None:
    s = CodeStrategy()
    assert s.requires_llm is False
    assert s.requires_query_embedding is False


@pytest.mark.asyncio
async def test_code_strategy_empty_query_returns_empty_list() -> None:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
    s = CodeStrategy()
    out = await s.retrieve(
        conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="",
        top_k=10,
    )
    assert out == []
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_code_strategy_empty_seed_short_circuits() -> None:
    """When the seed SELECT returns no rows, the neighbor CTE is skipped."""
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=MagicMock(fetchall=lambda: []))
    s = CodeStrategy()
    out = await s.retrieve(
        conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="main",
        top_k=10,
    )
    assert out == []
    # Only the seed query ran — neighbor CTE skipped on empty seed.
    assert conn.execute.await_count == 1

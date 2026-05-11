"""v0.19.1 — two-phase recall: SELECTs on read-conn, write-back on write-conn.

Pins the routing contract: when ``write_conn`` is supplied, every
write (counter bump + audit) goes there; reads stay on ``conn``.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.engine.recall import recall

recall_module = sys.modules["z3rno_core.engine.recall"]


@pytest.mark.asyncio
async def test_recall_routes_write_back_to_write_conn(monkeypatch) -> None:
    """When ``write_conn`` is set, the bump + audit land on it and not
    on the read connection."""
    # Patch the strategy so we don't run real retrieval.
    from z3rno_core.retrieval.base import StrategyResult

    fake_result = StrategyResult(
        memory_id=uuid4(),
        content="hi",
        summary=None,
        memory_type="episodic",
        importance_score=0.5,
        relevance_score=0.9,
        recall_count=0,
        created_at=None,  # type: ignore[arg-type]
        valid_from=None,  # type: ignore[arg-type]
        metadata={},
    )

    fake_strategy_inst = MagicMock()
    fake_strategy_inst.retrieve = AsyncMock(return_value=[fake_result])
    fake_strategy_inst.delegated_to = None

    fake_strategy_cls = MagicMock(return_value=fake_strategy_inst)

    async def _stub_audit(*_a, **_k):
        return None

    monkeypatch.setattr(recall_module, "get_strategy", lambda _: fake_strategy_cls)
    monkeypatch.setattr(recall_module, "create_audit_entry", _stub_audit)

    read_conn = MagicMock()
    read_conn.execute = AsyncMock()
    write_conn = MagicMock()
    write_conn.execute = AsyncMock()

    await recall(
        read_conn,
        write_conn=write_conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="hello",
        strategy="VECTOR",
    )

    # The counter-bump UPDATE is the only direct conn.execute call inside
    # recall.py; it must have gone to write_conn, not read_conn.
    update_calls = [
        c for c in write_conn.execute.await_args_list
        if "UPDATE public.memories" in c.args[0].text
        if hasattr(c.args[0], "text")
    ]
    assert len(update_calls) == 1, (
        "expected exactly one UPDATE on write_conn, "
        f"saw {len(update_calls)}"
    )

    # And the read connection should NOT have been used for any UPDATE.
    for call in read_conn.execute.await_args_list:
        sql = call.args[0]
        sql_str = sql.text if hasattr(sql, "text") else str(sql)
        assert "UPDATE public.memories" not in sql_str, (
            f"read_conn was asked to UPDATE: {sql_str}"
        )


@pytest.mark.asyncio
async def test_recall_falls_back_to_single_conn_when_write_conn_none(monkeypatch) -> None:
    """Back-compat: with ``write_conn=None``, the single ``conn`` handles
    both phases (legacy behaviour)."""
    from z3rno_core.retrieval.base import StrategyResult

    fake_result = StrategyResult(
        memory_id=uuid4(),
        content="hi",
        summary=None,
        memory_type="episodic",
        importance_score=0.5,
        relevance_score=0.9,
        recall_count=0,
        created_at=None,  # type: ignore[arg-type]
        valid_from=None,  # type: ignore[arg-type]
        metadata={},
    )
    fake_inst = MagicMock()
    fake_inst.retrieve = AsyncMock(return_value=[fake_result])
    fake_inst.delegated_to = None
    async def _stub_audit(*_a, **_k):
        return None

    monkeypatch.setattr(
        recall_module, "get_strategy", lambda _: MagicMock(return_value=fake_inst)
    )
    monkeypatch.setattr(recall_module, "create_audit_entry", _stub_audit)

    conn = MagicMock()
    conn.execute = AsyncMock()

    await recall(
        conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="hello",
        strategy="VECTOR",
    )

    update_calls = [
        c for c in conn.execute.await_args_list
        if hasattr(c.args[0], "text") and "UPDATE public.memories" in c.args[0].text
    ]
    assert len(update_calls) == 1

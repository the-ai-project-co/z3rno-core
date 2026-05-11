"""v0.20.5 — async counter bump unit tests.

Pins three branches:
  1. No write_engine + bump_counters_async=False → sync UPDATE on write_conn
     (legacy path, byte-identical to v0.19.1).
  2. write_engine + bump_counters_async=True → asyncio.create_task that runs
     UPDATE on a fresh conn; recall response returns without awaiting it.
  3. bump_counters_async=True but no write_engine → falls back to sync.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.engine.recall import recall

recall_module = sys.modules["z3rno_core.engine.recall"]


def _stub_strategy(monkeypatch, results: list[object]) -> MagicMock:
    from z3rno_core.retrieval.base import StrategyResult

    if results and not isinstance(results[0], StrategyResult):
        results = [
            StrategyResult(
                memory_id=uuid4(),
                content="x",
                summary=None,
                memory_type="episodic",
                importance_score=0.5,
                relevance_score=0.5,
                recall_count=0,
                created_at=None,  # type: ignore[arg-type]
                valid_from=None,  # type: ignore[arg-type]
                metadata={},
            )
            for _ in results
        ]
    inst = MagicMock()
    inst.retrieve = AsyncMock(return_value=results)
    inst.delegated_to = None
    monkeypatch.setattr(
        recall_module, "get_strategy", lambda _: MagicMock(return_value=inst)
    )

    async def _stub_audit(*_a, **_k):
        return None

    monkeypatch.setattr(recall_module, "create_audit_entry", _stub_audit)
    return inst


@pytest.mark.asyncio
async def test_sync_path_when_async_flag_off(monkeypatch) -> None:
    """Default behavior: bump UPDATE runs synchronously on write_conn."""
    _stub_strategy(monkeypatch, results=[object()])
    write_conn = MagicMock()
    write_conn.execute = AsyncMock()
    read_conn = MagicMock()
    read_conn.execute = AsyncMock()

    await recall(
        read_conn,
        write_conn=write_conn,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="hi",
        strategy="VECTOR",
    )
    # write_conn.execute should have been awaited at least twice
    # (UPDATE memories + audit row).
    update_calls = [
        c
        for c in write_conn.execute.await_args_list
        if hasattr(c.args[0], "text") and "UPDATE public.memories" in c.args[0].text
    ]
    assert len(update_calls) == 1, "expected sync UPDATE on write_conn"


@pytest.mark.asyncio
async def test_async_path_spawns_task(monkeypatch) -> None:
    """``bump_counters_async=True`` + write_engine → asyncio.create_task."""
    _stub_strategy(monkeypatch, results=[object()])
    write_conn = MagicMock()
    write_conn.execute = AsyncMock()
    read_conn = MagicMock()
    read_conn.execute = AsyncMock()

    # Mock the engine + the bump task — assert it gets called.
    fake_engine = MagicMock()
    bump_called = asyncio.Event()

    async def _stub_bump(eng, ids, org):
        bump_called.set()

    monkeypatch.setattr(recall_module, "_bump_counters_async", _stub_bump)

    await recall(
        read_conn,
        write_conn=write_conn,
        write_engine=fake_engine,
        bump_counters_async=True,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="hi",
        strategy="VECTOR",
    )

    # The bump task was scheduled; wait briefly for it to run.
    try:
        await asyncio.wait_for(bump_called.wait(), timeout=1.0)
    except TimeoutError:
        pytest.fail("async bump task did not run within 1s")

    # No SYNC UPDATE on write_conn — it should be on the bump task only.
    update_calls = [
        c
        for c in write_conn.execute.await_args_list
        if hasattr(c.args[0], "text") and "UPDATE public.memories" in c.args[0].text
    ]
    assert update_calls == [], (
        "sync UPDATE should be skipped when async path is active; "
        f"saw {len(update_calls)}"
    )


@pytest.mark.asyncio
async def test_async_flag_without_engine_falls_back_to_sync(monkeypatch) -> None:
    """If the operator sets bump_counters_async=True but doesn't pass
    write_engine, we keep the sync path (don't silently drop the bump)."""
    _stub_strategy(monkeypatch, results=[object()])
    write_conn = MagicMock()
    write_conn.execute = AsyncMock()
    read_conn = MagicMock()
    read_conn.execute = AsyncMock()

    await recall(
        read_conn,
        write_conn=write_conn,
        write_engine=None,
        bump_counters_async=True,
        org_id=uuid4(),
        agent_id=uuid4(),
        query="hi",
        strategy="VECTOR",
    )

    update_calls = [
        c
        for c in write_conn.execute.await_args_list
        if hasattr(c.args[0], "text") and "UPDATE public.memories" in c.args[0].text
    ]
    assert len(update_calls) == 1, "expected sync fallback when no write_engine"

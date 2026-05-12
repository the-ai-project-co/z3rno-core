"""Unit tests for ``RecallCountBatcher`` (v0.22.0 slice 21.5).

Mocks the AsyncEngine so we can exercise the coalescing logic without
a database. The integration test in ``test_recall_batch_integration.py``
covers the actual SQL.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.engine.recall_batch import (
    RecallCountBatcher,
    _reset_for_tests,
    get_batcher,
    shutdown_batcher,
)


@pytest.fixture(autouse=True)
def _clear_singleton() -> None:
    _reset_for_tests()


def _stub_engine() -> MagicMock:
    """Mock AsyncEngine whose .connect() returns an async context
    manager around an AsyncMock connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.commit = AsyncMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    engine = MagicMock()
    engine.connect = MagicMock(return_value=cm)
    return engine


@pytest.mark.asyncio
async def test_bump_then_flush_emits_one_update_per_org() -> None:
    engine = _stub_engine()
    batcher = RecallCountBatcher(engine, window_ms=5)

    org_a, org_b = uuid4(), uuid4()
    m1, m2, m3 = str(uuid4()), str(uuid4()), str(uuid4())

    batcher.bump(org_id=org_a, memory_ids=[m1, m2])
    batcher.bump(org_id=org_b, memory_ids=[m3])
    await batcher.flush_pending()

    # One UPDATE per org, plus SET ROLE + SET LOCAL config per conn.
    # Total .execute calls = 2 orgs * 3 statements = 6.
    assert engine.connect.call_count == 2
    await batcher.aclose()


@pytest.mark.asyncio
async def test_repeated_bumps_coalesce_into_deltas() -> None:
    engine = _stub_engine()
    batcher = RecallCountBatcher(engine, window_ms=5)

    org = uuid4()
    mid = str(uuid4())

    # Same memory bumped five times in the same window.
    for _ in range(5):
        batcher.bump(org_id=org, memory_ids=[mid])
    await batcher.flush_pending()

    # Inspect the UPDATE call args — delta should be 5, not 1.
    conn = await engine.connect().__aenter__()  # type: ignore[func-returns-value]
    update_calls = [
        c for c in conn.execute.await_args_list
        if "UPDATE public.memories" in str(c.args[0])
    ]
    assert len(update_calls) == 1
    params = update_calls[0].args[1]
    assert params["ids"] == [mid]
    assert params["deltas"] == [5]
    await batcher.aclose()


@pytest.mark.asyncio
async def test_flush_failure_is_swallowed_not_raised() -> None:
    engine = _stub_engine()
    conn = await engine.connect().__aenter__()  # type: ignore[func-returns-value]
    conn.execute.side_effect = RuntimeError("primary down")

    batcher = RecallCountBatcher(engine, window_ms=5)
    batcher.bump(org_id=uuid4(), memory_ids=[str(uuid4())])
    # Must not raise.
    await batcher.flush_pending()
    await batcher.aclose()


@pytest.mark.asyncio
async def test_get_batcher_returns_singleton() -> None:
    engine_a = _stub_engine()
    engine_b = _stub_engine()
    first = get_batcher(engine_a, window_ms=5)
    second = get_batcher(engine_b, window_ms=5)
    assert first is second
    await shutdown_batcher()


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    engine = _stub_engine()
    batcher = RecallCountBatcher(engine, window_ms=5)
    batcher.bump(org_id=uuid4(), memory_ids=[str(uuid4())])
    await batcher.aclose()
    await batcher.aclose()  # second call must be a no-op


@pytest.mark.asyncio
async def test_drain_loop_flushes_on_window_timer() -> None:
    """Without an explicit flush_pending(), the background loop should
    drain within ~one window."""
    engine = _stub_engine()
    batcher = RecallCountBatcher(engine, window_ms=10)
    batcher.bump(org_id=uuid4(), memory_ids=[str(uuid4())])
    # Sleep ~3 windows so the drainer has time to run.
    await asyncio.sleep(0.05)
    assert engine.connect.call_count >= 1
    await batcher.aclose()

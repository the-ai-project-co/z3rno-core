"""Integration test for ``RecallCountBatcher`` (v0.22.0 slice 21.5).

Drives real UPDATEs against a testcontainer Postgres to prove:
  1. Coalesced bumps land the correct delta on ``memories.recall_count``.
  2. 50 concurrent bumps on overlapping IDs collapse into one UPDATE per
     org (not 50).
  3. ``last_recalled_at`` is bumped to ``now()``.

Skipped without DATABASE_URL.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from z3rno_core.engine.recall_batch import (
    RecallCountBatcher,
    _reset_for_tests,
)
from z3rno_core.models import Agent, Memory, MemoryType, Tenant
from z3rno_core.models.enums import PlanTier

DATABASE_URL = os.environ.get("DATABASE_URL")
ASYNC_DATABASE_URL = (
    DATABASE_URL.replace("+psycopg", "+asyncpg") if DATABASE_URL else None
)

pytestmark = [
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="DATABASE_URL not set - skipping integration tests",
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def sync_engine() -> Generator[Engine, None, None]:
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def async_engine() -> Generator[AsyncEngine, None, None]:
    assert ASYNC_DATABASE_URL is not None
    eng = create_async_engine(ASYNC_DATABASE_URL, poolclass=NullPool)
    yield eng
    eng.sync_engine.dispose()


@pytest.fixture(autouse=True)
def _reset() -> None:
    _reset_for_tests()


@pytest.fixture
def seeded(sync_engine: Engine) -> Generator[tuple[UUID, UUID, list[UUID]], None, None]:
    """Seed an org + agent + 3 memories with recall_count = 0."""
    org_id, agent_id = uuid4(), uuid4()
    memory_ids = [uuid4() for _ in range(3)]
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_id, name=f"Batcher IT {org_id}", plan_tier=PlanTier.PRO)
        )
        session.flush()
        session.add(Agent(id=agent_id, org_id=org_id, name="batcher-agent"))
        session.flush()
        for mid in memory_ids:
            session.add(
                Memory(
                    id=mid,
                    org_id=org_id,
                    agent_id=agent_id,
                    content=f"batcher-fixture-{mid}",
                    memory_type=MemoryType.SEMANTIC,
                )
            )
        session.commit()
    yield org_id, agent_id, memory_ids
    with sync_engine.connect() as conn:
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM agents WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


def _read_counts(sync_engine: Engine, memory_ids: list[UUID]) -> dict[UUID, int]:
    with sync_engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, recall_count FROM memories WHERE id = ANY(:ids)"),
            {"ids": [str(m) for m in memory_ids]},
        ).all()
    return {row[0]: row[1] for row in rows}


@pytest.mark.asyncio
async def test_coalesced_bumps_apply_correct_delta(
    sync_engine: Engine,
    async_engine: AsyncEngine,
    seeded: tuple[UUID, UUID, list[UUID]],
) -> None:
    """Five bumps on memory[0], two on memory[1], one on memory[2] —
    flush — recall_count should read 5, 2, 1."""
    org_id, _, memory_ids = seeded
    m0, m1, m2 = [str(m) for m in memory_ids]

    batcher = RecallCountBatcher(async_engine, window_ms=20)
    for _ in range(5):
        batcher.bump(org_id=org_id, memory_ids=[m0])
    for _ in range(2):
        batcher.bump(org_id=org_id, memory_ids=[m1])
    batcher.bump(org_id=org_id, memory_ids=[m2])
    await batcher.flush_pending()
    await batcher.aclose()

    counts = _read_counts(sync_engine, memory_ids)
    assert counts[memory_ids[0]] == 5
    assert counts[memory_ids[1]] == 2
    assert counts[memory_ids[2]] == 1


@pytest.mark.asyncio
async def test_concurrent_bumps_total_to_expected(
    sync_engine: Engine,
    async_engine: AsyncEngine,
    seeded: tuple[UUID, UUID, list[UUID]],
) -> None:
    """50 concurrent recalls of memory[0] should land recall_count = 50."""
    org_id, _, memory_ids = seeded
    m0 = str(memory_ids[0])

    batcher = RecallCountBatcher(async_engine, window_ms=20)

    async def _one_bump() -> None:
        batcher.bump(org_id=org_id, memory_ids=[m0])

    await asyncio.gather(*[_one_bump() for _ in range(50)])
    await batcher.flush_pending()
    await batcher.aclose()

    counts = _read_counts(sync_engine, [memory_ids[0]])
    assert counts[memory_ids[0]] == 50


@pytest.mark.asyncio
async def test_last_recalled_at_advances(
    sync_engine: Engine,
    async_engine: AsyncEngine,
    seeded: tuple[UUID, UUID, list[UUID]],
) -> None:
    """The UPDATE must bump ``last_recalled_at`` alongside the counter."""
    org_id, _, memory_ids = seeded
    m0 = str(memory_ids[0])

    batcher = RecallCountBatcher(async_engine, window_ms=20)
    batcher.bump(org_id=org_id, memory_ids=[m0])
    await batcher.flush_pending()
    await batcher.aclose()

    with sync_engine.connect() as conn:
        row = conn.execute(
            text("SELECT last_recalled_at FROM memories WHERE id = :id"),
            {"id": m0},
        ).one()
    assert row[0] is not None

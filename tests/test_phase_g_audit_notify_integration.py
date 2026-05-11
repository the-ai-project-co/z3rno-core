"""v0.20.2 — end-to-end test for the NOTIFY/LISTEN audit drain trigger.

Asserts:
  1. Migration 032's trigger fires on every INSERT into
     audit_log_pending.
  2. ``listen_for_audit_pending`` wakes its callback within ~1s of
     the insert (latency budget — the chart's new fallback poll is
     60s; LISTEN should beat that by 60x).

Skipped without ``DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from z3rno_core.engine.audit import (
    AUDIT_NOTIFY_CHANNEL,
    enqueue_audit_entry,
    listen_for_audit_pending,
)
from z3rno_core.models import Agent, Tenant
from z3rno_core.models.enums import PlanTier

DATABASE_URL = os.environ.get("DATABASE_URL")
ASYNC_DATABASE_URL = (
    DATABASE_URL.replace("+psycopg", "+asyncpg") if DATABASE_URL else None
)

pytestmark = [
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="DATABASE_URL not set — skipping NOTIFY/LISTEN integration",
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def engine() -> Generator[Engine, None, None]:
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@pytest.fixture
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    assert ASYNC_DATABASE_URL is not None
    eng = create_async_engine(ASYNC_DATABASE_URL, poolclass=NullPool)
    yield eng
    await eng.dispose()


@pytest.fixture
def seed(engine: Engine) -> Generator[dict[str, UUID], None, None]:
    org = uuid4()
    agent = uuid4()
    with Session(engine) as session:
        session.add(Tenant(org_id=org, name="NotifyTest", plan_tier=PlanTier.PRO))
        session.flush()
        session.add(Agent(id=agent, org_id=org, name="a", agent_metadata={}))
        session.commit()
    yield {"org": org, "agent": agent}
    with engine.connect() as conn:
        for tbl in ("audit_log", "audit_log_pending", "agents", "tenants"):
            conn.execute(text(f"DELETE FROM {tbl} WHERE org_id = '{org}'"))
        conn.commit()


# ---------------------------------------------------------------------------
# Trigger fires on INSERT
# ---------------------------------------------------------------------------


def test_notify_channel_constant_matches_migration() -> None:
    """Catch drift between the constant + the trigger payload channel."""
    assert AUDIT_NOTIFY_CHANNEL == "z3rno_audit_pending"


@pytest.mark.asyncio
async def test_listen_receives_notify_within_latency_budget(
    async_engine: AsyncEngine,
    seed: dict[str, UUID],
) -> None:
    """The listener callback must fire within ~1s of an
    enqueue_audit_entry call. 60s would still beat the new poll
    fallback; the budget is set tight to catch regressions."""
    assert ASYNC_DATABASE_URL is not None
    received: list[str | None] = []
    received_event = asyncio.Event()

    async def _on_notify(payload: str | None) -> None:
        received.append(payload)
        received_event.set()

    stop_event = asyncio.Event()

    async def _producer() -> None:
        # Wait a beat to ensure LISTEN is registered before we INSERT.
        await asyncio.sleep(0.1)
        async with async_engine.connect() as conn:
            await conn.execute(
                text(f"SET LOCAL app.current_org_id = '{seed['org']}'")
            )
            await enqueue_audit_entry(
                conn,
                org_id=seed["org"],
                operation="recall",
                agent_id=seed["agent"],
                details={"latency_test": True},
            )
            await conn.commit()

    listener_task = asyncio.create_task(
        listen_for_audit_pending(
            ASYNC_DATABASE_URL,
            on_notify=_on_notify,
            stop_event=stop_event,
        )
    )
    producer_task = asyncio.create_task(_producer())

    try:
        await asyncio.wait_for(received_event.wait(), timeout=2.0)
    finally:
        stop_event.set()
        await asyncio.gather(producer_task, listener_task, return_exceptions=True)

    assert received, "listener never received the NOTIFY"
    # Payload is the org_id as a text string.
    assert received[0] == str(seed["org"])

"""Regression test for the AUTO empty-graph downgrade probe.

The v0.21.3 ``_has_graph_corpus`` probe SQL filtered on a non-existent
``memory_relationships.agent_id`` column, threw, and hit the
conservative ``except → return True`` path — the downgrade never fired
in production. Unit tests didn't catch it because they mocked the
connection. v0.21.4 dropped the agent_id predicate.

This test exercises the probe against the real schema. It would have
failed loudly on 0.21.3.

Skipped without DATABASE_URL.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from z3rno_core.models import Agent, Memory, MemoryRelationship, MemoryType, Tenant
from z3rno_core.models.enums import PlanTier, RelationshipType
from z3rno_core.retrieval.strategies.auto import (
    _GRAPH_CORPUS_CACHE,
    _has_graph_corpus,
)

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


@pytest.fixture
def test_org(sync_engine: Engine) -> Generator[UUID, None, None]:
    org_id = uuid4()
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_id, name=f"AutoProbe IT {org_id}", plan_tier=PlanTier.PRO)
        )
        session.commit()
    yield org_id
    with sync_engine.connect() as conn:
        conn.execute(
            text(f"DELETE FROM memory_relationships WHERE org_id = '{org_id}'")
        )
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM agents WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> AsyncGenerator[None, None]:
    _GRAPH_CORPUS_CACHE.clear()
    yield
    _GRAPH_CORPUS_CACHE.clear()


@pytest.mark.asyncio
async def test_probe_returns_false_on_empty_corpus(
    async_engine: AsyncEngine, test_org: UUID
) -> None:
    """The 0.21.3 regression: probe must execute cleanly and return False,
    not fall through the ``except`` path to a conservative True."""
    async with async_engine.connect() as conn:
        result = await _has_graph_corpus(
            conn, org_id=test_org, agent_id=uuid4()
        )
    assert result is False, (
        "Empty org should report no graph corpus. If this fails with True, "
        "the probe SQL is throwing — check column references against the "
        "real memory_relationships schema."
    )


@pytest.mark.asyncio
async def test_probe_returns_true_when_corpus_exists(
    sync_engine: Engine, async_engine: AsyncEngine, test_org: UUID
) -> None:
    """Inverse case: a single edge in the org's memory_relationships
    flips the probe to True."""
    src_id, tgt_id, agent_id = uuid4(), uuid4(), uuid4()
    with Session(sync_engine) as session:
        session.add(Agent(id=agent_id, org_id=test_org, name="probe-fixture-agent"))
        session.flush()
        for mid in (src_id, tgt_id):
            session.add(
                Memory(
                    id=mid,
                    org_id=test_org,
                    agent_id=agent_id,
                    content=f"probe-fixture-{mid}",
                    memory_type=MemoryType.SEMANTIC,
                )
            )
        session.flush()
        session.add(
            MemoryRelationship(
                org_id=test_org,
                source_memory_id=src_id,
                target_memory_id=tgt_id,
                relationship_type=RelationshipType.RELATED_TO,
                weight=1.0,
            )
        )
        session.commit()

    async with async_engine.connect() as conn:
        result = await _has_graph_corpus(
            conn, org_id=test_org, agent_id=uuid4()
        )
    assert result is True

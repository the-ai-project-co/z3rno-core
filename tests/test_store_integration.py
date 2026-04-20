"""Integration tests for the store() engine function.

Requires a running postgres with migrations applied. Set DATABASE_URL
env var or these tests will be skipped.

Usage:
    DATABASE_URL=postgresql+psycopg://z3rno:z3rno@localhost:5433/z3rno \
        uv run pytest tests/test_store_integration.py -v
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session

from z3rno_core.engine import NoOpEmbeddingProvider, StoreError, store
from z3rno_core.engine.store import StoreResult
from z3rno_core.models import MemoryType, Tenant
from z3rno_core.models.enums import PlanTier

DATABASE_URL = os.environ.get("DATABASE_URL")
ASYNC_DATABASE_URL = DATABASE_URL.replace("+psycopg", "+asyncpg") if DATABASE_URL else None

pytestmark = [
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="DATABASE_URL not set - skipping integration tests",
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def sync_engine() -> Generator[Engine, None, None]:
    """Sync engine for setup/teardown."""
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def async_engine() -> Generator[AsyncEngine, None, None]:
    """Async engine for store() tests (sync fixture to avoid event-loop scope issues)."""
    assert ASYNC_DATABASE_URL is not None
    eng = create_async_engine(ASYNC_DATABASE_URL, pool_size=1, max_overflow=0)
    yield eng
    eng.sync_engine.dispose()


@pytest.fixture(scope="module")
def test_org(sync_engine: Engine) -> Generator[UUID, None, None]:
    """Create a test tenant and return its org_id."""
    org_id = uuid4()
    with Session(sync_engine) as session:
        session.add(Tenant(org_id=org_id, name="Store Test Tenant", plan_tier=PlanTier.PRO))
        session.commit()
    yield org_id
    with sync_engine.connect() as conn:
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM memory_relationships WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


async def _do_store(
    eng: AsyncEngine,
    org_id: UUID,
    **kwargs: Any,
) -> StoreResult:
    """Helper to run store() in a transaction."""
    async with eng.connect() as conn, conn.begin():
        return await store(
            conn,
            org_id=org_id,
            embedding_provider=NoOpEmbeddingProvider(),
            **kwargs,
        )


async def test_store_simple_memory(async_engine: AsyncEngine, test_org: UUID) -> None:
    """Store a basic memory and verify it exists."""
    result = await _do_store(
        async_engine,
        org_id=test_org,
        agent_id=uuid4(),
        content="The sky is blue.",
        memory_type=MemoryType.SEMANTIC,
    )
    assert result.memory_id is not None
    assert result.importance_score == 0.5
    assert result.embedding_model == "noop"


async def test_store_with_metadata(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Store with metadata and verify JSONB is stored correctly."""
    meta = {"source": "test", "tags": ["important", "verified"]}
    result = await _do_store(
        async_engine,
        org_id=test_org,
        agent_id=uuid4(),
        content="Metadata test memory.",
        memory_type=MemoryType.EPISODIC,
        metadata=meta,
    )

    with sync_engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT metadata FROM memories WHERE id = '{result.memory_id}'")
        ).fetchone()
        assert row is not None
        assert row[0]["source"] == "test"
        assert row[0]["tags"] == ["important", "verified"]


async def test_store_with_ttl(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Store with TTL and verify ttl_expires_at is set."""
    result = await _do_store(
        async_engine,
        org_id=test_org,
        agent_id=uuid4(),
        content="Ephemeral working memory.",
        memory_type=MemoryType.WORKING,
        ttl_seconds=3600,
    )

    with sync_engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT ttl_expires_at FROM memories WHERE id = '{result.memory_id}'")
        ).fetchone()
        assert row is not None
        assert row[0] is not None


async def test_store_with_custom_importance(async_engine: AsyncEngine, test_org: UUID) -> None:
    """Store with explicit importance score."""
    result = await _do_store(
        async_engine,
        org_id=test_org,
        agent_id=uuid4(),
        content="High importance memory.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.95,
    )
    assert result.importance_score == 0.95


async def test_store_creates_audit_entry(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Verify that store() creates an audit log entry."""
    result = await _do_store(
        async_engine,
        org_id=test_org,
        agent_id=uuid4(),
        content="Audit test memory.",
        memory_type=MemoryType.EPISODIC,
    )

    with sync_engine.connect() as conn:
        row = conn.execute(
            text(
                f"SELECT operation, memory_id, row_hash FROM audit_log "
                f"WHERE org_id = '{test_org}' AND memory_id = '{result.memory_id}'"
            )
        ).fetchone()
        assert row is not None
        assert row[0] == "store"
        assert row[1] == result.memory_id
        assert row[2] is not None


async def test_store_empty_content_raises(async_engine: AsyncEngine, test_org: UUID) -> None:
    """Store with empty content should raise StoreError."""
    with pytest.raises(StoreError, match="non-empty"):
        await _do_store(
            async_engine,
            org_id=test_org,
            agent_id=uuid4(),
            content="",
            memory_type=MemoryType.EPISODIC,
        )


async def test_store_invalid_importance_raises(async_engine: AsyncEngine, test_org: UUID) -> None:
    """Store with importance > 1.0 should raise StoreError."""
    with pytest.raises(StoreError, match="importance"):
        await _do_store(
            async_engine,
            org_id=test_org,
            agent_id=uuid4(),
            content="Bad importance.",
            memory_type=MemoryType.EPISODIC,
            importance=1.5,
        )

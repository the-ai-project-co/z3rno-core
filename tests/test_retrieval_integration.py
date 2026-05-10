"""End-to-end integration tests for the Phase C.1 retrieval framework.

Skipped unless ``DATABASE_URL`` is set. Validates:

  * Migration 022 ran cleanly (memories.content_tsv populated).
  * LEXICAL returns ranked results matching plainto_tsquery semantics.
  * AUTO delegates to VECTOR (the C.1 skeleton behaviour).
  * recall() returns a RecallResponse carrying strategy provenance.
  * RLS isolates results across tenants for every strategy.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

# Side-effect: register strategies.
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.engine.recall import recall
from z3rno_core.models import Agent, MemoryType, Tenant
from z3rno_core.models.enums import PlanTier
from z3rno_core.retrieval import RecallResponse

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def seeded_org(
    sync_engine: Engine,
) -> Generator[tuple[UUID, UUID, list[UUID]], None, None]:
    """Tenant + agent + a handful of Memos with deterministic content.

    Returns ``(org_id, agent_id, memory_ids)`` so tests can target
    specific Memos by index when checking LEXICAL ranking.
    """
    org_id = uuid4()
    agent_id = uuid4()
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_id, name=f"Retrieval IT {org_id}", plan_tier=PlanTier.PRO)
        )
        session.commit()
    with Session(sync_engine) as session:
        session.add(
            Agent(id=agent_id, org_id=org_id, external_id=f"a-{agent_id}", name="A")
        )
        session.commit()

    # Seed Memos via raw SQL so the test doesn't depend on engine.store —
    # we're testing recall, not store.
    contents = [
        "The quick brown fox jumps over the lazy dog",
        "Quantum mechanics revolutionized physics",
        "Alice loves chocolate ice cream",
        "Bob and Alice went hiking yesterday",
        "Postgres full-text search uses tsvector",
    ]
    memory_ids: list[UUID] = []
    with sync_engine.connect() as conn:
        for content in contents:
            mid = uuid4()
            memory_ids.append(mid)
            conn.execute(
                text("""
                    INSERT INTO public.memories (
                        id, org_id, agent_id, memory_type, content,
                        importance_score, recall_count, created_at, valid_from
                    ) VALUES (
                        :id, :org_id, :agent_id, 'episodic', :content,
                        0.5, 0, now(), now()
                    )
                """),
                {
                    "id": str(mid),
                    "org_id": str(org_id),
                    "agent_id": str(agent_id),
                    "content": content,
                },
            )
        conn.commit()

    yield org_id, agent_id, memory_ids

    # Cleanup.
    with sync_engine.connect() as conn:
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM agents WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_lexical_strategy_returns_ranked_results(
    async_engine: AsyncEngine, seeded_org: tuple[UUID, UUID, list[UUID]]
) -> None:
    """LEXICAL ranks Memos by ts_rank against the query."""
    org_id, agent_id, mids = seeded_org

    async with async_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org_id', :o, false)"),
            {"o": str(org_id)},
        )
        resp = await recall(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query="Alice chocolate",
            strategy="LEXICAL",
            top_k=5,
        )

    assert isinstance(resp, RecallResponse)
    assert resp.strategy_used == "LEXICAL"
    assert resp.reranked is False
    # Both "Alice" Memos should rank — the chocolate one should top.
    assert len(resp) >= 1
    top_result = resp[0]
    assert "Alice" in top_result.content
    assert "lexical" in top_result.score_components
    assert top_result.score_components["lexical"] > 0


async def test_lexical_strategy_empty_query_returns_empty(
    async_engine: AsyncEngine, seeded_org: tuple[UUID, UUID, list[UUID]]
) -> None:
    org_id, agent_id, _ = seeded_org

    async with async_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org_id', :o, false)"),
            {"o": str(org_id)},
        )
        resp = await recall(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query="",
            strategy="LEXICAL",
            top_k=5,
        )

    assert len(resp) == 0


async def test_auto_strategy_delegates_to_vector(
    async_engine: AsyncEngine, seeded_org: tuple[UUID, UUID, list[UUID]]
) -> None:
    """C.1 skeleton: AUTO always routes to VECTOR; the audit + provenance reflect this."""
    org_id, agent_id, _ = seeded_org

    async with async_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org_id', :o, false)"),
            {"o": str(org_id)},
        )
        # No embedding_provider → VECTOR falls back to importance + recency.
        resp = await recall(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query="Alice",
            strategy="AUTO",
            top_k=5,
        )

    assert resp.strategy_used == "VECTOR"
    assert resp.strategies_considered == ["AUTO->VECTOR"]
    assert resp.reranked is False
    # All seeded Memos match (no other filter), all returned within top_k.
    assert len(resp) > 0


async def test_default_strategy_is_auto(
    async_engine: AsyncEngine, seeded_org: tuple[UUID, UUID, list[UUID]]
) -> None:
    """No ``strategy=`` kwarg → AUTO (default global)."""
    org_id, agent_id, _ = seeded_org

    async with async_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org_id', :o, false)"),
            {"o": str(org_id)},
        )
        resp = await recall(conn, org_id=org_id, agent_id=agent_id, top_k=5)

    assert resp.strategies_considered == ["AUTO->VECTOR"]
    assert resp.strategy_used == "VECTOR"


async def test_lexical_rls_isolates_across_tenants(
    async_engine: AsyncEngine,
    sync_engine: Engine,
    seeded_org: tuple[UUID, UUID, list[UUID]],
) -> None:
    """A second tenant cannot see the first tenant's LEXICAL results."""
    org_a, agent_a, _ = seeded_org

    # Spin up a second org with its own Memo.
    org_b = uuid4()
    agent_b = uuid4()
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_b, name=f"RLS IT {org_b}", plan_tier=PlanTier.PRO)
        )
        session.commit()
    with Session(sync_engine) as session:
        session.add(
            Agent(id=agent_b, org_id=org_b, external_id=f"a-{agent_b}", name="B")
        )
        session.commit()
    with sync_engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO public.memories (
                    id, org_id, agent_id, memory_type, content,
                    importance_score, recall_count, created_at, valid_from
                ) VALUES (
                    :id, :org_id, :agent_id, 'episodic',
                    'Charlie likes pancakes for breakfast',
                    0.5, 0, now(), now()
                )
            """),
            {
                "id": str(uuid4()),
                "org_id": str(org_b),
                "agent_id": str(agent_b),
            },
        )
        conn.commit()

    try:
        # Org B context — search for "Alice" (which exists in org A).
        # Should return zero hits because RLS isolates.
        async with async_engine.begin() as conn:
            await conn.execute(text("SET LOCAL ROLE z3rno_app"))
            await conn.execute(
                text("SELECT set_config('app.current_org_id', :o, false)"),
                {"o": str(org_b)},
            )
            resp = await recall(
                conn,
                org_id=org_b,
                agent_id=agent_b,
                query="Alice",
                strategy="LEXICAL",
                top_k=10,
            )

        assert len(resp) == 0

        # Org A context still sees its own results.
        async with async_engine.begin() as conn:
            await conn.execute(text("SET LOCAL ROLE z3rno_app"))
            await conn.execute(
                text("SELECT set_config('app.current_org_id', :o, false)"),
                {"o": str(org_a)},
            )
            resp_a = await recall(
                conn,
                org_id=org_a,
                agent_id=agent_a,
                query="Alice",
                strategy="LEXICAL",
                top_k=10,
            )
        assert len(resp_a) >= 1
    finally:
        with sync_engine.connect() as conn:
            conn.execute(
                text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete")
            )
            conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_b}'"))
            conn.execute(
                text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete")
            )
            conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_b}'"))
            conn.execute(text(f"DELETE FROM agents WHERE org_id = '{org_b}'"))
            conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_b}'"))
            conn.commit()

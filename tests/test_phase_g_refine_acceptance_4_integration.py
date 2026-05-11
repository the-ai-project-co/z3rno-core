"""v0.20.4 — Phase D acceptance #4 end-to-end against real Postgres.

Closes the integration gap left by ``test_phase_g_refine_acceptance_4.py``
(unit-level chain proven; integration fixture was stubbed pending this
slice). Seeds two memory_relationships edges with equal starting
weight, votes ``-1`` three times on edge A and ``+1`` three times on
edge B, runs one ``run_reweight`` cycle, asserts edge A's weight is
now strictly lower than edge B's.

Skipped without ``DATABASE_URL``.
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

from z3rno_core.models import Agent, Memory, MemoryType, Tenant
from z3rno_core.models.enums import PlanTier
from z3rno_core.refine.reweight import run_reweight

DATABASE_URL = os.environ.get("DATABASE_URL")
ASYNC_DATABASE_URL = (
    DATABASE_URL.replace("+psycopg", "+asyncpg") if DATABASE_URL else None
)

pytestmark = [
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="DATABASE_URL not set — skipping refine integration",
    ),
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Two memos linked by two edges with equal starting weight.

    edge_down  vote -1 (x3) -> expect weight to drop after refine.
    edge_up    vote +1 (x3) -> expect weight to rise.
    """
    org = uuid4()
    agent = uuid4()
    mem_a = uuid4()
    mem_b = uuid4()
    mem_c = uuid4()
    edge_down = uuid4()
    edge_up = uuid4()

    with Session(engine) as session:
        session.add(Tenant(org_id=org, name="Refine A4", plan_tier=PlanTier.PRO))
        session.flush()
        session.add(Agent(id=agent, org_id=org, name="agent", agent_metadata={}))
        session.flush()
        for mid in (mem_a, mem_b, mem_c):
            session.add(
                Memory(
                    id=mid,
                    org_id=org,
                    agent_id=agent,
                    memory_type=MemoryType.SEMANTIC,
                    content=f"memo {mid.hex[:6]}",
                    memory_metadata={},
                )
            )
        session.commit()

    # Edges + feedback land via raw SQL to avoid having to import the
    # whole SA model graph for relationships + feedback.
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO memory_relationships (
                    id, org_id, source_memory_id, target_memory_id,
                    relationship_type, weight, metadata, created_at, updated_at
                ) VALUES
                    (:e_down, :org, :a, :b, 'related_to', 0.5, '{}'::jsonb, now(), now()),
                    (:e_up,   :org, :a, :c, 'related_to', 0.5, '{}'::jsonb, now(), now())
            """),
            {"e_down": str(edge_down), "e_up": str(edge_up), "org": str(org),
             "a": str(mem_a), "b": str(mem_b), "c": str(mem_c)},
        )
        # 3 down-votes on edge_down, 3 up-votes on edge_up.
        for _ in range(3):
            conn.execute(
                text("""
                    INSERT INTO feedback (id, org_id, agent_id, edge_id, signal, created_at)
                    VALUES (gen_random_uuid(), :org, :agent, :edge, -1, now())
                """),
                {"org": str(org), "agent": str(agent), "edge": str(edge_down)},
            )
            conn.execute(
                text("""
                    INSERT INTO feedback (id, org_id, agent_id, edge_id, signal, created_at)
                    VALUES (gen_random_uuid(), :org, :agent, :edge, 1, now())
                """),
                {"org": str(org), "agent": str(agent), "edge": str(edge_up)},
            )
        conn.commit()

    yield {
        "org": org,
        "agent": agent,
        "edge_down": edge_down,
        "edge_up": edge_up,
        "mem_a": mem_a,
        "mem_b": mem_b,
        "mem_c": mem_c,
    }

    # Teardown: rip everything in reverse FK order.
    with engine.connect() as conn:
        for tbl in ("feedback", "memory_relationships", "memories", "agents", "tenants"):
            conn.execute(text(f"DELETE FROM {tbl} WHERE org_id = '{org}'"))
        conn.commit()


# ---------------------------------------------------------------------------
# Acceptance bar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_shifts_edge_weights_within_one_cycle(
    async_engine: AsyncEngine,
    seed: dict[str, UUID],
) -> None:
    """One refine cycle should move edge_down's weight strictly below
    edge_up's. Both start equal at 0.5; negative feedback EMA-blends
    toward 0.0, positive toward 1.0."""
    org_id = seed["org"]

    async with async_engine.connect() as conn:
        await conn.execute(text("SET LOCAL ROLE z3rno_app"))
        await conn.execute(
            text(f"SET LOCAL app.current_org_id = '{org_id}'")
        )
        result = await run_reweight(conn, org_id=org_id, decay=0.5)
        # decay=0.5 amplifies the per-cycle shift so a single cycle
        # produces a visible delta. Defaults (0.95) would still pass
        # but with a tighter margin.
        await conn.commit()
        # Read back the two edges.
        row = (
            await conn.execute(
                text("""
                    SELECT
                        (SELECT weight FROM memory_relationships WHERE id = :e_down) AS w_down,
                        (SELECT weight FROM memory_relationships WHERE id = :e_up) AS w_up
                """),
                {
                    "e_down": str(seed["edge_down"]),
                    "e_up": str(seed["edge_up"]),
                },
            )
        ).fetchone()

    assert result.edges_reweighted == 2, (
        f"both edges should have been reweighted; got {result.edges_reweighted}"
    )
    assert result.feedback_drained == 6, (
        f"3 down-votes + 3 up-votes = 6 drained; got {result.feedback_drained}"
    )

    w_down, w_up = float(row[0]), float(row[1])
    assert w_down < 0.5 < w_up, (
        f"expected w_down < 0.5 < w_up after one refine cycle; "
        f"got w_down={w_down:.4f}, w_up={w_up:.4f}"
    )
    # Sanity: both are in [0, 1].
    assert 0.0 <= w_down <= 1.0
    assert 0.0 <= w_up <= 1.0

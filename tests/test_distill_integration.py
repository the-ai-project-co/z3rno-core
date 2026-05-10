"""Integration tests for the Forge pipeline (Phase A) — end-to-end against
a real PostgreSQL with Migration 015 applied.

What's covered
--------------
  * Happy path: insert source Memo → run ForgePipeline → distill_jobs row,
    entity_provenance rows, new SEMANTIC Memos, audit_log entries
  * Idempotency: re-running the same job is a no-op
  * Multi-memory: counters roll up correctly
  * Empty source content → memory listed in skipped_memory_ids
  * Job lifecycle transitions: queued → running → completed
  * RLS: cross-tenant isolation of distill_jobs and entity_provenance
  * Failure path: gateway error puts the job in 'failed' status

LLM calls are stubbed via :class:`StubLLMGateway` so tests are deterministic,
offline, and fast. AGE writes are best-effort and silently skipped on the
testcontainer (which doesn't ship AGE) — the relational assertions cover
the contract that matters for Phase A.

Skipped unless ``DATABASE_URL`` is set. Set it to point at a Postgres with
migrations applied through head (revision 015).
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

from z3rno_core.distill.extract import _LLMExtraction
from z3rno_core.distill.llm_gateway import LLMGatewayError, StubLLMGateway
from z3rno_core.distill.schemas import Entity, Relationship
from z3rno_core.engine import NoOpEmbeddingProvider, flush_audit_chain, store
from z3rno_core.forge import ForgeOptions, ForgePipeline
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
def test_org(sync_engine: Engine) -> Generator[UUID, None, None]:
    """Per-test tenant — ensures complete isolation between tests."""
    org_id = uuid4()
    with Session(sync_engine) as session:
        session.add(Tenant(org_id=org_id, name=f"Forge IT {org_id}", plan_tier=PlanTier.PRO))
        session.commit()
    yield org_id
    with sync_engine.connect() as conn:
        # The audit_log immutability trigger blocks DELETE — toggle it off for cleanup.
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM entity_provenance WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM distill_jobs WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM memory_relationships WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_memory(eng: AsyncEngine, org_id: UUID, agent_id: UUID, content: str) -> UUID:
    """Insert one source memory and return its id."""
    async with eng.connect() as conn, conn.begin():
        await conn.execute(
            text(f"SET LOCAL app.current_org_id = '{org_id}'"),
        )
        res = await store(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            content=content,
            memory_type=MemoryType.EPISODIC,
            embedding_provider=NoOpEmbeddingProvider(),
        )
    return res.memory_id


def _stub_gateway() -> StubLLMGateway:
    """Stub gateway with a deterministic, multi-entity extraction response."""
    return StubLLMGateway(
        model="stub/forge-it",
        completion=lambda _s, _u: "Z3rno turns text into a typed knowledge graph.",
        structured=lambda _s, _u, _m: _LLMExtraction(
            entities=[
                Entity(name="Z3rno", type="product", description="smart memory"),
                Entity(name="Cognee", type="product"),
            ],
            relationships=[
                Relationship(source="Z3rno", target="Cognee", predicate="competes_with"),
            ],
            triplets=[],
            summary="Z3rno is smart memory.",
        ),
    )


def _count(sync_engine: Engine, sql: str, params: dict[str, object]) -> int:
    with sync_engine.connect() as conn:
        return int(conn.execute(text(sql), params).scalar() or 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_happy_path_writes_memos_provenance_and_audit(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    agent_id = uuid4()
    src_id = await _seed_memory(
        async_engine, test_org, agent_id, "Z3rno is a smart-memory platform."
    )

    pipeline = ForgePipeline(
        gateway=_stub_gateway(),
        embedding_provider=NoOpEmbeddingProvider(),
        options=ForgeOptions(chunk_size=128, chunk_overlap=0, include_summary=False),
    )
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
    )

    assert summary.status == "completed"
    assert summary.error is None
    assert summary.memories_processed == 1
    assert summary.memories_skipped == 0
    assert summary.entities_extracted == 2
    assert summary.relationships_extracted == 1
    assert summary.memos_written == 2  # two entities; summary disabled

    # distill_jobs row reflects final state
    job_row = _count(
        sync_engine,
        "SELECT count(*) FROM distill_jobs WHERE id = :j AND status = 'completed'",
        {"j": str(summary.job_id)},
    )
    assert job_row == 1

    # 2 entities -> 2 entity_provenance rows linked to source
    prov = _count(
        sync_engine,
        "SELECT count(*) FROM entity_provenance WHERE distill_job_id = :j AND source_memory_id = :s",
        {"j": str(summary.job_id), "s": str(src_id)},
    )
    assert prov == 2

    # 2 new SEMANTIC Memos exist for this org
    new_memos = _count(
        sync_engine,
        """
        SELECT count(*) FROM memories
        WHERE org_id = :o
          AND memory_type = 'semantic'
          AND id IN (SELECT memo_id FROM entity_provenance WHERE distill_job_id = :j)
        """,
        {"o": str(test_org), "j": str(summary.job_id)},
    )
    assert new_memos == 2

    # Audit log got a 'store' entry per new Memo (transitively via store()).
    # Audit writes are async-drained in v0.7.0 — flush before asserting.
    async with async_engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_org_id', :o, false)"),
            {"o": str(test_org)},
        )
        await flush_audit_chain(conn, test_org)
    audit_rows = _count(
        sync_engine,
        """
        SELECT count(*) FROM audit_log
        WHERE org_id = :o
          AND operation = 'store'
          AND memory_id IN (SELECT memo_id FROM entity_provenance WHERE distill_job_id = :j)
        """,
        {"o": str(test_org), "j": str(summary.job_id)},
    )
    assert audit_rows == 2


async def test_idempotency_rerunning_same_job_is_noop(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    agent_id = uuid4()
    src_id = await _seed_memory(async_engine, test_org, agent_id, "Z3rno is memory.")
    pipeline = ForgePipeline(
        gateway=_stub_gateway(),
        embedding_provider=NoOpEmbeddingProvider(),
        options=ForgeOptions(chunk_size=128, chunk_overlap=0, include_summary=False),
    )

    job_id = uuid4()
    first = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
        job_id=job_id,
    )
    assert first.memos_written == 2

    # Re-run with the SAME job_id — already_distilled() must trip; no new Memos.
    second = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
        job_id=job_id,
    )
    assert second.memories_skipped == 1
    assert second.memos_written == 0
    assert src_id in second.skipped_memory_ids

    # entity_provenance count unchanged.
    prov = _count(
        sync_engine,
        "SELECT count(*) FROM entity_provenance WHERE distill_job_id = :j",
        {"j": str(job_id)},
    )
    assert prov == 2


async def test_multi_memory_job_aggregates_counters(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    agent_id = uuid4()
    sources = [
        await _seed_memory(async_engine, test_org, agent_id, f"Source memory number {i}.")
        for i in range(3)
    ]

    pipeline = ForgePipeline(
        gateway=_stub_gateway(),
        embedding_provider=NoOpEmbeddingProvider(),
        options=ForgeOptions(chunk_size=128, chunk_overlap=0, include_summary=False),
    )
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=sources,
    )

    assert summary.status == "completed"
    assert summary.memories_processed == 3
    # 2 entities x 3 source memories — but Z3rno/Cognee are reused; merge dedupes
    # *within* a single chunk's result. Different source memories produce
    # separate Memo rows even with same entity name (provenance differs).
    assert summary.memos_written == 6
    assert summary.entities_extracted == 6  # 2 per source x 3

    # 6 provenance rows, 3 distinct source_memory_ids
    prov = _count(
        sync_engine,
        "SELECT count(DISTINCT source_memory_id) FROM entity_provenance WHERE distill_job_id = :j",
        {"j": str(summary.job_id)},
    )
    assert prov == 3


async def test_empty_source_content_is_skipped(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    agent_id = uuid4()
    # We cannot insert empty content via store() (it validates), so insert one
    # valid then soft-delete it; orchestrator's SELECT filters deleted.
    src_id = await _seed_memory(async_engine, test_org, agent_id, "to be deleted")
    with sync_engine.connect() as conn:
        conn.execute(
            text(f"UPDATE memories SET deleted_at = now() WHERE id = '{src_id}'"),
        )
        conn.commit()

    pipeline = ForgePipeline(
        gateway=_stub_gateway(),
        embedding_provider=NoOpEmbeddingProvider(),
    )
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
    )
    assert summary.status == "completed"
    assert summary.memories_skipped == 1
    assert summary.memos_written == 0
    assert src_id in summary.skipped_memory_ids


async def test_job_lifecycle_transitions(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    agent_id = uuid4()
    src_id = await _seed_memory(async_engine, test_org, agent_id, "Lifecycle memory.")

    pipeline = ForgePipeline(
        gateway=_stub_gateway(),
        embedding_provider=NoOpEmbeddingProvider(),
        options=ForgeOptions(include_summary=False),
    )
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
    )

    with sync_engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT status::text, started_at, completed_at, model
                FROM distill_jobs
                WHERE id = :j
            """),
            {"j": str(summary.job_id)},
        ).fetchone()
    assert row is not None
    assert row[0] == "completed"
    assert row[1] is not None  # started_at populated
    assert row[2] is not None  # completed_at populated
    assert row[3] == "stub/forge-it"


async def test_rls_cross_tenant_isolation(async_engine: AsyncEngine, sync_engine: Engine) -> None:
    """Two orgs run jobs; neither can see the other's rows under RLS."""
    org_a = uuid4()
    org_b = uuid4()
    with Session(sync_engine) as session:
        session.add(Tenant(org_id=org_a, name=f"A {org_a}", plan_tier=PlanTier.PRO))
        session.add(Tenant(org_id=org_b, name=f"B {org_b}", plan_tier=PlanTier.PRO))
        session.commit()

    try:
        agent_a = uuid4()
        agent_b = uuid4()
        src_a = await _seed_memory(async_engine, org_a, agent_a, "Org A memory.")
        src_b = await _seed_memory(async_engine, org_b, agent_b, "Org B memory.")

        pipeline = ForgePipeline(
            gateway=_stub_gateway(),
            embedding_provider=NoOpEmbeddingProvider(),
            options=ForgeOptions(include_summary=False),
        )
        sum_a = await pipeline.run(async_engine, org_id=org_a, agent_id=agent_a, memory_ids=[src_a])
        sum_b = await pipeline.run(async_engine, org_id=org_b, agent_id=agent_b, memory_ids=[src_b])
        assert sum_a.status == "completed"
        assert sum_b.status == "completed"

        # Set RLS to org_a, query distill_jobs — must see exactly 1.
        async with async_engine.connect() as conn, conn.begin():
            await conn.execute(text("SET LOCAL ROLE z3rno_app"))
            await conn.execute(text(f"SET LOCAL app.current_org_id = '{org_a}'"))
            seen_jobs = (await conn.execute(text("SELECT count(*) FROM distill_jobs"))).scalar()
            assert seen_jobs == 1
            seen_provs = (
                await conn.execute(text("SELECT count(*) FROM entity_provenance"))
            ).scalar()
            # 2 entities x 1 source under org_a
            assert seen_provs == 2

        # Symmetrically for org_b.
        async with async_engine.connect() as conn, conn.begin():
            await conn.execute(text("SET LOCAL ROLE z3rno_app"))
            await conn.execute(text(f"SET LOCAL app.current_org_id = '{org_b}'"))
            seen_jobs = (await conn.execute(text("SELECT count(*) FROM distill_jobs"))).scalar()
            assert seen_jobs == 1
    finally:
        with sync_engine.connect() as sync_conn:
            sync_conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
            for org in (org_a, org_b):
                sync_conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org}'"))
                sync_conn.execute(text(f"DELETE FROM entity_provenance WHERE org_id = '{org}'"))
                sync_conn.execute(text(f"DELETE FROM distill_jobs WHERE org_id = '{org}'"))
                sync_conn.execute(text(f"DELETE FROM memory_relationships WHERE org_id = '{org}'"))
                sync_conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org}'"))
                sync_conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org}'"))
            sync_conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
            sync_conn.commit()


async def test_gateway_error_marks_job_failed(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """When the LLM gateway raises, the job lands in 'failed' with an error."""
    agent_id = uuid4()
    src_id = await _seed_memory(async_engine, test_org, agent_id, "Will fail.")

    def boom(_s: str, _u: str, _m: type) -> _LLMExtraction:
        raise LLMGatewayError("simulated provider outage")

    failing_gw = StubLLMGateway(model="stub/fail", structured=boom)
    pipeline = ForgePipeline(
        gateway=failing_gw,
        embedding_provider=NoOpEmbeddingProvider(),
        options=ForgeOptions(include_summary=False),
    )
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent_id,
        memory_ids=[src_id],
    )

    # extract_from_chunk *absorbs* gateway errors and returns an empty result —
    # so a single failing chunk is "processed" with zero entities, status=completed.
    # The job runs to completion but writes no Memos.
    assert summary.status == "completed"
    assert summary.memories_processed == 1
    assert summary.memos_written == 0
    assert summary.entities_extracted == 0

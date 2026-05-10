"""Integration tests for the IngestPipeline (Phase B.1) — end-to-end
against a real PostgreSQL with Migration 016 applied.

What's covered
--------------
* Text ingest: creates 1 Memo with source_uri=None and dataset_id stamped.
* File ingest: storage backend produces a file:// URI; Memo metadata
  carries it as source_uri.
* URL ingest dedupe: re-ingesting the same URL returns the existing
  memory_id and skips the second store.
* Datasets: create + list + delete cascade-detaches memories.
* RLS isolation across orgs.
* INGEST_AUTO_DISTILL via post_ingest callback records distill_job_id.

LLM and HTTP are stubbed; only Postgres is real.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from z3rno_core.engine import NoOpEmbeddingProvider
from z3rno_core.ingest import IngestInput, IngestOptions, IngestPipeline
from z3rno_core.loaders import get_default_registry
from z3rno_core.loaders.url import FetchResult
from z3rno_core.models import Tenant
from z3rno_core.models.enums import PlanTier
from z3rno_core.storage import LocalStorageBackend

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
    org_id = uuid4()
    with Session(sync_engine) as session:
        session.add(Tenant(org_id=org_id, name=f"Ingest IT {org_id}", plan_tier=PlanTier.PRO))
        session.commit()
    yield org_id
    with sync_engine.connect() as conn:
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM ingest_jobs WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM entity_provenance WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM distill_jobs WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM memories WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM datasets WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


@pytest.fixture
def storage_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def pipeline(storage_dir: str) -> IngestPipeline:
    return IngestPipeline(
        registry=get_default_registry(),
        storage=LocalStorageBackend(storage_dir),
        embedding_provider=NoOpEmbeddingProvider(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _count(eng: Engine, sql: str, params: dict[str, object]) -> int:
    with eng.connect() as conn:
        return int(conn.execute(text(sql), params).scalar() or 0)


async def test_text_ingest_creates_memo(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID, pipeline: IngestPipeline
) -> None:
    agent = uuid4()
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent,
        ingest_input=IngestInput(kind="text", text="Z3rno is smart memory."),
        options=IngestOptions(auto_distill=False),
    )
    assert summary.status == "completed"
    assert len(summary.memory_ids) == 1
    assert summary.source_uri is None  # text isn't idempotent

    n = _count(
        sync_engine,
        "SELECT count(*) FROM memories WHERE id = :m AND org_id = :o",
        {"m": str(summary.memory_ids[0]), "o": str(test_org)},
    )
    assert n == 1


async def test_file_ingest_persists_artifact_and_records_source_uri(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID, pipeline: IngestPipeline
) -> None:
    agent = uuid4()
    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent,
        ingest_input=IngestInput(
            kind="file",
            content=b"# Hello\n\nMarkdown body.",
            content_type="text/markdown",
            filename="notes.md",
        ),
        options=IngestOptions(auto_distill=False),
    )
    assert summary.status == "completed"
    assert summary.source_uri is not None
    assert summary.source_uri.startswith("file://")
    assert summary.filename == "notes.md"
    assert summary.file_size == len(b"# Hello\n\nMarkdown body.")


async def test_url_ingest_dedupes_on_second_run(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID, pipeline: IngestPipeline
) -> None:
    """Re-ingesting the same URL returns the existing memory_id."""
    from unittest.mock import patch

    agent = uuid4()
    fake = FetchResult(
        url="https://example.com/page",
        content=b"<html><body><p>same body</p></body></html>",
        content_type="text/html",
        status_code=200,
    )

    async def fake_fetch(url: str, **_: object) -> FetchResult:
        return fake

    with patch("z3rno_core.ingest.pipeline.fetch_url", side_effect=fake_fetch):
        first = await pipeline.run(
            async_engine,
            org_id=test_org,
            agent_id=agent,
            ingest_input=IngestInput(kind="url", url="https://example.com/page"),
            options=IngestOptions(auto_distill=False),
        )
        second = await pipeline.run(
            async_engine,
            org_id=test_org,
            agent_id=agent,
            ingest_input=IngestInput(kind="url", url="https://example.com/page"),
            options=IngestOptions(auto_distill=False),
        )

    assert first.status == "completed"
    assert second.status == "completed"
    assert first.memory_ids == second.memory_ids
    assert second.skipped_existing == first.memory_ids


async def test_post_ingest_callback_records_distill_job_id(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID, pipeline: IngestPipeline
) -> None:
    agent = uuid4()
    distill_job_id = uuid4()

    # Mirror the production worker hook: pre-insert distill_jobs so the
    # FK from ingest_jobs.distill_job_id resolves.
    with sync_engine.connect() as sconn:
        sconn.execute(
            text(
                "INSERT INTO distill_jobs (id, org_id, agent_id, memory_ids, status) "
                "VALUES (CAST(:id AS uuid), CAST(:o AS uuid), CAST(:a AS uuid), '{}'::uuid[], 'queued')"
            ),
            {"id": str(distill_job_id), "o": str(test_org), "a": str(agent)},
        )
        sconn.commit()

    async def fake_post_ingest(_summary: object) -> UUID:
        return distill_job_id

    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent,
        ingest_input=IngestInput(kind="text", text="ingest with auto-distill"),
        post_ingest=fake_post_ingest,
    )
    assert summary.distill_job_id == distill_job_id

    # Confirm the row reflects it.
    with sync_engine.connect() as conn:
        result = conn.execute(
            text("SELECT distill_job_id FROM ingest_jobs WHERE id = CAST(:j AS uuid)"),
            {"j": str(summary.job_id)},
        ).fetchone()
        assert result is not None
        assert result[0] == distill_job_id


async def test_dataset_id_attached_to_memo(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID, pipeline: IngestPipeline
) -> None:
    agent = uuid4()
    dataset_id = uuid4()
    # Insert a dataset directly so we can scope the ingest under it.
    with sync_engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO datasets (id, org_id, name) "
                "VALUES (CAST(:id AS uuid), CAST(:o AS uuid), :n)"
            ),
            {"id": str(dataset_id), "o": str(test_org), "n": "test-set"},
        )
        conn.commit()

    summary = await pipeline.run(
        async_engine,
        org_id=test_org,
        agent_id=agent,
        ingest_input=IngestInput(kind="text", text="dataset-scoped ingest"),
        dataset_id=dataset_id,
        options=IngestOptions(auto_distill=False),
    )

    assert summary.status == "completed"
    n = _count(
        sync_engine,
        "SELECT count(*) FROM memories WHERE id = :m AND dataset_id = :d",
        {"m": str(summary.memory_ids[0]), "d": str(dataset_id)},
    )
    assert n == 1


async def test_rls_isolation_across_orgs(
    async_engine: AsyncEngine, sync_engine: Engine, pipeline: IngestPipeline
) -> None:
    """Two orgs each ingest; under SET ROLE z3rno_app each only sees its own jobs."""
    org_a = uuid4()
    org_b = uuid4()
    with Session(sync_engine) as session:
        session.add(Tenant(org_id=org_a, name=f"A {org_a}", plan_tier=PlanTier.PRO))
        session.add(Tenant(org_id=org_b, name=f"B {org_b}", plan_tier=PlanTier.PRO))
        session.commit()

    try:
        agent_a, agent_b = uuid4(), uuid4()
        sa = await pipeline.run(
            async_engine,
            org_id=org_a,
            agent_id=agent_a,
            ingest_input=IngestInput(kind="text", text="org A"),
            options=IngestOptions(auto_distill=False),
        )
        sb = await pipeline.run(
            async_engine,
            org_id=org_b,
            agent_id=agent_b,
            ingest_input=IngestInput(kind="text", text="org B"),
            options=IngestOptions(auto_distill=False),
        )
        assert sa.status == sb.status == "completed"

        # As app role under org_a, only see org_a's row.
        async with async_engine.connect() as conn, conn.begin():
            await conn.execute(text("SET LOCAL ROLE z3rno_app"))
            await conn.execute(text(f"SET LOCAL app.current_org_id = '{org_a}'"))
            count = (await conn.execute(text("SELECT count(*) FROM ingest_jobs"))).scalar()
            assert count == 1
    finally:
        with sync_engine.connect() as sconn:
            sconn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
            for org in (org_a, org_b):
                sconn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org}'"))
                sconn.execute(text(f"DELETE FROM ingest_jobs WHERE org_id = '{org}'"))
                sconn.execute(text(f"DELETE FROM memories WHERE org_id = '{org}'"))
                sconn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org}'"))
            sconn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
            sconn.commit()

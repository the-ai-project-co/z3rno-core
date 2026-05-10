"""Integration test for ``mark_stale_running_jobs_failed``.

Verifies the watchdog actually transitions stale ``running`` rows to
``failed`` and leaves fresh / non-running rows alone.
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

from z3rno_core.ingest.state import (
    insert_ingest_job,
    mark_stale_running_jobs_failed,
    update_ingest_job,
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
def test_org_and_agent(
    sync_engine: Engine,
) -> Generator[tuple[UUID, UUID], None, None]:
    org_id = uuid4()
    agent_id = uuid4()
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_id, name=f"Watchdog IT {org_id}", plan_tier=PlanTier.PRO)
        )
        session.commit()
    with Session(sync_engine) as session:
        session.add(
            Agent(id=agent_id, org_id=org_id, external_id=f"a-{agent_id}", name="A")
        )
        session.commit()
    yield org_id, agent_id
    with sync_engine.connect() as conn:
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM ingest_jobs WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM agents WHERE org_id = '{org_id}'"))
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


async def test_marks_only_stale_running_jobs_failed(
    async_engine: AsyncEngine,
    sync_engine: Engine,
    test_org_and_agent: tuple[UUID, UUID],
) -> None:
    """Stale ``running`` rows go to ``failed``; fresh + non-running rows untouched."""
    org_id, agent_id = test_org_and_agent

    stale_id = uuid4()
    fresh_id = uuid4()
    completed_id = uuid4()
    queued_id = uuid4()

    async with async_engine.begin() as conn:
        for jid, status in (
            (stale_id, "running"),
            (fresh_id, "running"),
            (completed_id, "completed"),
            (queued_id, "queued"),
        ):
            await insert_ingest_job(
                conn,
                job_id=jid,
                org_id=org_id,
                agent_id=agent_id,
                kind="text",
                status=status,
            )

        # Force ``stale_id``'s updated_at backwards in time so the watchdog
        # sees it as stale. ``fresh_id`` stays at its insert-time updated_at
        # (just now).
        await conn.execute(
            text(
                "UPDATE ingest_jobs SET updated_at = now() - interval '2 hours' "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": str(stale_id)},
        )

    async with async_engine.begin() as conn:
        failed = await mark_stale_running_jobs_failed(
            conn, stale_after_seconds=3600
        )

    # Only the stale running row transitioned.
    assert failed == [stale_id]

    # Verify in the DB.
    with sync_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, status::text, error FROM ingest_jobs "
                "WHERE org_id = :o ORDER BY status"
            ),
            {"o": str(org_id)},
        ).fetchall()
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    assert by_id[stale_id][0] == "failed"
    assert "watchdog" in (by_id[stale_id][1] or "")
    assert by_id[fresh_id][0] == "running"
    assert by_id[completed_id][0] == "completed"
    assert by_id[queued_id][0] == "queued"


async def test_returns_empty_when_no_stale_jobs(
    async_engine: AsyncEngine,
    test_org_and_agent: tuple[UUID, UUID],
) -> None:
    """No stale rows ⇒ returns empty list, no DB writes."""
    org_id, agent_id = test_org_and_agent

    job_id = uuid4()
    async with async_engine.begin() as conn:
        await insert_ingest_job(
            conn,
            job_id=job_id,
            org_id=org_id,
            agent_id=agent_id,
            kind="text",
            status="running",
        )

    async with async_engine.begin() as conn:
        # Threshold above the row's age (the row was just inserted).
        failed = await mark_stale_running_jobs_failed(
            conn, stale_after_seconds=3600
        )

    assert failed == []


async def test_does_not_overwrite_existing_error(
    async_engine: AsyncEngine,
    sync_engine: Engine,
    test_org_and_agent: tuple[UUID, UUID],
) -> None:
    """If a stale row already has an ``error`` set, preserve it."""
    org_id, agent_id = test_org_and_agent

    job_id = uuid4()
    async with async_engine.begin() as conn:
        await insert_ingest_job(
            conn,
            job_id=job_id,
            org_id=org_id,
            agent_id=agent_id,
            kind="text",
            status="running",
        )
        await update_ingest_job(
            conn, job_id=job_id, error="parser blew up halfway"
        )
        # Backdate.
        await conn.execute(
            text(
                "UPDATE ingest_jobs SET updated_at = now() - interval '2 hours' "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {"id": str(job_id)},
        )

    async with async_engine.begin() as conn:
        failed = await mark_stale_running_jobs_failed(
            conn, stale_after_seconds=3600
        )

    assert failed == [job_id]

    with sync_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status::text, error FROM ingest_jobs WHERE id = :i"
            ),
            {"i": str(job_id)},
        ).fetchone()
    assert row is not None
    assert row[0] == "failed"
    # COALESCE in the SQL preserved the pre-existing error.
    assert row[1] == "parser blew up halfway"

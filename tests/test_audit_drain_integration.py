"""End-to-end chain integrity tests for the v0.7.0 audit drain.

Skipped unless ``DATABASE_URL`` is set (testcontainer or local Postgres
with all migrations applied through head).

The point of these tests is to *prove* — not just by mock — that:

1. ``enqueue_audit_entry`` lands rows in ``audit_log_pending`` and
   ``drain_audit_chain`` then chains them into ``audit_log`` correctly.

2. The chain at rest in ``audit_log`` is *byte-identical* to what the
   pre-v0.7.0 synchronous path would have produced. This is the
   security-relevant invariant: an offline auditor walking
   ``audit_log`` ordered by ``id`` and recomputing
   ``SHA-256(prev_hash || canonical_json(row))`` MUST get the same
   ``row_hash`` value the drainer wrote.

3. The drainer is safe under concurrent invocation on the same org —
   the per-org advisory lock means only one drainer makes progress at
   a time; the other returns 0.

4. Rows are deleted from pending after they're drained; queue depth
   stays bounded by the producer rate, not unbounded.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from z3rno_core.engine.audit import (
    compute_row_hash,
    drain_audit_chain,
    enqueue_audit_entry,
    flush_audit_chain,
    list_orgs_with_pending,
)
from z3rno_core.models import Tenant
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
    """Per-test tenant — full isolation between tests."""
    org_id = uuid4()
    with Session(sync_engine) as session:
        session.add(
            Tenant(org_id=org_id, name=f"Audit IT {org_id}", plan_tier=PlanTier.PRO)
        )
        session.commit()
    yield org_id
    # Cleanup. audit_log has an immutability trigger; toggle it off briefly.
    with sync_engine.connect() as conn:
        conn.execute(text("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete"))
        conn.execute(text(f"DELETE FROM audit_log WHERE org_id = '{org_id}'"))
        conn.execute(text("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_delete"))
        conn.execute(
            text(f"DELETE FROM audit_log_pending WHERE org_id = '{org_id}'")
        )
        conn.execute(text(f"DELETE FROM tenants WHERE org_id = '{org_id}'"))
        conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _set_org_context(conn: object, org_id: UUID) -> None:
    """Set RLS context on a connection."""
    # conn is an AsyncConnection; .execute is the relevant method.
    await conn.execute(  # type: ignore[attr-defined]
        text("SELECT set_config('app.current_org_id', :o, false)"),
        {"o": str(org_id)},
    )


def _read_audit_log_for(
    sync_engine: Engine, org_id: UUID
) -> list[tuple[int, str, dict, bytes | None, bytes, datetime]]:
    """Return audit_log rows for one org in id-order: (id, op, details, prev_hash, row_hash, created_at)."""
    with sync_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, operation::text, details, prev_hash, row_hash, created_at
                FROM audit_log
                WHERE org_id = :o
                ORDER BY id
                """
            ),
            {"o": str(org_id)},
        ).fetchall()
    return [
        (r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows
    ]


def _expected_row_hash(
    org_id: UUID,
    operation: str,
    details: dict,
    created_at: datetime,
    prev_hash: bytes | None,
) -> bytes:
    """Recompute the row_hash the way an auditor would. Mirrors
    ``drain_audit_chain``'s ``row_data`` shape exactly."""
    row_data = {
        "org_id": str(org_id),
        "operation": operation,
        "agent_id": None,
        "user_id": None,
        "memory_id": None,
        "memory_type": None,
        "details": details or {},
        "created_at": created_at.isoformat(),
    }
    return compute_row_hash(prev_hash, row_data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_enqueue_then_drain_lands_in_audit_log(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Three enqueues + one flush ⇒ three rows in audit_log, none in pending."""
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        for i in range(3):
            await enqueue_audit_entry(
                conn,
                org_id=test_org,
                operation="store",
                details={"i": i},
            )

    # Pending should hold three rows now (drain hasn't run).
    with sync_engine.connect() as conn:
        pending_count = conn.execute(
            text(
                "SELECT count(*) FROM audit_log_pending WHERE org_id = :o"
            ),
            {"o": str(test_org)},
        ).scalar()
    assert pending_count == 3

    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        drained = await flush_audit_chain(conn, test_org)
    assert drained == 3

    # Pending now empty; audit_log has the three rows in id-order.
    rows = _read_audit_log_for(sync_engine, test_org)
    assert len(rows) == 3
    assert [r[1] for r in rows] == ["store", "store", "store"]
    assert [r[2]["i"] for r in rows] == [0, 1, 2]

    with sync_engine.connect() as conn:
        pending_after = conn.execute(
            text(
                "SELECT count(*) FROM audit_log_pending WHERE org_id = :o"
            ),
            {"o": str(test_org)},
        ).scalar()
    assert pending_after == 0


async def test_chain_at_rest_verifies_offline(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Recompute the SHA-256 chain offline; every row's row_hash must match.

    This is the security-relevant invariant — an auditor walking
    audit_log ordered by id and recomputing SHA-256(prev_hash || row_data)
    gets the same row_hash the drainer wrote. If this passes, the async
    path is byte-identical to the synchronous path at rest.
    """
    operations = ["store", "recall", "store", "forget", "store"]
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        for i, op in enumerate(operations):
            await enqueue_audit_entry(
                conn,
                org_id=test_org,
                operation=op,
                details={"seq": i},
            )

    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        await flush_audit_chain(conn, test_org)

    rows = _read_audit_log_for(sync_engine, test_org)
    assert len(rows) == len(operations)

    # Walk the chain. First row's prev_hash must be NULL (or chain to
    # a pre-existing hash, but since this is a fresh org, it's NULL).
    prev_hash: bytes | None = None
    for _id, op, details, observed_prev, observed_hash, created_at in rows:
        assert observed_prev == prev_hash, (
            f"chain break: row.prev_hash {observed_prev!r} != expected {prev_hash!r}"
        )
        expected = _expected_row_hash(test_org, op, details, created_at, prev_hash)
        assert observed_hash == expected, (
            f"row_hash mismatch on op={op}: got {observed_hash.hex()}, "
            f"expected {expected.hex()}"
        )
        prev_hash = observed_hash


async def test_drain_idempotent_when_no_pending(
    async_engine: AsyncEngine, test_org: UUID
) -> None:
    """flush_audit_chain on an empty queue returns 0 and does no work."""
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        n = await flush_audit_chain(conn, test_org)
    assert n == 0


async def test_concurrent_drainers_do_not_double_chain(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """Two drainers racing on the same org → exactly one wins per call.

    This is the critical correctness property of the per-org advisory
    lock. If both drainers proceed, they'd produce two competing
    chains and the row_hash continuity would break.
    """
    # Seed pending events.
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        for i in range(20):
            await enqueue_audit_entry(
                conn,
                org_id=test_org,
                operation="store",
                details={"i": i},
            )

    # Two concurrent drainers, each in its own transaction.
    async def _one_drainer() -> int:
        async with async_engine.begin() as conn:
            await _set_org_context(conn, test_org)
            return await drain_audit_chain(conn, test_org, batch_size=20)

    n1, n2 = await asyncio.gather(_one_drainer(), _one_drainer())

    # Exactly one drainer made progress; the other lost the lock and
    # returned 0. (Or one drained everything and the other found an
    # empty queue. Either way: total drained = 20, no overlap.)
    assert n1 + n2 == 20
    assert (n1 == 20 and n2 == 0) or (n1 == 0 and n2 == 20)

    # Final chain must still verify offline.
    rows = _read_audit_log_for(sync_engine, test_org)
    assert len(rows) == 20
    prev: bytes | None = None
    for _id, op, details, observed_prev, observed_hash, created_at in rows:
        assert observed_prev == prev
        expected = _expected_row_hash(test_org, op, details, created_at, prev)
        assert observed_hash == expected
        prev = observed_hash


async def test_pending_rows_deleted_after_drain(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """audit_log_pending must not grow without bound — drained rows are deleted."""
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        for _ in range(5):
            await enqueue_audit_entry(
                conn, org_id=test_org, operation="store"
            )

    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        await flush_audit_chain(conn, test_org)

    with sync_engine.connect() as conn:
        depth = conn.execute(
            text(
                "SELECT count(*) FROM audit_log_pending WHERE org_id = :o"
            ),
            {"o": str(test_org)},
        ).scalar()
    assert depth == 0


async def test_list_orgs_with_pending_finds_only_orgs_with_backlog(
    async_engine: AsyncEngine, sync_engine: Engine, test_org: UUID
) -> None:
    """The drain task uses list_orgs_with_pending to discover work.

    With one org that has pending rows, the result includes it. After
    flush, the org no longer appears.
    """
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        await enqueue_audit_entry(conn, org_id=test_org, operation="store")

    # Discovery query needs a session that can read across tenants. RLS
    # restricts to current_org_id; the worker DB role bypasses RLS in
    # practice. For this test, we set context explicitly to confirm
    # the query at least returns *this* org.
    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        orgs = await list_orgs_with_pending(conn)
    assert test_org in orgs

    async with async_engine.begin() as conn:
        await _set_org_context(conn, test_org)
        await flush_audit_chain(conn, test_org)
        orgs_after = await list_orgs_with_pending(conn)
    assert test_org not in orgs_after

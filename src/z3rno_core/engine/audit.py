"""Audit log: tamper-evident hash chain with async drain.

The audit log uses a SHA-256 hash chain per-tenant for tamper evidence:

    row_hash = SHA-256(prev_hash || canonical_json(row_data))

Until v0.6.x the chain was computed inline on every request. Each
operation did `SELECT prev_hash` → compute → `INSERT`, all under the
request's connection. That serialized every store/recall/forget for a
single org and was the bottleneck identified in v0.6.0 stress testing.

v0.7.0 splits the write path:

  * Engine operations (store / recall / forget / update / lifecycle) call
    :func:`enqueue_audit_entry`, which appends an unchained row to
    ``audit_log_pending``. This is a single INSERT — fully parallel and
    independent of any other operation in the same org.

  * A single-consumer drain task (`z3rno.audit_drain` in z3rno-server)
    calls :func:`drain_audit_chain` periodically. The drainer takes a
    per-org advisory lock, reads pending rows in id-order, chains them,
    inserts into ``audit_log``, and deletes from pending. Two drainers
    on the same org cannot interleave — the lock serializes them safely.

End-to-end tamper-evidence on ``audit_log`` is unchanged. The chain at
rest is identical to what the synchronous path produced; the only
difference is a small (sub-second) lag between the operation and the
chained row landing in ``audit_log``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


def compute_row_hash(prev_hash: bytes | None, data: dict[str, Any]) -> bytes:
    """Compute the SHA-256 hash for an audit log row.

    Args:
        prev_hash: The row_hash of the previous audit entry (None for first).
        data: The row data to hash (canonical JSON serialization).

    Returns:
        32-byte SHA-256 digest.
    """
    h = hashlib.sha256()
    if prev_hash:
        h.update(prev_hash)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    h.update(canonical.encode("utf-8"))
    return h.digest()


async def get_latest_hash(conn: AsyncConnection, org_id: UUID) -> bytes | None:
    """Fetch the most recent row_hash for a tenant's audit chain.

    Returns None if this is the first audit entry for the tenant.
    """
    result = await conn.execute(
        text("""
            SELECT row_hash
            FROM audit_log
            WHERE org_id = :org_id
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """),
        {"org_id": str(org_id)},
    )
    row = result.fetchone()
    if row is None:
        return None
    return row[0]  # type: ignore[no-any-return]


async def enqueue_audit_entry(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    operation: str,
    agent_id: UUID | None = None,
    user_id: UUID | None = None,
    memory_id: UUID | None = None,
    memory_type: str | None = None,
    details: dict[str, Any] | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> None:
    """Append an audit event to the pending queue. Fast, unchained, parallel-safe.

    The event is *not* visible in ``audit_log`` until the drainer has
    chained it. For most callers (compliance, regular audit queries) the
    sub-second lag is invisible; if you need synchronous visibility (e.g.
    in tests), call :func:`flush_audit_chain` after the operation.
    """
    await conn.execute(
        text("""
            INSERT INTO audit_log_pending (
                org_id, agent_id, user_id, operation, memory_id,
                memory_type, details,
                api_key_id, ip_address, user_agent, request_id
            ) VALUES (
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:user_id AS uuid),
                CAST(:operation AS audit_operation_enum),
                CAST(:memory_id AS uuid),
                CAST(:memory_type AS memory_type_enum),
                CAST(:details AS jsonb),
                CAST(:api_key_id AS uuid),
                CAST(:ip_address AS inet),
                :user_agent,
                :request_id
            )
        """),
        {
            "org_id": str(org_id),
            "agent_id": str(agent_id) if agent_id else None,
            "user_id": str(user_id) if user_id else None,
            "operation": operation,
            "memory_id": str(memory_id) if memory_id else None,
            "memory_type": memory_type,
            "details": json.dumps(details or {}),
            "api_key_id": str(api_key_id) if api_key_id else None,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_id": request_id,
        },
    )


# -----------------------------------------------------------------------------
# Backwards compatibility
# -----------------------------------------------------------------------------
# Older callers (and external integrations) import ``create_audit_entry``.
# Keep the name as an alias for ``enqueue_audit_entry``: the call still
# *records* the event and the chained row still lands in ``audit_log``,
# just asynchronously instead of synchronously. Synchronous-visibility
# tests should add a ``flush_audit_chain(conn, org_id)`` call before
# asserting on ``audit_log`` rows.
create_audit_entry = enqueue_audit_entry


# -----------------------------------------------------------------------------
# Drainer — chains pending rows into audit_log under a per-org lock
# -----------------------------------------------------------------------------

# A 64-bit Postgres advisory-lock key derived from a UUID. Postgres
# requires bigint, so we take the first 8 bytes of the UUID's 16 and
# coerce to a signed int64. Distinct orgs map to distinct keys with
# overwhelming probability (random UUIDs); collision risk is acceptable
# because a collision merely serializes two unrelated drainers briefly.
def _advisory_lock_key(org_id: UUID) -> int:
    raw = org_id.bytes[:8]
    n = int.from_bytes(raw, "big", signed=False)
    # Map to int64 range; pg_try_advisory_xact_lock takes signed bigint.
    return n - (1 << 63)


async def drain_audit_chain(
    conn: AsyncConnection,
    org_id: UUID,
    *,
    batch_size: int = 100,
) -> int:
    """Drain up to ``batch_size`` pending rows for one org.

    Acquires a per-org transactional advisory lock so concurrent drainers
    on the same org do not interleave (which would corrupt the chain
    order). Returns the number of rows drained — 0 if the lock was
    contended or the queue was empty.

    The caller's transaction must be open and will be committed/rolled-back
    as a unit. The lock is released when the transaction ends.
    """
    lock_key = _advisory_lock_key(org_id)
    locked_row = (
        await conn.execute(
            text("SELECT pg_try_advisory_xact_lock(:k)"),
            {"k": lock_key},
        )
    ).fetchone()
    if not locked_row or not locked_row[0]:
        return 0

    pending = (
        await conn.execute(
            text("""
                SELECT id, agent_id, user_id, operation, memory_id, memory_type,
                       details, api_key_id, ip_address, user_agent, request_id,
                       created_at
                FROM audit_log_pending
                WHERE org_id = :org_id
                ORDER BY id
                LIMIT :batch_size
            """),
            {"org_id": str(org_id), "batch_size": batch_size},
        )
    ).fetchall()

    if not pending:
        return 0

    prev_hash = await get_latest_hash(conn, org_id)
    drained_ids: list[int] = []

    for row in pending:
        (
            pending_id,
            agent_id,
            user_id,
            operation,
            memory_id,
            memory_type,
            details,
            api_key_id,
            ip_address,
            user_agent,
            request_id,
            created_at,
        ) = row

        # Canonical row data — must match the synchronous path's shape so
        # consumers verifying the chain offline get identical hashes.
        row_data = {
            "org_id": str(org_id),
            "operation": operation,
            "agent_id": str(agent_id) if agent_id else None,
            "user_id": str(user_id) if user_id else None,
            "memory_id": str(memory_id) if memory_id else None,
            "memory_type": memory_type,
            "details": details or {},
            "created_at": (
                created_at.isoformat()
                if isinstance(created_at, datetime)
                else str(created_at)
            ),
        }
        row_hash = compute_row_hash(prev_hash, row_data)

        await conn.execute(
            text("""
                INSERT INTO audit_log (
                    org_id, agent_id, user_id, operation, memory_id,
                    memory_type, details, prev_hash, row_hash,
                    api_key_id, ip_address, user_agent, request_id,
                    created_at, updated_at
                ) VALUES (
                    CAST(:org_id AS uuid),
                    CAST(:agent_id AS uuid),
                    CAST(:user_id AS uuid),
                    CAST(:operation AS audit_operation_enum),
                    CAST(:memory_id AS uuid),
                    CAST(:memory_type AS memory_type_enum),
                    CAST(:details AS jsonb),
                    :prev_hash,
                    :row_hash,
                    CAST(:api_key_id AS uuid),
                    CAST(:ip_address AS inet),
                    :user_agent,
                    :request_id,
                    :created_at,
                    :created_at
                )
            """),
            {
                "org_id": str(org_id),
                "agent_id": str(agent_id) if agent_id else None,
                "user_id": str(user_id) if user_id else None,
                "operation": operation,
                "memory_id": str(memory_id) if memory_id else None,
                "memory_type": memory_type,
                "details": (
                    json.dumps(details) if not isinstance(details, str) else details
                ),
                "prev_hash": prev_hash,
                "row_hash": row_hash,
                "api_key_id": str(api_key_id) if api_key_id else None,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "request_id": request_id,
                "created_at": created_at,
            },
        )

        prev_hash = row_hash
        drained_ids.append(pending_id)

    await conn.execute(
        text("DELETE FROM audit_log_pending WHERE id = ANY(:ids)"),
        {"ids": drained_ids},
    )
    return len(drained_ids)


async def flush_audit_chain(
    conn: AsyncConnection,
    org_id: UUID,
    *,
    batch_size: int = 500,
    max_iterations: int = 1000,
) -> int:
    """Drain *all* pending rows for one org. Loops until the queue is empty.

    Used by tests (synchronous-visibility) and by the periodic drain task.
    ``max_iterations`` caps the loop so a runaway producer can't pin a
    drainer forever — set it high enough to clear normal backlogs.
    """
    total = 0
    for _ in range(max_iterations):
        n = await drain_audit_chain(conn, org_id, batch_size=batch_size)
        if n == 0:
            break
        total += n
    return total


async def list_orgs_with_pending(
    conn: AsyncConnection,
    *,
    limit: int = 1000,
) -> list[UUID]:
    """Return distinct org_ids that have pending audit rows. RLS-bypass.

    The drain task uses this to discover which orgs to drain. The caller
    must be running under a role with ``BYPASSRLS`` *or* must have RLS
    disabled on the session — otherwise the query returns only the
    current session's org.
    """
    result = await conn.execute(
        text("SELECT DISTINCT org_id FROM audit_log_pending LIMIT :limit"),
        {"limit": limit},
    )
    return [UUID(str(row[0])) for row in result.fetchall()]


# ---------------------------------------------------------------------------
# v0.20.2 — NOTIFY/LISTEN drain trigger
# ---------------------------------------------------------------------------


AUDIT_NOTIFY_CHANNEL = "z3rno_audit_pending"


async def listen_for_audit_pending(
    dsn: str,
    *,
    on_notify: Callable[[str | None], Coroutine[Any, Any, None]],
    stop_event: asyncio.Event | None = None,
) -> None:
    """Long-lived listener for ``NOTIFY z3rno_audit_pending``.

    Opens an asyncpg connection, issues ``LISTEN``, and invokes
    ``on_notify(org_id_or_none)`` each time a notification arrives.
    Caller is expected to ``celery_app.send_task("z3rno.audit_drain")``
    from inside the callback — this module deliberately doesn't import
    Celery so the listener stays runnable in unit tests without a
    broker.

    Migration 032 installs the trigger that emits these notifications
    on every ``INSERT INTO audit_log_pending``. The legacy periodic
    poll stays in place as a fallback (longer interval) so a stuck
    listener can't grow the queue unbounded.

    Loop semantics:
      * The function returns when ``stop_event`` (if supplied) is set.
        Without an event, it runs indefinitely.
      * Connection errors raise back to the caller — operators
        typically restart the listener pod / Celery worker.
    """
    try:
        import asyncpg  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for the LISTEN-based audit drain. "
            "It's already a runtime dep of z3rno-core but make sure "
            "the listener process can import it."
        ) from exc

    # Strip the SQLAlchemy "+asyncpg" prefix asyncpg.connect expects raw libpq DSN.
    raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(raw_dsn)
    try:
        # asyncpg's listener gives us a callback on every NOTIFY. The
        # callback is sync — schedule the user's async callback on the
        # event loop.
        loop = asyncio.get_running_loop()
        # Strong refs on dispatched tasks so they aren't GC'd mid-flight.
        in_flight: set[asyncio.Task[None]] = set()

        def _dispatch(_conn: Any, _pid: int, _channel: str, payload: str) -> None:
            org_id = payload or None
            task = loop.create_task(on_notify(org_id))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        await conn.add_listener(AUDIT_NOTIFY_CHANNEL, _dispatch)
        # Idle: yield until stop_event fires or the connection dies.
        if stop_event is None:
            stop_event = asyncio.Event()
        await stop_event.wait()
    finally:
        await conn.close()

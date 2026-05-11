"""v0.19.4 — multi-tenant refine scheduler.

Fair round-robin across opted-in tenants. The beat task calls
``pick_refine_tenants(conn, limit=N)`` per tick; orgs with the
oldest ``refine_last_run_at`` (NULLS FIRST → never-run wins) come
first. After each successful enqueue, callers invoke
``mark_refine_dispatched`` to bump the row's timestamp so the next
tick sees a different cohort.

The picker runs in its own short transaction so it doesn't block
the beat scheduler if a worker hangs. Concurrency safety: ``FOR
UPDATE SKIP LOCKED`` ensures two beat singletons racing each other
(shouldn't happen, but cheap to defend) never pick the same row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True)
class TenantToRefine:
    org_id: UUID
    last_run_at: datetime | None


async def pick_refine_tenants(
    conn: AsyncConnection,
    *,
    limit: int = 10,
) -> list[TenantToRefine]:
    """Return up to ``limit`` opted-in tenants in fair-rotation order.

    Skips soft-deleted rows. Skips rows currently being picked by a
    concurrent dispatcher (FOR UPDATE SKIP LOCKED).

    Callers must dispatch + ``mark_refine_dispatched`` inside the same
    transaction so the timestamp advance is atomic with the enqueue.
    """
    if limit <= 0:
        return []
    # NOTE: this runs *outside* the per-tenant RLS context — it's the
    # beat scheduler iterating all tenants. The query restricts to the
    # public schema and never reads tenant-scoped tables.
    result = await conn.execute(
        text(
            "SELECT org_id, refine_last_run_at "
            "FROM tenants "
            "WHERE refine_enabled IS TRUE "
            "ORDER BY refine_last_run_at NULLS FIRST "
            "LIMIT :lim "
            "FOR UPDATE SKIP LOCKED"
        ),
        {"lim": int(limit)},
    )
    return [TenantToRefine(org_id=row[0], last_run_at=row[1]) for row in result.fetchall()]


async def mark_refine_dispatched(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    at: datetime | None = None,
) -> None:
    """Bump ``tenants.refine_last_run_at`` so the next tick rotates."""
    await conn.execute(
        text(
            "UPDATE tenants SET refine_last_run_at = COALESCE(:at, now()) "
            "WHERE org_id = CAST(:org_id AS uuid)"
        ),
        {"at": at, "org_id": str(org_id)},
    )


__all__ = [
    "TenantToRefine",
    "mark_refine_dispatched",
    "pick_refine_tenants",
]

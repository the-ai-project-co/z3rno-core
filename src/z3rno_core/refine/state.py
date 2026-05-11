"""``refine_jobs`` state helpers (Phase D slice 3).

Mirrors :mod:`z3rno_core.ingest.state` so the operator surface stays
consistent across Forge / Ingest / Refine. RLS is enforced at the DB
tier; callers must set ``app.current_org_id`` before invoking these.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


async def insert_refine_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    org_id: UUID,
    dataset_id: UUID | None = None,
    trigger: str = "api",
    status: str = "queued",
) -> None:
    """Insert a fresh ``refine_jobs`` row."""
    await conn.execute(
        text("""
            INSERT INTO public.refine_jobs (
                id, org_id, dataset_id, status, trigger,
                created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:dataset_id AS uuid),
                CAST(:status AS refine_job_status),
                :trigger,
                now(), now()
            )
        """),
        {
            "id": str(job_id),
            "org_id": str(org_id),
            "dataset_id": str(dataset_id) if dataset_id else None,
            "status": status,
            "trigger": trigger,
        },
    )


async def update_refine_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    status: str | None = None,
    memos_scanned: int | None = None,
    memos_deduped: int | None = None,
    edges_reweighted: int | None = None,
    edges_pruned: int | None = None,
    feedback_drained: int | None = None,
    job_metadata: dict[str, Any] | None = None,
    error: str | None = None,
    started_at_now: bool = False,
    completed_at_now: bool = False,
) -> None:
    """Patch a ``refine_jobs`` row. Only non-None columns are written."""
    sets: list[str] = ["updated_at = now()"]
    params: dict[str, object] = {"id": str(job_id)}

    def _add(col: str, val: object | None) -> None:
        if val is not None:
            sets.append(f"{col} = :{col}")
            params[col] = val

    if status is not None:
        sets.append("status = CAST(:status AS refine_job_status)")
        params["status"] = status
    _add("memos_scanned", memos_scanned)
    _add("memos_deduped", memos_deduped)
    _add("edges_reweighted", edges_reweighted)
    _add("edges_pruned", edges_pruned)
    _add("feedback_drained", feedback_drained)
    _add("error", error)
    if job_metadata is not None:
        sets.append("job_metadata = CAST(:job_metadata AS jsonb)")
        params["job_metadata"] = json.dumps(job_metadata)
    if started_at_now:
        sets.append("started_at = now()")
    if completed_at_now:
        sets.append("completed_at = now()")

    sql = "UPDATE refine_jobs SET " + ", ".join(sets) + " WHERE id = CAST(:id AS uuid)"  # noqa: S608
    await conn.execute(text(sql), params)

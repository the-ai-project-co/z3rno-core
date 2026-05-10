"""``ingest_jobs`` state helpers (Phase B.1).

Mirrors the shape of :mod:`z3rno_core.distill.graph_writer` so the
Phase A and Phase B.1 Celery tasks can use a consistent vocabulary.

The actual table is created in Migration 016. These helpers are
exclusively SQL-level — RLS context must be set by the caller before
any of them runs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


# ---------------------------------------------------------------------------
# Insert / update
# ---------------------------------------------------------------------------


async def insert_ingest_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    org_id: UUID,
    agent_id: UUID,
    kind: str,
    dataset_id: UUID | None = None,
    source_uri: str | None = None,
    content_type: str | None = None,
    filename: str | None = None,
    file_size: int | None = None,
    status: str = "queued",
    search_batch_id: UUID | None = None,
) -> None:
    """Insert a fresh ``ingest_jobs`` row.

    ``status`` defaults to ``"queued"`` for the standard inline-enqueue
    path. The Phase B.2.1 direct-to-S3 flow passes ``"awaiting_upload"``
    so the row exists while the client PUTs bytes to the presigned URL.
    """
    await conn.execute(
        text("""
            INSERT INTO public.ingest_jobs (
                id, org_id, agent_id, dataset_id,
                kind, source_uri, content_type, filename, file_size,
                status, memory_ids, memos_written,
                search_batch_id,
                created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST(:dataset_id AS uuid),
                CAST(:kind AS ingest_job_kind),
                :source_uri, :content_type, :filename, :file_size,
                CAST(:status AS ingest_job_status),
                :memory_ids, 0,
                CAST(:search_batch_id AS uuid),
                now(), now()
            )
        """),
        {
            "id": str(job_id),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "dataset_id": str(dataset_id) if dataset_id else None,
            "kind": kind,
            "source_uri": source_uri,
            "content_type": content_type,
            "filename": filename,
            "file_size": file_size,
            "status": status,
            "memory_ids": [],
            "search_batch_id": str(search_batch_id) if search_batch_id else None,
        },
    )


async def update_ingest_job(
    conn: AsyncConnection,
    *,
    job_id: UUID,
    status: str | None = None,
    source_uri: str | None = None,
    content_type: str | None = None,
    filename: str | None = None,
    file_size: int | None = None,
    memory_ids: list[UUID] | None = None,
    memos_written: int | None = None,
    distill_job_id: UUID | None = None,
    error: str | None = None,
    warnings: list[dict[str, Any]] | None = None,
    started_at_now: bool = False,
    completed_at_now: bool = False,
) -> None:
    """Patch an ``ingest_jobs`` row. Only non-None columns are written."""
    sets: list[str] = ["updated_at = now()"]
    params: dict[str, object] = {"id": str(job_id)}

    def _add(col: str, val: object | None) -> None:
        if val is not None:
            sets.append(f"{col} = :{col}")
            params[col] = val

    _add("status", status)
    _add("source_uri", source_uri)
    _add("content_type", content_type)
    _add("filename", filename)
    _add("file_size", file_size)
    _add("memos_written", memos_written)
    _add("error", error)
    if memory_ids is not None:
        sets.append("memory_ids = :memory_ids")
        params["memory_ids"] = [str(m) for m in memory_ids]
    if distill_job_id is not None:
        sets.append("distill_job_id = CAST(:distill_job_id AS uuid)")
        params["distill_job_id"] = str(distill_job_id)
    if warnings is not None:
        sets.append("warnings = CAST(:warnings AS jsonb)")
        params["warnings"] = json.dumps(warnings)
    if started_at_now:
        sets.append("started_at = now()")
    if completed_at_now:
        sets.append("completed_at = now()")

    sql = "UPDATE ingest_jobs SET " + ", ".join(sets) + " WHERE id = CAST(:id AS uuid)"  # noqa: S608
    await conn.execute(text(sql), params)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


async def mark_stale_running_jobs_failed(
    conn: AsyncConnection,
    *,
    stale_after_seconds: int = 3600,
    limit: int = 100,
) -> list[UUID]:
    """Find and fail ingest_jobs stuck in ``running`` past the threshold.

    The watchdog for orphaned ingest runs: if ``IngestPipeline.run()``
    crashes before it can call ``update_ingest_job(status="failed")``,
    the row stays in ``running`` forever. This helper scans for rows
    whose ``updated_at`` is older than ``stale_after_seconds`` and
    transitions them to ``failed`` with an explanatory error.

    Returns the list of job_ids it transitioned. Empty list when none
    are stale. The watchdog runs RLS-bypassing (worker DB role has
    ``BYPASSRLS``) so it sees jobs across all tenants.

    The threshold is intentionally generous — even multi-GB ingests
    finish well inside an hour. Operators can tighten it via the
    ``INGEST_WATCHDOG_STALE_AFTER_SECONDS`` env var on the server
    side.
    """
    rows = (
        await conn.execute(
            text("""
                SELECT id
                FROM public.ingest_jobs
                WHERE status = 'running'
                  AND updated_at < now() - make_interval(secs => :secs)
                ORDER BY updated_at
                LIMIT :limit
            """),
            {"secs": stale_after_seconds, "limit": limit},
        )
    ).fetchall()

    if not rows:
        return []

    stale_ids = [r[0] for r in rows]
    error_msg = (
        f"watchdog: stale running >{stale_after_seconds}s; row never transitioned"
    )
    await conn.execute(
        text("""
            UPDATE public.ingest_jobs
            SET status = 'failed',
                error = COALESCE(error, :error_msg),
                updated_at = now(),
                completed_at = now()
            WHERE id = ANY(CAST(:ids AS uuid[]))
              AND status = 'running'
        """),
        {"ids": [str(i) for i in stale_ids], "error_msg": error_msg},
    )
    return [UUID(str(i)) for i in stale_ids]


async def get_search_batch_aggregate(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    batch_id: UUID,
) -> dict[str, Any] | None:
    """Return aggregate status for all jobs in a search batch.

    Returns a dict with ``total``, per-status counts, and the list of
    ``job_ids`` so the caller can drill in. Returns ``None`` when the
    batch_id doesn't exist for this org (RLS-scoped).
    """
    row = (
        await conn.execute(
            text("""
                SELECT
                    count(*) AS total,
                    count(*) FILTER (WHERE status = 'queued') AS queued,
                    count(*) FILTER (WHERE status = 'running') AS running,
                    count(*) FILTER (WHERE status = 'completed') AS completed,
                    count(*) FILTER (WHERE status = 'failed') AS failed,
                    array_agg(id ORDER BY created_at) AS job_ids
                FROM public.ingest_jobs
                WHERE org_id = CAST(:org_id AS uuid)
                  AND search_batch_id = CAST(:batch_id AS uuid)
            """),
            {"org_id": str(org_id), "batch_id": str(batch_id)},
        )
    ).fetchone()

    if row is None or row[0] == 0:
        return None

    return {
        "total": row[0],
        "queued": row[1],
        "running": row[2],
        "completed": row[3],
        "failed": row[4],
        "job_ids": [UUID(str(jid)) for jid in (row[5] or [])],
    }


async def find_memory_by_source_uri(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    source_uri: str,
    dataset_id: UUID | None = None,
) -> UUID | None:
    """Return the existing Memo id matching ``(source_uri, dataset_id)``.

    Used for idempotent re-ingestion of URLs (and any future content-
    addressable artifact that produces a stable source_uri). Returns
    ``None`` when no match exists or the Memo has been deleted.
    """
    if not source_uri:
        return None
    if dataset_id is None:
        sql = text("""
            SELECT id FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND deleted_at IS NULL
              AND valid_to IS NULL
              AND dataset_id IS NULL
              AND metadata->>'source_uri' = :source_uri
            ORDER BY created_at DESC
            LIMIT 1
        """)
        params: dict[str, object] = {
            "org_id": str(org_id),
            "source_uri": source_uri,
        }
    else:
        sql = text("""
            SELECT id FROM memories
            WHERE org_id = CAST(:org_id AS uuid)
              AND deleted_at IS NULL
              AND valid_to IS NULL
              AND dataset_id = CAST(:dataset_id AS uuid)
              AND metadata->>'source_uri' = :source_uri
            ORDER BY created_at DESC
            LIMIT 1
        """)
        params = {
            "org_id": str(org_id),
            "dataset_id": str(dataset_id),
            "source_uri": source_uri,
        }
    row = (await conn.execute(sql, params)).fetchone()
    return row[0] if row else None

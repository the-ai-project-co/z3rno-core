"""``ingest_jobs`` state helpers (Phase B.1).

Mirrors the shape of :mod:`z3rno_core.distill.graph_writer` so the
Phase A and Phase B.1 Celery tasks can use a consistent vocabulary.

The actual table is created in Migration 016. These helpers are
exclusively SQL-level — RLS context must be set by the caller before
any of them runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
) -> None:
    """Insert a fresh ``ingest_jobs`` row.

    ``status`` defaults to ``"queued"`` for the standard inline-enqueue
    path. The Phase B.2.1 direct-to-S3 flow passes ``"awaiting_upload"``
    so the row exists while the client PUTs bytes to the presigned URL.
    """
    await conn.execute(
        text("""
            INSERT INTO ingest_jobs (
                id, org_id, agent_id, dataset_id,
                kind, source_uri, content_type, filename, file_size,
                status, memory_ids, memos_written,
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
    if started_at_now:
        sets.append("started_at = now()")
    if completed_at_now:
        sets.append("completed_at = now()")

    sql = "UPDATE ingest_jobs SET " + ", ".join(sets) + " WHERE id = CAST(:id AS uuid)"  # noqa: S608
    await conn.execute(text(sql), params)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


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

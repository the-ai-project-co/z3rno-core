"""Summarize stage of the Refine pipeline (Phase D slice 4).

Clusters connected Memos into small subgraphs and emits one
``memo_type='SUMMARY'`` Memo per cluster. The summary Memo content is
LLM-generated from the cluster members' contents.

Cache key
---------
The subgraph hash is the sorted, joined Memo-ID set of the cluster. If
a subsequent cycle produces the same hash, we skip the LLM call. The
cache lives in ``memories.metadata.refine.summary_of`` on existing
summary Memos — when we find one matching the hash, we don't re-emit.

Opt-in: ``REFINE_SUMMARIZE_ENABLED=true`` AND an LLM gateway.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from sqlalchemy import text

from z3rno_core.distill.summarize import summarize_text

log = structlog.get_logger(__name__)

_MIN_CLUSTER_SIZE = 2

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from z3rno_core.distill.llm_gateway import LLMGateway


@dataclass(frozen=True)
class SummarizeResult:
    clusters_examined: int
    summaries_written: int
    summaries_skipped_cached: int


def _cluster_hash(member_ids: list[UUID]) -> str:
    """Stable hash of a cluster's membership set."""
    joined = ",".join(sorted(str(m) for m in member_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


async def _cluster_memos_by_components(
    conn: AsyncConnection,
    org_id: UUID,
    dataset_id: UUID | None,
    max_clusters: int,
) -> list[list[UUID]]:
    """v0.19.5 — connected-component clustering over ``memory_relationships``.

    Finds graph connected components via a recursive union-find on
    ``memory_relationships``. Each component (≥ ``_MIN_CLUSTER_SIZE``)
    becomes one cluster. The result is the AGE community equivalent
    expressed in plain SQL — no AGE extension required at refine time.

    Isolated Memos (no edges) are skipped intentionally — the
    summarize stage is for grouping *related* knowledge, not lone
    facts.
    """
    where_dataset = (
        "AND m.dataset_id = CAST(:dataset_id AS uuid)"
        if dataset_id
        else "AND m.dataset_id IS NULL"
    )
    params: dict[str, object] = {
        "org_id": str(org_id),
        "limit": max_clusters * 20,
        "min_size": _MIN_CLUSTER_SIZE,
    }
    if dataset_id:
        params["dataset_id"] = str(dataset_id)

    # Recursive walk: seed = each Memo; step = follow any relationship.
    # The lowest-ID member of each component is its canonical "root".
    rows = (
        await conn.execute(
            text(f"""
                WITH RECURSIVE
                    seeds AS (
                        SELECT m.id
                        FROM public.memories m
                        WHERE m.org_id = CAST(:org_id AS uuid)
                          AND m.valid_to IS NULL
                          AND m.deleted_at IS NULL
                          AND m.memo_type IS NOT NULL
                          AND m.memo_type != 'SUMMARY'
                          {where_dataset}
                        LIMIT :limit
                    ),
                    walk(memo_id, root_id) AS (
                        SELECT id, id FROM seeds
                        UNION
                        SELECT
                            CASE WHEN r.source_memory_id = w.memo_id
                                 THEN r.target_memory_id
                                 ELSE r.source_memory_id
                            END,
                            LEAST(w.root_id,
                                  CASE WHEN r.source_memory_id = w.memo_id
                                       THEN r.target_memory_id
                                       ELSE r.source_memory_id
                                  END)
                        FROM walk w
                        JOIN public.memory_relationships r
                          ON r.source_memory_id = w.memo_id
                          OR r.target_memory_id = w.memo_id
                        WHERE r.org_id = CAST(:org_id AS uuid)
                    )
                SELECT (SELECT MIN(root_id) FROM walk w2 WHERE w2.memo_id = w.memo_id) AS comp,
                       w.memo_id
                FROM walk w
                GROUP BY w.memo_id
            """),  # noqa: S608 — where_dataset interpolated from constants
            params,
        )
    ).fetchall()

    by_root: dict[UUID, list[UUID]] = {}
    for comp, mid in rows:
        by_root.setdefault(comp, []).append(mid)

    clusters = [v for v in by_root.values() if len(v) >= _MIN_CLUSTER_SIZE]
    # Largest first — most likely to produce a useful summary.
    clusters.sort(key=len, reverse=True)
    return clusters[:max_clusters]


async def _cluster_memos(
    conn: AsyncConnection,
    org_id: UUID,
    dataset_id: UUID | None,
    max_clusters: int,
) -> list[list[UUID]]:
    """Cluster currently-valid Memos by the ``memo_type`` they share.

    Coarse grouping — fast and predictable. Operators who want
    graph-aware clustering can flip ``RefineOptions.cluster_strategy
    = "connected_components"`` to switch to the recursive-CTE walk.
    """
    where_dataset = (
        "AND dataset_id = CAST(:dataset_id AS uuid)" if dataset_id else "AND dataset_id IS NULL"
    )
    params: dict[str, object] = {"org_id": str(org_id), "limit": max_clusters * 20}
    if dataset_id:
        params["dataset_id"] = str(dataset_id)

    rows = (
        await conn.execute(
            text(f"""
                SELECT memo_type, id
                FROM public.memories
                WHERE org_id = CAST(:org_id AS uuid)
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
                  AND memo_type IS NOT NULL
                  AND memo_type != 'SUMMARY'
                  {where_dataset}
                ORDER BY memo_type
                LIMIT :limit
            """),  # noqa: S608 — interpolated identifier is constant
            params,
        )
    ).fetchall()

    by_type: dict[str, list[UUID]] = {}
    for memo_type, mid in rows:
        by_type.setdefault(memo_type, []).append(mid)

    clusters = [v for v in by_type.values() if len(v) >= _MIN_CLUSTER_SIZE]
    return clusters[:max_clusters]


async def _existing_summary_hash(conn: AsyncConnection, org_id: UUID, h: str) -> bool:
    """Return True when a SUMMARY Memo with this cluster hash already exists."""
    row = (
        await conn.execute(
            text("""
                SELECT 1 FROM public.memories
                WHERE org_id = CAST(:org_id AS uuid)
                  AND memo_type = 'SUMMARY'
                  AND deleted_at IS NULL
                  AND valid_to IS NULL
                  AND metadata->'refine'->>'cluster_hash' = :h
                LIMIT 1
            """),
            {"org_id": str(org_id), "h": h},
        )
    ).fetchone()
    return row is not None


async def _fetch_contents(conn: AsyncConnection, ids: list[UUID]) -> str:
    rows = (
        await conn.execute(
            text("""
                SELECT content FROM public.memories
                WHERE id = ANY(CAST(:ids AS uuid[]))
                ORDER BY valid_from ASC
            """),
            {"ids": [str(i) for i in ids]},
        )
    ).fetchall()
    return "\n\n".join(r[0] for r in rows if r[0])


async def _write_summary_memo(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    cluster_hash: str,
    member_ids: list[UUID],
    summary: str,
) -> None:
    """Insert a SUMMARY Memo carrying cluster lineage in metadata.

    Bypasses ``engine.store()`` so we can set ``memo_type='SUMMARY'``
    in one statement without an UPDATE chase.
    """
    from uuid import uuid4  # noqa: PLC0415 — local to keep imports tidy

    metadata = {
        "kind": "summary",
        "refine": {
            "cluster_hash": cluster_hash,
            "summary_of": [str(m) for m in member_ids],
        },
    }
    import json  # noqa: PLC0415

    # agent_id is required by the schema; the SUMMARY is org-scoped, so we
    # pick the most common agent_id in the cluster.
    rows = (
        await conn.execute(
            text("""
                SELECT agent_id, COUNT(*)
                FROM public.memories
                WHERE id = ANY(CAST(:ids AS uuid[]))
                GROUP BY agent_id
                ORDER BY COUNT(*) DESC
                LIMIT 1
            """),
            {"ids": [str(m) for m in member_ids]},
        )
    ).fetchall()
    if not rows:
        return
    agent_id = rows[0][0]

    await conn.execute(
        text("""
            INSERT INTO public.memories (
                id, org_id, agent_id, memory_type, content, metadata,
                memo_type, importance_score, valid_from, created_at, updated_at
            ) VALUES (
                CAST(:id AS uuid),
                CAST(:org_id AS uuid),
                CAST(:agent_id AS uuid),
                CAST('semantic' AS memory_type_enum),
                :content,
                CAST(:metadata AS jsonb),
                'SUMMARY',
                0.6,
                now(), now(), now()
            )
        """),
        {
            "id": str(uuid4()),
            "org_id": str(org_id),
            "agent_id": str(agent_id),
            "content": summary,
            "metadata": json.dumps(metadata),
        },
    )


async def run_summarize(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    gateway: LLMGateway | None,
    dataset_id: UUID | None = None,
    max_clusters: int = 10,
    cluster_strategy: str = "memo_type",
) -> SummarizeResult:
    if gateway is None:
        return SummarizeResult(0, 0, 0)

    # v0.19.5 — graph-aware clustering when operators opt in.
    if cluster_strategy == "connected_components":
        clusters = await _cluster_memos_by_components(
            conn, org_id, dataset_id, max_clusters
        )
    else:
        clusters = await _cluster_memos(conn, org_id, dataset_id, max_clusters)
    summaries_written = 0
    cached_skips = 0

    for member_ids in clusters:
        h = _cluster_hash(member_ids)
        if await _existing_summary_hash(conn, org_id, h):
            cached_skips += 1
            continue

        try:
            body = await _fetch_contents(conn, member_ids)
            if not body.strip():
                continue
            summary = await summarize_text(body, gateway=gateway, style="concise")
        except Exception as exc:
            log.warning("refine.summarize.llm_failed", error=str(exc))
            continue
        if not summary.strip():
            continue

        try:
            await _write_summary_memo(
                conn,
                org_id=org_id,
                cluster_hash=h,
                member_ids=member_ids,
                summary=summary,
            )
        except Exception as exc:
            log.warning("refine.summarize.write_failed", error=str(exc))
            continue
        summaries_written += 1

    return SummarizeResult(
        clusters_examined=len(clusters),
        summaries_written=summaries_written,
        summaries_skipped_cached=cached_skips,
    )

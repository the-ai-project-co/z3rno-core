"""recall() - the core memory retrieval operation.

Combines vector similarity search, metadata filtering, temporal queries,
and optional graph context enrichment into a single ranked result set.

Every recall():
  1. Embeds the query text (if provided)
  2. Builds a filtered SQL query (org_id, memory_type, metadata, temporal)
  3. Executes pgvector cosine similarity search
  4. Optionally enriches with graph context (N-hop traversal)
  5. Ranks results by fused relevance score
  6. Updates recall_count / last_recalled_at on returned memories
  7. Logs an audit entry

Default safety filters (always applied unless overridden):
  - valid_to IS NULL (only current versions)
  - deleted_at IS NULL (exclude soft-deleted)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.audit import create_audit_entry
from z3rno_core.engine.embedding import EmbeddingProvider


@dataclass(frozen=True)
class RecallResult:
    """A single result from a recall() query."""

    memory_id: UUID
    content: str
    summary: str | None
    memory_type: str
    similarity_score: float
    importance_score: float
    relevance_score: float  # Fused final score
    recall_count: int
    created_at: datetime
    valid_from: datetime
    metadata: dict[str, Any]
    graph_context: list[dict[str, Any]] = field(default_factory=list)


class RecallError(Exception):
    """Raised when recall() fails."""


async def recall(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    query: str | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    memory_type: str | None = None,
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    similarity_threshold: float = 0.0,
    time_range: tuple[datetime, datetime] | None = None,
    as_of: datetime | None = None,
    include_deleted: bool = False,
    # Relevance scoring weights (must sum to ~1.0)
    similarity_weight: float = 0.60,
    importance_weight: float = 0.25,
    recency_weight: float = 0.15,
    # Audit context
    user_id: UUID | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> list[RecallResult]:
    """Recall memories with vector search, filtering, and ranking.

    Args:
        conn: Active async connection (must be in a transaction).
        org_id: Tenant org_id.
        agent_id: Agent performing the recall.
        query: Text query to embed for similarity search. If None,
            returns memories by recency/importance only.
        embedding_provider: Provider to embed the query text.
        memory_type: Filter to a specific memory type.
        filters: JSONB metadata containment filter (e.g. {"tag": "important"}).
        top_k: Maximum number of results to return.
        similarity_threshold: Minimum cosine similarity (0-1). Only applies
            when query is provided.
        time_range: Filter memories created in (start, end) range.
        as_of: SCD Type 2 point-in-time query. Returns versions valid
            at this timestamp instead of current versions.
        include_deleted: If True, includes soft-deleted memories.
        similarity_weight: Weight for cosine similarity in fused scoring (default 0.60).
        importance_weight: Weight for importance score in fused scoring (default 0.25).
        recency_weight: Weight for recency score in fused scoring (default 0.15).
        user_id: For audit log.
        api_key_id: For audit log.
        ip_address: For audit log.
        user_agent: For audit log.
        request_id: For audit log.

    Returns:
        List of RecallResult ordered by relevance_score descending.

    Raises:
        RecallError: If scoring weights do not sum to approximately 1.0.
    """
    # Validate scoring weights sum to ~1.0
    weight_sum = similarity_weight + importance_weight + recency_weight
    if abs(weight_sum - 1.0) > 0.01:
        raise RecallError(
            f"Scoring weights must sum to ~1.0 (got {weight_sum:.4f}): "
            f"similarity={similarity_weight}, importance={importance_weight}, "
            f"recency={recency_weight}"
        )

    # --- Build the query ---
    conditions: list[str] = ["org_id = CAST(:org_id AS uuid)"]
    params: dict[str, Any] = {"org_id": str(org_id)}

    # Agent filter
    conditions.append("agent_id = CAST(:agent_id AS uuid)")
    params["agent_id"] = str(agent_id)

    # Temporal: as_of overrides default "current only" filter
    if as_of:
        conditions.append("valid_from <= CAST(:as_of AS timestamptz)")
        conditions.append("(valid_to IS NULL OR valid_to > CAST(:as_of AS timestamptz))")
        params["as_of"] = as_of
    else:
        conditions.append("valid_to IS NULL")

    # Soft-delete filter
    if not include_deleted:
        conditions.append("deleted_at IS NULL")

    # Memory type filter
    if memory_type:
        conditions.append("memory_type = CAST(:memory_type AS memory_type_enum)")
        params["memory_type"] = memory_type

    # Time range filter
    if time_range:
        conditions.append("created_at >= CAST(:time_start AS timestamptz)")
        conditions.append("created_at <= CAST(:time_end AS timestamptz)")
        params["time_start"] = time_range[0]
        params["time_end"] = time_range[1]

    # Metadata JSONB containment filter
    if filters:
        conditions.append("metadata @> CAST(:meta_filter AS jsonb)")
        params["meta_filter"] = json.dumps(filters)

    where_clause = " AND ".join(conditions)

    # --- Vector similarity search or fallback to recency ---
    # Pre-filtering optimization: org_id is always the FIRST condition in the
    # WHERE clause so that PostgreSQL can use it to narrow the candidate set
    # BEFORE the HNSW index scan. This avoids scanning vectors belonging to
    # other tenants, significantly improving recall latency in multi-tenant
    # deployments.
    #
    # TODO(perf): For large multi-tenant deployments, create partial HNSW
    # indexes per org_id to allow truly tenant-scoped ANN search:
    #
    #   CREATE INDEX ix_memories_embedding_hnsw_<org_id>
    #   ON memories USING hnsw (embedding vector_cosine_ops)
    #   WITH (m = 16, ef_construction = 200)
    #   WHERE org_id = '<org_id>';
    #
    # This requires a migration per tenant (or a dynamic DDL job). Until then,
    # the pre-filter WHERE clause + ix_memories_org_agent_valid B-tree index
    # gives PostgreSQL enough information to narrow the scan via a bitmap
    # intersection with the HNSW index.
    if query and embedding_provider:
        query_embedding = await embedding_provider.embed_text(query)
        if query_embedding:
            vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            # The WHERE clause starts with org_id = ... so PostgreSQL can
            # pre-filter via the B-tree index before the HNSW vector scan.
            sql = f"""
                SELECT id, content, summary, memory_type, importance_score,
                       recall_count, created_at, valid_from, metadata,
                       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM memories
                WHERE {where_clause}
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :top_k
            """
            params["query_vec"] = vector_str
            params["top_k"] = top_k
        else:
            sql = _fallback_query(where_clause)
            params["top_k"] = top_k
    else:
        sql = _fallback_query(where_clause)
        params["top_k"] = top_k

    result = await conn.execute(text(sql), params)
    rows = result.fetchall()

    # --- Build results with fused relevance score ---
    now = datetime.now().astimezone()
    results: list[RecallResult] = []
    for row in rows:
        similarity = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0
        importance = float(row[4])

        # Apply similarity threshold
        if query and embedding_provider and similarity < similarity_threshold:
            continue

        # Fused relevance scoring with configurable weights
        recency_days = max((now - row[6]).total_seconds() / 86400, 0.01)
        recency_score = min(1.0, 1.0 / recency_days)  # More recent = higher

        if query and embedding_provider:
            relevance = (
                similarity_weight * similarity
                + importance_weight * importance
                + recency_weight * recency_score
            )
        else:
            # No similarity available; redistribute weight between importance and recency
            relevance = 0.50 * importance + 0.50 * recency_score

        results.append(
            RecallResult(
                memory_id=row[0],
                content=row[1],
                summary=row[2],
                memory_type=row[3],
                similarity_score=similarity,
                importance_score=importance,
                relevance_score=round(relevance, 4),
                recall_count=row[5],
                created_at=row[6],
                valid_from=row[7],
                metadata=row[8] if row[8] else {},
            )
        )

    # Sort by relevance
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    # --- Update recall_count and last_recalled_at ---
    if results:
        memory_ids = [str(r.memory_id) for r in results]
        id_list = ",".join(f"'{mid}'" for mid in memory_ids)
        await conn.execute(
            text(f"""
                UPDATE memories
                SET recall_count = recall_count + 1,
                    last_recalled_at = now(),
                    updated_at = now()
                WHERE id IN ({id_list})
            """)
        )

    # --- Audit log ---
    await create_audit_entry(
        conn,
        org_id=org_id,
        operation="recall",
        agent_id=agent_id,
        user_id=user_id,
        details={
            "query_length": len(query) if query else 0,
            "memory_type_filter": memory_type,
            "top_k": top_k,
            "result_count": len(results),
            "similarity_threshold": similarity_threshold,
        },
        api_key_id=api_key_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
    )

    return results


def _fallback_query(where_clause: str) -> str:
    """Query without vector similarity (fallback to importance + recency)."""
    return f"""
        SELECT id, content, summary, memory_type, importance_score,
               recall_count, created_at, valid_from, metadata,
               NULL AS similarity
        FROM memories
        WHERE {where_clause}
        ORDER BY importance_score DESC, created_at DESC
        LIMIT :top_k
    """

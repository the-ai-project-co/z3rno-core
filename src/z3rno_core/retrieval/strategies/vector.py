"""VECTOR retrieval — cosine similarity over the pgvector HNSW index.

Behavioural parity with the pre-Phase-C ``engine.recall`` vector
path. Lifted out into a strategy class so AUTO can delegate, AUTO+
re-rank can compose, and the engine layer becomes a thin dispatcher.

Score composition is identical to pre-C.1:
    relevance = 0.60 * similarity + 0.25 * importance + 0.15 * recency

Operators can tune the weights via the ``similarity_weight`` /
``importance_weight`` / ``recency_weight`` kwargs in ``**extra``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.engine.embedding import EmbeddingProvider
from z3rno_core.retrieval._filters import build_where_clause
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)


@register_strategy
class VectorStrategy(RetrievalStrategy):
    name = "VECTOR"
    requires_query_embedding = True
    requires_llm = False

    async def retrieve(
        self,
        conn: AsyncConnection,
        *,
        org_id: UUID,
        agent_id: UUID,
        query: str,
        top_k: int,
        memory_type: str | None = None,
        filters: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        # Strategy-specific kwargs.
        embedding_provider: EmbeddingProvider | None = extra.get("embedding_provider")
        time_range: tuple[datetime, datetime] | None = extra.get("time_range")
        as_of: datetime | None = extra.get("as_of")
        include_deleted: bool = bool(extra.get("include_deleted", False))
        similarity_weight: float = float(extra.get("similarity_weight", 0.60))
        importance_weight: float = float(extra.get("importance_weight", 0.25))
        recency_weight: float = float(extra.get("recency_weight", 0.15))

        if not _weights_sum_to_one(similarity_weight, importance_weight, recency_weight):
            raise ValueError(
                f"VECTOR weights must sum to ~1.0; got "
                f"similarity={similarity_weight}, importance={importance_weight}, "
                f"recency={recency_weight}"
            )

        where_clause, params = build_where_clause(
            org_id=org_id,
            agent_id=agent_id,
            memory_type=memory_type,
            filters=filters,
            time_range=time_range,
            as_of=as_of,
            include_deleted=include_deleted,
            conversation_id=extra.get("conversation_id"),
        )

        # Vector path requires query + provider; otherwise fall back to
        # importance / recency ranking (mirrors pre-Phase-C behaviour).
        vector_search = False
        if query and embedding_provider is not None:
            query_embedding = await embedding_provider.embed_text(query)
            if query_embedding:
                vector_search = True
                vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                params["query_vec"] = vector_str
                # The WHERE clause starts with org_id so Postgres can
                # pre-filter via the B-tree index before the HNSW scan.
                sql = f"""
                    SELECT id, content, summary, memory_type, importance_score,
                           recall_count, created_at, valid_from, metadata,
                           1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                    FROM public.memories
                    WHERE {where_clause}
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:query_vec AS vector)
                    LIMIT :top_k
                """
        if not vector_search:
            sql = _fallback_query(where_clause)

        params["top_k"] = top_k

        rows = (await conn.execute(text(sql), params)).fetchall()

        now = datetime.now().astimezone()
        results: list[StrategyResult] = []
        for row in rows:
            similarity = float(row[9]) if len(row) > 9 and row[9] is not None else 0.0
            importance = float(row[4])

            if vector_search and similarity < similarity_threshold:
                continue

            recency_days = max((now - row[6]).total_seconds() / 86400, 0.01)
            recency_score = min(1.0, 1.0 / recency_days)

            if vector_search:
                relevance = (
                    similarity_weight * similarity
                    + importance_weight * importance
                    + recency_weight * recency_score
                )
                score_components = {
                    "vector": round(similarity, 4),
                    "importance": round(importance, 4),
                    "recency": round(recency_score, 4),
                }
            else:
                # No similarity signal — redistribute weight between
                # importance and recency. Mirrors the pre-Phase-C
                # fallback exactly.
                relevance = 0.50 * importance + 0.50 * recency_score
                score_components = {
                    "importance": round(importance, 4),
                    "recency": round(recency_score, 4),
                }

            results.append(
                StrategyResult(
                    memory_id=row[0],
                    content=row[1],
                    summary=row[2],
                    memory_type=row[3],
                    importance_score=importance,
                    relevance_score=round(relevance, 4),
                    recall_count=row[5],
                    created_at=row[6],
                    valid_from=row[7],
                    metadata=row[8] if row[8] else {},
                    score_components=score_components,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results


def _fallback_query(where_clause: str) -> str:
    """Importance + recency ordering when no vector signal is available."""
    return f"""
        SELECT id, content, summary, memory_type, importance_score,
               recall_count, created_at, valid_from, metadata,
               NULL AS similarity
        FROM public.memories
        WHERE {where_clause}
        ORDER BY importance_score DESC, created_at DESC
        LIMIT :top_k
    """


def _weights_sum_to_one(*weights: float, tolerance: float = 0.01) -> bool:
    # Same tolerance as the legacy engine.recall validation (0.01) so the
    # two checks agree. Engine layer validates first with a RecallError;
    # this is defence-in-depth.
    return math.isclose(sum(weights), 1.0, abs_tol=tolerance)

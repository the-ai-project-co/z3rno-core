"""LEXICAL retrieval — Postgres full-text BM25-style ranking.

Uses ``plainto_tsquery`` so callers pass natural-language queries
without learning tsquery syntax (no ``&``, ``|``, ``!`` operators —
those are escaped to plain words). ``ts_rank`` produces a 0..N
relevance score; we normalise it to [0,1] for consistency with the
other strategies' ``relevance_score`` contract.

Migration 022 added ``memories.content_tsv`` as a GENERATED column
plus a GIN index — lookups are O(log n) on big tenants.

LEXICAL ignores ``similarity_threshold`` (that's a vector concept).
Callers wanting "min relevance" should filter the response client-
side or compose with re-ranker in C.3.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.retrieval._filters import build_where_clause
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)


@register_strategy
class LexicalStrategy(RetrievalStrategy):
    name = "LEXICAL"
    requires_query_embedding = False
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
        time_range: tuple[datetime, datetime] | None = extra.get("time_range")
        as_of: datetime | None = extra.get("as_of")
        include_deleted: bool = bool(extra.get("include_deleted", False))
        # Postgres text-search config — 'english' default with stemming +
        # stop-word handling. Operators serving other languages can
        # override per-request without a schema change.
        ts_config: str = str(extra.get("ts_config", "english"))

        if not query or not query.strip():
            # Empty query means "no lexical signal" — return nothing rather
            # than dumping the whole table at LIMIT :top_k. Callers that
            # want recency-only should use VECTOR with no query.
            return []

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

        params["query"] = query
        params["top_k"] = top_k
        params["ts_config"] = ts_config

        # plainto_tsquery is intentionally forgiving — empty result on
        # a stop-words-only query rather than a syntax error.
        sql = f"""
            SELECT id, content, summary, memory_type, importance_score,
                   recall_count, created_at, valid_from, metadata,
                   ts_rank(content_tsv, plainto_tsquery(:ts_config, :query)) AS rank
            FROM public.memories
            WHERE {where_clause}
              AND content_tsv @@ plainto_tsquery(:ts_config, :query)
            ORDER BY rank DESC, created_at DESC
            LIMIT :top_k
        """

        rows = (await conn.execute(text(sql), params)).fetchall()

        now = datetime.now().astimezone()
        results: list[StrategyResult] = []
        for row in rows:
            raw_rank = float(row[9]) if row[9] is not None else 0.0
            # ts_rank values are unbounded above; in practice they stay
            # well below 1 for typical content, but pathological cases
            # can exceed it. Clamp + linear normalise to [0, 1] using
            # a soft cap of 1.0 — anything above 1.0 maps to 1.0.
            normalised_rank = min(raw_rank, 1.0)

            importance = float(row[4])
            recency_days = max((now - row[6]).total_seconds() / 86400, 0.01)
            recency_score = min(1.0, 1.0 / recency_days)

            # 70% lexical, 20% importance, 10% recency. Lexical relevance
            # is the primary signal for this strategy; importance + recency
            # tie-break similarly-ranked rows.
            relevance = 0.70 * normalised_rank + 0.20 * importance + 0.10 * recency_score

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
                    score_components={
                        "lexical": round(normalised_rank, 4),
                        "importance": round(importance, 4),
                        "recency": round(recency_score, 4),
                    },
                )
            )

        return results

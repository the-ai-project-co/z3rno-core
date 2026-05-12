"""CODE retrieval strategy (Phase D slice 5).

Surfaces code-graph Memos written by ``z3rno_core.codegraph``. The
strategy combines:

  1. A lexical filter on the qualified name / content (cheap, narrows
     to the entity the caller likely meant).
  2. A one-hop neighborhood pull over ``memory_relationships`` filtered
     to codegraph edges — so a query for ``"main"`` returns ``main``
     plus its CALLS / DEFINES / IMPORTS neighbors.

The strategy reads the edge ``metadata.codegraph_kind`` discriminator,
so it does not require any schema extension to the
``relationship_type_enum``. Other strategies are unaffected.

CODE inherits no LLM dependency — it is plain SQL, cheap to run, and
safe to register alongside the Phase C strategies.
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
class CodeStrategy(RetrievalStrategy):
    name = "CODE"
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
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float = 0.0,
        **extra: Any,
    ) -> list[StrategyResult]:
        time_range: tuple[datetime, datetime] | None = extra.get("time_range")
        as_of: datetime | None = extra.get("as_of")
        include_deleted: bool = bool(extra.get("include_deleted", False))
        # Include neighbors of every Memo that lexically matches the
        # query. Default 1 — the call graph extends one hop. Operators
        # can pass `hops=2` for module-level summaries.
        hops: int = max(1, int(extra.get("hops", 1)))

        if not query or not query.strip():
            return []

        where_clause, params = build_where_clause(
            org_id=org_id,
            agent_id=agent_id,
            memory_type=memory_type,
            user_id=extra.get("user_id"),
            metadata_filter=metadata_filter,
            conversation_id=extra.get("conversation_id"),
            time_range=time_range,
            as_of=as_of,
            include_deleted=include_deleted,
        )
        params["query"] = query
        params["top_k"] = top_k

        # Seed: every code-Memo whose qualified_name or content contains
        # the query as a substring (case-insensitive). The
        # `memo_type LIKE 'CODE_%'` filter is the cheap discriminator
        # between code-graph rows and other Memos.
        seed_sql = f"""
            SELECT id, content, summary, memory_type, importance_score,
                   recall_count, created_at, valid_from, metadata
            FROM public.memories
            WHERE {where_clause}
              AND memo_type LIKE 'CODE_%'
              AND (
                    content ILIKE '%' || :query || '%'
                 OR metadata->>'qualified_name' ILIKE '%' || :query || '%'
              )
            ORDER BY importance_score DESC, created_at DESC
            LIMIT :top_k
        """
        seed_rows = (await conn.execute(text(seed_sql), params)).fetchall()
        seed_ids = [row[0] for row in seed_rows]
        if not seed_ids:
            return []

        # Neighbor pull. Up to ``hops`` away, following only codegraph
        # edges. Implemented as a recursive CTE for portability —
        # works regardless of whether AGE is loaded.
        neighbor_sql = """
            WITH RECURSIVE walk(memory_id, hop) AS (
                SELECT id, 0 FROM public.memories
                WHERE id = ANY(CAST(:seed_ids AS uuid[]))

                UNION

                SELECT
                    CASE WHEN mr.source_memory_id = w.memory_id
                         THEN mr.target_memory_id
                         ELSE mr.source_memory_id
                    END,
                    w.hop + 1
                FROM walk w
                JOIN public.memory_relationships mr
                  ON (mr.source_memory_id = w.memory_id
                      OR mr.target_memory_id = w.memory_id)
                 AND mr.metadata ? 'codegraph_kind'
                WHERE w.hop < :hops
            )
            SELECT m.id, m.content, m.summary, m.memory_type, m.importance_score,
                   m.recall_count, m.created_at, m.valid_from, m.metadata,
                   MIN(w.hop) AS min_hop
            FROM walk w
            JOIN public.memories m ON m.id = w.memory_id
            WHERE m.deleted_at IS NULL
            GROUP BY m.id, m.content, m.summary, m.memory_type,
                     m.importance_score, m.recall_count,
                     m.created_at, m.valid_from, m.metadata
            ORDER BY min_hop ASC, m.importance_score DESC
            LIMIT :top_k
        """
        rows = (
            await conn.execute(
                text(neighbor_sql),
                {"seed_ids": [str(i) for i in seed_ids], "hops": hops, "top_k": top_k},
            )
        ).fetchall()

        results: list[StrategyResult] = []
        for row in rows:
            hop = int(row[9])
            # Closer to the seed → higher relevance. Hop 0 → 1.0, hop 1
            # → 0.66, etc. Clamped to [0, 1].
            hop_score = max(0.0, 1.0 - hop / (hops + 1))
            importance = float(row[4])
            relevance = 0.70 * hop_score + 0.30 * importance
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
                        "hop_score": round(hop_score, 4),
                        "importance": round(importance, 4),
                        "min_hop": float(hop),
                    },
                )
            )
        return results

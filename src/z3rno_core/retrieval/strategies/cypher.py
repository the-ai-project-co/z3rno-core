"""CYPHER retrieval — raw operator-supplied Cypher passthrough.

Production-default OFF. The strategy itself enforces the gate via
``allow_cypher_query`` in ``**extra`` (defaults to ``False``); the
server config flag ``ALLOW_CYPHER_QUERY=true`` flips this on for
deployments that trust the caller. With the gate off, calling
``CYPHER`` raises ``CypherDisabledError``; the server maps that to
HTTP 403.

The caller passes their own Cypher via ``raw_cypher`` in ``**extra``.
We still apply the same read-only validator as ``ASK`` (no CREATE /
MERGE / DELETE / etc.) — even an operator-trusted query path
shouldn't be able to mutate the graph through a retrieval endpoint.
Operators who need mutating Cypher should use the graph_writer API
directly, not recall.

The pattern is intentionally narrow: no LLM, no NL→Cypher translation
— just "execute this exact Cypher and return rows mapped to
StrategyResult". For NL→Cypher use ``strategy="ASK"`` which lands the
LLM-translation layer.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)
from z3rno_core.retrieval.strategies.ask import (
    _execute_cypher_sync,
    _validate_cypher,
)

logger = logging.getLogger(__name__)


class CypherDisabledError(Exception):
    """Raised when CYPHER is invoked but ``allow_cypher_query`` is False.

    Server-side this becomes a 403 — the strategy exists but the
    operator has disabled it.
    """


class CypherValidationError(ValueError):
    """The supplied Cypher failed the read-only validator."""


@register_strategy
class CypherStrategy(RetrievalStrategy):
    name = "CYPHER"
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
        del agent_id, memory_type, metadata_filter, similarity_threshold, query

        allow = bool(extra.get("allow_cypher_query", False))
        if not allow:
            raise CypherDisabledError(
                "CYPHER strategy is disabled on this server "
                "(set ALLOW_CYPHER_QUERY=true to enable)"
            )

        raw_cypher: str = str(extra.get("raw_cypher") or "").strip()
        if not raw_cypher:
            raise CypherValidationError(
                "CYPHER requires raw_cypher in extra; got empty/None"
            )

        invalid = _validate_cypher(raw_cypher)
        if invalid:
            raise CypherValidationError(f"Cypher rejected: {invalid}")

        try:
            ids, aggregate = await conn.run_sync(
                lambda sync_conn: _execute_cypher_sync(
                    sync_conn, org_id=org_id, cypher=raw_cypher, limit=top_k
                )
            )
        except (DBAPIError, ValueError) as exc:
            logger.warning(
                "cypher.execute.failed", extra={"error": str(exc)[:200]}
            )
            return []

        if not ids:
            # Pure aggregate result (e.g. count) — surface as a synthetic
            # row mirroring ASK's pattern. Aggregate value rides on
            # results[0].metadata["cypher_aggregate"].
            if aggregate is not None:
                from z3rno_core.retrieval.strategies.ask import _aggregate_result  # noqa: PLC0415

                synthetic = _aggregate_result(
                    query="<raw cypher>",
                    aggregate=aggregate,
                    cypher=raw_cypher,
                )
                # Re-key the metadata so consumers can distinguish ASK vs CYPHER aggregates.
                metadata = dict(synthetic.metadata)
                metadata["cypher_aggregate"] = metadata.pop("ask_aggregate")
                metadata["cypher_query"] = metadata.pop("ask_query")
                metadata["cypher"] = metadata.pop("ask")["cypher"]
                return [
                    StrategyResult(
                        memory_id=synthetic.memory_id,
                        content=synthetic.content,
                        summary=synthetic.summary,
                        memory_type=synthetic.memory_type,
                        importance_score=synthetic.importance_score,
                        relevance_score=synthetic.relevance_score,
                        recall_count=synthetic.recall_count,
                        created_at=synthetic.created_at,
                        valid_from=synthetic.valid_from,
                        metadata=metadata,
                        score_components=synthetic.score_components,
                    )
                ]
            return []

        # Resolve ids → full memos.
        rows = (
            await conn.execute(
                text("""
                    SELECT id, content, summary, memory_type, importance_score,
                           recall_count, created_at, valid_from, metadata
                    FROM public.memories
                    WHERE org_id = CAST(:org_id AS uuid)
                      AND id = ANY(CAST(:ids AS uuid[]))
                      AND deleted_at IS NULL
                      AND valid_to IS NULL
                """),
                {"org_id": str(org_id), "ids": [str(i) for i in ids]},
            )
        ).fetchall()

        out: list[StrategyResult] = []
        for i, row in enumerate(rows):
            metadata = dict(row[8] or {})
            if i == 0:
                metadata["cypher"] = raw_cypher
            out.append(
                StrategyResult(
                    memory_id=UUID(str(row[0])),
                    content=row[1],
                    summary=row[2],
                    memory_type=row[3],
                    importance_score=float(row[4]),
                    relevance_score=round(1.0 / (i + 1), 4),
                    recall_count=row[5],
                    created_at=row[6],
                    valid_from=row[7],
                    metadata=metadata,
                    score_components={"cypher_rank": round(1.0 / (i + 1), 4)},
                )
            )
        return out

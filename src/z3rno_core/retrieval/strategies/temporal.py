"""TEMPORAL retrieval — SCD-2 time-travel via ``valid_from`` / ``valid_to``.

Two ways to specify the point in time:

  1. Caller passes ``as_of=datetime`` in ``**extra``. Direct path; no
     LLM needed.
  2. Query contains a natural-language time hint ("last week",
     "before March 2025", "what did the user know in 2023"). When an
     ``llm_gateway`` is configured, TEMPORAL extracts a timestamp via
     structured output and uses it.

After resolving the timestamp, TEMPORAL delegates to ``VectorStrategy``
with ``as_of=resolved_ts``. VectorStrategy already enforces SCD-2 in
its WHERE clause (``valid_from <= :as_of AND (valid_to IS NULL OR
valid_to > :as_of)``), so this strategy is a thin wrapper plus
time-parsing.

When neither ``as_of`` nor an LLM is available, TEMPORAL collapses to
current-time recall — functionally equivalent to ``strategy="VECTOR"``
but documents the intent in the audit row.

Result enrichment: ``results[i].metadata["temporal_as_of"]`` carries
the ISO-8601 timestamp the strategy resolved to. Useful for caller
auditing ("which moment in history did the recall actually use?").
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import LLMGateway, LLMGatewayError
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)
from z3rno_core.retrieval.strategies.vector import VectorStrategy

logger = logging.getLogger(__name__)


class _ResolvedTimestamp(BaseModel):
    """LLM-extracted timestamp from a natural-language time hint.

    ``timestamp`` is ISO-8601 UTC. Empty string signals "no time hint
    detected in the query — use current time".
    """

    timestamp: str = Field(
        "",
        description=(
            "ISO-8601 UTC timestamp (e.g. '2024-03-05T00:00:00Z') the "
            "query refers to. Empty string when the query has no time "
            "hint."
        ),
        max_length=64,
    )
    rationale: str = Field("", description="One-sentence reason.", max_length=240)


@register_strategy
class TemporalStrategy(RetrievalStrategy):
    name = "TEMPORAL"
    requires_query_embedding = True  # delegates to VECTOR
    requires_llm = False  # time-parsing is optional

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
        embedding_provider = extra.get("embedding_provider")
        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        as_of: datetime | None = extra.get("as_of")

        # TEMPORAL is meaningless without a query — and the delegated
        # VECTOR path needs an embedder. Mirror VECTOR's preconditions
        # so we don't waste a fallback SELECT on an obviously-empty
        # request.
        if not query.strip() or embedding_provider is None:
            return []

        # Caller didn't supply a timestamp → ask the LLM to extract one.
        # Without an LLM we run a current-time recall (still useful — at
        # least we get back today's view, and the audit row records
        # the temporal intent).
        if as_of is None and llm_gateway is not None:
            try:
                resolved = await self._extract_timestamp(llm_gateway, query=query)
                if resolved:
                    as_of = resolved
            except LLMGatewayError:
                logger.warning("temporal.extract.llm_failed", exc_info=True)
            except Exception:
                logger.warning("temporal.extract.unexpected", exc_info=True)

        # Forward to VECTOR with the resolved timestamp. Mutate a copy of
        # extra so the inner call sees as_of even when the caller didn't
        # pass one.
        delegated_extra = dict(extra)
        if as_of is not None:
            delegated_extra["as_of"] = as_of

        vector_results = await VectorStrategy().retrieve(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold,
            **delegated_extra,
        )

        # Tag the resolved timestamp on every result so callers can show
        # "as of <date>" badges.
        if as_of is None:
            return vector_results

        ts_iso = as_of.isoformat()

        # Phase F slice 3 — when a memo_versions row exists for the
        # Memo at ``as_of``, merge those historical properties into
        # the result metadata. The row-level SCD-2 already handled
        # content / valid_from; this layer adds graph-projected
        # properties (memo_type, ontology_uri, refine lineage, etc.).
        from z3rno_core.temporal.memo_versioning import (  # noqa: PLC0415
            get_memo_at,
        )

        out: list[StrategyResult] = []
        for r in vector_results:
            new_metadata = dict(r.metadata)
            new_metadata["temporal_as_of"] = ts_iso
            new_components = dict(r.score_components)
            new_components["temporal"] = 1.0

            try:
                version = await get_memo_at(conn, memo_id=r.memory_id, as_of=as_of)
            except Exception:
                version = None
            if version is not None:
                new_metadata["memo_version"] = version.version
                new_metadata["memo_version_properties"] = version.properties
                new_components["memo_version_hit"] = 1.0

            out.append(
                StrategyResult(
                    memory_id=r.memory_id,
                    content=r.content,
                    summary=r.summary,
                    memory_type=r.memory_type,
                    importance_score=r.importance_score,
                    relevance_score=r.relevance_score,
                    recall_count=r.recall_count,
                    created_at=r.created_at,
                    valid_from=r.valid_from,
                    metadata=new_metadata,
                    score_components=new_components,
                    graph_context=r.graph_context,
                )
            )
        return out

    async def _extract_timestamp(self, gateway: LLMGateway, *, query: str) -> datetime | None:
        """LLM extracts an ISO-8601 timestamp from time hints in ``query``."""
        system = (
            "Extract the point-in-time the user is asking about. Return an "
            "ISO-8601 UTC timestamp (e.g. '2024-03-05T00:00:00Z'). If the "
            "query has no time hint, return an empty string. Examples:\n"
            "  'what did the user say last week' → "
            "approximate one week before today, ISO-8601.\n"
            "  'memories from March 2024' → '2024-03-15T00:00:00Z' "
            "(pick a representative point in the month).\n"
            "  'recent thoughts on X' → '' (no specific time).\n"
            "  'on 2023-06-12' → '2023-06-12T00:00:00Z'."
        )
        user = f"Query: {query}"
        resolved = await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_ResolvedTimestamp,
            max_tokens=120,
            temperature=0.0,
        )
        raw = resolved.timestamp.strip()
        if not raw:
            return None
        try:
            # Accept both 'Z' and '+00:00' suffixes.
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            logger.warning("temporal.extract.parse_failed", extra={"raw": raw})
            return None

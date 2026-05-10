"""AUTO retrieval — the router that picks the best strategy per query.

**Phase C.1 ships the skeleton.** AUTO delegates to VECTOR
unconditionally and tags the response so consumers know they got the
default. This keeps Phase C.1 byte-identical to pre-Phase-C behaviour
when the caller uses ``strategy="AUTO"`` (the new global default).

**Phase C.3** replaces ``_classify`` with an LLM-driven classifier
that returns one of the registered strategy names. The interface
documented here is the contract C.3 will fulfill — feel free to call
``AutoStrategy().retrieve()`` from production today and get VECTOR
behaviour; the upgrade is invisible to consumers.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    get_strategy,
    register_strategy,
)


@register_strategy
class AutoStrategy(RetrievalStrategy):
    name = "AUTO"

    # ``requires_llm`` flips to True in C.3 when the classifier becomes
    # real. Today AUTO is no-op routing — keep it False so deployments
    # without an LLM gateway can still use AUTO transparently.
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
        chosen = await self._classify(query=query, **extra)
        strategy_cls = get_strategy(chosen)
        return await strategy_cls().retrieve(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            filters=filters,
            similarity_threshold=similarity_threshold,
            **extra,
        )

    async def _classify(self, *, query: str, **extra: Any) -> str:
        """Pick the best strategy for ``query``.

        Phase C.1: returns ``"VECTOR"`` unconditionally. The engine
        layer reads the response's ``strategies_considered`` to log
        the routing decision — we tag the AUTO→VECTOR fall-through
        as ``"AUTO->VECTOR"`` for clarity.

        Phase C.3: calls ``llm_gateway`` (from ``**extra``) to classify
        the query into one of the registered strategy names. Returns
        ``"VECTOR"`` on classifier timeout / unavailable / unknown
        result — fail-safe by default.
        """
        del query, extra
        return "VECTOR"

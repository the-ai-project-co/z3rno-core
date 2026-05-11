"""AUTO retrieval — LLM-driven router that picks the best strategy per query.

Phase C.3 replaces the C.1 skeleton's "always VECTOR" with a real
LLM classifier:

  1. Ask ``llm_gateway`` (Pydantic-validated structured output) to
     classify the query into one of the *currently registered* strategy
     names — VECTOR / LEXICAL / GRAPH / TRIPLET / TRACE / ASK / TEMPORAL.
     CYPHER is excluded because it requires a hand-written query.
  2. Validate the classifier's choice against the registry. Unknown
     → fall back to VECTOR.
  3. Delegate ``retrieve()`` to the chosen strategy with the original
     kwargs.

Fail-safe defaults: any failure of the LLM call (timeout, network,
rate limit, validation error) silently falls back to VECTOR rather
than 500ing the caller. AUTO is meant to be the "always works" entry
point — degrading to a sensible default beats refusing the request.

Engine integration: the engine reads ``self.delegated_to`` after
``retrieve()`` to populate ``RecallResponse.strategies_considered``
with ``[f"AUTO->{chosen}"]``. Audit rows record the actual delegate
so historical queries can be analysed by strategy.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import LLMGateway, LLMGatewayError
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    UnknownStrategyError,
    get_strategy,
    register_strategy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier I/O
# ---------------------------------------------------------------------------


class _ClassifierChoice(BaseModel):
    """LLM-emitted strategy choice + brief reason.

    ``strategy`` must be one of the canonical names. Any other value
    triggers a registry lookup which raises and the AUTO router falls
    back to VECTOR. Reasoning is purely diagnostic — it lives in the
    audit row's ``details`` field via ``classifier_reason``.
    """

    strategy: str = Field(
        ...,
        description=(
            "Canonical strategy name (UPPERCASE). One of: VECTOR, LEXICAL, "
            "GRAPH, TRIPLET, TRACE, TEMPORAL, ASK. Use VECTOR for "
            "open-ended natural-language semantic search."
        ),
    )
    reason: str = Field(
        "",
        description="One-sentence rationale for the chosen strategy.",
        max_length=240,
    )


# Strategy names AUTO is allowed to choose from. ``AUTO`` is excluded
# (recursive!), ``CYPHER`` is excluded (raw query, not router-picked).
# This list is intentionally explicit rather than derived from the
# registry — we don't want a future strategy to silently become an
# AUTO target without operator review.
_AUTO_CANDIDATE_STRATEGIES = (
    "VECTOR",
    "LEXICAL",
    "GRAPH",
    "TRIPLET",
    "TRACE",
    "TEMPORAL",
    "ASK",
)


@register_strategy
class AutoStrategy(RetrievalStrategy):
    name = "AUTO"

    # C.3: AUTO can call the LLM when available, but degrades to VECTOR
    # without one — so `requires_llm=False` is correct (AUTO can run
    # without it). The classifier picks the best strategy *when* an LLM
    # is configured.
    requires_llm = False

    def __init__(self) -> None:
        # Populated after retrieve() so the engine layer can read what
        # AUTO actually chose for audit + response provenance.
        self.delegated_to: str = "VECTOR"
        self.classifier_reason: str = ""

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
        self.delegated_to = chosen

        try:
            strategy_cls = get_strategy(chosen)
        except UnknownStrategyError:
            logger.warning(
                "auto.classifier.unknown_strategy",
                extra={"chosen": chosen, "fallback": "VECTOR"},
            )
            self.delegated_to = "VECTOR"
            strategy_cls = get_strategy("VECTOR")

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

        Returns one of the registered strategy names. Falls back to
        ``"VECTOR"`` on any classifier failure — AUTO must never raise.

        Caching: callers can pass a dict-like ``classifier_cache`` in
        ``**extra`` with ``get(key)`` / ``set(key, value)``. The cache
        key is the query string (callers wanting hash-keyed Valkey
        caches wrap the dict-like adapter accordingly).
        """
        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        if llm_gateway is None or not query.strip():
            # No LLM → byte-identical to C.1 behaviour. Empty / whitespace-
            # only queries never benefit from classification.
            self.classifier_reason = "no_llm_gateway"
            return "VECTOR"

        # Optional cache.
        cache = extra.get("classifier_cache")
        cache_key = query.strip()
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                self.classifier_reason = "cache_hit"
                return str(cached)

        try:
            choice = await self._invoke_classifier(llm_gateway, query)
        except LLMGatewayError as exc:
            logger.warning(
                "auto.classifier.llm_failed",
                extra={"error": str(exc)[:200]},
            )
            self.classifier_reason = f"llm_failed:{type(exc).__name__}"
            return "VECTOR"
        except Exception as exc:
            # Pydantic ValidationError + anything else. Never propagate.
            logger.warning(
                "auto.classifier.unexpected_error",
                extra={"error": str(exc)[:200]},
            )
            self.classifier_reason = f"unexpected:{type(exc).__name__}"
            return "VECTOR"

        chosen = (choice.strategy or "").upper().strip()
        if chosen not in _AUTO_CANDIDATE_STRATEGIES:
            logger.warning(
                "auto.classifier.out_of_set",
                extra={"chosen": chosen, "fallback": "VECTOR"},
            )
            self.classifier_reason = f"out_of_set:{chosen}"
            chosen = "VECTOR"
        else:
            self.classifier_reason = choice.reason

        if cache is not None:
            cache.set(cache_key, chosen)

        return chosen

    async def _invoke_classifier(
        self, gateway: LLMGateway, query: str
    ) -> _ClassifierChoice:
        """LLM call to classify ``query`` into a strategy + reason."""
        system = _CLASSIFIER_SYSTEM_PROMPT
        user = f"Query: {query}"
        return await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_ClassifierChoice,
            max_tokens=120,
            temperature=0.0,
        )


# Constructed once at module load — used as the classifier system prompt.
# Keeping it module-level (not inside the class) so test introspection
# can verify the prompt shape without instantiating AutoStrategy.
_CLASSIFIER_SYSTEM_PROMPT = """\
You are a query-router for a memory database. Decide which retrieval
strategy best fits the user's query. You MUST choose from this
exact set:

  VECTOR    Open-ended semantic search ("things related to X").
  LEXICAL   Exact-word / phrase match ("contains the word Postgres").
  GRAPH     "Explain how X and Y are connected" — needs relationships.
  TRIPLET   Direct factual question of the form
            "What/Who/Where (subject) (predicate) ?" or
            "Who/What ? (predicate) (object)".
  TRACE     Multi-step reasoning — "find X, then explain why".
  TEMPORAL  Time-aware ("what did the user know on date X").
  ASK       Natural-language analytics ("how many memories about Y").

Return JSON: {"strategy": "<NAME>", "reason": "<one sentence>"}.

Default to VECTOR when in doubt — it's the safest fallback.
""".strip()

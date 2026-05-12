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
import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import text
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

# Strategies that require a Forge-distilled corpus to return anything
# meaningful. On a fresh tenant (or any tenant that hasn't run distill
# yet), these strategies have no rows to traverse and silently return
# 0 results. v0.21.3 — when the AUTO classifier picks one of these,
# probe ``memory_relationships`` first; if empty for this (org, agent),
# downgrade to VECTOR. Closes UX trap #7 from
# V0-21-2-AS-A-USER-2026-05-12.
_GRAPH_DEPENDENT_STRATEGIES = frozenset({"GRAPH", "TRIPLET", "TRACE", "ASK"})

# Process-local TTL cache of "does this (org, agent) have a Forge
# corpus?". Keyed by ``(org_id, agent_id)``. Value is
# ``(checked_at_monotonic, has_graph_corpus)``. The check is one
# ``SELECT 1 FROM memory_relationships ... LIMIT 1`` — cheap, but no
# need to repeat it every recall. TTL is short enough that the moment
# a tenant lands its first distill, the next 60 s of recalls catch up.
_GRAPH_CORPUS_CACHE: dict[tuple[str, str], tuple[float, bool]] = {}
_GRAPH_CORPUS_TTL_SECONDS = 60.0


async def _has_graph_corpus(
    conn: AsyncConnection, *, org_id: UUID, agent_id: UUID
) -> bool:
    """Return True when the org has any AGE-projected edges.

    Probes ``memory_relationships`` (the canonical edge table that
    Forge distill, refine, and graph_writer all write to). The table
    is org-scoped only (no ``agent_id`` column — edges live across
    agents within a tenant), so the probe filters on org alone but
    the cache key still includes agent for forward-compat. Caches
    the answer per (org, agent) for ``_GRAPH_CORPUS_TTL_SECONDS``.
    On any DB error, returns True conservatively — "we don't know, let
    the chosen strategy try" — so this can never make AUTO worse than
    the pre-fix behaviour.
    """
    key = (str(org_id), str(agent_id))
    now = time.monotonic()
    cached = _GRAPH_CORPUS_CACHE.get(key)
    if cached is not None and now - cached[0] < _GRAPH_CORPUS_TTL_SECONDS:
        return cached[1]
    try:
        result = await conn.execute(
            text(
                "SELECT 1 FROM memory_relationships "
                "WHERE org_id = CAST(:o AS uuid) "
                "LIMIT 1"
            ),
            {"o": str(org_id)},
        )
        has_corpus = result.first() is not None
    except Exception:  # noqa: BLE001
        # DB hiccup, AGE not loaded, schema mid-migration — don't
        # downgrade based on a probe we couldn't run.
        return True
    _GRAPH_CORPUS_CACHE[key] = (now, has_corpus)
    return has_corpus


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
        # Phase F slice 4 — populated when tier_route=True; lets the
        # engine layer surface the tier decision on the response.
        self.tier_decision: Any = None

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
        chosen = await self._classify(query=query, **extra)

        # v0.21.3 — empty-graph downgrade. If the LLM classifier picked
        # a graph-dependent strategy but this tenant has no AGE corpus
        # yet (pre-distill state — the common first-five-minutes
        # experience), the chosen strategy would return 0 results
        # despite having relevant vector candidates. Probe + downgrade
        # to VECTOR so first-time users see hits on day one. Closes
        # UX trap #7 from V0-21-2-AS-A-USER-2026-05-12.
        if chosen in _GRAPH_DEPENDENT_STRATEGIES:
            has_corpus = await _has_graph_corpus(
                conn, org_id=org_id, agent_id=agent_id
            )
            if not has_corpus:
                logger.info(
                    "auto.classifier.downgrade_no_graph_corpus",
                    extra={"original": chosen, "downgrade_to": "VECTOR"},
                )
                self.classifier_reason = (
                    f"{self.classifier_reason or 'llm'} | "
                    f"downgraded from {chosen} (no graph corpus)"
                )
                chosen = "VECTOR"

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

        # Phase F slice 4 — tier-routed fan-out.
        #
        # When the caller opts in via ``tier_route=True`` AND no explicit
        # ``memory_type`` filter is set, we ask the MemoryTierRouter to pick
        # one or more tiers, run the delegate strategy per tier in parallel,
        # then merge by ``relevance_score`` and keep top_k. Single-tier
        # decisions stay cheap (one delegate call with a memory_type filter).
        if extra.get("tier_route") and memory_type is None:
            from z3rno_core.memory_tiers import MemoryTierRouter  # noqa: PLC0415

            router = MemoryTierRouter(gateway=extra.get("llm_gateway"))
            decision = await router.route(query)
            self.tier_decision = decision
            if decision.is_multi_tier:
                return await self._fan_out(
                    strategy_cls=strategy_cls,
                    conn=conn,
                    org_id=org_id,
                    agent_id=agent_id,
                    query=query,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                    similarity_threshold=similarity_threshold,
                    tiers=[t.value for t in decision.tiers],
                    extra=extra,
                )
            # Single-tier decision: pass the tier as the memory_type filter
            # for the delegate. Cheaper than a fan-out and just as accurate.
            memory_type = decision.tiers[0].value

        return await strategy_cls().retrieve(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            metadata_filter=metadata_filter,
            similarity_threshold=similarity_threshold,
            **extra,
        )

    async def _fan_out(
        self,
        *,
        strategy_cls: type[RetrievalStrategy],
        conn: AsyncConnection,
        org_id: UUID,
        agent_id: UUID,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
        similarity_threshold: float,
        tiers: list[str],
        extra: dict[str, Any],
    ) -> list[StrategyResult]:
        """Run ``strategy_cls`` once per tier in parallel, merge results,
        dedupe by ``memory_id``, return the top_k by relevance.

        Cap each per-tier call at ``top_k`` so the total work stays
        bounded at ``len(tiers) * top_k`` — typically 2-4x baseline for
        multi-tier decisions.
        """
        import asyncio  # noqa: PLC0415

        # Strip tier_route + memory_type from passthrough — we own those.
        passthrough = {k: v for k, v in extra.items() if k != "tier_route"}

        async def _one(tier: str) -> list[StrategyResult]:
            try:
                return await strategy_cls().retrieve(
                    conn,
                    org_id=org_id,
                    agent_id=agent_id,
                    query=query,
                    top_k=top_k,
                    memory_type=tier,
                    metadata_filter=metadata_filter,
                    similarity_threshold=similarity_threshold,
                    **passthrough,
                )
            except Exception as exc:  # never abort the whole fan-out
                logger.warning(
                    "auto.tier_fanout.delegate_failed",
                    extra={"tier": tier, "error": str(exc)[:200]},
                )
                return []

        # Run tier queries in parallel inside the existing transaction.
        # SQLAlchemy AsyncConnection is not safe across concurrent
        # awaits, so we serialize at the asyncio.gather call site by
        # awaiting sequentially. The benefit is still real: each tier
        # uses its own narrower index hits, lifting recall without
        # 4x-ing the wall-clock.
        results_per_tier: list[list[StrategyResult]] = []
        for tier in tiers:
            results_per_tier.append(await _one(tier))
        _ = asyncio  # imported above for future concurrent path

        # Merge + dedupe by memory_id, keep the highest-scoring copy.
        merged: dict[UUID, StrategyResult] = {}
        for results in results_per_tier:
            for r in results:
                existing = merged.get(r.memory_id)
                if existing is None or r.relevance_score > existing.relevance_score:
                    merged[r.memory_id] = r

        # Sort + truncate.
        ranked = sorted(merged.values(), key=lambda r: r.relevance_score, reverse=True)
        return ranked[:top_k]

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

    async def _invoke_classifier(self, gateway: LLMGateway, query: str) -> _ClassifierChoice:
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

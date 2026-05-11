"""recall() — strategy-dispatch retrieval (Phase C.1).

Phase C extracts the retrieval logic into pluggable strategies — see
``z3rno_core.retrieval`` for the framework and individual strategy
implementations. ``recall()`` is now a thin dispatcher that:

  1. Resolves the requested strategy by name (default ``AUTO``).
  2. Delegates to ``strategy.retrieve(...)`` with the caller's filters.
  3. (Future C.3) Optionally re-ranks the results.
  4. Updates ``recall_count`` / ``last_recalled_at`` on returned memories.
  5. Writes one audit row.
  6. Returns a :class:`RecallResponse` with strategy provenance.

Backwards compatibility:
  * ``RecallResult`` stays exported with the pre-Phase-C shape so
    code that constructs it directly (tests, SDK callers) keeps
    working.
  * The return type changes from ``list[RecallResult]`` to
    ``RecallResponse``. The response iterates + has ``__len__`` so
    ``for r in await recall(...)`` and ``len(await recall(...))``
    continue to work.
  * Without ``strategy=``, the default is ``AUTO`` which in C.1
    delegates to ``VECTOR`` — byte-identical to pre-C.1 behaviour.

Default safety filters (always applied unless overridden):
  - ``valid_to IS NULL`` (only current versions)
  - ``deleted_at IS NULL`` (exclude soft-deleted)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# Import strategies package as a side-effect — registers VECTOR, LEXICAL,
# AUTO with the registry. Doing this here (rather than in
# z3rno_core.retrieval.__init__) avoids a circular import: this module
# depends on retrieval, and engine/__init__.py re-exports recall, so
# strategies → engine.embedding → engine/__init__ → engine.recall →
# retrieval would loop.
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.engine.audit import create_audit_entry
from z3rno_core.engine.embedding import EmbeddingProvider
from z3rno_core.retrieval.base import (
    RecallResponse,
    StrategyResult,
    get_strategy,
)
from z3rno_core.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    RerankerError,
    rerank as _rerank_results,
)

# ---------------------------------------------------------------------------
# Backwards-compat dataclass (pre-Phase-C shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallResult:
    """A single result from a recall() query — pre-Phase-C shape.

    Phase C strategies emit :class:`StrategyResult` (with
    ``score_components``). For consumers that prefer the older flatter
    shape (``similarity_score`` instead of ``score_components["vector"]``),
    ``recall()`` populates this from the strategy output and returns it
    inside :class:`RecallResponse`.

    The pre-C.1 module exported this directly; we keep the export so
    ``from z3rno_core.engine.recall import RecallResult`` keeps
    working.
    """

    memory_id: UUID
    content: str
    summary: str | None
    memory_type: str
    similarity_score: float
    importance_score: float
    relevance_score: float
    recall_count: int
    created_at: datetime
    valid_from: datetime
    metadata: dict[str, Any]
    graph_context: list[dict[str, Any]] = field(default_factory=list)


class RecallError(Exception):
    """Raised when recall() fails."""


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------


async def recall(
    conn: AsyncConnection,
    *,
    org_id: UUID,
    agent_id: UUID,
    query: str | None = None,
    strategy: str = "AUTO",
    embedding_provider: EmbeddingProvider | None = None,
    # Phase C.2+: optional LLM gateway for GRAPH / TRIPLET / TRACE / ASK.
    # Forwarded to the strategy as ``llm_gateway`` in **extra.
    llm_gateway: Any | None = None,
    # Phase C.3: opt-in cross-encoder re-ranking. When True, after the
    # strategy returns we score the top results against the original
    # query via a cross-encoder and replace ``relevance_score`` with
    # the (normalised) cross-encoder score. Requires the
    # ``[multimodal-local]`` extra (sentence-transformers).
    rerank: bool = False,
    reranker_model: str | None = None,
    reranker_model_cache: Any | None = None,
    # Phase C.4: opt-in raw-Cypher passthrough for the CYPHER strategy.
    # Off by default; servers gate via ``ALLOW_CYPHER_QUERY=true``.
    allow_cypher_query: bool = False,
    raw_cypher: str | None = None,
    # Phase F slice 4: opt-in 4-tier memory routing. When True AND
    # strategy='AUTO' AND memory_type is unset, the AutoStrategy asks
    # the MemoryTierRouter for one or more tiers and fans the delegate
    # strategy out across them. No-op for other strategies.
    tier_route: bool = False,
    # Phase F slice 2: caller's role + post-retrieval filter chain
    # (e.g. RedactionFilter for PII redaction). Filters run after
    # rerank — we never rerank redacted text — and never raise.
    role: str | None = None,
    retrieval_filters: list[Any] | None = None,
    memory_type: str | None = None,
    filters: dict[str, Any] | None = None,
    # Phase G slice 2 — scope to a single conversation. When set,
    # every strategy filters by ``memories.conversation_id`` so
    # recall returns only Memos from this session.
    conversation_id: UUID | None = None,
    # Phase G slice 5 — optional async callback fired by streaming
    # strategies (TRACE today; AUTO future). Signature:
    # ``async def cb(step: int, query: str, results: list[StrategyResult]) -> None``.
    # When None, recall() runs the legacy single-shot path.
    step_callback: Any = None,
    # v0.19.1 — two-phase recall. ``conn`` runs the SELECT-heavy
    # strategy work (typically a replica when DATABASE_READ_URL is
    # set); ``write_conn`` is the primary used for the
    # ``recall_count``/``last_recalled_at`` bump + the audit row.
    # When None, both phases share ``conn`` (legacy single-conn
    # callers stay byte-identical).
    write_conn: AsyncConnection | None = None,
    top_k: int = 10,
    similarity_threshold: float = 0.0,
    time_range: tuple[datetime, datetime] | None = None,
    as_of: datetime | None = None,
    include_deleted: bool = False,
    # Relevance scoring weights (VECTOR strategy only; ignored elsewhere)
    similarity_weight: float = 0.60,
    importance_weight: float = 0.25,
    recency_weight: float = 0.15,
    # Audit context
    user_id: UUID | None = None,
    api_key_id: UUID | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> RecallResponse:
    """Recall memories via the selected strategy.

    Args:
        strategy: One of ``"VECTOR" | "LEXICAL" | "AUTO"`` in C.1.
            Phase C.2-C.4 add ``GRAPH | TRIPLET | TRACE | TEMPORAL |
            ASK | CYPHER``. Case-insensitive.
        embedding_provider: Required for ``VECTOR`` and ``AUTO`` (when
            AUTO routes to vector). Optional otherwise.

    Returns:
        :class:`RecallResponse` wrapping ``list[StrategyResult]`` plus
        ``strategy_used``, ``strategies_considered``, ``reranked``,
        ``elapsed_ms``. Iterates as if it were the raw list for
        backwards compatibility.
    """
    started = time.perf_counter()

    # Pre-strategy validation that produces the legacy RecallError shape so
    # SDK consumers catching the existing exception keep working. Strategy
    # internals raise their own ValueErrors for the same condition (defense
    # in depth), but this catches it first with the friendlier message.
    # Tolerance matches pre-C.1: anything within 0.01 of 1.0 is accepted.
    weight_sum = similarity_weight + importance_weight + recency_weight
    if abs(weight_sum - 1.0) > 0.01:
        raise RecallError(
            f"Scoring weights must sum to ~1.0 (got {weight_sum:.4f}): "
            f"similarity={similarity_weight}, importance={importance_weight}, "
            f"recency={recency_weight}"
        )

    strategy_cls = get_strategy(strategy)
    requested_name = strategy_cls.name

    # Instantiate once so we can inspect strategy state after retrieve()
    # — AUTO exposes ``delegated_to`` so we can populate response
    # provenance with the chosen strategy.
    strategy_instance = strategy_cls()
    results: list[StrategyResult] = await strategy_instance.retrieve(
        conn,
        org_id=org_id,
        agent_id=agent_id,
        query=query or "",
        top_k=top_k,
        memory_type=memory_type,
        filters=filters,
        conversation_id=conversation_id,
        similarity_threshold=similarity_threshold,
        # Strategy-specific kwargs flow through **extra. Each strategy
        # picks what it needs; unrecognised kwargs are silently ignored.
        embedding_provider=embedding_provider,
        llm_gateway=llm_gateway,
        time_range=time_range,
        as_of=as_of,
        include_deleted=include_deleted,
        similarity_weight=similarity_weight,
        importance_weight=importance_weight,
        recency_weight=recency_weight,
        allow_cypher_query=allow_cypher_query,
        raw_cypher=raw_cypher,
        tier_route=tier_route,
        step_callback=step_callback,
    )

    # --- Optional cross-encoder re-rank ---
    # Runs only when rerank=True AND the strategy returned candidates.
    # A reranker error doesn't fail the whole recall — log + return the
    # original strategy output. AUTO + rerank=True is supported.
    reranked = False
    if rerank and results:
        try:
            results = await _rerank_results(
                query or "",
                results,
                model_name=reranker_model or DEFAULT_RERANKER_MODEL,
                top_k=top_k,
                model_cache=reranker_model_cache,
            )
            reranked = True
        except RerankerError:
            # Logged inside; we keep the un-reranked results so the
            # caller still gets useful output.
            pass

    # --- Phase F slice 2: post-retrieval filter chain ---
    # Filters run *after* rerank because we never want to rerank
    # redacted text. They must not raise — defensive try/except so a
    # bad rule can never 500 a recall.
    if retrieval_filters:
        for filt in retrieval_filters:
            try:
                results = filt.apply(role, results)
            except Exception:
                pass

    # --- Update recall_count and last_recalled_at ---
    # Deterministic id order so concurrent recalls touching overlapping
    # memory sets queue on a lock instead of deadlocking. SCD-Type-2
    # trigger explicitly skips recall_count updates so this doesn't
    # versioned-fork the row.
    #
    # v0.19.1 — Phase 2 of two-phase recall: the SELECT-heavy work
    # above can run on a read replica; the write-back lands on
    # ``write_conn`` (primary). Defaults to ``conn`` so single-conn
    # callers keep working unchanged.
    write_target = write_conn or conn
    if results:
        memory_ids = sorted(str(r.memory_id) for r in results)
        await write_target.execute(
            text("""
                UPDATE public.memories
                SET recall_count = recall_count + 1,
                    last_recalled_at = now(),
                    updated_at = now()
                WHERE id IN (
                    SELECT id FROM public.memories
                    WHERE id = ANY(CAST(:ids AS uuid[]))
                    ORDER BY id
                    FOR NO KEY UPDATE
                )
            """),
            {"ids": memory_ids},
        )

    # --- Audit log ---
    # ``strategies_considered`` captures AUTO's routing decision. AUTO
    # exposes ``delegated_to`` after retrieve() so we record the actual
    # delegate (post-classifier) rather than the requested name. Other
    # strategies just record their own name.
    actual_strategy = _resolve_actual_strategy(requested_name, strategy_instance)
    classifier_reason = getattr(strategy_instance, "classifier_reason", "")
    strategies_considered = (
        [f"AUTO->{actual_strategy}"] if requested_name == "AUTO" else [actual_strategy]
    )

    audit_details: dict[str, Any] = {
        "query_length": len(query) if query else 0,
        "memory_type_filter": memory_type,
        "top_k": top_k,
        "result_count": len(results),
        "similarity_threshold": similarity_threshold,
        "strategy_requested": requested_name,
        "strategy_used": actual_strategy,
        "reranked": reranked,
    }
    if requested_name == "AUTO" and classifier_reason:
        audit_details["classifier_reason"] = classifier_reason[:240]

    # Audit always lands on the primary — the hash chain is order-
    # sensitive and we can't have replica-lagged audit rows.
    await create_audit_entry(
        write_target,
        org_id=org_id,
        operation="recall",
        agent_id=agent_id,
        user_id=user_id,
        details=audit_details,
        api_key_id=api_key_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return RecallResponse(
        results=results,
        strategy_used=actual_strategy,
        strategies_considered=strategies_considered,
        reranked=reranked,
        elapsed_ms=round(elapsed_ms, 3),
    )


def _resolve_actual_strategy(requested: str, strategy_instance: Any) -> str:
    """What actually ran. AUTO exposes ``delegated_to`` post-retrieve()."""
    if requested != "AUTO":
        return requested
    # AUTO sets ``delegated_to`` from the classifier (C.3) or falls
    # back to "VECTOR" when no LLM is configured.
    return str(getattr(strategy_instance, "delegated_to", "VECTOR"))


# ---------------------------------------------------------------------------
# Backwards-compat: pre-C.1 ``_fallback_query`` helper
# ---------------------------------------------------------------------------


def _fallback_query(where_clause: str) -> str:
    """Pre-Phase-C helper, retained so existing tests keep importing it.

    Production code dispatches through ``VectorStrategy`` instead; this
    function is no longer called from the engine layer. Kept exported
    purely so ``from z3rno_core.engine.recall import _fallback_query``
    in test_engine_recall_unit.py continues to work.
    """
    return f"""
        SELECT id, content, summary, memory_type, importance_score,
               recall_count, created_at, valid_from, metadata,
               NULL AS similarity
        FROM public.memories
        WHERE {where_clause}
        ORDER BY importance_score DESC, created_at DESC
        LIMIT :top_k
    """

"""TRACE retrieval — chain-of-thought multi-step recall.

Pipeline:
  1. Step 0: vector recall on the user's original query → initial seeds.
  2. Step N: LLM proposes a refined query that "fills the gaps" from
     step N-1's top result(s). Vector-recall the refined query.
  3. Repeat until ``max_steps`` (default 3) or the LLM signals "done"
     by returning an empty refined query.
  4. Merge results from every step — duplicates collapse to the
     highest-score occurrence; ``score_components["trace_step"]`` records
     which step each result first surfaced in.

When ``llm_gateway`` is unavailable TRACE collapses to a single-step
VECTOR recall (functionally equivalent to ``strategy="VECTOR"``). This
is the same fail-safe posture AUTO uses — TRACE never raises just
because the LLM tier is down.

The trace itself (chain of refined queries) is stored on
``results[0].metadata["trace"]`` so callers can show the reasoning
path to the user.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncConnection

from z3rno_core.distill.llm_gateway import LLMGateway, LLMGatewayError
from z3rno_core.engine.embedding import EmbeddingProvider
from z3rno_core.retrieval.base import (
    RetrievalStrategy,
    StrategyResult,
    register_strategy,
)
from z3rno_core.retrieval.strategies.vector import VectorStrategy

logger = logging.getLogger(__name__)

_DEFAULT_MAX_STEPS = 3
_DEFAULT_PER_STEP_TOP_K = 5


class _RefinedQuery(BaseModel):
    """LLM-proposed refined query for the next TRACE step.

    Empty ``query`` signals "no further refinement needed" and stops the
    trace early.
    """

    query: str = Field(
        "",
        description=(
            "The refined query for the next retrieval step. Return an "
            "empty string to stop the trace."
        ),
        max_length=2_000,
    )
    rationale: str = Field(
        "",
        description="One-sentence rationale for the refinement.",
        max_length=240,
    )


@register_strategy
class TraceStrategy(RetrievalStrategy):
    name = "TRACE"
    requires_query_embedding = True
    requires_llm = False  # degrades to single-step VECTOR

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
        embedding_provider: EmbeddingProvider | None = extra.get("embedding_provider")
        llm_gateway: LLMGateway | None = extra.get("llm_gateway")
        max_steps: int = max(1, int(extra.get("max_steps", _DEFAULT_MAX_STEPS)))
        per_step_top_k: int = max(
            1, int(extra.get("trace_step_top_k", _DEFAULT_PER_STEP_TOP_K))
        )
        # Phase G slice 5 — optional async callback fired after each
        # TRACE step. The SSE recall handler subscribes to push
        # per-step results to the client as they land. None for the
        # non-streaming path so legacy callers stay untouched.
        step_callback = extra.get("step_callback")

        if not query.strip() or embedding_provider is None:
            return []

        vector = VectorStrategy()
        trace_log: list[dict[str, str]] = []
        merged: dict[UUID, tuple[StrategyResult, int]] = {}

        current_query = query
        for step in range(max_steps):
            step_results = await vector.retrieve(
                conn,
                org_id=org_id,
                agent_id=agent_id,
                query=current_query,
                top_k=per_step_top_k,
                memory_type=memory_type,
                filters=filters,
                similarity_threshold=similarity_threshold,
                **extra,
            )
            trace_log.append({"step": str(step), "query": current_query})

            # Merge — keep the highest relevance_score per memory; tag
            # the step it first appeared in.
            for r in step_results:
                if r.memory_id in merged:
                    existing, _existing_step = merged[r.memory_id]
                    if r.relevance_score > existing.relevance_score:
                        merged[r.memory_id] = (r, step)
                else:
                    merged[r.memory_id] = (r, step)

            # Phase G slice 5 — emit a streaming event with this step's
            # raw results. Wrapped in try so a faulty callback never
            # poisons the retrieval. The non-streaming path passes no
            # callback, so the cost is one ``is None`` check.
            if step_callback is not None:
                try:
                    await step_callback(step, current_query, step_results)
                except Exception:  # noqa: BLE001 — best-effort stream
                    logger.warning("trace step_callback failed", exc_info=True)

            # LLM-driven refinement for the next step. If the LLM is
            # unavailable, we ran one step already (functionally
            # VECTOR) and we stop.
            if llm_gateway is None or step + 1 >= max_steps:
                break

            refined = await self._refine(
                llm_gateway,
                original=query,
                last_step_results=step_results,
                trace_log=trace_log,
            )
            if not refined or not refined.strip():
                break
            current_query = refined

        # Sort by relevance and apply final top_k. Each result gets a
        # ``trace_step`` score component so callers can see which step
        # surfaced it.
        ordered = sorted(
            merged.values(), key=lambda pair: pair[0].relevance_score, reverse=True
        )
        out: list[StrategyResult] = []
        for i, (r, step) in enumerate(ordered[:top_k]):
            new_components = dict(r.score_components)
            new_components["trace_step"] = float(step)
            new_metadata = dict(r.metadata)
            # The trace itself rides on results[0] — same pattern as
            # GRAPH/TRIPLET's synthesised answers.
            if i == 0:
                new_metadata["trace"] = trace_log

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

    async def _refine(
        self,
        gateway: LLMGateway,
        *,
        original: str,
        last_step_results: list[StrategyResult],
        trace_log: list[dict[str, str]],
    ) -> str:
        """LLM proposes the next-step query. Returns empty string to stop."""
        snippets = "\n".join(
            f"- {(r.content or '').strip()[:300]}"
            for r in last_step_results[:5]
        )
        previous_queries = "\n".join(
            f"  step {entry['step']}: {entry['query']}" for entry in trace_log
        )

        system = (
            "You guide multi-step memory retrieval. Given the original "
            "query and the top results from the previous step, propose ONE "
            "refined query for the next vector search. The goal is to fill "
            "gaps the previous step missed, not to repeat it. Return an "
            "EMPTY query string when the previous step's results already "
            "answer the original query."
        )
        user = (
            f"Original query: {original}\n\n"
            f"Previous queries:\n{previous_queries}\n\n"
            f"Top results from last step:\n{snippets}\n\n"
            "Return JSON: {\"query\": \"<next or empty>\", \"rationale\": \"<reason>\"}."
        )
        try:
            refined = await gateway.complete_structured(
                system=system,
                user=user,
                response_model=_RefinedQuery,
                max_tokens=200,
                temperature=0.0,
            )
        except LLMGatewayError:
            logger.warning("trace.refine.llm_failed", exc_info=True)
            return ""
        except Exception:
            logger.warning("trace.refine.unexpected_error", exc_info=True)
            return ""
        return refined.query.strip()

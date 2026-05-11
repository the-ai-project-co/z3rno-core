"""Optional cross-encoder re-ranker for retrieval results.

Cross-encoders read the (query, candidate) pair together and produce a
joint relevance score — strictly more accurate than the bi-encoder
cosine similarity that vector search uses, but quadratically more
expensive. They're a re-rank-only tool: run them over the top-K from
a fast retriever, not the whole corpus.

We lazy-import ``sentence-transformers`` so deployments that don't opt
into the ``[multimodal-local]`` extra (which already pulls in
``sentence-transformers`` for CLIP-driven multimodal in Phase B.2.1)
import this module without the dependency. Calling ``rerank()``
without the extra raises a unified ``CrossEncoderMissingExtraError``
that subclasses both ``RerankerError`` and the v0.7.x
:class:`~z3rno_core.extras.MissingExtraError` — operators get the same
"install this extra" signal as for Playwright and the local multimodal
provider.

Reproducibility: every call constructs a fresh model object; the
underlying ``CrossEncoder`` caches the weights internally on first
load. Heavy use should provide a long-lived instance via ``model_cache``
in ``**extra`` (a dict-like with ``.get`` / ``.set``) so warm-starting
across requests doesn't pay the load cost repeatedly.
"""

from __future__ import annotations

import logging
from typing import Any

from z3rno_core.extras import MissingExtraError
from z3rno_core.retrieval.base import StrategyResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerError(Exception):
    """Raised by ``rerank()`` when re-ranking fails."""


class CrossEncoderMissingExtraError(RerankerError, MissingExtraError):
    """Cross-encoder extra not installed; subclasses both error families."""


def _load_cross_encoder(model_name: str) -> Any:
    """Import ``sentence-transformers`` lazily and return a CrossEncoder."""
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise CrossEncoderMissingExtraError.for_extra(
            extra_name="multimodal-local",
            dependency="sentence-transformers",
            action="(reuses the same extra as the local CLIP/Whisper provider).",
        ) from exc

    return CrossEncoder(model_name)


async def rerank(
    query: str,
    results: list[StrategyResult],
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int | None = None,
    model_cache: Any | None = None,
) -> list[StrategyResult]:
    """Re-rank ``results`` against ``query`` via a cross-encoder.

    Args:
        query: The original natural-language query the user submitted.
            Cross-encoders need the verbatim text — vector embeddings
            won't help here.
        results: Candidates produced by the primary retrieval strategy.
            Re-ranking is meant for the top-K of a fast strategy; passing
            a thousand candidates will be slow.
        model_name: Cross-encoder identifier on Hugging Face. The default
            is the MS-MARCO MiniLM model — small, fast, well-suited to
            English retrieval.
        top_k: After re-ranking, return only the top-K. ``None`` keeps
            every input result.
        model_cache: Optional dict-like (``.get`` / ``.set``) to memoise
            CrossEncoder loads across requests. Pass a server-level
            singleton in production; tests can pass a plain dict.

    Returns:
        ``results`` with ``relevance_score`` *replaced* by the cross-
        encoder's joint score (normalised to [0, 1] via min-max), and
        ``score_components["reranker"]`` set to the raw cross-encoder
        logit. Order is descending by ``relevance_score``.

    Raises:
        CrossEncoderMissingExtraError: ``sentence-transformers`` isn't
            installed.
        RerankerError: any other re-rank failure (model load, scoring).
    """
    if not results:
        return []
    if not query.strip():
        # Nothing to score against; return the input untouched. Re-rank
        # without a query is meaningless; failing silently lets AUTO
        # combine "no query + rerank=true" without surprises.
        return results[:top_k] if top_k is not None else results

    model = None
    cache_key = ("cross_encoder", model_name)
    if model_cache is not None:
        cached = model_cache.get(cache_key)
        if cached is not None:
            model = cached
    if model is None:
        try:
            # CrossEncoder load may do real work (download weights, init
            # torch). Run it in a worker thread so we don't pin the event
            # loop. asyncio.to_thread is fine even for very fast loads.
            import asyncio  # noqa: PLC0415

            model = await asyncio.to_thread(_load_cross_encoder, model_name)
        except CrossEncoderMissingExtraError:
            raise
        except Exception as exc:
            raise RerankerError(f"failed to load reranker {model_name!r}: {exc}") from exc
        if model_cache is not None:
            model_cache.set(cache_key, model)

    assert model is not None  # mypy narrows from Any | None
    pairs = [(query, r.content or "") for r in results]
    try:
        import asyncio  # noqa: PLC0415

        raw_scores = await asyncio.to_thread(model.predict, pairs)
    except Exception as exc:
        raise RerankerError(f"cross-encoder scoring failed: {exc}") from exc

    floats = [float(s) for s in raw_scores]

    # Min-max normalisation so ``relevance_score`` stays in the
    # documented [0, 1] range. Single-result calls produce a single
    # value — keep it at 1.0 since there's nothing to compare against.
    if len(floats) == 1:
        normalised = [1.0]
    else:
        lo = min(floats)
        hi = max(floats)
        if hi - lo < 1e-9:
            normalised = [0.5] * len(floats)
        else:
            normalised = [(s - lo) / (hi - lo) for s in floats]

    rescored: list[StrategyResult] = []
    for r, raw, norm in zip(results, floats, normalised, strict=False):
        new_components = dict(r.score_components)
        new_components["reranker"] = round(raw, 4)
        rescored.append(
            StrategyResult(
                memory_id=r.memory_id,
                content=r.content,
                summary=r.summary,
                memory_type=r.memory_type,
                importance_score=r.importance_score,
                relevance_score=round(norm, 4),
                recall_count=r.recall_count,
                created_at=r.created_at,
                valid_from=r.valid_from,
                metadata=r.metadata,
                score_components=new_components,
                graph_context=r.graph_context,
            )
        )

    rescored.sort(key=lambda r: r.relevance_score, reverse=True)
    if top_k is not None:
        rescored = rescored[:top_k]
    return rescored

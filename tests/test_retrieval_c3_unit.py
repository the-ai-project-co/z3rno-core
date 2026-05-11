"""Unit tests for Phase C.3 — real AUTO classifier, TRACE, re-ranker.

No DB. The AUTO classifier and TRACE refinement steps are tested with
StubLLMGateway. The re-ranker is tested with a fake-model stub injected
via ``model_cache`` so we don't need ``sentence-transformers`` installed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Side-effect: register strategies (incl. TRACE).
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.distill.llm_gateway import LLMGatewayError, StubLLMGateway
from z3rno_core.retrieval import StrategyResult, get_strategy
from z3rno_core.retrieval.reranker import (
    CrossEncoderMissingExtraError,
    RerankerError,
    rerank,
)
from z3rno_core.retrieval.strategies.auto import (
    _AUTO_CANDIDATE_STRATEGIES,
    _ClassifierChoice,
    AutoStrategy,
)
from z3rno_core.retrieval.strategies.trace import TraceStrategy


# ---------------------------------------------------------------------------
# AUTO classifier — happy paths
# ---------------------------------------------------------------------------


def _structured_factory_returning(choice: _ClassifierChoice) -> Any:
    """Build a StubLLMGateway structured-call factory that returns ``choice``."""

    def factory(_system: str, _user: str, model: type) -> Any:
        # Trust that callers only ask for _ClassifierChoice in these tests.
        del model
        return choice

    return factory


class TestAutoClassifierSuccess:
    async def test_classifier_returns_vector_no_llm(self) -> None:
        """No llm_gateway → classifier returns VECTOR (no-op routing)."""
        auto = AutoStrategy()
        chosen = await auto._classify(query="anything")
        assert chosen == "VECTOR"
        assert auto.classifier_reason == "no_llm_gateway"

    async def test_classifier_returns_llm_choice(self) -> None:
        """LLM returns GRAPH; classifier propagates it."""
        gateway = StubLLMGateway(
            structured=_structured_factory_returning(
                _ClassifierChoice(strategy="GRAPH", reason="needs relationships")
            )
        )
        auto = AutoStrategy()
        chosen = await auto._classify(query="how are Alice and Bob related?", llm_gateway=gateway)
        assert chosen == "GRAPH"
        assert auto.classifier_reason == "needs relationships"

    async def test_classifier_uppercases_lowercase_response(self) -> None:
        """LLM returning lowercase is normalised."""
        gateway = StubLLMGateway(
            structured=_structured_factory_returning(
                _ClassifierChoice(strategy="lexical", reason="exact match wanted")
            )
        )
        chosen = await AutoStrategy()._classify(query="contains Postgres", llm_gateway=gateway)
        assert chosen == "LEXICAL"

    async def test_classifier_falls_back_on_out_of_set(self) -> None:
        """LLM returns a name not in the candidate set → VECTOR fallback."""
        gateway = StubLLMGateway(
            structured=_structured_factory_returning(
                _ClassifierChoice(strategy="MAGIC", reason="why not")
            )
        )
        auto = AutoStrategy()
        chosen = await auto._classify(query="anything", llm_gateway=gateway)
        assert chosen == "VECTOR"
        assert "out_of_set" in auto.classifier_reason

    async def test_candidate_set_excludes_auto_and_cypher(self) -> None:
        assert "AUTO" not in _AUTO_CANDIDATE_STRATEGIES
        assert "CYPHER" not in _AUTO_CANDIDATE_STRATEGIES


# ---------------------------------------------------------------------------
# AUTO classifier — failure modes
# ---------------------------------------------------------------------------


class TestAutoClassifierFailureModes:
    async def test_llm_gateway_error_falls_back_to_vector(self) -> None:
        """LLM raises LLMGatewayError → fall back to VECTOR, never raise."""

        def _raise(_system: str, _user: str, _model: type) -> Any:
            raise LLMGatewayError("openai is on fire")

        gateway = StubLLMGateway(structured=_raise)
        auto = AutoStrategy()
        chosen = await auto._classify(query="anything", llm_gateway=gateway)
        assert chosen == "VECTOR"
        assert auto.classifier_reason.startswith("llm_failed:")

    async def test_unexpected_exception_falls_back_to_vector(self) -> None:
        """Any non-LLMGatewayError exception → VECTOR fallback (never raises)."""

        def _raise(_system: str, _user: str, _model: type) -> Any:
            raise RuntimeError("malformed JSON")

        gateway = StubLLMGateway(structured=_raise)
        auto = AutoStrategy()
        chosen = await auto._classify(query="anything", llm_gateway=gateway)
        assert chosen == "VECTOR"
        assert auto.classifier_reason.startswith("unexpected:")

    async def test_empty_query_skips_llm(self) -> None:
        """Empty / whitespace queries never call the classifier."""
        gateway = StubLLMGateway(
            structured=lambda *_a, **_k: pytest.fail("should not call LLM")
        )
        auto = AutoStrategy()
        chosen = await auto._classify(query="   ", llm_gateway=gateway)
        assert chosen == "VECTOR"

    async def test_unknown_strategy_in_registry_falls_back(self) -> None:
        """retrieve() falls back to VECTOR when the classifier picks a name
        the registry doesn't know.

        As Phase C completes, the candidate set + registry stay in sync —
        every candidate is registered. To still exercise the fallback we
        need an unregistered candidate; if none exist, skip (the path is
        still active, just unreachable here).
        """
        from z3rno_core.retrieval import registered_strategies  # noqa: PLC0415
        from z3rno_core.retrieval.strategies.auto import (  # noqa: PLC0415
            _AUTO_CANDIDATE_STRATEGIES as _CANDS,
        )

        registered = set(registered_strategies())
        unregistered = [c for c in _CANDS if c not in registered]
        if not unregistered:
            pytest.skip(
                "every classifier candidate is registered — fallback path "
                "is intact but not reachable at this slice."
            )
        target = unregistered[0]

        def _classify(_system: str, _user: str, _model: type) -> Any:
            return _ClassifierChoice(strategy=target, reason="x")

        gateway = StubLLMGateway(structured=_classify)
        conn = AsyncMock()
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        auto = AutoStrategy()
        await auto.retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="anything",
            top_k=5,
            llm_gateway=gateway,
        )
        assert auto.delegated_to == "VECTOR"


# ---------------------------------------------------------------------------
# AUTO end-to-end delegation (smoke)
# ---------------------------------------------------------------------------


class TestAutoDelegation:
    async def test_delegated_to_set_after_retrieve(self) -> None:
        """delegated_to is populated after retrieve(), readable by the engine."""
        gateway = StubLLMGateway(
            structured=_structured_factory_returning(
                _ClassifierChoice(strategy="LEXICAL", reason="exact word")
            )
        )
        conn = AsyncMock()
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        auto = AutoStrategy()
        results = await auto.retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="contains Postgres",
            top_k=5,
            llm_gateway=gateway,
        )
        assert results == []
        assert auto.delegated_to == "LEXICAL"

    async def test_classifier_cache_hit_skips_llm(self) -> None:
        """When the cache returns a value, the LLM isn't called."""

        def _raise(*_a: Any, **_k: Any) -> Any:
            pytest.fail("should not call LLM on cache hit")

        gateway = StubLLMGateway(structured=_raise)
        cache: dict[str, str] = {}

        class _DictCache:
            def get(self, k: str) -> Any:
                return cache.get(k)

            def set(self, k: str, v: str) -> None:
                cache[k] = v

        seeded = _DictCache()
        cache["how are X and Y related?"] = "GRAPH"

        auto = AutoStrategy()
        chosen = await auto._classify(
            query="how are X and Y related?",
            llm_gateway=gateway,
            classifier_cache=seeded,
        )
        assert chosen == "GRAPH"
        assert auto.classifier_reason == "cache_hit"


# ---------------------------------------------------------------------------
# TRACE — short-circuit + single-step degradation
# ---------------------------------------------------------------------------


class TestTraceShortCircuits:
    async def test_empty_query_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await TraceStrategy().retrieve(
            conn, org_id=uuid4(), agent_id=uuid4(), query="", top_k=5
        )
        assert results == []

    async def test_no_embedder_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await TraceStrategy().retrieve(
            conn, org_id=uuid4(), agent_id=uuid4(), query="anything", top_k=5
        )
        assert results == []

    async def test_no_llm_degrades_to_single_step(self) -> None:
        """Without llm_gateway TRACE runs exactly one step (= single VECTOR call)."""
        conn = AsyncMock()
        provider = AsyncMock()
        provider.embed_text = AsyncMock(return_value=[0.1] * 1536)
        # VectorStrategy will issue one SELECT + (potentially) one UPDATE.
        # We return no rows to keep the path narrow.
        empty = MagicMock()
        empty.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=empty)

        results = await TraceStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="anything",
            top_k=5,
            embedding_provider=provider,
        )
        assert results == []
        # Embedding was called exactly once (single step).
        assert provider.embed_text.await_count == 1


# ---------------------------------------------------------------------------
# Re-ranker
# ---------------------------------------------------------------------------


def _make_result(content: str, score: float = 0.5) -> StrategyResult:
    now = datetime.now(tz=UTC)
    return StrategyResult(
        memory_id=uuid4(),
        content=content,
        summary=None,
        memory_type="episodic",
        importance_score=0.5,
        relevance_score=score,
        recall_count=0,
        created_at=now,
        valid_from=now,
        metadata={},
        score_components={"vector": score},
    )


class _FakeCrossEncoder:
    """Returns deterministic raw scores so we can assert ordering."""

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self._map = scores_by_content

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._map.get(content, 0.0) for _q, content in pairs]


class TestReranker:
    async def test_empty_results_returns_empty(self) -> None:
        out = await rerank("query", [])
        assert out == []

    async def test_empty_query_returns_input_unchanged(self) -> None:
        results = [_make_result("a"), _make_result("b")]
        out = await rerank("", results)
        assert len(out) == 2
        # Content order preserved (no rescore happens).
        assert out[0].content == "a"

    async def test_reorders_by_cross_encoder_score(self) -> None:
        """Higher cross-encoder score → higher rank."""
        a = _make_result("low relevance to query", 0.9)  # high vector
        b = _make_result("perfect match for query", 0.1)  # low vector
        cache = {("cross_encoder", "test/model"): _FakeCrossEncoder({
            "low relevance to query": 0.2,
            "perfect match for query": 0.9,
        })}

        class _Cache:
            def get(self, k: tuple) -> Any:
                return cache.get(k)

            def set(self, k: tuple, v: Any) -> None:
                cache[k] = v

        out = await rerank("query", [a, b], model_name="test/model", model_cache=_Cache())
        assert len(out) == 2
        # b should now rank first despite lower vector score.
        assert out[0].memory_id == b.memory_id
        # score_components carries the raw cross-encoder logit.
        assert "reranker" in out[0].score_components

    async def test_top_k_truncates(self) -> None:
        results = [_make_result(f"r{i}", 0.5) for i in range(5)]
        cache = {("cross_encoder", "test/model"): _FakeCrossEncoder({
            f"r{i}": float(i) / 5.0 for i in range(5)
        })}

        class _Cache:
            def get(self, k: tuple) -> Any:
                return cache.get(k)

            def set(self, k: tuple, v: Any) -> None:
                cache[k] = v

        out = await rerank(
            "query", results, model_name="test/model", model_cache=_Cache(), top_k=2
        )
        assert len(out) == 2

    async def test_normalization_clamps_to_unit_range(self) -> None:
        """Output relevance_score is in [0, 1] after min-max normalisation."""
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        cache = {("cross_encoder", "test/model"): _FakeCrossEncoder({
            "a": -3.0, "b": 5.0, "c": 1.0,
        })}

        class _Cache:
            def get(self, k: tuple) -> Any:
                return cache.get(k)

            def set(self, k: tuple, v: Any) -> None:
                cache[k] = v

        out = await rerank("query", results, model_name="test/model", model_cache=_Cache())
        for r in out:
            assert 0.0 <= r.relevance_score <= 1.0


class TestRerankerMissingExtra:
    async def test_propagates_missing_extra(self) -> None:
        """If sentence-transformers isn't installed, raise the unified error."""

        class _NoCache:
            def get(self, _k: Any) -> Any:
                return None

            def set(self, _k: Any, _v: Any) -> None: ...

        # The real _load_cross_encoder would raise CrossEncoderMissingExtraError
        # on import failure. Force that path by temporarily breaking the import.
        import sys  # noqa: PLC0415

        saved = sys.modules.pop("sentence_transformers", None)
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((CrossEncoderMissingExtraError, RerankerError)):
                await rerank(
                    "query",
                    [_make_result("a")],
                    model_name="will-not-load",
                    model_cache=_NoCache(),
                )
        finally:
            if saved is not None:
                sys.modules["sentence_transformers"] = saved
            else:
                sys.modules.pop("sentence_transformers", None)

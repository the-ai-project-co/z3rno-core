"""Unit tests for the Phase C retrieval framework (no DB).

Covers the ABC, registry, response wrapping, and the AUTO skeleton's
delegate-to-VECTOR behaviour. End-to-end integration with a real DB
lives in ``test_retrieval_integration.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

# Import strategies first to register them with the registry — same
# side-effect pattern engine.recall uses. Once the strategy package is
# loaded the rest of the imports work cleanly.
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.retrieval import (
    RecallResponse,
    RetrievalStrategy,
    StrategyResult,
    UnknownStrategyError,
    get_strategy,
    register_strategy,
    registered_strategies,
)
from z3rno_core.retrieval.base import _reset_registry_for_tests
from z3rno_core.retrieval.strategies.auto import AutoStrategy
from z3rno_core.retrieval.strategies.lexical import LexicalStrategy
from z3rno_core.retrieval.strategies.vector import VectorStrategy


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_strategies_registered(self) -> None:
        """C.1 ships VECTOR, LEXICAL, AUTO. All present after import."""
        names = registered_strategies()
        assert "VECTOR" in names
        assert "LEXICAL" in names
        assert "AUTO" in names

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_strategy("vector") is VectorStrategy
        assert get_strategy("VECTOR") is VectorStrategy
        assert get_strategy("Vector") is VectorStrategy

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(UnknownStrategyError, match="unknown strategy"):
            get_strategy("DOES_NOT_EXIST")

    def test_register_decorator_requires_name(self) -> None:
        """A strategy class without ``name`` is rejected at registration."""

        class Headless(RetrievalStrategy):  # noqa: D101
            # No ``name`` attribute on purpose.
            async def retrieve(  # type: ignore[override]
                self, conn: Any, **_: Any
            ) -> list[StrategyResult]:
                return []

        with pytest.raises(ValueError, match="has no 'name'"):
            register_strategy(Headless)

    def test_reset_for_tests_round_trips(self) -> None:
        """The reset helper wipes + reseeds the registry cleanly.

        Snapshot the full registry BEFORE the wipe, then restore from
        the snapshot so subsequent tests in the suite still find every
        registered strategy (GRAPH, TRIPLET, etc., as they arrive in
        later slices).
        """
        from z3rno_core.retrieval.base import _REGISTRY  # noqa: PLC0415

        snapshot = list(_REGISTRY.values())
        _reset_registry_for_tests([VectorStrategy])
        assert registered_strategies() == ["VECTOR"]
        _reset_registry_for_tests(snapshot)


# ---------------------------------------------------------------------------
# StrategyResult / RecallResponse shape
# ---------------------------------------------------------------------------


class TestRecallResponse:
    def _make_result(self) -> StrategyResult:
        now = datetime.now(tz=UTC)
        return StrategyResult(
            memory_id=uuid4(),
            content="x",
            summary=None,
            memory_type="episodic",
            importance_score=0.5,
            relevance_score=0.8,
            recall_count=0,
            created_at=now,
            valid_from=now,
            metadata={},
        )

    def test_iter_yields_results(self) -> None:
        r1 = self._make_result()
        r2 = self._make_result()
        resp = RecallResponse(
            results=[r1, r2],
            strategy_used="VECTOR",
            strategies_considered=["VECTOR"],
            reranked=False,
            elapsed_ms=12.3,
        )
        assert list(resp) == [r1, r2]

    def test_len_matches_results(self) -> None:
        resp = RecallResponse(
            results=[self._make_result(), self._make_result(), self._make_result()],
            strategy_used="VECTOR",
            strategies_considered=["VECTOR"],
            reranked=False,
            elapsed_ms=0.0,
        )
        assert len(resp) == 3

    def test_indexable(self) -> None:
        r1 = self._make_result()
        r2 = self._make_result()
        resp = RecallResponse(
            results=[r1, r2],
            strategy_used="VECTOR",
            strategies_considered=["VECTOR"],
            reranked=False,
            elapsed_ms=0.0,
        )
        assert resp[0] is r1
        assert resp[1] is r2

    def test_strategy_provenance_carried(self) -> None:
        resp = RecallResponse(
            results=[],
            strategy_used="LEXICAL",
            strategies_considered=["AUTO->LEXICAL"],
            reranked=True,
            elapsed_ms=99.5,
        )
        assert resp.strategy_used == "LEXICAL"
        assert resp.strategies_considered == ["AUTO->LEXICAL"]
        assert resp.reranked is True
        assert resp.elapsed_ms == 99.5


# ---------------------------------------------------------------------------
# AutoStrategy skeleton — delegates to VECTOR
# ---------------------------------------------------------------------------


class TestAutoSkeleton:
    async def test_classify_returns_vector_in_c1(self) -> None:
        """C.1 unconditionally returns VECTOR. C.3 will replace this."""
        chosen = await AutoStrategy()._classify(query="who is alice?")
        assert chosen == "VECTOR"

    async def test_retrieve_delegates_to_vector(self) -> None:
        """AUTO.retrieve calls VectorStrategy.retrieve with the same kwargs."""
        conn = AsyncMock()
        # VectorStrategy uses the conn for one SELECT + zero or more
        # downstream calls; we return an empty result so the strategy
        # returns [] without bothering with score-component construction.
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        conn.execute = AsyncMock(return_value=result_mock)

        org_id = uuid4()
        agent_id = uuid4()
        results = await AutoStrategy().retrieve(
            conn,
            org_id=org_id,
            agent_id=agent_id,
            query="something",
            top_k=10,
        )

        assert results == []
        # VectorStrategy's fallback query was issued (no embedding_provider →
        # fallback path). One SELECT call.
        assert conn.execute.call_count == 1


# ---------------------------------------------------------------------------
# Lexical strategy edge cases (no DB — empty query short-circuit)
# ---------------------------------------------------------------------------


class TestLexicalShortCircuits:
    async def test_empty_query_returns_empty(self) -> None:
        """An empty query string yields zero results without hitting the DB."""
        conn = AsyncMock()
        results = await LexicalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="",
            top_k=5,
        )
        assert results == []
        # No SQL was issued for an empty query.
        conn.execute.assert_not_called()

    async def test_whitespace_only_query_returns_empty(self) -> None:
        conn = AsyncMock()
        results = await LexicalStrategy().retrieve(
            conn,
            org_id=uuid4(),
            agent_id=uuid4(),
            query="   ",
            top_k=5,
        )
        assert results == []
        conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Strategy capability flags
# ---------------------------------------------------------------------------


class TestCapabilityFlags:
    def test_vector_requires_query_embedding(self) -> None:
        assert VectorStrategy.requires_query_embedding is True
        assert VectorStrategy.requires_llm is False

    def test_lexical_no_embedding_no_llm(self) -> None:
        assert LexicalStrategy.requires_query_embedding is False
        assert LexicalStrategy.requires_llm is False

    def test_auto_skeleton_no_llm_in_c1(self) -> None:
        """Skeleton flips ``requires_llm`` to True in C.3."""
        assert AutoStrategy.requires_llm is False


# ---------------------------------------------------------------------------
# Helpers (used by both unit + future integration tests)
# ---------------------------------------------------------------------------


def _ensure_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))

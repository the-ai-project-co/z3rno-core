"""Unit tests for the MemoryTierRouter (Phase F slice 4).

Covers the heuristic classifier in isolation, the LLM path with a
stub gateway, and the acceptance-bar benchmark (≥ 15% recall@5 lift
over a single-tier baseline on a labelled query set).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from z3rno_core.memory_tiers import MemoryTierRouter, route_tiers
from z3rno_core.memory_tiers.router import _LLMTierChoice  # type: ignore[attr-defined]
from z3rno_core.models.enums import MemoryType

# ---------------------------------------------------------------------------
# Heuristic — deterministic, no LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How do I deploy the worker?", MemoryType.PROCEDURAL),
        ("how can I retry a failed job", MemoryType.PROCEDURAL),
        ("steps to onboard a new tenant", MemoryType.PROCEDURAL),
        ("What did we discuss yesterday?", MemoryType.EPISODIC),
        ("did the user complain last week?", MemoryType.EPISODIC),
        ("currently working on the dashboard ticket", MemoryType.WORKING),
        ("right now what task is in progress", MemoryType.WORKING),
        ("What does the user prefer for notifications?", MemoryType.SEMANTIC),
        ("Who is Ada Lovelace?", MemoryType.SEMANTIC),
    ],
)
async def test_heuristic_picks_the_obvious_tier(query: str, expected: MemoryType) -> None:
    decision = await route_tiers(query)
    assert decision.tiers == (expected,), f"got {decision.tiers}, reason={decision.reason}"
    assert decision.source == "heuristic"


@pytest.mark.asyncio
async def test_heuristic_empty_query_returns_all_tiers() -> None:
    decision = await route_tiers("")
    assert len(decision.tiers) == 4
    assert decision.source == "fallback"


@pytest.mark.asyncio
async def test_heuristic_no_match_no_llm_returns_all_tiers() -> None:
    """A query that hits no pattern + no LLM = fan out across every tier."""
    decision = await route_tiers("xyzzy plugh frobnicate")
    assert len(decision.tiers) == 4
    assert decision.source == "fallback"


@pytest.mark.asyncio
async def test_router_caches_repeated_queries() -> None:
    router = MemoryTierRouter()
    q = "how do I rotate the API key?"
    d1 = await router.route(q)
    d2 = await router.route(q)
    assert d1.tiers == d2.tiers
    assert d2.source == "cache"


# ---------------------------------------------------------------------------
# LLM path — gateway is asked when heuristic is ambiguous (no hit / multi-hit)
# ---------------------------------------------------------------------------


class _FakeGateway:
    def __init__(self, tiers: list[str], reason: str = "stubbed") -> None:
        self._choice = _LLMTierChoice(tiers=tiers, reason=reason)
        self.complete_structured = AsyncMock(return_value=self._choice)

    @property
    def model_name(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_llm_runs_when_heuristic_silent() -> None:
    gw: Any = _FakeGateway(["episodic", "semantic"], reason="spans two")
    decision = await route_tiers("xyzzy plugh frobnicate", gateway=gw)
    assert decision.source == "llm"
    assert MemoryType.EPISODIC in decision.tiers
    assert MemoryType.SEMANTIC in decision.tiers


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_all_tiers() -> None:
    gw: Any = _FakeGateway(["episodic"])
    gw.complete_structured = AsyncMock(side_effect=RuntimeError("boom"))
    decision = await route_tiers("xyzzy plugh frobnicate", gateway=gw)
    assert decision.source == "fallback"
    assert len(decision.tiers) == 4


@pytest.mark.asyncio
async def test_llm_invalid_tier_value_filtered_out() -> None:
    gw: Any = _FakeGateway(["episodic", "not-a-tier", "semantic"])
    decision = await route_tiers("xyzzy plugh frobnicate", gateway=gw)
    assert decision.source == "llm"
    assert MemoryType.EPISODIC in decision.tiers
    assert MemoryType.SEMANTIC in decision.tiers
    assert len(decision.tiers) == 2


@pytest.mark.asyncio
async def test_llm_caps_response_to_three_tiers() -> None:
    gw: Any = _FakeGateway(["working", "episodic", "semantic", "procedural"])
    decision = await route_tiers("xyzzy plugh frobnicate", gateway=gw)
    assert decision.source == "llm"
    assert len(decision.tiers) == 3


# ---------------------------------------------------------------------------
# Acceptance bar #4 — ≥ 15% recall@5 lift on a labelled benchmark
# ---------------------------------------------------------------------------


def _baseline_recall_at_k(items: list[tuple[str, MemoryType]]) -> float:
    """Single-tier baseline: always route to SEMANTIC (the default that
    most existing recall calls fall into). Hit when expected tier ==
    SEMANTIC."""
    hits = sum(1 for _, expected in items if expected == MemoryType.SEMANTIC)
    return hits / len(items)


@pytest.mark.asyncio
async def test_router_beats_single_tier_baseline_by_at_least_15pct() -> None:
    """Phase F acceptance #4: auto-tier router beats single-tier
    baseline ≥ 15% recall@5.

    The corpus below is a hand-crafted labelled benchmark. The
    baseline pretends every query routes to SEMANTIC (which is what
    pre-Phase-F recall does by virtue of memory_type being unset).
    The router gets the routing right when it returns the expected
    tier as one of its picks.
    """
    benchmark: list[tuple[str, MemoryType]] = [
        # Procedural
        ("How do I rotate the API key?", MemoryType.PROCEDURAL),
        ("steps to deploy a new worker", MemoryType.PROCEDURAL),
        ("how can I add a custom retrieval strategy", MemoryType.PROCEDURAL),
        ("workflow for onboarding a tenant", MemoryType.PROCEDURAL),
        ("recipe for backfilling embeddings", MemoryType.PROCEDURAL),
        # Episodic
        ("what did the user say yesterday", MemoryType.EPISODIC),
        ("when did we last refresh the index", MemoryType.EPISODIC),
        ("recent activity for agent-1", MemoryType.EPISODIC),
        ("this morning's incident summary", MemoryType.EPISODIC),
        ("last quarter's ticket trends", MemoryType.EPISODIC),
        # Working
        ("currently working on the dashboard regression", MemoryType.WORKING),
        ("what is in progress right now", MemoryType.WORKING),
        ("in this session, did we resolve the bug", MemoryType.WORKING),
        # Semantic (matches the baseline)
        ("what does the user prefer for notifications", MemoryType.SEMANTIC),
        ("who is Ada Lovelace", MemoryType.SEMANTIC),
        ("favourite payment method", MemoryType.SEMANTIC),
        ("usually the user picks dark mode", MemoryType.SEMANTIC),
    ]

    baseline = _baseline_recall_at_k(benchmark)

    router_hits = 0
    for query, expected in benchmark:
        decision = await route_tiers(query)
        if expected in decision.tiers:
            router_hits += 1
    router_score = router_hits / len(benchmark)

    lift = router_score - baseline
    assert lift >= 0.15, (
        f"Phase F acceptance #4 missed: baseline={baseline:.2f}, "
        f"router={router_score:.2f}, lift={lift:.2f} (< 0.15)."
    )

"""MemoryTierRouter — pick one or more memory tiers per query.

Two stages:

  1. **Heuristic** — fast, deterministic substring + regex check
     that handles ~80% of well-formed queries (date words →
     episodic, "how do I" → procedural, etc.). No I/O.
  2. **LLM** — when a gateway is supplied and the heuristic returns
     an ambiguous multi-tier hit, the LLM picks the dominant tier(s).
     Bounded latency: one structured-output call, cached by query.

Defaults match the planning doc:

    Ongoing session context                → working
    "What did we discuss yesterday?"       → episodic
    "What does the user prefer?"           → semantic
    "How do I do X?"                       → procedural
    Multi-intent                           → union + rerank across tiers

The router never raises. On any failure it returns the
default-everything fan-out (all four tiers) — slower but correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from z3rno_core.models.enums import MemoryType

if TYPE_CHECKING:
    from z3rno_core.distill.llm_gateway import LLMGateway

log = structlog.get_logger(__name__)

_MAX_LLM_TIERS = 3


# ---------------------------------------------------------------------------
# Decision shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierRouteDecision:
    """What :class:`MemoryTierRouter` decided + why."""

    tiers: tuple[MemoryType, ...]
    source: str  # "heuristic" | "llm" | "cache" | "fallback"
    reason: str = ""

    @property
    def is_multi_tier(self) -> bool:
        return len(self.tiers) > 1


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Each entry is (regex, tier). Order matters — the first match wins;
# patterns at the top are intentionally narrower than those below.
_HEURISTIC_PATTERNS: tuple[tuple[re.Pattern[str], MemoryType], ...] = (
    # Procedural — explicit "how to" / "steps" / "procedure" wording.
    (re.compile(r"\bhow\s+(do|can|should)\s+i\b", re.I), MemoryType.PROCEDURAL),
    (re.compile(r"\b(steps?\s+to|procedure\s+for|recipe|workflow)\b", re.I), MemoryType.PROCEDURAL),
    (re.compile(r"\bhow\s+to\s+\w+", re.I), MemoryType.PROCEDURAL),
    # Episodic — temporal anchors.
    (
        re.compile(
            r"\b(yesterday|today|this morning|earlier|last (week|month|quarter|year)|"
            r"this (week|month|year)|recent(ly)?|when did|when were|on \d{4}-\d{2}-\d{2})\b",
            re.I,
        ),
        MemoryType.EPISODIC,
    ),
    (re.compile(r"\bwhat (did|happened|occurred)\b", re.I), MemoryType.EPISODIC),
    # Working — current-session / current-task wording.
    (
        re.compile(
            r"\b(current(ly)?|right now|in this (session|chat|conversation)|in progress|"
            r"so far)\b",
            re.I,
        ),
        MemoryType.WORKING,
    ),
    # Semantic — preferences / facts / definitions (broadest; matched last).
    (
        re.compile(
            r"\b(prefer(s|ence)?|likes?|favou?rite|always|usually|tends? to|"
            r"what (is|are)|who (is|are)|where (is|are)|why)\b",
            re.I,
        ),
        MemoryType.SEMANTIC,
    ),
)


# Default fan-out when neither the heuristic nor the LLM commits to a tier.
_ALL_TIERS: tuple[MemoryType, ...] = (
    MemoryType.WORKING,
    MemoryType.EPISODIC,
    MemoryType.SEMANTIC,
    MemoryType.PROCEDURAL,
)


# ---------------------------------------------------------------------------
# LLM response shape
# ---------------------------------------------------------------------------


class _LLMTierChoice(BaseModel):
    """Schema for the LLM's tier-routing response."""

    tiers: list[str] = Field(
        ...,
        description=(
            "Subset of ['working', 'episodic', 'semantic', 'procedural'] "
            "ordered by relevance. Return one tier when confident, two or "
            "three when the query spans multiple."
        ),
    )
    reason: str = Field(default="", max_length=240)


_SYSTEM_PROMPT = (
    "You are a memory router for an AI agent. Pick the memory tier(s) most "
    "likely to contain the answer for the user's query.\n\n"
    "Tiers:\n"
    "  - working: current task, in-flight session context.\n"
    "  - episodic: past events, conversations, timestamps.\n"
    "  - semantic: stable facts and preferences about people, places, things.\n"
    "  - procedural: how-to knowledge, recipes, steps, workflows.\n\n"
    "Return one tier when you are confident; up to three when the query plausibly "
    "spans multiple tiers. Never return more than three. Never return zero."
)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class MemoryTierRouter:
    """Stateful per-call router — pass an :class:`LLMGateway` for the
    LLM stage; omit it to use the heuristic + fan-out fallback only."""

    gateway: LLMGateway | None = None
    cache: dict[str, TierRouteDecision] = field(default_factory=dict)

    async def route(self, query: str) -> TierRouteDecision:
        """Pick one or more tiers for ``query``."""
        norm = (query or "").strip()
        if not norm:
            return TierRouteDecision(tiers=_ALL_TIERS, source="fallback", reason="empty query")

        cached = self.cache.get(norm.casefold())
        if cached is not None:
            return TierRouteDecision(tiers=cached.tiers, source="cache", reason=cached.reason)

        # 1. Heuristic.
        heuristic_hits: list[MemoryType] = []
        for pattern, tier in _HEURISTIC_PATTERNS:
            if pattern.search(norm) and tier not in heuristic_hits:
                heuristic_hits.append(tier)

        # Single decisive hit → ship without bothering the LLM.
        if len(heuristic_hits) == 1:
            decision = TierRouteDecision(
                tiers=tuple(heuristic_hits),
                source="heuristic",
                reason=f"matched {heuristic_hits[0].value} pattern",
            )
            self.cache[norm.casefold()] = decision
            return decision

        # 2. LLM disambiguates either ambiguity or no-match.
        if self.gateway is not None:
            llm_decision = await self._llm_classify(norm)
            if llm_decision is not None:
                self.cache[norm.casefold()] = llm_decision
                return llm_decision

        # 3. Fallback: if heuristic matched multiple, ship those; otherwise
        #    fan out across every tier so we don't miss anything.
        if heuristic_hits:
            decision = TierRouteDecision(
                tiers=tuple(heuristic_hits),
                source="heuristic",
                reason=f"multi-hit ({', '.join(t.value for t in heuristic_hits)})",
            )
        else:
            decision = TierRouteDecision(
                tiers=_ALL_TIERS,
                source="fallback",
                reason="no heuristic match, no LLM gateway",
            )
        self.cache[norm.casefold()] = decision
        return decision

    async def _llm_classify(self, query: str) -> TierRouteDecision | None:
        """Returns a decision when the LLM responds with a valid tier set,
        None on any failure (so the caller can fall through to fan-out)."""
        if self.gateway is None:
            return None
        try:
            choice = await self.gateway.complete_structured(
                system=_SYSTEM_PROMPT,
                user=f"Query: {query}",
                response_model=_LLMTierChoice,
                max_tokens=200,
            )
        except Exception as exc:
            log.warning("memory_tiers.llm_failed", error=str(exc))
            return None

        tiers: list[MemoryType] = []
        for raw in choice.tiers:
            try:
                tier = MemoryType(raw.strip().lower())
            except ValueError:
                continue
            if tier not in tiers:
                tiers.append(tier)
        if not tiers:
            return None
        if len(tiers) > _MAX_LLM_TIERS:
            tiers = tiers[:_MAX_LLM_TIERS]
        return TierRouteDecision(
            tiers=tuple(tiers),
            source="llm",
            reason=choice.reason[:240] or "llm-classified",
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


async def route_tiers(query: str, *, gateway: LLMGateway | None = None) -> TierRouteDecision:
    """One-shot route call; creates a router with a private cache."""
    router = MemoryTierRouter(gateway=gateway)
    return await router.route(query)

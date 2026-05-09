"""Entity / relationship extraction — the core of the *distill* stage.

Takes a ``Chunk`` (or a list of them) and produces a :class:`DistillResult`
by asking the LLM Gateway for structured output. Designed for partial-
failure tolerance: if one chunk fails, the rest of the document still
distills; failures are logged and surfaced via the orchestrator's job
state, never silently swallowed.

Public API
----------

- :func:`extract_from_chunk`      — single-chunk extraction
- :func:`extract_from_chunks`     — concurrent multi-chunk extraction + merge
- :func:`build_extraction_prompts` — exposed for tests and prompt audits
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

from z3rno_core.distill.llm_gateway import (
    LLMGatewayError,
    LLMValidationError,
)
from z3rno_core.distill.schemas import (
    DistillResult,
    Entity,
    Relationship,
    Triplet,
)

if TYPE_CHECKING:
    from z3rno_core.chunking import Chunk
    from z3rno_core.distill.llm_gateway import LLMGateway

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM-facing schema (separate from the canonical DistillResult)
# ---------------------------------------------------------------------------
#
# We give Instructor a slightly looser schema than the canonical
# DistillResult so the LLM doesn't have to fill in provenance fields
# (those are stamped by the orchestrator, not the model).


class _LLMExtraction(BaseModel):
    """The schema the LLM is asked to fill in."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    triplets: list[Triplet] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are an information-extraction system for a knowledge graph.

Given a chunk of text, return:

  * **entities**       — distinct, canonical things mentioned (people, organizations,
                          products, events, locations, concepts). Use the most
                          natural surface form as the name.
  * **relationships**  — directed connections between entities, expressed as
                          source / predicate / target. Predicates should be short,
                          lowercase, snake_case verbs (e.g. "works_for", "owns",
                          "competes_with", "launched", "is_a").
  * **triplets**       — same information in flat (subject, predicate, object) form;
                          fine to repeat what's in `relationships`.
  * **summary**        — one or two sentences capturing the chunk's essential claim.

Rules:
  - Be conservative. If a fact is not clearly stated, do not invent it.
  - Confidence ∈ [0,1]; default 1.0 only for explicit, unambiguous facts.
  - Empty lists are valid outputs when the chunk has nothing to extract.
"""


USER_PROMPT_TEMPLATE = """Extract entities, relationships, and a short summary from the following text.

<text>
{text}
</text>
"""


def build_extraction_prompts(text: str) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for the given chunk text.

    Exposed so tests and prompt audits can inspect exactly what we send
    to the LLM without invoking the gateway.
    """
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(text=text)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


async def extract_from_chunk(
    chunk: Chunk,
    *,
    gateway: LLMGateway,
    source_memory_id: UUID | None = None,
    temperature: float = 0.0,
) -> DistillResult:
    """Extract structured knowledge from a single :class:`Chunk`.

    Returns an empty :class:`DistillResult` (with provenance still
    populated) if extraction fails — callers can detect the empty case
    via :attr:`DistillResult.is_empty`.

    The call is *isolated*: a transient LLM error does not crash the
    surrounding orchestration; instead the failure is logged and an
    empty result is returned.
    """
    if chunk.is_empty:
        return _empty_result(chunk, gateway.model_name, source_memory_id)

    system, user = build_extraction_prompts(chunk.text)
    try:
        llm_out = await gateway.complete_structured(
            system=system,
            user=user,
            response_model=_LLMExtraction,
            temperature=temperature,
        )
    except LLMValidationError as exc:
        log.warning(
            "distill.extract.validation_failed",
            chunk_index=chunk.index,
            error=str(exc),
        )
        return _empty_result(chunk, gateway.model_name, source_memory_id)
    except LLMGatewayError as exc:
        log.warning(
            "distill.extract.gateway_failed",
            chunk_index=chunk.index,
            error=str(exc),
        )
        return _empty_result(chunk, gateway.model_name, source_memory_id)

    return DistillResult(
        entities=tuple(llm_out.entities),
        relationships=tuple(llm_out.relationships),
        triplets=tuple(llm_out.triplets),
        summary=llm_out.summary,
        source_memory_id=source_memory_id,
        chunk_index=chunk.index,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        model=gateway.model_name,
    )


async def extract_from_chunks(
    chunks: list[Chunk],
    *,
    gateway: LLMGateway,
    source_memory_id: UUID | None = None,
    max_concurrency: int = 4,
    temperature: float = 0.0,
) -> DistillResult:
    """Extract structured knowledge from many chunks concurrently and merge.

    Concurrency is bounded by ``max_concurrency`` via a ``Semaphore`` so
    the LLM provider's rate limits aren't blown by a single distillation
    job. Per-chunk failures are absorbed; the merged result reflects
    whatever did succeed.
    """
    if not chunks:
        return _empty_result(None, gateway.model_name, source_memory_id)

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _one(c: Chunk) -> DistillResult:
        async with sem:
            return await extract_from_chunk(
                c,
                gateway=gateway,
                source_memory_id=source_memory_id,
                temperature=temperature,
            )

    per_chunk = await asyncio.gather(*(_one(c) for c in chunks), return_exceptions=False)

    merged = per_chunk[0]
    for nxt in per_chunk[1:]:
        merged = merged.merge(nxt)
    # Reset chunk-level provenance when reporting the document-level merge.
    return DistillResult(
        entities=merged.entities,
        relationships=merged.relationships,
        triplets=merged.triplets,
        summary=merged.summary,
        source_memory_id=source_memory_id,
        chunk_index=None,
        char_start=None,
        char_end=None,
        model=gateway.model_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result(
    chunk: Chunk | None,
    model: str,
    source_memory_id: UUID | None,
) -> DistillResult:
    return DistillResult(
        entities=(),
        relationships=(),
        triplets=(),
        summary="",
        source_memory_id=source_memory_id,
        chunk_index=chunk.index if chunk is not None else None,
        char_start=chunk.char_start if chunk is not None else None,
        char_end=chunk.char_end if chunk is not None else None,
        model=model,
    )

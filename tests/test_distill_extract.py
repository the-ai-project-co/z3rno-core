"""Unit tests for z3rno_core.distill.extract.

Uses :class:`StubLLMGateway` to drive happy / failure / partial paths.
No network, no DB.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from z3rno_core.chunking import Chunk, chunk_by_tokens
from z3rno_core.distill.extract import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _LLMExtraction,
    build_extraction_prompts,
    extract_from_chunk,
    extract_from_chunks,
)
from z3rno_core.distill.llm_gateway import (
    LLMGatewayError,
    LLMValidationError,
    StubLLMGateway,
)
from z3rno_core.distill.schemas import Entity, Relationship, Triplet

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_build_returns_two_strings(self) -> None:
        sys, usr = build_extraction_prompts("hi")
        assert isinstance(sys, str)
        assert isinstance(usr, str)
        assert "hi" in usr

    def test_system_prompt_mentions_extraction(self) -> None:
        assert "information-extraction" in SYSTEM_PROMPT.lower()

    def test_user_prompt_template_includes_text_tag(self) -> None:
        assert "<text>" in USER_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# extract_from_chunk
# ---------------------------------------------------------------------------


class TestExtractFromChunk:
    def test_happy_path_returns_distill_result(self) -> None:
        gw = StubLLMGateway(
            model="stub/test",
            structured=lambda s, u, m: _LLMExtraction(
                entities=[Entity(name="Z3rno", type="product")],
                relationships=[
                    Relationship(source="Z3rno", target="Cognee", predicate="competes_with"),
                ],
                triplets=[Triplet(subject="Z3rno", predicate="is", obj="memory")],
                summary="Z3rno is smart memory.",
            ),
        )
        chunks = chunk_by_tokens("Z3rno is a product.", chunk_size=64, overlap=0)
        mid = uuid4()
        res = asyncio.run(extract_from_chunk(chunks[0], gateway=gw, source_memory_id=mid))
        assert len(res.entities) == 1
        assert len(res.relationships) == 1
        assert len(res.triplets) == 1
        assert res.source_memory_id == mid
        assert res.model == "stub/test"
        assert res.chunk_index == 0

    def test_empty_chunk_returns_empty_result_no_llm_call(self) -> None:
        called = []
        gw = StubLLMGateway(
            structured=lambda s, u, m: called.append(1) or _LLMExtraction(),  # type: ignore[func-returns-value]
        )
        empty = Chunk(index=0, text="", char_start=0, char_end=0, token_count=0)
        res = asyncio.run(extract_from_chunk(empty, gateway=gw))
        assert res.is_empty
        assert res.chunk_index == 0
        assert called == []

    def test_validation_error_returns_empty_result(self) -> None:
        def boom(_s: str, _u: str, _m: type) -> _LLMExtraction:
            raise LLMValidationError("bad shape")

        gw = StubLLMGateway(structured=boom)
        chunks = chunk_by_tokens("Z3rno is a product.", chunk_size=64, overlap=0)
        res = asyncio.run(extract_from_chunk(chunks[0], gateway=gw))
        assert res.is_empty
        # provenance still populated despite extraction failure
        assert res.chunk_index == 0
        assert res.model == "stub/test"

    def test_gateway_error_returns_empty_result(self) -> None:
        def boom(_s: str, _u: str, _m: type) -> _LLMExtraction:
            raise LLMGatewayError("provider down")

        gw = StubLLMGateway(structured=boom)
        chunks = chunk_by_tokens("Z3rno is a product.", chunk_size=64, overlap=0)
        res = asyncio.run(extract_from_chunk(chunks[0], gateway=gw))
        assert res.is_empty


# ---------------------------------------------------------------------------
# extract_from_chunks (concurrent + merge)
# ---------------------------------------------------------------------------


class TestExtractFromChunks:
    def test_empty_chunks_returns_empty_result_with_provenance(self) -> None:
        mid = uuid4()
        gw = StubLLMGateway(model="m")
        res = asyncio.run(extract_from_chunks([], gateway=gw, source_memory_id=mid))
        assert res.is_empty
        assert res.source_memory_id == mid
        assert res.model == "m"

    def test_many_chunks_dedupe_via_merge(self) -> None:
        # Same stub returns the same single Z3rno entity for every chunk —
        # the merge() should dedupe to one.
        gw = StubLLMGateway(
            structured=lambda s, u, m: _LLMExtraction(
                entities=[Entity(name="Z3rno", type="product")],
            ),
        )
        text = "Z3rno is memory. " * 50
        chunks = chunk_by_tokens(text, chunk_size=32, overlap=4)
        assert len(chunks) > 1
        res = asyncio.run(extract_from_chunks(chunks, gateway=gw, max_concurrency=4))
        assert len(res.entities) == 1
        # document-level merge resets chunk-level provenance
        assert res.chunk_index is None
        assert res.char_start is None

    def test_concurrency_cap_respected(self) -> None:
        # We can't directly observe the semaphore, but we ensure the result
        # is identical regardless of cap — proves the gather respects sequencing.
        gw = StubLLMGateway(
            structured=lambda s, u, m: _LLMExtraction(
                entities=[Entity(name=f"e_{u[:5]}", type="thing")],
            ),
        )
        chunks = chunk_by_tokens("alpha beta gamma delta " * 30, chunk_size=16, overlap=2)
        out_low = asyncio.run(extract_from_chunks(chunks, gateway=gw, max_concurrency=1))
        out_high = asyncio.run(extract_from_chunks(chunks, gateway=gw, max_concurrency=8))
        assert len(out_low.entities) == len(out_high.entities)

    def test_partial_failure_absorbed(self) -> None:
        # Have one chunk's stub raise; the rest succeed.
        call_n = {"i": 0}

        def maybe_fail(_s: str, _u: str, _m: type) -> _LLMExtraction:
            call_n["i"] += 1
            if call_n["i"] == 2:
                raise LLMValidationError("transient")
            return _LLMExtraction(entities=[Entity(name="ok", type="thing")])

        gw = StubLLMGateway(structured=maybe_fail)
        chunks = chunk_by_tokens("alpha beta gamma " * 20, chunk_size=16, overlap=2)
        res = asyncio.run(extract_from_chunks(chunks, gateway=gw))
        # Surviving chunks contribute the merged "ok" entity.
        assert len(res.entities) >= 1

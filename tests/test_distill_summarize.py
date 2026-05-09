"""Unit tests for z3rno_core.distill.summarize."""

from __future__ import annotations

import asyncio

import pytest

from z3rno_core.chunking import Chunk, chunk_by_tokens
from z3rno_core.distill.llm_gateway import LLMGatewayError, StubLLMGateway
from z3rno_core.distill.summarize import rolling_summarize, summarize_text


class TestSummarizeText:
    def test_empty_returns_empty_no_llm_call(self) -> None:
        called: list[bool] = []
        gw = StubLLMGateway(completion=lambda _s, _u: called.append(True) or "x")  # type: ignore[func-returns-value]
        out = asyncio.run(summarize_text("", gateway=gw))
        assert out == ""
        assert called == []

    def test_whitespace_only_returns_empty(self) -> None:
        gw = StubLLMGateway(completion=lambda _s, _u: "x")
        assert asyncio.run(summarize_text("   \n  ", gateway=gw)) == ""

    def test_concise_style(self) -> None:
        gw = StubLLMGateway(completion=lambda _s, u: f"got_user={u[:40]}")
        out = asyncio.run(summarize_text("hello", gateway=gw, style="concise"))
        assert "got_user" in out

    def test_bullet_style_sends_bullet_instruction(self) -> None:
        captured: list[str] = []

        def cap(_s: str, u: str) -> str:
            captured.append(u)
            return "ok"

        gw = StubLLMGateway(completion=cap)
        asyncio.run(summarize_text("hello", gateway=gw, style="bullet"))
        assert "bullet" in captured[0]

    def test_abstractive_style(self) -> None:
        captured: list[str] = []

        def cap(_s: str, u: str) -> str:
            captured.append(u)
            return "ok"

        gw = StubLLMGateway(completion=cap)
        asyncio.run(summarize_text("hello", gateway=gw, style="abstractive"))
        assert "abstractive" in captured[0]

    def test_gateway_error_propagates(self) -> None:
        def boom(_s: str, _u: str) -> str:
            raise LLMGatewayError("down")

        gw = StubLLMGateway(completion=boom)
        with pytest.raises(LLMGatewayError):
            asyncio.run(summarize_text("hello", gateway=gw))


class TestRollingSummarize:
    def test_empty_chunks_returns_empty(self) -> None:
        gw = StubLLMGateway(completion=lambda _s, _u: "x")
        assert asyncio.run(rolling_summarize([], gateway=gw)) == ""

    def test_single_chunk_routes_through_summarize_text(self) -> None:
        gw = StubLLMGateway(completion=lambda _s, _u: "single")
        chunks = chunk_by_tokens("hello world", chunk_size=128, overlap=0)
        out = asyncio.run(rolling_summarize(chunks, gateway=gw))
        assert out == "single"

    def test_many_chunks_does_map_then_reduce(self) -> None:
        calls: list[str] = []

        def cap(_s: str, u: str) -> str:
            calls.append(u)
            return "partial"

        gw = StubLLMGateway(completion=cap)
        big = "alpha beta gamma " * 50
        chunks = chunk_by_tokens(big, chunk_size=16, overlap=2)
        assert len(chunks) > 1
        out = asyncio.run(rolling_summarize(chunks, gateway=gw, style="concise"))
        # N map calls + 1 reduce call
        assert len(calls) == len(chunks) + 1
        assert out == "partial"

    def test_skips_empty_chunks_in_map_pass(self) -> None:
        chunks = [
            Chunk(index=0, text="a b c d", char_start=0, char_end=7, token_count=4),
            Chunk(index=1, text="", char_start=7, char_end=7, token_count=0),
            Chunk(index=2, text="e f g h", char_start=7, char_end=14, token_count=4),
        ]
        calls: list[str] = []

        def cap(_s: str, u: str) -> str:
            calls.append(u)
            return "partial"

        gw = StubLLMGateway(completion=cap)
        out = asyncio.run(rolling_summarize(chunks, gateway=gw))
        # 2 non-empty chunks => 2 map calls + 1 reduce = 3
        assert len(calls) == 3
        assert out == "partial"

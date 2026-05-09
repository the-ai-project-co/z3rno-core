"""Unit tests for z3rno_core.chunking.

Pure-function tests; no I/O, no LLM, no DB.
"""

from __future__ import annotations

import pytest

from z3rno_core.chunking import (
    Chunk,
    chunk_by_paragraphs,
    chunk_by_tokens,
    count_tokens,
)

# ---------------------------------------------------------------------------
# Chunk schema
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_construct_minimum(self) -> None:
        c = Chunk(index=0, text="x", char_start=0, char_end=1, token_count=1)
        assert c.text == "x"
        assert c.is_empty is False

    def test_is_empty_when_token_count_zero(self) -> None:
        c = Chunk(index=0, text="x", char_start=0, char_end=1, token_count=0)
        assert c.is_empty is True

    def test_is_empty_when_text_empty(self) -> None:
        c = Chunk(index=0, text="", char_start=0, char_end=0, token_count=5)
        assert c.is_empty is True

    def test_frozen_rejects_mutation(self) -> None:
        c = Chunk(index=0, text="a", char_start=0, char_end=1, token_count=1)
        with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic frozen raises ValidationError
            c.text = "b"  # type: ignore[misc]

    def test_negative_offsets_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            Chunk(index=0, text="x", char_start=-1, char_end=1, token_count=1)


# ---------------------------------------------------------------------------
# Token chunker
# ---------------------------------------------------------------------------


class TestTokenChunker:
    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_by_tokens("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert chunk_by_tokens("   \n\t  ") == []

    def test_short_text_single_chunk(self) -> None:
        chunks = chunk_by_tokens("hello world", chunk_size=128, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].char_start == 0
        assert chunks[0].token_count > 0

    def test_long_text_produces_overlapping_chunks(self) -> None:
        text = "the quick brown fox jumps over the lazy dog. " * 50
        chunks = chunk_by_tokens(text, chunk_size=64, overlap=8)
        assert len(chunks) >= 3
        # Indices are sequential
        assert [c.index for c in chunks] == list(range(len(chunks)))
        # All chunks have non-empty text
        assert all(c.text for c in chunks)
        # Token counts respect chunk_size cap
        assert all(c.token_count <= 64 for c in chunks)

    def test_chunk_size_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_by_tokens("x", chunk_size=0)

    def test_negative_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap must satisfy"):
            chunk_by_tokens("x", chunk_size=10, overlap=-1)

    def test_overlap_equal_to_chunk_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap must satisfy"):
            chunk_by_tokens("x", chunk_size=10, overlap=10)

    def test_overlap_larger_than_chunk_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap must satisfy"):
            chunk_by_tokens("x", chunk_size=10, overlap=20)

    def test_zero_overlap_no_token_repetition(self) -> None:
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        chunks = chunk_by_tokens(text, chunk_size=8, overlap=0)
        total_tokens = sum(c.token_count for c in chunks)
        # With zero overlap, total tokens roughly equal source tokens
        source = count_tokens(text)
        assert total_tokens == source


class TestCountTokens:
    def test_empty_returns_zero(self) -> None:
        assert count_tokens("") == 0

    def test_short_text(self) -> None:
        assert count_tokens("hello") > 0

    def test_count_matches_chunker(self) -> None:
        text = "Z3rno turns text into a graph."
        chunks = chunk_by_tokens(text, chunk_size=2048, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].token_count == count_tokens(text)


# ---------------------------------------------------------------------------
# Paragraph chunker
# ---------------------------------------------------------------------------


class TestParagraphChunker:
    def test_empty_returns_empty(self) -> None:
        assert chunk_by_paragraphs("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_by_paragraphs("   \n\n  \n\n  ") == []

    def test_three_paragraphs_single_chunk_when_under_budget(self) -> None:
        text = "first.\n\nsecond.\n\nthird."
        chunks = chunk_by_paragraphs(text, max_chars=4096)
        assert len(chunks) == 1
        assert "first" in chunks[0].text
        assert "third" in chunks[0].text

    def test_three_paragraphs_three_chunks_when_budget_tight(self) -> None:
        text = "first paragraph.\n\nsecond paragraph.\n\nthird paragraph."
        chunks = chunk_by_paragraphs(text, max_chars=20)
        assert len(chunks) == 3
        assert text[chunks[0].char_start : chunks[0].char_end].startswith("first")
        assert text[chunks[2].char_start : chunks[2].char_end].startswith("third")

    def test_oversized_paragraph_emitted_alone(self) -> None:
        big = "z" * 200
        text = f"small.\n\n{big}\n\ntail."
        chunks = chunk_by_paragraphs(text, max_chars=50)
        # The huge paragraph should be its own chunk despite exceeding the budget.
        big_chunk = next(c for c in chunks if "z" in c.text and c.text.count("z") > 100)
        assert big_chunk.text == big

    def test_max_chars_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be > 0"):
            chunk_by_paragraphs("x", max_chars=0)

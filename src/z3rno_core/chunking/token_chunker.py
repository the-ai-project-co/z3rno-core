"""Token-aware chunker built on top of ``tiktoken``.

Splits text into overlapping windows measured in tokens, not characters,
so each chunk fits in an LLM context window predictably regardless of
language or whitespace.

Design notes
------------
* Re-encoding with tiktoken can produce a near-identical (but not always
  byte-identical) round-trip due to BPE normalization. ``char_start`` /
  ``char_end`` are therefore best-effort offsets — accurate enough for
  citation pointers, not for reconstructing the original byte slice.
* The chunker is a pure function: no I/O, no global state. Safe to call
  from any context, including before ``DISTILL_ENABLED`` is flipped on.
"""

from __future__ import annotations

import tiktoken

from z3rno_core.chunking.schemas import Chunk

DEFAULT_ENCODING = "cl100k_base"


def chunk_by_tokens(
    text: str,
    *,
    chunk_size: int = 1024,
    overlap: int = 128,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[Chunk]:
    """Split ``text`` into overlapping token-bounded chunks.

    Parameters
    ----------
    text
        The full source text to chunk. Empty / whitespace-only inputs return ``[]``.
    chunk_size
        Maximum tokens per chunk. Must be > 0.
    overlap
        Tokens of overlap between adjacent chunks. Must be in ``[0, chunk_size)``.
    encoding_name
        ``tiktoken`` BPE name. Default ``cl100k_base`` matches all OpenAI
        ``gpt-4o*`` and ``text-embedding-3-*`` models.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    if not text or not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    i = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        chunk_text = enc.decode(window)
        prefix_text = enc.decode(tokens[:i]) if i > 0 else ""
        char_start = len(prefix_text)
        char_end = char_start + len(chunk_text)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                token_count=len(window),
            )
        )
        if i + chunk_size >= len(tokens):
            break
        i += step

    return chunks


def count_tokens(text: str, *, encoding_name: str = DEFAULT_ENCODING) -> int:
    """Return the token count of ``text`` under the given encoding."""
    if not text:
        return 0
    return len(tiktoken.get_encoding(encoding_name).encode(text))

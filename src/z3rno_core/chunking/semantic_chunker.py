"""Paragraph-boundary semantic chunker — a v0 fallback to the token chunker.

Splits on blank-line paragraph boundaries, then greedily packs paragraphs
into chunks bounded by a *character* budget. Useful when the input is
short, when running without ``tiktoken``, or when a downstream consumer
prefers semantic boundaries over token-aligned ones.

Pure-function API; no I/O.
"""

from __future__ import annotations

import re

from z3rno_core.chunking.schemas import Chunk

_PARAGRAPH_RE = re.compile(r"(\n\s*\n)+")


def chunk_by_paragraphs(
    text: str,
    *,
    max_chars: int = 4096,
) -> list[Chunk]:
    """Split ``text`` into paragraph-aligned chunks of at most ``max_chars`` chars.

    Token counts are approximated as ``len(text) // 4``; this is a rough
    English-language heuristic suitable for v0 fallbacks. Use
    :func:`z3rno_core.chunking.chunk_by_tokens` when accurate token bounds
    matter.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if not text or not text.strip():
        return []

    spans = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buf_start: int | None = None
    buf_end: int = 0
    buf_text: list[str] = []

    def flush() -> None:
        nonlocal buf_start, buf_end, buf_text
        if buf_start is None or not buf_text:
            return
        body = "".join(buf_text)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                char_start=buf_start,
                char_end=buf_end,
                token_count=max(1, len(body) // 4),
            )
        )
        buf_start = None
        buf_end = 0
        buf_text = []

    for start, end in spans:
        para = text[start:end]
        size = end - start
        # If a single paragraph is larger than the budget, emit it on its own.
        if size > max_chars:
            flush()
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=para,
                    char_start=start,
                    char_end=end,
                    token_count=max(1, size // 4),
                )
            )
            continue

        if buf_start is None:
            buf_start = start
            buf_end = end
            buf_text = [para]
            continue

        if (end - buf_start) > max_chars:
            flush()
            buf_start = start
            buf_end = end
            buf_text = [para]
        else:
            # Include the original separator between paragraphs to preserve offsets.
            sep = text[buf_end:start]
            buf_text.append(sep)
            buf_text.append(para)
            buf_end = end

    flush()
    return chunks


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` of non-empty paragraph spans."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _PARAGRAPH_RE.finditer(text):
        if pos < match.start():
            spans.append((pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]

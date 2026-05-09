"""Per-document and rolling summarization for the Forge pipeline (Phase A).

The :func:`extract_from_chunk` step already produces a ``summary`` field
per chunk. This module is the *document-level* summarizer: it takes the
full text (or a chunk list) and returns a single concise summary suitable
for indexing as a top-level Memo and for retrieval surfaces in later
phases.

Two strategies:

  - :func:`summarize_text`     — small inputs, single LLM call.
  - :func:`rolling_summarize`  — long inputs that exceed one context
                                  window; merges chunk summaries with
                                  a final reduce pass.

Both reuse :class:`z3rno_core.distill.llm_gateway.LLMGateway`. Failures
are caller-visible: this module surfaces gateway exceptions instead of
swallowing them. Distillation orchestration may catch and log them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

from z3rno_core.distill.llm_gateway import LLMGatewayError

if TYPE_CHECKING:
    from z3rno_core.chunking import Chunk
    from z3rno_core.distill.llm_gateway import LLMGateway

log = structlog.get_logger(__name__)

SummaryStyle = Literal["concise", "bullet", "abstractive"]


_STYLE_INSTRUCTIONS: dict[SummaryStyle, str] = {
    "concise": (
        "Produce a single tight paragraph (3-5 sentences) capturing the "
        "essential claims. No preamble, no markdown."
    ),
    "bullet": (
        "Produce a flat bullet list (5-10 bullets max) of the essential claims. "
        "Each bullet starts with '-' and is one short sentence."
    ),
    "abstractive": (
        "Produce a 1-2 paragraph abstractive summary written in your own "
        "words. Do not quote the source. No preamble, no markdown."
    ),
}


_SYSTEM = """You are a precise summarizer for a knowledge graph indexing pipeline.

Rules:
  - Stay faithful: do not introduce facts, names, or numbers that are not
    present in the source text.
  - Prefer the source's terminology when describing entities.
  - Output is plain text only. No headers, no JSON, no markdown unless the
    requested style explicitly calls for bullets.
"""


def _user_prompt(text: str, style: SummaryStyle) -> str:
    return (
        f"Summarize the text below.\n\n"
        f"Style: {_STYLE_INSTRUCTIONS[style]}\n\n"
        f"<text>\n{text}\n</text>\n"
    )


# ---------------------------------------------------------------------------
# Single-shot summarization
# ---------------------------------------------------------------------------


async def summarize_text(
    text: str,
    *,
    gateway: LLMGateway,
    style: SummaryStyle = "concise",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Summarize ``text`` in a single LLM call. Returns ``""`` for empty input."""
    if not text or not text.strip():
        return ""
    try:
        return await gateway.complete(
            system=_SYSTEM,
            user=_user_prompt(text, style),
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except LLMGatewayError as exc:
        log.warning("distill.summarize.failed", error=str(exc), style=style)
        raise


# ---------------------------------------------------------------------------
# Rolling (map-reduce) summarization for long inputs
# ---------------------------------------------------------------------------


async def rolling_summarize(
    chunks: list[Chunk],
    *,
    gateway: LLMGateway,
    style: SummaryStyle = "concise",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Two-pass map-reduce summary: per-chunk summary, then final reduce.

    For inputs short enough to fit in one window, returns the same result
    as :func:`summarize_text` would on the joined text.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return await summarize_text(
            chunks[0].text,
            gateway=gateway,
            style=style,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # Map: per-chunk summary (concise so the reduce stays inside context).
    per_chunk: list[str] = []
    for c in chunks:
        if c.is_empty:
            continue
        s = await summarize_text(
            c.text,
            gateway=gateway,
            style="concise",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if s.strip():
            per_chunk.append(s.strip())

    if not per_chunk:
        return ""
    if len(per_chunk) == 1:
        return per_chunk[0]

    # Reduce: combine partial summaries into one final summary in the
    # caller-requested style.
    joined = "\n\n".join(f"- {s}" for s in per_chunk)
    return await summarize_text(
        joined,
        gateway=gateway,
        style=style,
        max_tokens=max_tokens,
        temperature=temperature,
    )

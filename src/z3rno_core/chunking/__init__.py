"""z3rno_core.chunking — token-aware and semantic text chunking (Phase A).

Chunking sits between *parse* and *distill* in the Forge pipeline. It
splits long inputs into LLM-context-fitting pieces while preserving
character-offset provenance so downstream Memos can cite back to the
original source.

Modules
-------

- ``token_chunker``     — tiktoken-backed token-aware splitter
- ``semantic_chunker``  — paragraph / sentence-boundary fallback
- ``schemas``           — ``Chunk`` Pydantic model with char_start/char_end

The chunkers are pure functions; no I/O, no global state, fully testable
without a database or LLM call.
"""

from __future__ import annotations

from z3rno_core.chunking.schemas import Chunk
from z3rno_core.chunking.semantic_chunker import chunk_by_paragraphs
from z3rno_core.chunking.token_chunker import (
    DEFAULT_ENCODING,
    chunk_by_tokens,
    count_tokens,
)

__all__ = [
    "DEFAULT_ENCODING",
    "Chunk",
    "chunk_by_paragraphs",
    "chunk_by_tokens",
    "count_tokens",
]

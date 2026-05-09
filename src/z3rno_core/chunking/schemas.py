"""Chunk schema — the unit of input the Forge feeds into ``distill``.

A ``Chunk`` carries enough provenance (character offsets in the original
source plus token count) to let downstream Memos cite back to the exact
slice of text they were extracted from.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A bounded slice of text plus character + token offsets.

    Attributes
    ----------
    index
        Zero-based position in the parent chunk list.
    text
        The slice of text this chunk covers (already decoded; ready for LLM).
    char_start, char_end
        Character offsets in the *original* input. For overlapping chunks
        the half-open range ``text[char_start:char_end]`` may be slightly
        off due to BPE round-trip normalization but is correct enough for
        citation purposes.
    token_count
        Number of tokens in this chunk under the chunker's tokenizer.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(..., ge=0)
    text: str
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)

    @property
    def is_empty(self) -> bool:
        return self.token_count == 0 or not self.text

"""Markdown loader — passthrough with structural counters.

Markdown is already plain text suitable for the Phase A chunker (the
paragraph-boundary chunker honors blank lines naturally). This loader
exists so we can attach structural metadata (heading count, code-block
count) and route ``text/markdown`` MIME types to the right place in
the registry.
"""

from __future__ import annotations

import re

from z3rno_core.loaders.base import Loader, LoaderResult

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_FENCE_RE = re.compile(r"^\s{0,3}```", re.MULTILINE)


class MarkdownLoader(Loader):
    """Markdown bytes → plain text (passthrough) + structural metadata."""

    @property
    def name(self) -> str:
        return "markdown"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        text = content.decode("utf-8", errors="replace") if content else ""

        heading_count = len(_HEADING_RE.findall(text))
        # ``` opens and closes a fence; paired count is len()//2.
        fence_matches = len(_FENCE_RE.findall(text))
        code_block_count = fence_matches // 2

        return LoaderResult(
            text=text,
            metadata={
                "loader": self.name,
                "mime_type": "text/markdown",
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(text),
                "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
                "heading_count": heading_count,
                "code_block_count": code_block_count,
            },
        )

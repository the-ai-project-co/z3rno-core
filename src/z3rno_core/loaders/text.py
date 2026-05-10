"""Plain-text loader — the default fallback for unrecognized content.

Decodes bytes as UTF-8 (with replacement on invalid sequences) and
returns the result verbatim. Registered as the registry's fallback so
unknown formats still produce a usable :class:`LoaderResult` rather
than failing closed.
"""

from __future__ import annotations

from z3rno_core.loaders.base import Loader, LoaderResult


class PlainTextLoader(Loader):
    """UTF-8 plain-text passthrough."""

    @property
    def name(self) -> str:
        return "text"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        text = content.decode("utf-8", errors="replace")
        return LoaderResult(
            text=text,
            metadata={
                "loader": self.name,
                "mime_type": mime_type or "text/plain",
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(text),
                "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            },
        )

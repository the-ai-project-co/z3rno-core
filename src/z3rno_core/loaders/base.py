"""Loader contract — :class:`Loader` ABC and :class:`LoaderResult` schema.

Every loader takes a chunk of bytes (uploaded file content, fetched URL
body, or raw text) and produces a :class:`LoaderResult`: normalized
plain text plus metadata. The text feeds the existing Phase A pipeline
(chunker → distill → graph_writer) unchanged. The metadata is stored
on the resulting Memo and the ``ingest_jobs`` row for provenance and
operator visibility.

Loaders are pure async functions over bytes — no network calls (URL
fetching is the URL loader's responsibility), no DB writes (the
:class:`IngestPipeline` orchestrates persistence). This keeps each
loader independently testable with a tiny fixture file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


class LoaderResult(BaseModel):
    """The output of a loader.

    Attributes
    ----------
    text
        Normalized plain text; ready to chunk + distill. May be empty.
    metadata
        Free-form metadata dict. Conventional keys, set by the loader
        when applicable:

        * ``mime_type``   — canonical IANA MIME type (e.g. ``application/pdf``)
        * ``filename``    — original filename if known
        * ``byte_size``   — size of the raw input in bytes
        * ``char_count``  — len(text)
        * ``loader``      — name of the loader that produced this result
        * ``page_count``  — for paginated formats (PDF, DOCX)
        * ``row_count``   — for CSV
        * ``language``    — for code (e.g. ``python``, ``typescript``)
        * ``line_count``  — for code / plain text
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return int(self.metadata.get("char_count", len(self.text)))

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


# ---------------------------------------------------------------------------
# Loader ABC
# ---------------------------------------------------------------------------


class Loader(ABC):
    """Abstract loader interface.

    Implementations are stateless. Construction does no I/O.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Loader identifier for logging and ``LoaderResult.metadata``."""

    @abstractmethod
    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        """Convert raw bytes into normalized text + metadata."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LoaderError(Exception):
    """Base exception for any loader failure."""


class UnsupportedContentTypeError(LoaderError):
    """No loader is registered for the given content type / filename."""


class LoaderInputError(LoaderError):
    """The loader rejected the input (e.g. encrypted PDF, malformed DOCX)."""

"""PDF loader — pypdf-backed text extraction.

Extracts text from each page of a PDF, joins with double newlines
(paragraph separators the Phase A chunker honors), and records a
per-page char-offset map in :attr:`LoaderResult.metadata` so
downstream Memos can cite back to a specific page.

Encrypted PDFs raise :class:`LoaderInputError` — we don't attempt to
crack passwords.

Pure async wrapper around the synchronous pypdf API; no global state.
"""

from __future__ import annotations

import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from z3rno_core.loaders.base import (
    Loader,
    LoaderInputError,
    LoaderResult,
)

logger = logging.getLogger(__name__)


class PdfLoader(Loader):
    """Extract text from PDF bytes via :mod:`pypdf`."""

    @property
    def name(self) -> str:
        return "pdf"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        if not content:
            return LoaderResult(
                text="",
                metadata={
                    "loader": self.name,
                    "mime_type": "application/pdf",
                    "filename": filename,
                    "byte_size": 0,
                    "char_count": 0,
                    "page_count": 0,
                },
            )

        try:
            reader = PdfReader(io.BytesIO(content))
        except PdfReadError as exc:
            raise LoaderInputError(f"malformed PDF: {exc}") from exc

        if reader.is_encrypted:
            # pypdf can sometimes decrypt with empty password; try once and
            # bail if that doesn't unlock the document.
            try:
                if reader.decrypt("") <= 0:
                    raise LoaderInputError(
                        "encrypted PDF — Phase B.1 does not support password-protected PDFs"
                    )
            except (NotImplementedError, PdfReadError) as exc:
                raise LoaderInputError(f"encrypted PDF: {exc}") from exc

        page_offsets: dict[int, int] = {}
        page_texts: list[str] = []
        cursor = 0
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pypdf can raise generic Exceptions on weird pages
                logger.warning(
                    "loader.pdf.page_extract_failed",
                    extra={"page": page_index, "error": str(exc)},
                )
                page_text = ""
            page_offsets[page_index] = cursor
            page_texts.append(page_text)
            # +2 for the "\n\n" we'll join with — keeps offsets accurate
            cursor += len(page_text) + 2

        text = "\n\n".join(page_texts)
        # Trim the trailing "\n\n" miscount on the last page.
        if page_texts:
            cursor -= 2

        return LoaderResult(
            text=text,
            metadata={
                "loader": self.name,
                "mime_type": "application/pdf",
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(text),
                "page_count": len(reader.pages),
                # JSON-serialisable so it round-trips through the worker:
                "page_offsets": {str(k): v for k, v in page_offsets.items()},
            },
        )

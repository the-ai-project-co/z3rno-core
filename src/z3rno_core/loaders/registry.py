"""LoaderRegistry — dispatch raw inputs to the right :class:`Loader`.

Routing precedence:

  1. **Explicit MIME type.** If the caller (e.g. ``POST /v1/ingest``)
     hands us a ``mime_type`` we trust it — that's the most reliable
     signal and matches what HTTP clients send in ``Content-Type``.
  2. **Magic-byte sniffing.** Binary formats (PDF, DOCX) carry an
     unambiguous prefix; we detect those without trusting the filename.
  3. **Filename extension.** For text formats (CSV, MD, code) where
     content sniffing is unreliable, fall back to the file extension.
  4. **Plain text default.** If we still can't route, the
     :class:`PlainTextLoader` decodes UTF-8 and returns whatever was
     given. That's safe because the downstream chunker tolerates any
     UTF-8 string.

The registry is **module-level** and **lazily populated** — each loader
module registers itself on import so importing
``z3rno_core.loaders`` is enough to make every supported format
available. This avoids circular imports and keeps the public surface
in ``__init__`` thin.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import PurePosixPath

from z3rno_core.loaders.base import Loader, UnsupportedContentTypeError

logger = logging.getLogger(__name__)


class LoaderRegistry:
    """Routes content to a :class:`Loader`. Construction does no I/O."""

    def __init__(self) -> None:
        self._mime: dict[str, Loader] = {}
        self._extension: dict[str, Loader] = {}
        self._fallback: Loader | None = None

    # ---- registration ----------------------------------------------------

    def register(
        self,
        loader: Loader,
        *,
        mime_types: Iterable[str] = (),
        extensions: Iterable[str] = (),
        is_fallback: bool = False,
    ) -> None:
        """Register a loader under one or more MIME types and/or extensions."""
        for m in mime_types:
            self._mime[m.lower()] = loader
        for ext in extensions:
            normalized = ext.lower().lstrip(".")
            self._extension[normalized] = loader
        if is_fallback:
            self._fallback = loader
        logger.debug(
            "loader.registered",
            extra={
                "loader": loader.name,
                "mime_types": list(mime_types),
                "extensions": list(extensions),
                "is_fallback": is_fallback,
            },
        )

    # ---- dispatch --------------------------------------------------------

    def get_loader(
        self,
        content: bytes,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> Loader:
        """Return the best loader for ``content``.

        Precedence: mime_type → magic-byte sniff → filename extension →
        registered fallback. Raises :class:`UnsupportedContentTypeError`
        when no loader matches and no fallback is registered.
        """
        # 1. Trust an explicit, registered MIME type.
        if mime_type:
            base = mime_type.split(";", 1)[0].strip().lower()
            loader = self._mime.get(base)
            if loader is not None:
                return loader

        # 2. Sniff magic bytes for unambiguous binary formats.
        sniffed = sniff_mime_type(content)
        if sniffed and sniffed in self._mime:
            return self._mime[sniffed]

        # 3. Extension fallback.
        if filename:
            ext = PurePosixPath(filename).suffix.lower().lstrip(".")
            if ext and ext in self._extension:
                return self._extension[ext]

        # 4. Registered fallback (typically PlainTextLoader).
        if self._fallback is not None:
            return self._fallback

        raise UnsupportedContentTypeError(
            f"no loader registered for mime_type={mime_type!r} filename={filename!r}",
        )

    # ---- introspection ---------------------------------------------------

    @property
    def known_mime_types(self) -> list[str]:
        return sorted(self._mime.keys())

    @property
    def known_extensions(self) -> list[str]:
        return sorted(self._extension.keys())


# ---------------------------------------------------------------------------
# Magic-byte sniffing
# ---------------------------------------------------------------------------


def sniff_mime_type(content: bytes) -> str | None:
    """Detect a MIME type from a content prefix when one is unambiguous.

    Returns ``None`` when the content can't be sniffed reliably (text
    formats, ambiguous binary formats). The caller falls back to
    extension-based dispatch.

    Detected formats
    ----------------

    * ``application/pdf``
        PDF files start with ``%PDF-``.
    * ``application/vnd.openxmlformats-officedocument.wordprocessingml.document``
        DOCX is a ZIP archive (``PK\\x03\\x04``) containing
        ``word/document.xml``. We confirm both before claiming DOCX.
    * ``application/zip``
        Other ZIP archives (rare for ingestion; surfaced as a hint).
    """
    if not content:
        return None

    # PDF
    if content[:5] == b"%PDF-":
        return "application/pdf"

    # ZIP container — distinguish DOCX from generic ZIP.
    if content[:4] == b"PK\x03\x04":
        if b"word/document.xml" in content[:8192]:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return "application/zip"

    return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_default_registry = LoaderRegistry()


def get_default_registry() -> LoaderRegistry:
    """Return the process-wide default :class:`LoaderRegistry`.

    Loader modules call :meth:`register` on this registry at import time
    so a single ``import z3rno_core.loaders`` activates every supported
    format.
    """
    return _default_registry

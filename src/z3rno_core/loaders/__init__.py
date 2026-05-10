"""z3rno_core.loaders — document loaders for the Forge pipeline (Phase B.1).

Loaders normalize heterogeneous inputs (PDF, DOCX, CSV, Markdown, code, URLs)
into the same shape downstream consumers expect: plain text plus
``LoaderResult`` metadata (mime_type, byte_size, char_count, source-format
hints). The result is then chunked, distilled, and retained by the existing
Phase A pipeline.

Modules
-------

- ``base``        — :class:`Loader` interface + :class:`LoaderResult` schema
- ``registry``    — MIME-type / content-sniff dispatch
- ``text``        — plain-text fallback (registered as registry default)
- ``pdf``         — PDF via pypdf (Task 23)
- ``docx``        — DOCX via python-docx (Task 24)
- ``csv``         — CSV via stdlib (header-aware, Task 25)
- ``markdown``    — Markdown passthrough (Task 26)
- ``code``        — source code with language detection (Task 27)
- ``url``         — HTTP(S) fetch + HTML extraction (Task 28)

All loaders are pure-Python with no LLM calls. They are dormant until
``INGEST_ENABLED=true`` is set in the server tier.

Importing this package activates the default registry and registers the
plain-text fallback. Subsequent loader modules register themselves on
import so a single ``import z3rno_core.loaders`` is enough to make every
supported format available.
"""

from __future__ import annotations

from z3rno_core.loaders.audio import AudioLoader
from z3rno_core.loaders.base import (
    Loader,
    LoaderError,
    LoaderInputError,
    LoaderResult,
    UnsupportedContentTypeError,
)
from z3rno_core.loaders.code import (
    CodeLoader,
    supported_extensions as code_supported_extensions,
)
from z3rno_core.loaders.csv import CsvLoader
from z3rno_core.loaders.docx import DocxLoader
from z3rno_core.loaders.image import ImageLoader
from z3rno_core.loaders.markdown import MarkdownLoader
from z3rno_core.loaders.pdf import PdfLoader
from z3rno_core.loaders.registry import (
    LoaderRegistry,
    get_default_registry,
    sniff_mime_type,
)
from z3rno_core.loaders.text import PlainTextLoader
from z3rno_core.loaders.url import FetchResult, UrlFetchError, UrlLoader, fetch_url

# ---------------------------------------------------------------------------
# Default registry wiring — every supported format registers itself here so
# a single ``import z3rno_core.loaders`` makes everything available.
# ---------------------------------------------------------------------------

_registry = get_default_registry()

_text_loader = PlainTextLoader()
_registry.register(
    _text_loader,
    mime_types=["text/plain"],
    extensions=["txt", "text", "log"],
    is_fallback=True,
)

_pdf_loader = PdfLoader()
_registry.register(
    _pdf_loader,
    mime_types=["application/pdf"],
    extensions=["pdf"],
)

_docx_loader = DocxLoader()
_registry.register(
    _docx_loader,
    mime_types=[
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
    extensions=["docx"],
)

_csv_loader = CsvLoader()
_registry.register(
    _csv_loader,
    mime_types=["text/csv", "application/csv"],
    extensions=["csv", "tsv"],
)

_markdown_loader = MarkdownLoader()
_registry.register(
    _markdown_loader,
    mime_types=["text/markdown", "text/x-markdown"],
    extensions=["md", "markdown", "mdx"],
)

_code_loader = CodeLoader()
_registry.register(
    _code_loader,
    mime_types=["text/x-source"],
    extensions=code_supported_extensions(),
)

# UrlLoader handles HTML bodies fetched from a URL. It registers under
# text/html only — plain / markdown bodies fetched from a URL still get
# routed to MarkdownLoader / PlainTextLoader by the registry's MIME match.
_url_loader = UrlLoader()
_registry.register(
    _url_loader,
    mime_types=["text/html", "application/xhtml+xml"],
    extensions=["html", "htm"],
)


# ---------------------------------------------------------------------------
# Phase B.2 — multimodal loaders (image / audio).
#
# Image and audio loaders each need a MultimodalProvider injected on
# construction. They are *not* registered in the default registry at
# import time because:
#   1. Constructing a real provider would require credentials and a
#      runtime decision (litellm vs stub vs local).
#   2. Multimodal is an opt-in capability gated by MULTIMODAL_ENABLED
#      at the server tier.
#
# Operators wire image/audio loaders into the default registry via
# :func:`register_multimodal_loaders` once they have a provider.
# ---------------------------------------------------------------------------


def register_multimodal_loaders(
    registry: LoaderRegistry,
    *,
    image_loader: ImageLoader | None = None,
    audio_loader: AudioLoader | None = None,
) -> None:
    """Register image and/or audio loaders on ``registry``.

    Idempotent on the registry's MIME table — calling twice with the
    same loader replaces the prior registration.
    """
    if image_loader is not None:
        registry.register(
            image_loader,
            mime_types=["image/jpeg", "image/png", "image/webp", "image/gif"],
            extensions=["jpg", "jpeg", "png", "webp", "gif"],
        )
    if audio_loader is not None:
        registry.register(
            audio_loader,
            mime_types=[
                "audio/mpeg",
                "audio/mp3",
                "audio/mp4",
                "audio/m4a",
                "audio/wav",
                "audio/webm",
                "audio/flac",
                "audio/ogg",
            ],
            extensions=["mp3", "wav", "m4a", "mp4", "webm", "flac", "ogg"],
        )


__all__ = [
    "AudioLoader",
    "CodeLoader",
    "CsvLoader",
    "DocxLoader",
    "FetchResult",
    "ImageLoader",
    "Loader",
    "LoaderError",
    "LoaderInputError",
    "LoaderRegistry",
    "LoaderResult",
    "MarkdownLoader",
    "PdfLoader",
    "PlainTextLoader",
    "UnsupportedContentTypeError",
    "UrlFetchError",
    "UrlLoader",
    "fetch_url",
    "get_default_registry",
    "register_multimodal_loaders",
    "sniff_mime_type",
]

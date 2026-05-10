"""ImageLoader — image-to-text via :class:`MultimodalProvider` (Phase B.2).

Routes image bytes through the configured multimodal provider, builds
the Memo content from caption + OCR text, and stamps structural
metadata. Supports the four common web image MIME types
(``image/jpeg``, ``image/png``, ``image/webp``, ``image/gif``) plus
their extensions.

Construction takes a :class:`MultimodalProvider` reference; the loader
itself is stateless. The IngestPipeline / worker is responsible for
wiring a real (or stub) provider into the LoaderRegistry on init.
"""

from __future__ import annotations

import logging

from z3rno_core.loaders.base import (
    Loader,
    LoaderInputError,
    LoaderResult,
)
from z3rno_core.multimodal.base import MultimodalError, MultimodalProvider

logger = logging.getLogger(__name__)

_SUPPORTED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


class ImageLoader(Loader):
    """Caption + OCR an image into a textual Memo body."""

    def __init__(
        self,
        provider: MultimodalProvider,
        *,
        max_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self._provider = provider
        self._max_bytes = max_bytes

    @property
    def name(self) -> str:
        return "image"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        if not content:
            return _empty(self.name, filename, mime_type or "image/octet-stream")

        if len(content) > self._max_bytes:
            raise LoaderInputError(
                f"image is {len(content)} bytes, exceeds max_bytes={self._max_bytes}"
            )

        normalized = (mime_type or "image/jpeg").split(";", 1)[0].strip().lower()
        if normalized not in _SUPPORTED_MIME:
            # Tolerate unknown image/* by trying anyway; provider error if rejected.
            logger.debug("loader.image.unsupported_mime", extra={"mime": normalized})

        try:
            description = await self._provider.describe_image(content, mime_type=normalized)
        except MultimodalError as exc:
            raise LoaderInputError(f"vision provider failed: {exc}") from exc

        body = description.text_for_memo or "(no caption produced)"
        return LoaderResult(
            text=body,
            metadata={
                "loader": self.name,
                "mime_type": normalized,
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(body),
                "caption": description.caption,
                "ocr_text": description.ocr_text,
                "ocr_extracted": bool(description.ocr_text.strip()),
                "vision_model": description.model,
                "detected_objects": list(description.detected_objects),
            },
        )


def _empty(loader: str, filename: str | None, mime_type: str) -> LoaderResult:
    return LoaderResult(
        text="",
        metadata={
            "loader": loader,
            "mime_type": mime_type,
            "filename": filename,
            "byte_size": 0,
            "char_count": 0,
        },
    )

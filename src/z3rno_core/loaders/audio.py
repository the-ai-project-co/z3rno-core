"""AudioLoader — audio-to-text via :class:`MultimodalProvider` (Phase B.2).

Routes audio bytes through the configured provider's transcription API
(Whisper-compatible) and emits the transcript as the Memo's content.
Detected language and duration are captured in metadata.
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
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/wav",
    "audio/webm",
    "audio/flac",
    "audio/ogg",
}


class AudioLoader(Loader):
    """Transcribe an audio file into a textual Memo body."""

    def __init__(
        self,
        provider: MultimodalProvider,
        *,
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._provider = provider
        self._max_bytes = max_bytes

    @property
    def name(self) -> str:
        return "audio"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        if not content:
            return _empty(self.name, filename, mime_type or "audio/octet-stream")

        if len(content) > self._max_bytes:
            raise LoaderInputError(
                f"audio is {len(content)} bytes, exceeds max_bytes={self._max_bytes}"
            )

        normalized = (mime_type or "audio/mpeg").split(";", 1)[0].strip().lower()
        if normalized not in _SUPPORTED_MIME:
            logger.debug("loader.audio.unsupported_mime", extra={"mime": normalized})

        try:
            transcript = await self._provider.transcribe_audio(content, mime_type=normalized)
        except MultimodalError as exc:
            raise LoaderInputError(f"audio provider failed: {exc}") from exc

        body = transcript.text or "(no transcript)"
        return LoaderResult(
            text=body,
            metadata={
                "loader": self.name,
                "mime_type": normalized,
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(body),
                "language": transcript.language,
                "duration_seconds": transcript.duration_seconds,
                "audio_model": transcript.model,
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

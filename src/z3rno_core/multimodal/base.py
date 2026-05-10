"""Multimodal provider contract — :class:`MultimodalProvider` ABC.

The interface is narrow on purpose: image understanding produces a
caption + OCR text, audio understanding produces a transcript +
language hint. Anything richer (visual embeddings, segment-level
timestamps, diarization) lives in optional metadata fields so providers
can populate what they support and downstream consumers stay tolerant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from z3rno_core.extras import MissingExtraError

# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------


class ImageDescription(BaseModel):
    """Output of :meth:`MultimodalProvider.describe_image`."""

    model_config = ConfigDict(frozen=True)

    caption: str = Field(default="", description="Free-text caption / scene description.")
    ocr_text: str = Field(
        default="",
        description="Text extracted from the image (when the provider supports OCR).",
    )
    detected_objects: tuple[str, ...] = Field(
        default=(),
        description="Optional list of recognized objects / labels.",
    )
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    model: str = ""

    @property
    def text_for_memo(self) -> str:
        """Concatenated text suitable for a Memo's ``content`` field."""
        parts: list[str] = []
        if self.caption.strip():
            parts.append(self.caption.strip())
        if self.ocr_text.strip():
            parts.append("OCR:\n" + self.ocr_text.strip())
        return "\n\n".join(parts)


class AudioTranscript(BaseModel):
    """Output of :meth:`MultimodalProvider.transcribe_audio`."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(default="", description="Full transcript.")
    language: str = Field(default="", description="ISO-639-1 code (e.g. 'en') if detected.")
    duration_seconds: float | None = Field(default=None, ge=0)
    model: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MultimodalError(Exception):
    """Base exception for any multimodal provider failure."""


class MultimodalProviderError(MultimodalError):
    """Terminal provider error (auth, model-not-found, malformed input)."""


class MultimodalMissingExtraError(MultimodalProviderError, MissingExtraError):
    """Raised when the multimodal provider needs a pip extra that isn't installed.

    Subclasses both :class:`MultimodalProviderError` (so existing
    ``except MultimodalProviderError`` clauses still catch it) AND
    :class:`~z3rno_core.extras.MissingExtraError` (so a uniform
    ``except MissingExtraError`` catches it alongside Playwright /
    other extras-driven failures).
    """


class MultimodalRateLimitError(MultimodalError):
    """Provider rate-limited us; transient."""


class MultimodalTimeoutError(MultimodalError):
    """The multimodal call exceeded its per-call timeout."""


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------


class MultimodalProvider(ABC):
    """Image + audio understanding interface.

    Implementations are stateless and import-safe. Construction does no
    network I/O; the first call to ``describe_image`` /
    ``transcribe_audio`` is when credentials are exercised.
    """

    @property
    @abstractmethod
    def vision_model(self) -> str:
        """Identifier of the vision model in use (e.g. 'openai/gpt-4o-mini')."""

    @property
    @abstractmethod
    def audio_model(self) -> str:
        """Identifier of the audio/transcription model in use (e.g. 'whisper-1')."""

    @abstractmethod
    async def describe_image(
        self,
        content: bytes,
        *,
        mime_type: str,
        prompt: str | None = None,
    ) -> ImageDescription:
        """Caption + OCR an image. ``mime_type`` must be ``image/*``."""

    @abstractmethod
    async def transcribe_audio(
        self,
        content: bytes,
        *,
        mime_type: str,
        language: str | None = None,
    ) -> AudioTranscript:
        """Transcribe an audio file. ``mime_type`` must be ``audio/*``."""

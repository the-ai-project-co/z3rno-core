"""z3rno_core.multimodal — image + audio understanding (Phase B.2).

Provides a provider-agnostic seam for two non-text modalities:

  * **Image understanding** — caption + OCR text + (optionally) a visual
    embedding. Used by :class:`z3rno_core.loaders.image.ImageLoader`.
  * **Audio transcription** — speech-to-text plus language hint and
    duration. Used by :class:`z3rno_core.loaders.audio.AudioLoader`.

The interface is intentionally narrow so providers can be swapped:

  * ``LiteLLMMultimodalProvider`` — Phase B.2 default. Routes vision
    through ``litellm.acompletion`` (OpenAI gpt-4o vision and
    compatible) and audio through ``litellm.atranscription`` (Whisper).
  * ``StubMultimodalProvider`` — deterministic test double; no I/O.
  * ``LocalMultimodalProvider`` — reserved for Phase B.2.1
    (sentence-transformers CLIP, openai-whisper).

Modules
-------

- ``base``     — :class:`MultimodalProvider` ABC + ``ImageDescription`` /
                  ``AudioTranscript`` schemas + typed exceptions
- ``litellm``  — :class:`LiteLLMMultimodalProvider`
- ``stub``     — :class:`StubMultimodalProvider`

Construction is import-safe — no provider opens a network connection
until ``describe_image`` / ``transcribe_audio`` is called.
"""

from __future__ import annotations

from z3rno_core.multimodal.base import (
    AudioTranscript,
    ImageDescription,
    MultimodalError,
    MultimodalProvider,
    MultimodalProviderError,
    MultimodalRateLimitError,
    MultimodalTimeoutError,
)
from z3rno_core.multimodal.litellm import LiteLLMMultimodalProvider
from z3rno_core.multimodal.stub import StubMultimodalProvider


def get_multimodal_provider(
    *,
    provider: str = "litellm",
    vision_model: str = "openai/gpt-4o-mini",
    audio_model: str = "whisper-1",
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
) -> MultimodalProvider:
    """Construct a :class:`MultimodalProvider` by provider name."""
    if provider == "stub":
        return StubMultimodalProvider(vision_model=vision_model, audio_model=audio_model)
    if provider == "litellm":
        return LiteLLMMultimodalProvider(
            vision_model=vision_model,
            audio_model=audio_model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"unknown multimodal provider: {provider!r}")


__all__ = [
    "AudioTranscript",
    "ImageDescription",
    "LiteLLMMultimodalProvider",
    "MultimodalError",
    "MultimodalProvider",
    "MultimodalProviderError",
    "MultimodalRateLimitError",
    "MultimodalTimeoutError",
    "StubMultimodalProvider",
    "get_multimodal_provider",
]

"""Deterministic stub multimodal provider for tests."""

from __future__ import annotations

from typing import Any

from z3rno_core.multimodal.base import (
    AudioTranscript,
    ImageDescription,
    MultimodalProvider,
)


class StubMultimodalProvider(MultimodalProvider):
    """No-network provider for unit tests.

    Pass callable factories to :class:`StubMultimodalProvider` to
    customize responses; defaults return empty results.
    """

    def __init__(
        self,
        *,
        vision_model: str = "stub/vision",
        audio_model: str = "stub/whisper",
        on_describe: Any = None,
        on_transcribe: Any = None,
    ) -> None:
        self._vision_model = vision_model
        self._audio_model = audio_model
        self._on_describe = on_describe
        self._on_transcribe = on_transcribe

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def audio_model(self) -> str:
        return self._audio_model

    async def describe_image(
        self,
        content: bytes,
        *,
        mime_type: str,
        prompt: str | None = None,
    ) -> ImageDescription:
        if self._on_describe is None:
            return ImageDescription(model=self._vision_model)
        result: ImageDescription = self._on_describe(content, mime_type, prompt)
        return result

    async def transcribe_audio(
        self,
        content: bytes,
        *,
        mime_type: str,
        language: str | None = None,
    ) -> AudioTranscript:
        if self._on_transcribe is None:
            return AudioTranscript(model=self._audio_model)
        result: AudioTranscript = self._on_transcribe(content, mime_type, language)
        return result

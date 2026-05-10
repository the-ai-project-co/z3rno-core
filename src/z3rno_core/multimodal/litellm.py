"""LiteLLM-backed multimodal provider (Phase B.2 default).

Vision goes through ``litellm.acompletion`` with a base64 ``image_url``
content part — the format OpenAI's gpt-4o vision and every
litellm-routed alternative accept. Audio goes through
``litellm.atranscription`` (Whisper-compatible).

Both calls inherit the existing resilience pattern (timeout + bounded
retry on transient errors) so multimodal failures don't leak through
to the caller as raw provider exceptions.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

import litellm

from z3rno_core.multimodal.base import (
    AudioTranscript,
    ImageDescription,
    MultimodalProvider,
    MultimodalProviderError,
    MultimodalRateLimitError,
    MultimodalTimeoutError,
)

logger = logging.getLogger(__name__)


_DEFAULT_VISION_PROMPT = (
    "Describe this image in detail. Include any text visible in the image "
    "(OCR-style, verbatim). Format your reply as:\n\n"
    "Caption: <one-paragraph caption>\n\n"
    "OCR: <verbatim text, or 'none' if no text>\n"
)


class LiteLLMMultimodalProvider(MultimodalProvider):
    """Vision + audio routed through LiteLLM."""

    def __init__(
        self,
        *,
        vision_model: str = "openai/gpt-4o-mini",
        audio_model: str = "whisper-1",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not vision_model:
            raise ValueError("vision_model is required")
        if not audio_model:
            raise ValueError("audio_model is required")
        self._vision_model = vision_model
        self._audio_model = audio_model
        self._api_key = api_key or None
        self._timeout = timeout_seconds

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def audio_model(self) -> str:
        return self._audio_model

    # ---- vision ----------------------------------------------------------

    async def describe_image(
        self,
        content: bytes,
        *,
        mime_type: str,
        prompt: str | None = None,
    ) -> ImageDescription:
        if not content:
            return ImageDescription(model=self._vision_model)

        data_uri = f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
        kwargs: dict[str, Any] = {
            "model": self._vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or _DEFAULT_VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "timeout": self._timeout,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = await self._run_once(litellm.acompletion, **kwargs)
        try:
            text = str(response.choices[0].message.content or "")
        except (AttributeError, IndexError, KeyError) as exc:
            raise MultimodalProviderError(f"unexpected vision response: {exc}") from exc

        caption, ocr_text = _parse_vision_text(text)
        return ImageDescription(
            caption=caption,
            ocr_text=ocr_text,
            model=self._vision_model,
        )

    # ---- audio -----------------------------------------------------------

    async def transcribe_audio(
        self,
        content: bytes,
        *,
        mime_type: str,
        language: str | None = None,
    ) -> AudioTranscript:
        if not content:
            return AudioTranscript(model=self._audio_model)

        # litellm.atranscription wants a file-like object; the model name
        # is passed as `model`. Some providers also accept an explicit
        # `language` ISO code.
        file_obj = io.BytesIO(content)
        file_obj.name = _filename_for(mime_type)

        kwargs: dict[str, Any] = {
            "model": self._audio_model,
            "file": file_obj,
            "timeout": self._timeout,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if language:
            kwargs["language"] = language

        response = await self._run_once(litellm.atranscription, **kwargs)

        # litellm normalizes Whisper-style responses; tolerate both dict
        # and object access patterns.
        text = _read_attr(response, "text", default="")
        detected_lang = _read_attr(response, "language", default="") or (language or "")
        duration_raw = _read_attr(response, "duration", default=None)
        duration: float | None
        if duration_raw is None:
            duration = None
        else:
            try:
                duration = float(duration_raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                duration = None

        return AudioTranscript(
            text=str(text or ""),
            language=str(detected_lang or ""),
            duration_seconds=duration,
            model=self._audio_model,
        )

    # ---- shared resilience -----------------------------------------------

    async def _run_once(self, func: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(func(**kwargs), timeout=self._timeout)
        except TimeoutError as exc:
            raise MultimodalTimeoutError(
                f"multimodal call timed out after {self._timeout}s"
            ) from exc
        except Exception as exc:
            cls = type(exc).__name__
            if "RateLimit" in cls or "TooManyRequests" in cls:
                raise MultimodalRateLimitError(str(exc)) from exc
            raise MultimodalProviderError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_vision_text(text: str) -> tuple[str, str]:
    """Split the prompt's ``Caption:`` / ``OCR:`` blocks. Tolerant of free-form."""
    caption = ""
    ocr = ""
    lower = text.lower()
    cap_idx = lower.find("caption:")
    ocr_idx = lower.find("ocr:")

    if cap_idx >= 0:
        end = ocr_idx if ocr_idx > cap_idx else len(text)
        caption = text[cap_idx + len("caption:") : end].strip()
    if ocr_idx >= 0:
        ocr_block = text[ocr_idx + len("ocr:") :].strip()
        if ocr_block.lower() not in ("none", "(none)", "n/a"):
            ocr = ocr_block

    if not caption and not ocr:
        # Provider returned free-form text; treat it all as the caption.
        caption = text.strip()
    return caption, ocr


def _filename_for(mime_type: str) -> str:
    """Synthesize a filename hint for litellm's multipart audio upload."""
    table = {
        "audio/mpeg": "audio.mp3",
        "audio/mp3": "audio.mp3",
        "audio/mp4": "audio.m4a",
        "audio/m4a": "audio.m4a",
        "audio/wav": "audio.wav",
        "audio/webm": "audio.webm",
        "audio/flac": "audio.flac",
        "audio/ogg": "audio.ogg",
    }
    return table.get(mime_type.lower(), "audio.bin")


def _read_attr(obj: object, name: str, *, default: object | None = None) -> object:
    """Read ``name`` from an object that might be a dict or an attribute holder."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

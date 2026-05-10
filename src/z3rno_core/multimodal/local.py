"""LocalMultimodalProvider — on-device CLIP + Whisper (Phase B.2.1).

The default :class:`LiteLLMMultimodalProvider` ships every byte of every
image and every second of every audio file to an external LLM. For
operators with privacy-sensitive workloads (regulated industries,
air-gapped deployments) that's a non-starter. This provider runs both
the vision and audio pipelines locally:

* **Image**: ``sentence-transformers`` CLIP (``clip-ViT-B-32``) for
  zero-shot label scoring against a fixed candidate vocabulary. The
  top-N labels become ``detected_objects``; the highest-scoring label
  is templated into a caption. CLIP is not a captioner — there is no
  free-form description — but the structured labels are reliable and
  good enough to surface a Memo's content for retrieval.
* **Audio**: ``openai-whisper`` (the local PyTorch model, not the
  hosted API). Transcribes bytes by writing them to a temp file and
  invoking ``model.transcribe`` on a worker thread so the asyncio loop
  stays responsive.

Heavy weight + heavy deps. Both libraries are gated behind the
``multimodal-local`` extra in ``pyproject.toml`` so the default install
stays lean. Construction is import-safe — model weights are loaded
lazily on the first call so unit tests that never touch images or audio
don't pay the load cost.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

from z3rno_core.multimodal.base import (
    AudioTranscript,
    ImageDescription,
    MultimodalProvider,
    MultimodalProviderError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# A small, deliberately generic candidate vocabulary. Operators with
# domain-specific needs (medical imagery, legal docs) should subclass
# and override ``label_vocabulary``.
_DEFAULT_VOCAB: tuple[str, ...] = (
    "a photo of a person",
    "a photo of a group of people",
    "a photo of a document",
    "a screenshot of software",
    "a chart or graph",
    "a diagram or illustration",
    "a photo of a building",
    "a photo of a landscape",
    "a photo of an animal",
    "a photo of a vehicle",
    "a photo of food",
    "a photo of a product",
    "a logo or icon",
    "a photo of text on a sign",
    "a piece of artwork",
)


_DEFAULT_TEMP_SUFFIX_BY_MIME: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
}


class LocalMultimodalProvider(MultimodalProvider):
    """On-device CLIP + Whisper.

    Vision and audio models are loaded lazily on the first
    :meth:`describe_image` / :meth:`transcribe_audio` call. Repeated
    calls reuse the cached models — load cost is paid once per worker
    process.
    """

    def __init__(
        self,
        *,
        vision_model: str = "clip-ViT-B-32",
        audio_model: str = "base",
        label_vocabulary: tuple[str, ...] | None = None,
        max_labels: int = 5,
        device: str | None = None,
    ) -> None:
        if not vision_model:
            raise ValueError("vision_model is required")
        if not audio_model:
            raise ValueError("audio_model is required")
        if max_labels < 1:
            raise ValueError("max_labels must be >= 1")
        self._vision_model_name = vision_model
        self._audio_model_name = audio_model
        self._vocab = label_vocabulary or _DEFAULT_VOCAB
        self._max_labels = max_labels
        self._device = device
        self._clip_model: Any = None
        self._vocab_embeddings: Any = None
        self._whisper_model: Any = None

    @property
    def vision_model(self) -> str:
        return f"local/clip:{self._vision_model_name}"

    @property
    def audio_model(self) -> str:
        return f"local/whisper:{self._audio_model_name}"

    # ---- vision ----------------------------------------------------------

    async def describe_image(
        self,
        content: bytes,
        *,
        mime_type: str,
        prompt: str | None = None,
    ) -> ImageDescription:
        del prompt  # CLIP is zero-shot scoring, no free-form prompt
        if not content:
            return ImageDescription(model=self.vision_model)

        try:
            return await asyncio.to_thread(self._describe_sync, content, mime_type)
        except MultimodalProviderError:
            raise
        except Exception as exc:
            raise MultimodalProviderError(f"local CLIP failed: {exc}") from exc

    def _describe_sync(self, content: bytes, mime_type: str) -> ImageDescription:
        del mime_type  # PIL infers the format from the bytes
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise MultimodalProviderError(
                "Pillow is required for LocalMultimodalProvider; "
                "install z3rno-core[multimodal-local]"
            ) from exc

        self._ensure_clip_loaded()
        assert self._clip_model is not None
        assert self._vocab_embeddings is not None

        try:
            image = Image.open(io.BytesIO(content))
            image.load()  # surface decode errors here, not deep in encode
        except Exception as exc:
            raise MultimodalProviderError(f"failed to decode image: {exc}") from exc

        width, height = image.size

        from sentence_transformers import util  # noqa: PLC0415

        img_emb = self._clip_model.encode(
            [image], convert_to_tensor=True, show_progress_bar=False
        )
        scores = util.cos_sim(img_emb, self._vocab_embeddings)[0]
        # Top-K by score
        scored = sorted(
            zip(self._vocab, scores.tolist(), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top = scored[: self._max_labels]
        objects = tuple(_strip_template(label) for label, _ in top)
        caption = top[0][0] if top else ""

        return ImageDescription(
            caption=caption,
            ocr_text="",  # CLIP does not OCR
            detected_objects=objects,
            width=width if width >= 1 else None,
            height=height if height >= 1 else None,
            model=self.vision_model,
        )

    def _ensure_clip_loaded(self) -> None:
        if self._clip_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise MultimodalProviderError(
                "sentence-transformers is required for LocalMultimodalProvider; "
                "install z3rno-core[multimodal-local]"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        model = SentenceTransformer(self._vision_model_name, **kwargs)
        vocab_emb = model.encode(
            list(self._vocab),
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        self._clip_model = model
        self._vocab_embeddings = vocab_emb

    # ---- audio -----------------------------------------------------------

    async def transcribe_audio(
        self,
        content: bytes,
        *,
        mime_type: str,
        language: str | None = None,
    ) -> AudioTranscript:
        if not content:
            return AudioTranscript(model=self.audio_model)
        try:
            return await asyncio.to_thread(
                self._transcribe_sync, content, mime_type, language
            )
        except MultimodalProviderError:
            raise
        except Exception as exc:
            raise MultimodalProviderError(f"local Whisper failed: {exc}") from exc

    def _transcribe_sync(
        self, content: bytes, mime_type: str, language: str | None
    ) -> AudioTranscript:
        self._ensure_whisper_loaded()
        assert self._whisper_model is not None

        suffix = _DEFAULT_TEMP_SUFFIX_BY_MIME.get(mime_type.lower(), ".bin")
        # Whisper wants a filesystem path because it shells out to ffmpeg.
        # Use delete=False so the model can read the file after we close it,
        # then unlink in finally.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(content)
            tmp_path = fh.name
        try:
            kwargs: dict[str, Any] = {"fp16": False}
            if language:
                kwargs["language"] = language
            result: dict[str, Any] = self._whisper_model.transcribe(tmp_path, **kwargs)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("local_multimodal.tmp_unlink_failed", extra={"path": tmp_path})

        text_value = result.get("text", "")
        detected_lang = result.get("language", "") or (language or "")
        # Whisper does not return duration; segments contain end timestamps.
        duration: float | None = None
        segments = result.get("segments")
        if isinstance(segments, list) and segments:
            last = segments[-1]
            end = last.get("end") if isinstance(last, dict) else None
            if isinstance(end, int | float):
                duration = float(end)

        return AudioTranscript(
            text=str(text_value or "").strip(),
            language=str(detected_lang or ""),
            duration_seconds=duration,
            model=self.audio_model,
        )

    def _ensure_whisper_loaded(self) -> None:
        if self._whisper_model is not None:
            return
        try:
            import whisper  # noqa: PLC0415
        except ImportError as exc:
            raise MultimodalProviderError(
                "openai-whisper is required for LocalMultimodalProvider; "
                "install z3rno-core[multimodal-local]"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device
        self._whisper_model = whisper.load_model(self._audio_model_name, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_template(label: str) -> str:
    """Strip the ``a photo of`` / ``a screenshot of`` template prefix."""
    lowered = label.lower()
    for prefix in (
        "a photo of a ",
        "a photo of an ",
        "a photo of ",
        "a screenshot of ",
        "a piece of ",
    ):
        if lowered.startswith(prefix):
            return label[len(prefix) :]
    return label

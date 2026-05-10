"""Unit tests for ImageLoader + AudioLoader (Phase B.2)."""

from __future__ import annotations

import asyncio

import pytest

from z3rno_core.loaders import AudioLoader, ImageLoader, register_multimodal_loaders
from z3rno_core.loaders.base import LoaderInputError
from z3rno_core.loaders.registry import LoaderRegistry, sniff_mime_type
from z3rno_core.multimodal import (
    AudioTranscript,
    ImageDescription,
    MultimodalProviderError,
    StubMultimodalProvider,
)


def _stub_image(caption: str = "", ocr: str = "") -> StubMultimodalProvider:
    return StubMultimodalProvider(
        on_describe=lambda c, m, p: ImageDescription(caption=caption, ocr_text=ocr),
    )


def _stub_audio(text: str = "", lang: str = "", dur: float | None = None) -> StubMultimodalProvider:
    return StubMultimodalProvider(
        on_transcribe=lambda c, m, lng: AudioTranscript(
            text=text, language=lang, duration_seconds=dur
        ),
    )


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------


class TestImageLoader:
    def test_happy_path(self) -> None:
        loader = ImageLoader(_stub_image(caption="a cat", ocr="WALK"))
        result = asyncio.run(
            loader.load(b"\x89PNG\r\n\x1a\nfake", filename="cat.png", mime_type="image/png")
        )
        assert "a cat" in result.text
        assert "WALK" in result.text
        assert result.metadata["caption"] == "a cat"
        assert result.metadata["ocr_extracted"] is True

    def test_empty_short_circuits_provider(self) -> None:
        called = {"hit": False}

        class CountingProvider(StubMultimodalProvider):
            async def describe_image(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                called["hit"] = True
                return ImageDescription()

        loader = ImageLoader(CountingProvider())
        result = asyncio.run(loader.load(b""))
        assert result.text == ""
        assert called["hit"] is False

    def test_oversize_rejected(self) -> None:
        loader = ImageLoader(_stub_image(), max_bytes=10)
        with pytest.raises(LoaderInputError, match="exceeds max_bytes"):
            asyncio.run(loader.load(b"x" * 100))

    def test_provider_error_translated(self) -> None:
        class FailingProvider(StubMultimodalProvider):
            async def describe_image(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise MultimodalProviderError("provider down")

        loader = ImageLoader(FailingProvider())
        with pytest.raises(LoaderInputError, match="vision provider failed"):
            asyncio.run(loader.load(b"img-bytes", mime_type="image/png"))

    def test_no_caption_falls_back(self) -> None:
        loader = ImageLoader(_stub_image())
        result = asyncio.run(loader.load(b"img-bytes", mime_type="image/png"))
        assert "no caption produced" in result.text


# ---------------------------------------------------------------------------
# AudioLoader
# ---------------------------------------------------------------------------


class TestAudioLoader:
    def test_happy_path(self) -> None:
        loader = AudioLoader(_stub_audio(text="hi", lang="en", dur=2.5))
        result = asyncio.run(loader.load(b"ID3audio", filename="x.mp3", mime_type="audio/mpeg"))
        assert result.text == "hi"
        assert result.metadata["language"] == "en"
        assert result.metadata["duration_seconds"] == 2.5

    def test_empty_short_circuits(self) -> None:
        loader = AudioLoader(_stub_audio())
        result = asyncio.run(loader.load(b""))
        assert result.text == ""

    def test_oversize_rejected(self) -> None:
        loader = AudioLoader(_stub_audio(), max_bytes=10)
        with pytest.raises(LoaderInputError, match="exceeds max_bytes"):
            asyncio.run(loader.load(b"x" * 100))

    def test_provider_error_translated(self) -> None:
        class FailingProvider(StubMultimodalProvider):
            async def transcribe_audio(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise MultimodalProviderError("provider down")

        loader = AudioLoader(FailingProvider())
        with pytest.raises(LoaderInputError, match="audio provider failed"):
            asyncio.run(loader.load(b"audio-bytes", mime_type="audio/mpeg"))

    def test_no_transcript_falls_back(self) -> None:
        loader = AudioLoader(_stub_audio())
        result = asyncio.run(loader.load(b"audio-bytes", mime_type="audio/mpeg"))
        assert "no transcript" in result.text


# ---------------------------------------------------------------------------
# Magic-byte sniffing for multimodal formats
# ---------------------------------------------------------------------------


class TestMultimodalSniffing:
    @pytest.mark.parametrize(
        ("prefix", "expected"),
        [
            (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
            (b"\x89PNG\r\n\x1a\n_extra_", "image/png"),
            (b"GIF89a___", "image/gif"),
            (b"GIF87a___", "image/gif"),
            (b"RIFF____WEBP____", "image/webp"),
            (b"ID3\x04\x00\x00\x00", "audio/mpeg"),
            (b"\xff\xfb\x90\x44", "audio/mpeg"),
            (b"RIFF____WAVE____", "audio/wav"),
            (b"fLaC___", "audio/flac"),
            (b"OggS___", "audio/ogg"),
        ],
    )
    def test_sniffs(self, prefix: bytes, expected: str) -> None:
        assert sniff_mime_type(prefix) == expected


# ---------------------------------------------------------------------------
# register_multimodal_loaders
# ---------------------------------------------------------------------------


class TestRegisterMultimodalLoaders:
    def test_registers_both(self) -> None:
        reg = LoaderRegistry()
        img = ImageLoader(_stub_image())
        aud = AudioLoader(_stub_audio())
        register_multimodal_loaders(reg, image_loader=img, audio_loader=aud)
        assert reg.get_loader(b"x", mime_type="image/png") is img
        assert reg.get_loader(b"x", mime_type="audio/mpeg") is aud

    def test_registers_image_only(self) -> None:
        reg = LoaderRegistry()
        img = ImageLoader(_stub_image())
        register_multimodal_loaders(reg, image_loader=img)
        assert reg.get_loader(b"x", mime_type="image/png") is img
        # No audio loader → falls back; need a fallback registered or it raises.
        # Just confirm the audio MIME isn't routed to the image loader.
        assert "image/png" in reg.known_mime_types
        assert "audio/mpeg" not in reg.known_mime_types

    def test_no_loaders_is_noop(self) -> None:
        reg = LoaderRegistry()
        register_multimodal_loaders(reg)
        assert reg.known_mime_types == []

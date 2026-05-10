"""Unit tests for z3rno_core.multimodal (Phase B.2)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from z3rno_core.multimodal import (
    AudioTranscript,
    ImageDescription,
    LiteLLMMultimodalProvider,
    MultimodalProvider,
    MultimodalProviderError,
    MultimodalRateLimitError,
    MultimodalTimeoutError,
    StubMultimodalProvider,
    get_multimodal_provider,
)
from z3rno_core.multimodal.litellm import _filename_for, _parse_vision_text

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestImageDescription:
    def test_text_for_memo_with_caption_only(self) -> None:
        d = ImageDescription(caption="a cat")
        assert d.text_for_memo == "a cat"

    def test_text_for_memo_with_caption_and_ocr(self) -> None:
        d = ImageDescription(caption="a sign", ocr_text="WALK")
        assert d.text_for_memo == "a sign\n\nOCR:\nWALK"

    def test_text_for_memo_empty(self) -> None:
        assert ImageDescription().text_for_memo == ""

    def test_frozen(self) -> None:
        d = ImageDescription()
        with pytest.raises(Exception):  # noqa: B017, PT011
            d.caption = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_stub(self) -> None:
        p = get_multimodal_provider(provider="stub")
        assert isinstance(p, StubMultimodalProvider)

    def test_litellm(self) -> None:
        p = get_multimodal_provider(provider="litellm", api_key="sk-fake")
        assert isinstance(p, LiteLLMMultimodalProvider)

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown multimodal"):
            get_multimodal_provider(provider="unknown")

    def test_litellm_requires_models(self) -> None:
        with pytest.raises(ValueError, match="vision_model"):
            LiteLLMMultimodalProvider(vision_model="", audio_model="whisper-1")
        with pytest.raises(ValueError, match="audio_model"):
            LiteLLMMultimodalProvider(vision_model="x", audio_model="")


# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class TestStubProvider:
    def test_default_describe(self) -> None:
        p = StubMultimodalProvider()
        out = asyncio.run(p.describe_image(b"x", mime_type="image/png"))
        assert isinstance(out, ImageDescription)

    def test_custom_describe(self) -> None:
        p = StubMultimodalProvider(on_describe=lambda c, m, prompt: ImageDescription(caption="hi"))
        out = asyncio.run(p.describe_image(b"x", mime_type="image/png"))
        assert out.caption == "hi"

    def test_default_transcribe(self) -> None:
        p = StubMultimodalProvider()
        out = asyncio.run(p.transcribe_audio(b"x", mime_type="audio/mpeg"))
        assert isinstance(out, AudioTranscript)

    def test_is_multimodal_provider(self) -> None:
        assert issubclass(StubMultimodalProvider, MultimodalProvider)


# ---------------------------------------------------------------------------
# LiteLLM provider helpers
# ---------------------------------------------------------------------------


class TestParseVisionText:
    def test_caption_and_ocr(self) -> None:
        cap, ocr = _parse_vision_text("Caption: a cat\nOCR: hello")
        assert cap == "a cat"
        assert ocr == "hello"

    def test_caption_only(self) -> None:
        cap, ocr = _parse_vision_text("Caption: a dog")
        assert cap == "a dog"
        assert ocr == ""

    def test_ocr_none_treated_as_empty(self) -> None:
        _cap, ocr = _parse_vision_text("Caption: x\nOCR: none")
        assert ocr == ""

    def test_freeform_falls_back_to_caption(self) -> None:
        cap, ocr = _parse_vision_text("a free-form description")
        assert cap == "a free-form description"
        assert ocr == ""


class TestFilenameFor:
    @pytest.mark.parametrize(
        ("mime", "expected"),
        [
            ("audio/mpeg", "audio.mp3"),
            ("audio/mp3", "audio.mp3"),
            ("audio/wav", "audio.wav"),
            ("audio/m4a", "audio.m4a"),
            ("audio/webm", "audio.webm"),
            ("audio/flac", "audio.flac"),
            ("audio/ogg", "audio.ogg"),
            ("audio/bogus", "audio.bin"),
        ],
    )
    def test_known_and_unknown(self, mime: str, expected: str) -> None:
        assert _filename_for(mime) == expected


# ---------------------------------------------------------------------------
# LiteLLM provider — resilience paths via mocked acompletion
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class TestLiteLLMVision:
    def test_describe_image_round_trip(self) -> None:
        p = LiteLLMMultimodalProvider(api_key="sk-fake")

        async def fake_acompletion(**kwargs: object) -> _FakeCompletionResponse:
            return _FakeCompletionResponse("Caption: a tree\nOCR: none")

        with patch("z3rno_core.multimodal.litellm.litellm.acompletion", new=fake_acompletion):
            out = asyncio.run(p.describe_image(b"png-bytes", mime_type="image/png"))
        assert out.caption == "a tree"
        assert out.ocr_text == ""
        assert out.model == p.vision_model

    def test_describe_empty_short_circuits(self) -> None:
        p = LiteLLMMultimodalProvider()
        out = asyncio.run(p.describe_image(b"", mime_type="image/png"))
        assert out.caption == ""
        assert out.model == p.vision_model

    def test_describe_timeout_maps_to_timeout_error(self) -> None:
        p = LiteLLMMultimodalProvider(timeout_seconds=0.01)

        async def slow(**kwargs: object) -> _FakeCompletionResponse:
            await asyncio.sleep(0.5)
            return _FakeCompletionResponse("x")

        with (
            patch("z3rno_core.multimodal.litellm.litellm.acompletion", new=slow),
            pytest.raises(MultimodalTimeoutError),
        ):
            asyncio.run(p.describe_image(b"x", mime_type="image/png"))

    def test_describe_rate_limit_class_mapped(self) -> None:
        p = LiteLLMMultimodalProvider()

        class FakeRateLimitError(Exception):
            pass

        async def fail(**kwargs: object) -> _FakeCompletionResponse:
            raise FakeRateLimitError("slow down")

        with (
            patch("z3rno_core.multimodal.litellm.litellm.acompletion", new=fail),
            pytest.raises(MultimodalRateLimitError),
        ):
            asyncio.run(p.describe_image(b"x", mime_type="image/png"))

    def test_describe_generic_exception_mapped(self) -> None:
        p = LiteLLMMultimodalProvider()

        async def fail(**kwargs: object) -> _FakeCompletionResponse:
            raise RuntimeError("boom")

        with (
            patch("z3rno_core.multimodal.litellm.litellm.acompletion", new=fail),
            pytest.raises(MultimodalProviderError),
        ):
            asyncio.run(p.describe_image(b"x", mime_type="image/png"))


class TestLiteLLMAudio:
    def test_transcribe_round_trip(self) -> None:
        p = LiteLLMMultimodalProvider(api_key="sk-fake")
        fake_response = MagicMock()
        fake_response.text = "hello world"
        fake_response.language = "en"
        fake_response.duration = 4.2

        async def fake_atranscription(**kwargs: object) -> object:
            return fake_response

        with patch(
            "z3rno_core.multimodal.litellm.litellm.atranscription",
            new=fake_atranscription,
        ):
            out = asyncio.run(p.transcribe_audio(b"mp3-bytes", mime_type="audio/mpeg"))
        assert out.text == "hello world"
        assert out.language == "en"
        assert out.duration_seconds == 4.2

    def test_transcribe_empty_short_circuits(self) -> None:
        p = LiteLLMMultimodalProvider()
        out = asyncio.run(p.transcribe_audio(b"", mime_type="audio/mpeg"))
        assert out.text == ""

    def test_transcribe_dict_response(self) -> None:
        p = LiteLLMMultimodalProvider()

        async def dict_response(**kwargs: object) -> dict[str, object]:
            return {"text": "spoken", "language": "es"}

        with patch("z3rno_core.multimodal.litellm.litellm.atranscription", new=dict_response):
            out = asyncio.run(p.transcribe_audio(b"x", mime_type="audio/mpeg"))
        assert out.text == "spoken"
        assert out.language == "es"
        assert out.duration_seconds is None

    def test_transcribe_with_language_hint_passed(self) -> None:
        p = LiteLLMMultimodalProvider()
        captured: dict[str, object] = {}

        async def cap(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"text": "x"}

        with patch("z3rno_core.multimodal.litellm.litellm.atranscription", new=cap):
            asyncio.run(p.transcribe_audio(b"x", mime_type="audio/wav", language="en"))
        assert captured.get("language") == "en"

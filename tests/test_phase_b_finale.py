"""Phase B finale (B.2.1 + B.2.2) — unit tests.

Covers:
  * Presigned upload flow on :class:`S3StorageBackend` and the
    :class:`PresignedUrlNotSupportedError` raised by the local backend.
  * Pipeline ``s3_uri`` materialization + validate_input dispatch.
  * URL loader Playwright auto-fallback gate (threshold + opt-in).
  * LocalMultimodalProvider lazy-load error paths and label templating
    helper.

No DB, no AWS, no network, no torch. aioboto3 + Playwright + heavy ML
deps are mocked at the seam where the code imports them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.ingest.pipeline import _REQUIRED_FIELD_BY_KIND, _validate_input
from z3rno_core.ingest.schemas import IngestInput
from z3rno_core.loaders.url import _extracted_text_chars, fetch_url
from z3rno_core.multimodal import LocalMultimodalProvider, get_multimodal_provider
from z3rno_core.multimodal.base import MultimodalProviderError
from z3rno_core.multimodal.local import _strip_template
from z3rno_core.storage import (
    LocalStorageBackend,
    PresignedUrlNotSupportedError,
    S3StorageBackend,
)

# ---------------------------------------------------------------------------
# Presigned upload — local raises, S3 issues a URL
# ---------------------------------------------------------------------------


class TestPresignedUpload:
    def test_local_raises_not_supported(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(tmp_path)
        with pytest.raises(PresignedUrlNotSupportedError):
            asyncio.run(
                backend.presigned_put_url(
                    org_id=uuid4(),
                    content_type="text/plain",
                )
            )

    def test_s3_returns_presigned_url(self) -> None:
        client = MagicMock()
        client.generate_presigned_url = AsyncMock(
            return_value="https://example-bucket.s3.amazonaws.com/z3rno/x?signed=1"
        )
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client)
        cm.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.client = MagicMock(return_value=cm)

        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            backend = S3StorageBackend(bucket="example-bucket", prefix="z3rno")
            org = uuid4()
            result = asyncio.run(
                backend.presigned_put_url(
                    org_id=org,
                    content_type="application/pdf",
                    filename="report.pdf",
                    ttl_seconds=600,
                )
            )

        assert result.method == "PUT"
        assert result.upload_url.startswith("https://")
        assert result.source_uri.startswith("s3://example-bucket/z3rno/")
        assert str(org) in result.source_uri
        assert result.source_uri.endswith(".pdf")
        assert result.content_type == "application/pdf"

    def test_s3_clamps_ttl_floor(self) -> None:
        client = MagicMock()
        client.generate_presigned_url = AsyncMock(return_value="https://x")
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=client)
        cm.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.client = MagicMock(return_value=cm)

        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            backend = S3StorageBackend(bucket="b")
            asyncio.run(
                backend.presigned_put_url(
                    org_id=uuid4(),
                    content_type="text/plain",
                    ttl_seconds=10,
                )
            )

        kwargs = client.generate_presigned_url.call_args.kwargs
        assert kwargs["ExpiresIn"] == 60  # clamped from 10


# ---------------------------------------------------------------------------
# Pipeline: validate_input dispatch table + s3_uri kind
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_required_field_table_covers_all_kinds(self) -> None:
        assert set(_REQUIRED_FIELD_BY_KIND) == {"text", "url", "file", "s3_uri"}

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown ingest kind"):
            _validate_input(IngestInput(kind="weird"))  # type: ignore[arg-type]

    def test_s3_uri_requires_source_uri(self) -> None:
        with pytest.raises(ValueError, match="source_uri"):
            _validate_input(IngestInput(kind="s3_uri"))

    def test_s3_uri_rejects_extras(self) -> None:
        with pytest.raises(ValueError, match="must not set"):
            _validate_input(
                IngestInput(
                    kind="s3_uri",
                    source_uri="s3://b/key",
                    text="extra",
                )
            )

    def test_s3_uri_accepts_valid(self) -> None:
        _validate_input(IngestInput(kind="s3_uri", source_uri="s3://b/key"))

    def test_text_still_works(self) -> None:
        _validate_input(IngestInput(kind="text", text="hello"))

    def test_file_still_works(self) -> None:
        _validate_input(IngestInput(kind="file", content=b"x"))


# ---------------------------------------------------------------------------
# URL loader: Playwright auto-fallback
# ---------------------------------------------------------------------------


_THIN_HTML = b"<html><body><div></div></body></html>"
_RICH_HTML = b"<html><body><article>" + b"x" * 500 + b"</article></body></html>"
_RENDERED = "<html><body><article>" + ("y" * 500) + "</article></body></html>"


class TestPlaywrightAutoFallback:
    def test_extracted_text_chars_threshold(self) -> None:
        assert _extracted_text_chars(_THIN_HTML) == 0
        assert _extracted_text_chars(_RICH_HTML) >= 500

    def test_disabled_by_default_no_render(self) -> None:
        # Pre-Phase-B.2.2 callers don't pass playwright_* kwargs and must
        # never hit the Playwright path even on thin HTML.
        async def _go() -> None:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = _THIN_HTML
            mock_response.headers = {"content-type": "text/html"}
            mock_response.url = "https://example.com"
            mock_response.reason_phrase = ""

            client = MagicMock()
            client.get = AsyncMock(return_value=mock_response)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=client)
            cm.__aexit__ = AsyncMock(return_value=None)

            with (
                patch("z3rno_core.loaders.url.httpx.AsyncClient", return_value=cm),
                patch(
                    "z3rno_core.loaders.url.render_with_playwright",
                    new=AsyncMock(return_value=_RENDERED),
                ) as render_mock,
            ):
                result = await fetch_url("https://example.com")
                assert result.content == _THIN_HTML
                render_mock.assert_not_called()

        asyncio.run(_go())

    def test_falls_back_when_thin_and_enabled(self) -> None:
        async def _go() -> None:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = _THIN_HTML
            mock_response.headers = {"content-type": "text/html"}
            mock_response.url = "https://example.com/"
            mock_response.reason_phrase = ""

            client = MagicMock()
            client.get = AsyncMock(return_value=mock_response)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=client)
            cm.__aexit__ = AsyncMock(return_value=None)

            with (
                patch("z3rno_core.loaders.url.httpx.AsyncClient", return_value=cm),
                patch(
                    "z3rno_core.loaders.url.render_with_playwright",
                    new=AsyncMock(return_value=_RENDERED),
                ) as render_mock,
            ):
                result = await fetch_url(
                    "https://example.com",
                    playwright_enabled=True,
                    playwright_min_chars=200,
                )
                assert result.content == _RENDERED.encode("utf-8")
                assert result.content_type == "text/html"
                render_mock.assert_called_once()

        asyncio.run(_go())

    def test_skips_when_static_meets_threshold(self) -> None:
        async def _go() -> None:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = _RICH_HTML
            mock_response.headers = {"content-type": "text/html"}
            mock_response.url = "https://example.com/"
            mock_response.reason_phrase = ""

            client = MagicMock()
            client.get = AsyncMock(return_value=mock_response)
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=client)
            cm.__aexit__ = AsyncMock(return_value=None)

            with (
                patch("z3rno_core.loaders.url.httpx.AsyncClient", return_value=cm),
                patch(
                    "z3rno_core.loaders.url.render_with_playwright",
                    new=AsyncMock(return_value=_RENDERED),
                ) as render_mock,
            ):
                result = await fetch_url(
                    "https://example.com",
                    playwright_enabled=True,
                    playwright_min_chars=200,
                )
                assert result.content == _RICH_HTML
                render_mock.assert_not_called()

        asyncio.run(_go())


# ---------------------------------------------------------------------------
# LocalMultimodalProvider — lazy-load error paths
# ---------------------------------------------------------------------------


class TestLocalMultimodalProvider:
    def test_factory_dispatches_to_local(self) -> None:
        provider = get_multimodal_provider(provider="local")
        assert isinstance(provider, LocalMultimodalProvider)
        assert provider.vision_model.startswith("local/clip:")
        assert provider.audio_model.startswith("local/whisper:")

    def test_construction_is_io_free(self) -> None:
        # Construction must NOT touch torch / sentence_transformers / whisper.
        # If it did, the test process would attempt to import those heavy
        # libs and fail on a lean dev install.
        provider = LocalMultimodalProvider()
        # Models stay None until first call.
        assert provider._clip_model is None
        assert provider._whisper_model is None

    def test_empty_image_returns_empty_description(self) -> None:
        provider = LocalMultimodalProvider()
        result = asyncio.run(provider.describe_image(b"", mime_type="image/jpeg"))
        assert result.caption == ""
        assert result.detected_objects == ()

    def test_empty_audio_returns_empty_transcript(self) -> None:
        provider = LocalMultimodalProvider()
        result = asyncio.run(provider.transcribe_audio(b"", mime_type="audio/mpeg"))
        assert result.text == ""

    def test_missing_pillow_yields_clear_error(self) -> None:
        provider = LocalMultimodalProvider()

        # Force the lazy PIL import to raise ImportError.
        with (
            patch.object(provider, "_describe_sync", side_effect=ImportError("PIL")),
            pytest.raises(MultimodalProviderError, match="local CLIP failed"),
        ):
            asyncio.run(provider.describe_image(b"x", mime_type="image/jpeg"))

    def test_label_template_stripping(self) -> None:
        assert _strip_template("a photo of a person") == "person"
        assert _strip_template("a photo of an animal") == "animal"
        assert _strip_template("a screenshot of software") == "software"
        assert _strip_template("a piece of artwork") == "artwork"
        # Nothing to strip -> returned as-is.
        assert _strip_template("custom label") == "custom label"

    def test_max_labels_validated(self) -> None:
        with pytest.raises(ValueError, match="max_labels"):
            LocalMultimodalProvider(max_labels=0)

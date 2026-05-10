"""Unit tests for z3rno_core.storage.s3 (Phase B.2).

aioboto3 is mocked; no AWS calls are made.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from z3rno_core.storage import (
    S3StorageBackend,
    StorageBackend,
    StorageError,
    StorageNotFoundError,
)
from z3rno_core.storage.s3 import _resolve_extension


class TestExtensionResolution:
    def test_filename_extension_wins(self) -> None:
        assert _resolve_extension("application/octet-stream", "x.md") == ".md"

    def test_mime_used_when_filename_has_none(self) -> None:
        assert _resolve_extension("image/jpeg", None) == ".jpg"

    def test_mime_with_charset(self) -> None:
        assert _resolve_extension("text/plain; charset=utf-8", None) == ".txt"

    def test_unknown_falls_back_to_bin(self) -> None:
        assert _resolve_extension("application/x-novel", None) == ".bin"


class TestS3StorageBackendConstruction:
    def test_construction_is_io_free(self) -> None:
        b = S3StorageBackend(bucket="my-bucket", region="us-west-2", prefix="z3rno")
        assert isinstance(b, StorageBackend)
        assert b.name == "s3"
        assert b.bucket == "my-bucket"

    def test_empty_bucket_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty bucket"):
            S3StorageBackend(bucket="")


class TestS3Parse:
    def test_rejects_non_s3_scheme(self) -> None:
        b = S3StorageBackend(bucket="b")
        with pytest.raises(StorageError, match="cannot resolve"):
            b._parse("file:///etc/passwd")

    def test_rejects_cross_bucket(self) -> None:
        b = S3StorageBackend(bucket="b")
        with pytest.raises(StorageError, match="cross-bucket"):
            b._parse("s3://other/key")

    def test_rejects_outside_prefix(self) -> None:
        b = S3StorageBackend(bucket="b", prefix="z3rno")
        with pytest.raises(StorageError, match="outside prefix"):
            b._parse("s3://b/random/key")

    def test_accepts_valid_uri(self) -> None:
        b = S3StorageBackend(bucket="b", prefix="z3rno")
        bucket, key = b._parse("s3://b/z3rno/foo/bar.pdf")
        assert bucket == "b"
        assert key == "z3rno/foo/bar.pdf"

    def test_empty_prefix_disables_sandbox(self) -> None:
        b = S3StorageBackend(bucket="b", prefix="")
        _bucket, key = b._parse("s3://b/anything/here.pdf")
        assert key == "anything/here.pdf"


# ---------------------------------------------------------------------------
# put / get / delete with mocked aioboto3 client
# ---------------------------------------------------------------------------


def _mock_s3_client(get_response: dict[str, object] | None = None) -> tuple[MagicMock, MagicMock]:
    """Return (session_mock, client_mock) where session_mock.client(...) returns
    an async context manager wrapping client_mock."""
    client = MagicMock()
    client.put_object = AsyncMock(return_value={"ETag": '"abc"'})
    client.delete_object = AsyncMock(return_value={})

    if get_response is not None:
        body_obj = MagicMock()
        body_obj.read = AsyncMock(return_value=b"content-bytes")
        client.get_object = AsyncMock(return_value={"Body": body_obj, **get_response})
    else:
        client.get_object = AsyncMock(return_value={"Body": MagicMock()})

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.client = MagicMock(return_value=cm)
    return session, client


class TestS3StoreReadDelete:
    def test_store_artifact_uri_shape(self) -> None:
        S3StorageBackend(bucket="my-bucket", prefix="z3rno")
        session, _client = _mock_s3_client()
        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            b2 = S3StorageBackend(bucket="my-bucket", prefix="z3rno")
            org = uuid4()
            uri = asyncio.run(
                b2.store_artifact(
                    org_id=org,
                    content=b"hello",
                    content_type="text/plain",
                    filename="note.txt",
                )
            )
        assert uri.startswith("s3://my-bucket/z3rno/")
        assert str(org) in uri
        assert uri.endswith(".txt")

    def test_read_artifact_round_trip(self) -> None:
        S3StorageBackend(bucket="my-bucket", prefix="z3rno")
        session, _client = _mock_s3_client(get_response={})
        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            b2 = S3StorageBackend(bucket="my-bucket", prefix="z3rno")
            data = asyncio.run(b2.read_artifact("s3://my-bucket/z3rno/foo.bin"))
        assert data == b"content-bytes"

    def test_read_missing_translates_to_not_found(self) -> None:
        from botocore.exceptions import ClientError

        S3StorageBackend(bucket="my-bucket", prefix="z3rno")
        session, client = _mock_s3_client()
        client.get_object = AsyncMock(
            side_effect=ClientError(
                error_response={"Error": {"Code": "NoSuchKey"}},
                operation_name="GetObject",
            )
        )
        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            b2 = S3StorageBackend(bucket="my-bucket", prefix="z3rno")
            with pytest.raises(StorageNotFoundError):
                asyncio.run(b2.read_artifact("s3://my-bucket/z3rno/missing.bin"))

    def test_delete_is_best_effort(self) -> None:
        from botocore.exceptions import BotoCoreError

        S3StorageBackend(bucket="my-bucket", prefix="z3rno")
        session, client = _mock_s3_client()
        client.delete_object = AsyncMock(side_effect=BotoCoreError())
        with patch("z3rno_core.storage.s3.aioboto3.Session", return_value=session):
            b2 = S3StorageBackend(bucket="my-bucket", prefix="z3rno")
            # Must NOT raise
            asyncio.run(b2.delete_artifact("s3://my-bucket/z3rno/foo.bin"))

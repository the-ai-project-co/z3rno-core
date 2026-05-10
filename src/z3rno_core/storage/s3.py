"""S3StorageBackend — managed-cloud artifact storage (Phase B.2).

Implements the existing :class:`StorageBackend` interface against AWS
S3 (and any S3-compatible service: MinIO, R2, etc.). Returns
``s3://<bucket>/<key>`` URIs that callers treat as opaque.

Layout::

    s3://<bucket>/<prefix>/<org_id>/<yyyy>/<mm>/<uuid>.<ext>

The prefix lets multiple Z3rno deployments share one bucket without
colliding. Org partitioning matches the local backend so backup,
retention, and forensic queries stay shape-stable across backends.

Phase B.2 uses single-shot ``put_object`` (good up to ~5 GB on S3).
Phase B.2.1 will add multipart upload + pre-signed URLs so large files
can stream from the client directly to S3 without round-tripping
through the API.
"""

from __future__ import annotations

import logging
import mimetypes
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

import aioboto3
from botocore.exceptions import BotoCoreError, ClientError

from z3rno_core.storage.base import (
    PresignedUpload,
    StorageBackend,
    StorageError,
    StorageNotFoundError,
)

logger = logging.getLogger(__name__)


# Mirrors the local backend's tight allowlist.
_FALLBACK_EXT_BY_MIME: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/x-source": ".txt",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
}


class S3StorageBackend(StorageBackend):
    """S3-backed :class:`StorageBackend`.

    Construction is import-safe: the ``aioboto3.Session`` is created
    eagerly but no AWS API calls are made until the first
    ``store_artifact`` / ``read_artifact`` / ``delete_artifact``.
    """

    def __init__(
        self,
        *,
        bucket: str,
        region: str = "us-east-1",
        prefix: str = "z3rno",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3StorageBackend requires a non-empty bucket")
        self._bucket = bucket
        self._region = region
        self._prefix = prefix.strip("/")
        self._endpoint_url = endpoint_url or None
        self._session = aioboto3.Session(
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
            region_name=region,
        )

    @property
    def name(self) -> str:
        return "s3"

    @property
    def bucket(self) -> str:
        return self._bucket

    # ---- public API ------------------------------------------------------

    async def store_artifact(
        self,
        *,
        org_id: UUID,
        content: bytes,
        content_type: str,
        filename: str | None = None,
    ) -> str:
        ext = _resolve_extension(content_type, filename)
        artifact_id = uuid4()
        now = datetime.now(UTC)
        key_parts: list[str] = []
        if self._prefix:
            key_parts.append(self._prefix)
        key_parts += [
            str(org_id),
            f"{now.year:04d}",
            f"{now.month:02d}",
            f"{artifact_id}{ext}",
        ]
        key = str(PurePosixPath(*key_parts))

        try:
            async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
                await s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=content,
                    ContentType=(content_type or "application/octet-stream")
                    .split(";", 1)[0]
                    .strip(),
                )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"S3 put_object failed: {exc}") from exc

        return f"s3://{self._bucket}/{key}"

    async def read_artifact(self, source_uri: str) -> bytes:
        bucket, key = self._parse(source_uri)
        try:
            async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
                response = await s3.get_object(Bucket=bucket, Key=key)
                body = await response["Body"].read()
                return bytes(body)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise StorageNotFoundError(f"artifact not found: {source_uri}") from exc
            raise StorageError(f"S3 get_object failed: {exc}") from exc
        except BotoCoreError as exc:
            raise StorageError(f"S3 get_object failed: {exc}") from exc

    async def delete_artifact(self, source_uri: str) -> None:
        bucket, key = self._parse(source_uri)
        try:
            async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
                await s3.delete_object(Bucket=bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            # Best-effort delete: an already-missing key is fine.
            logger.warning("s3.delete_failed", extra={"uri": source_uri, "error": str(exc)})

    async def presigned_put_url(
        self,
        *,
        org_id: UUID,
        content_type: str,
        filename: str | None = None,
        ttl_seconds: int = 900,
    ) -> PresignedUpload:
        """Issue a presigned PUT URL for the client to upload directly.

        The key layout matches :meth:`store_artifact` so subsequent
        :meth:`read_artifact` calls work immediately once the client
        completes the upload.
        """
        ext = _resolve_extension(content_type, filename)
        artifact_id = uuid4()
        now = datetime.now(UTC)
        key_parts: list[str] = []
        if self._prefix:
            key_parts.append(self._prefix)
        key_parts += [
            str(org_id),
            f"{now.year:04d}",
            f"{now.month:02d}",
            f"{artifact_id}{ext}",
        ]
        key = str(PurePosixPath(*key_parts))

        normalized_ct = (
            (content_type or "application/octet-stream").split(";", 1)[0].strip()
        )

        try:
            async with self._session.client("s3", endpoint_url=self._endpoint_url) as s3:
                upload_url = await s3.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": self._bucket,
                        "Key": key,
                        "ContentType": normalized_ct,
                    },
                    ExpiresIn=max(60, int(ttl_seconds)),
                    HttpMethod="PUT",
                )
        except (ClientError, BotoCoreError) as exc:
            raise StorageError(f"S3 generate_presigned_url failed: {exc}") from exc

        return PresignedUpload(
            upload_url=str(upload_url),
            source_uri=f"s3://{self._bucket}/{key}",
            expires_at=now + timedelta(seconds=max(60, int(ttl_seconds))),
            content_type=normalized_ct,
            method="PUT",
        )

    # ---- internals -------------------------------------------------------

    def _parse(self, source_uri: str) -> tuple[str, str]:
        parsed = urlparse(source_uri)
        if parsed.scheme != "s3":
            raise StorageError(
                f"S3StorageBackend cannot resolve {parsed.scheme!r} URI: {source_uri}"
            )
        bucket = parsed.netloc
        if not bucket:
            raise StorageError(f"S3 URI missing bucket: {source_uri}")
        if bucket != self._bucket:
            raise StorageError(
                f"refusing cross-bucket access: configured={self._bucket!r}, uri={bucket!r}"
            )
        key = unquote(parsed.path).lstrip("/")
        # Sandbox: refuse keys outside the configured prefix.
        if self._prefix and not key.startswith(f"{self._prefix}/"):
            raise StorageError(f"refusing access outside prefix {self._prefix!r}: key={key!r}")
        return bucket, key


# ---------------------------------------------------------------------------
# Helpers (mirror local backend)
# ---------------------------------------------------------------------------


def _resolve_extension(content_type: str, filename: str | None) -> str:
    """Mirror :func:`z3rno_core.storage.local._resolve_extension`."""
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix.lower()
    base = (content_type or "").split(";", 1)[0].strip().lower()
    if base in _FALLBACK_EXT_BY_MIME:
        return _FALLBACK_EXT_BY_MIME[base]
    guessed = mimetypes.guess_extension(base)
    return guessed.lower() if guessed else ".bin"

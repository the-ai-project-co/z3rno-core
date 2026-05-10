"""Storage backend contract.

The Forge's *retain* stage stamps every Memo with a ``source_uri`` so
downstream provenance lookups can reach back to the original artifact.
The :class:`StorageBackend` interface is the seam where that bytes-on-
disk concern is isolated from the rest of the engine.

Phase B.1 ships :class:`LocalStorageBackend`. Phase B.2 adds
:class:`S3StorageBackend`. Phase B.2.1 adds the
:meth:`StorageBackend.presigned_put_url` method for direct-to-storage
uploads — implemented for S3, raises :class:`PresignedUrlNotSupportedError`
on the local backend (which has no out-of-band upload mechanism).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


class StorageError(Exception):
    """Base exception for storage backends."""


class StorageNotFoundError(StorageError):
    """The requested artifact does not exist."""


class PresignedUrlNotSupportedError(StorageError):
    """The backend cannot issue presigned upload URLs (e.g. local FS)."""


@dataclass(frozen=True)
class PresignedUpload:
    """Output of :meth:`StorageBackend.presigned_put_url`.

    Clients PUT the artifact bytes to ``upload_url`` directly (no
    server-side relay), then call ``POST /v1/ingest/finalize/{job_id}``
    to trigger the worker. The worker reads the artifact from
    ``source_uri`` via the same backend.
    """

    upload_url: str
    source_uri: str
    expires_at: datetime
    content_type: str
    method: str = "PUT"


class StorageBackend(ABC):
    """Abstract artifact-storage interface.

    All operations are async — even when the underlying implementation
    is synchronous (local filesystem) — so callers can mix it with
    async pipelines and async S3 SDKs uniformly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier for logging."""

    @abstractmethod
    async def store_artifact(
        self,
        *,
        org_id: UUID,
        content: bytes,
        content_type: str,
        filename: str | None = None,
    ) -> str:
        """Persist ``content`` and return a canonical ``source_uri``.

        Implementations partition by ``org_id`` so a regulator can
        prove tenant isolation at the storage layer matches RLS at the
        DB layer. The returned URI must be opaque to callers — they
        round-trip it through :meth:`read_artifact` and never parse it.
        """

    @abstractmethod
    async def read_artifact(self, source_uri: str) -> bytes:
        """Fetch the bytes previously stored under ``source_uri``."""

    @abstractmethod
    async def delete_artifact(self, source_uri: str) -> None:
        """Hard-delete the artifact. No-op if it doesn't exist."""

    async def presigned_put_url(
        self,
        *,
        org_id: UUID,
        content_type: str,
        filename: str | None = None,
        ttl_seconds: int = 900,
    ) -> PresignedUpload:
        """Issue a presigned URL the client can PUT directly to.

        Default implementation raises :class:`PresignedUrlNotSupportedError`.
        S3-style backends override.
        """
        raise PresignedUrlNotSupportedError(
            f"{self.name} does not support presigned upload URLs"
        )

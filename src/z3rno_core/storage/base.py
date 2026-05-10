"""Storage backend contract.

The Forge's *retain* stage stamps every Memo with a ``source_uri`` so
downstream provenance lookups can reach back to the original artifact.
The :class:`StorageBackend` interface is the seam where that bytes-on-
disk concern is isolated from the rest of the engine.

Phase B.1 ships :class:`LocalStorageBackend`. Phase B.2 will add an S3
backend with the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID


class StorageError(Exception):
    """Base exception for storage backends."""


class StorageNotFoundError(StorageError):
    """The requested artifact does not exist."""


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

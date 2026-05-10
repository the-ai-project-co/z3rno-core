"""z3rno_core.storage — pluggable artifact storage backends (Phase B.1).

The Forge pipeline accepts raw artifacts (uploaded files, URL content) that
must persist somewhere outside Postgres so downstream provenance lookups
can retrieve the original. Storage backends abstract that destination.

Modules
-------

- ``base``   — :class:`StorageBackend` interface
- ``local``  — local filesystem backend (Phase B.1 default)

Phase B.2 adds ``s3`` for managed-cloud deployments. The choice is wired
through ``STORAGE_BACKEND`` in :class:`z3rno_server.config.Settings`.
"""

from __future__ import annotations

from z3rno_core.storage.base import (
    PresignedUpload,
    PresignedUrlNotSupportedError,
    StorageBackend,
    StorageError,
    StorageNotFoundError,
)
from z3rno_core.storage.local import LocalStorageBackend
from z3rno_core.storage.s3 import S3StorageBackend

__all__ = [
    "LocalStorageBackend",
    "PresignedUpload",
    "PresignedUrlNotSupportedError",
    "S3StorageBackend",
    "StorageBackend",
    "StorageError",
    "StorageNotFoundError",
]

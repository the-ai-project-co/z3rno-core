"""Unit tests for z3rno_core.storage (Phase B.1).

Filesystem-only; no DB. Each test gets its own temp dir.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from z3rno_core.storage import (
    LocalStorageBackend,
    StorageBackend,
    StorageError,
    StorageNotFoundError,
)
from z3rno_core.storage.local import _resolve_extension


@pytest.fixture
def backend(tmp_path: Path) -> LocalStorageBackend:
    return LocalStorageBackend(tmp_path)


class TestExtensionResolution:
    def test_filename_extension_wins(self) -> None:
        assert _resolve_extension("application/octet-stream", "note.md") == ".md"

    def test_mime_used_when_filename_has_none(self) -> None:
        assert _resolve_extension("text/plain", "noext") == ".txt"

    def test_mime_with_charset(self) -> None:
        assert _resolve_extension("text/plain; charset=utf-8", None) == ".txt"

    def test_unknown_falls_back_to_bin(self) -> None:
        assert _resolve_extension("application/x-something-novel", None) == ".bin"


class TestLocalStorageBackend:
    def test_is_storage_backend(self, backend: LocalStorageBackend) -> None:
        assert isinstance(backend, StorageBackend)
        assert backend.name == "local"

    def test_round_trip_pdf(self, backend: LocalStorageBackend) -> None:
        org = uuid4()
        src = b"%PDF-1.7 dummy bytes"
        uri = asyncio.run(
            backend.store_artifact(
                org_id=org, content=src, content_type="application/pdf", filename="d.pdf"
            )
        )
        assert uri.startswith("file://")
        assert str(org) in uri
        assert uri.endswith(".pdf")
        assert asyncio.run(backend.read_artifact(uri)) == src

    def test_extension_inferred_from_mime(self, backend: LocalStorageBackend) -> None:
        org = uuid4()
        uri = asyncio.run(
            backend.store_artifact(
                org_id=org, content=b"hello", content_type="text/plain", filename="noext"
            )
        )
        assert uri.endswith(".txt")

    def test_org_partitioning(self, backend: LocalStorageBackend) -> None:
        a, b = uuid4(), uuid4()
        ua = asyncio.run(backend.store_artifact(org_id=a, content=b"a", content_type="text/plain"))
        ub = asyncio.run(backend.store_artifact(org_id=b, content=b"b", content_type="text/plain"))
        assert str(a) in ua
        assert str(a) not in ub
        assert str(b) in ub

    def test_path_traversal_blocked(self, backend: LocalStorageBackend) -> None:
        with pytest.raises(StorageError):
            asyncio.run(backend.read_artifact("file:///etc/passwd"))

    def test_non_file_uri_rejected(self, backend: LocalStorageBackend) -> None:
        with pytest.raises(StorageError, match="cannot resolve"):
            asyncio.run(backend.read_artifact("s3://bucket/key"))

    def test_missing_artifact_raises(self, backend: LocalStorageBackend) -> None:
        target = backend.root_dir / "no" / "such" / "file.bin"
        with pytest.raises(StorageNotFoundError):
            asyncio.run(backend.read_artifact(target.as_uri()))

    def test_delete_idempotent(self, backend: LocalStorageBackend) -> None:
        org = uuid4()
        uri = asyncio.run(
            backend.store_artifact(org_id=org, content=b"x", content_type="text/plain")
        )
        asyncio.run(backend.delete_artifact(uri))
        with pytest.raises(StorageNotFoundError):
            asyncio.run(backend.read_artifact(uri))
        # second delete must NOT raise
        asyncio.run(backend.delete_artifact(uri))

    def test_atomic_write_no_partial_file(self, backend: LocalStorageBackend) -> None:
        # The .part staging file should never linger after a successful write.
        org = uuid4()
        uri = asyncio.run(
            backend.store_artifact(org_id=org, content=b"x", content_type="text/plain")
        )
        # No .part siblings in the resulting tree
        for p in backend.root_dir.rglob("*.part"):
            raise AssertionError(f"leftover staging file: {p}")
        assert asyncio.run(backend.read_artifact(uri)) == b"x"


class TestExpandUserAndAbsolutePath:
    def test_root_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = LocalStorageBackend(tmp)
            assert backend.root_dir == Path(tmp).resolve()

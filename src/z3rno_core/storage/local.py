"""LocalStorageBackend — filesystem artifact storage (Phase B.1 default).

Layout::

    <root>/<org_id>/<yyyy>/<mm>/<uuid>.<ext>

Org partitioning isn't a security boundary on the local filesystem
(any process with read access can see every tenant's artifacts) but
it does match the logical RLS partition so backups, retention sweeps,
and forensic queries operate on the same shape they will under S3.

``source_uri`` is a ``file://`` URL. Callers treat it as opaque —
:meth:`read_artifact` and :meth:`delete_artifact` are the only ways to
get bytes back.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

from z3rno_core.storage.base import (
    StorageBackend,
    StorageError,
    StorageNotFoundError,
)

logger = logging.getLogger(__name__)


# Default extensions when ``mimetypes`` doesn't know one. Keep tight —
# we'd rather store with a generic ``.bin`` than guess wrong.
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
}


class LocalStorageBackend(StorageBackend):
    """Filesystem-backed :class:`StorageBackend`.

    The root directory is created on first write. All filesystem
    operations are wrapped in :func:`asyncio.to_thread` so the event
    loop stays responsive even when storing or reading large files.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir).expanduser().resolve()

    @property
    def name(self) -> str:
        return "local"

    @property
    def root_dir(self) -> Path:
        return self._root

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
        relative = PurePosixPath(
            str(org_id),
            f"{now.year:04d}",
            f"{now.month:02d}",
            f"{artifact_id}{ext}",
        )
        target = self._root / relative

        await asyncio.to_thread(_write_bytes_atomic, target, content)
        logger.debug(
            "storage.local.stored",
            extra={"path": str(target), "byte_size": len(content)},
        )
        return target.as_uri()

    async def read_artifact(self, source_uri: str) -> bytes:
        path = self._resolve(source_uri)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise StorageNotFoundError(f"artifact not found: {source_uri}") from exc

    async def delete_artifact(self, source_uri: str) -> None:
        path = self._resolve(source_uri)
        await asyncio.to_thread(_unlink_quiet, path)

    # ---- internal --------------------------------------------------------

    def _resolve(self, source_uri: str) -> Path:
        parsed = urlparse(source_uri)
        if parsed.scheme != "file":
            raise StorageError(
                f"LocalStorageBackend cannot resolve {parsed.scheme!r} URI: {source_uri}"
            )
        # urlparse keeps the leading '/' on POSIX paths.
        path = Path(unquote(parsed.path)).resolve()
        # Defense in depth: never read outside the configured root.
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise StorageError(f"refusing to access path outside storage root: {path}") from exc
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_extension(content_type: str, filename: str | None) -> str:
    """Pick a file extension for the artifact.

    Precedence: existing extension on ``filename`` → MIME-derived → ``.bin``.
    """
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix.lower()

    base_mime = (content_type or "").split(";", 1)[0].strip().lower()
    if base_mime in _FALLBACK_EXT_BY_MIME:
        return _FALLBACK_EXT_BY_MIME[base_mime]

    guessed = mimetypes.guess_extension(base_mime)
    return guessed.lower() if guessed else ".bin"


def _write_bytes_atomic(target: Path, content: bytes) -> None:
    """Atomic write: stage to a sibling tempfile, then rename.

    Avoids partially written artifacts on a crash mid-write.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".part")
    staging.write_bytes(content)
    staging.replace(target)


def _unlink_quiet(path: Path) -> None:
    """Best-effort delete — ignore missing file."""
    try:
        path.unlink()
    except FileNotFoundError:
        return

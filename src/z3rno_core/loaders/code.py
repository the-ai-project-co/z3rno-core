"""Code loader — language-aware source-code passthrough.

Detects the source language from the filename extension, decodes the
bytes verbatim (no syntactic transformation), and stamps the language
into :attr:`LoaderResult.metadata`. Phase D's code-graph extraction
will pick this up later; for Phase B.1 the loader exists so code
ingest doesn't degrade to the plain-text fallback (and so the
language is recorded on the resulting Memo).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from z3rno_core.loaders.base import Loader, LoaderResult

# Extension → canonical language name.
_LANG_BY_EXT: dict[str, str] = {
    "py": "python",
    "pyi": "python",
    "ts": "typescript",
    "tsx": "typescript",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "c": "c",
    "h": "c",
    "cc": "cpp",
    "cpp": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "hh": "cpp",
    "cs": "csharp",
    "fs": "fsharp",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "lua": "lua",
    "pl": "perl",
    "r": "r",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "sql": "sql",
    "yml": "yaml",
    "yaml": "yaml",
    "toml": "toml",
    "json": "json",
    "xml": "xml",
    "html": "html",
    "css": "css",
}


class CodeLoader(Loader):
    """Source code passthrough with extension-based language detection."""

    @property
    def name(self) -> str:
        return "code"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        text = content.decode("utf-8", errors="replace") if content else ""
        language = _detect_language(filename) or "unknown"
        return LoaderResult(
            text=text,
            metadata={
                "loader": self.name,
                "mime_type": mime_type or "text/x-source",
                "filename": filename,
                "byte_size": len(content),
                "char_count": len(text),
                "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
                "language": language,
            },
        )


def _detect_language(filename: str | None) -> str | None:
    """Return the canonical language name for ``filename`` or ``None``."""
    if not filename:
        return None
    ext = PurePosixPath(filename).suffix.lower().lstrip(".")
    return _LANG_BY_EXT.get(ext)


def supported_extensions() -> list[str]:
    """List of file extensions :class:`CodeLoader` recognizes."""
    return sorted(_LANG_BY_EXT.keys())

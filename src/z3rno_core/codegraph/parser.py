"""tree-sitter wrapper (Phase D slice 5).

Hosts the language-loader logic and the single ``parse_source`` entry
point. tree-sitter and its grammar wheels are lazy-imported — call
sites that never enable codegraph pay no startup cost.

Per-process language cache keeps repeated parses cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

SUPPORTED_LANGUAGES: tuple[str, ...] = ("python", "typescript")


@dataclass(frozen=True)
class ParsedSource:
    """Thin wrapper around the tree-sitter Tree + raw source bytes."""

    language: str
    source: bytes
    tree: Any  # tree_sitter.Tree — kept loose to avoid an import at type-check time


def _ensure_language(language: str) -> str:
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"language {language!r} not supported; "
            f"got {language!r}, expected one of {SUPPORTED_LANGUAGES}"
        )
    return language


@lru_cache(maxsize=8)
def _load_language(language: str) -> Any:
    """Return the tree-sitter Language object for ``language``."""
    language = _ensure_language(language)
    try:
        from tree_sitter import Language  # noqa: PLC0415 — lazy
    except ImportError as exc:
        raise ImportError(
            "tree-sitter is required for codegraph extraction. "
            "Install with: pip install 'z3rno-core[codegraph]'"
        ) from exc

    if language == "python":
        import tree_sitter_python as ts_python  # noqa: PLC0415

        return Language(ts_python.language())
    if language == "typescript":
        import tree_sitter_typescript as ts_ts  # noqa: PLC0415

        # The TypeScript grammar wheel exposes both TS and TSX variants;
        # we use the TS one — TSX inherits its surface forms.
        return Language(ts_ts.language_typescript())

    # _ensure_language already gated against this — kept for completeness.
    raise ValueError(f"unsupported language: {language}")


@lru_cache(maxsize=8)
def _load_parser(language: str) -> Any:
    try:
        from tree_sitter import Parser  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tree-sitter is required for codegraph extraction. "
            "Install with: pip install 'z3rno-core[codegraph]'"
        ) from exc
    parser = Parser(_load_language(language))
    return parser


def parse_source(source: str | bytes, *, language: str) -> ParsedSource:
    """Parse ``source`` and return the wrapped Tree.

    ``language`` must be one of :data:`SUPPORTED_LANGUAGES`.
    """
    language = _ensure_language(language)
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    parser = _load_parser(language)
    tree = parser.parse(src_bytes)
    return ParsedSource(language=language, source=src_bytes, tree=tree)


def reset_cache() -> None:
    """Clear the parser/language cache (test/admin hook)."""
    _load_language.cache_clear()
    _load_parser.cache_clear()

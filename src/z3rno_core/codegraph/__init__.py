"""z3rno_core.codegraph — Phase D slice 5 code-graph extraction.

Lazy-loaded. tree-sitter + per-language grammar wheels ship under the
``codegraph`` optional-dependency group; importing this package does
not load them until :func:`parse_source` is called.

Public API: :func:`parse_source`, :class:`CodeMemo`, :class:`CodeEdge`,
:func:`extract` (parser → typed Memo + Edge lists), and the writer
helpers.
"""

from __future__ import annotations

from z3rno_core.codegraph.extractor import (
    CodeEdge,
    CodeEdgeKind,
    CodeMemo,
    CodeMemoKind,
    ExtractResult,
    extract,
)
from z3rno_core.codegraph.parser import SUPPORTED_LANGUAGES, parse_source
from z3rno_core.codegraph.writer import CodegraphWriteResult, write_extraction

__all__ = [
    "SUPPORTED_LANGUAGES",
    "CodeEdge",
    "CodeEdgeKind",
    "CodeMemo",
    "CodeMemoKind",
    "CodegraphWriteResult",
    "ExtractResult",
    "extract",
    "parse_source",
    "write_extraction",
]

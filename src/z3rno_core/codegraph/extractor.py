"""AST walker producing typed Memos + Edges (Phase D slice 5).

Pure transform — takes a :class:`ParsedSource` and returns lists of
:class:`CodeMemo` / :class:`CodeEdge`. No DB, no I/O. The writer
turns these into ``memories`` + ``memory_relationships`` rows.

What we extract
---------------
Memos (``CodeMemo.kind``):
  * MODULE   — one per parsed file
  * CLASS    — class / interface declarations
  * FUNCTION — function / method declarations
  * IMPORT   — each import statement target

Edges (``CodeEdge.kind``):
  * DEFINES  — module → top-level function/class; class → method
  * IMPORTS  — module → imported module name
  * CALLS    — function/method → callee identifier (best-effort by name)
  * INHERITS — class → base class name

Caveats
-------
Call resolution is *by name* — we record the call-site identifier but
not the resolved callee Memo. A later refine() cycle can re-bind the
edge once cross-module symbol resolution lands. This is the same
trade-off the Forge made for entity grounding in Phase A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from z3rno_core.codegraph.parser import ParsedSource

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class CodeMemoKind(StrEnum):
    MODULE = "MODULE"
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    IMPORT = "IMPORT"


class CodeEdgeKind(StrEnum):
    DEFINES = "DEFINES"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"


@dataclass(frozen=True)
class CodeMemo:
    """One emitted Memo. ``key`` is an in-extraction identifier the
    writer uses to wire edges before DB IDs exist."""

    key: str
    kind: CodeMemoKind
    name: str
    qualified_name: str  # module.path.name
    language: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeEdge:
    """One emitted edge between two CodeMemos (by ``key``).

    For CALLS edges the target may be unresolved (``target_key=None``
    + ``target_name`` set) — the writer creates a placeholder Memo.
    """

    src_key: str
    kind: CodeEdgeKind
    target_key: str | None = None
    target_name: str | None = None


@dataclass(frozen=True)
class ExtractResult:
    memos: tuple[CodeMemo, ...]
    edges: tuple[CodeEdge, ...]


# ---------------------------------------------------------------------------
# Language-specific node-type tables
# ---------------------------------------------------------------------------

_PYTHON_NODE_TYPES = {
    "module": "module",
    "class": "class_definition",
    "function": "function_definition",
    "import_stmt": ("import_statement", "import_from_statement"),
    "call": "call",
    "identifier": "identifier",
    "name_field": "name",
    "argument_list": "argument_list",
    "block": "block",
}

_TS_NODE_TYPES = {
    "module": "program",
    "class": ("class_declaration", "interface_declaration"),
    "function": ("function_declaration", "method_definition", "arrow_function"),
    "import_stmt": "import_statement",
    "call": "call_expression",
    "identifier": ("identifier", "type_identifier", "property_identifier"),
    "name_field": "name",
    "argument_list": "arguments",
    "block": ("statement_block", "class_body"),
}


def _types_for(language: str) -> dict[str, Any]:
    if language == "python":
        return _PYTHON_NODE_TYPES
    if language == "typescript":
        return _TS_NODE_TYPES
    raise ValueError(f"no node-type table for language: {language}")


# ---------------------------------------------------------------------------
# Walk helpers
# ---------------------------------------------------------------------------


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _matches(node: Any, expected: Any) -> bool:
    """Match a node type against a string or tuple of strings."""
    if isinstance(expected, tuple):
        return bool(node.type in expected)
    return bool(node.type == expected)


def _child_named(node: Any, types: dict[str, Any], key: str) -> Any | None:
    target = types[key]
    for child in node.named_children:
        if _matches(child, target):
            return child
    return None


def _identifier_of(node: Any, types: dict[str, Any], source: bytes) -> str | None:
    """Pull the leading identifier out of a class/function/call node."""
    # Prefer the named "name" field when tree-sitter exposes it.
    try:
        named = node.child_by_field_name("name")
    except Exception:
        named = None
    if named is not None:
        return _node_text(named, source)
    # Fallback: first identifier-shaped child.
    for child in node.children:
        if _matches(child, types["identifier"]):
            return _node_text(child, source)
    return None


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


def extract(parsed: ParsedSource, *, module_name: str = "<anonymous>") -> ExtractResult:
    """Walk the parsed tree and emit Memos + edges."""
    types = _types_for(parsed.language)
    source = parsed.source

    memos: list[CodeMemo] = []
    edges: list[CodeEdge] = []
    # In-extraction key → CodeMemo, so multiple emits with the same key
    # collapse cleanly.
    seen_keys: dict[str, CodeMemo] = {}

    def _emit_memo(m: CodeMemo) -> None:
        if m.key in seen_keys:
            return
        seen_keys[m.key] = m
        memos.append(m)

    def _emit_edge(e: CodeEdge) -> None:
        edges.append(e)

    # --- 1. module Memo ---
    module_key = f"module::{module_name}"
    _emit_memo(
        CodeMemo(
            key=module_key,
            kind=CodeMemoKind.MODULE,
            name=module_name,
            qualified_name=module_name,
            language=parsed.language,
            start_line=0,
            end_line=parsed.tree.root_node.end_point[0],
        )
    )

    # --- 2. recursive walk ---
    def walk(node: Any, scope_qname: str, parent_key: str) -> None:
        for child in node.named_children:
            # Class / interface declaration
            if _matches(child, types["class"]):
                name = _identifier_of(child, types, source) or "<anonymous>"
                qname = f"{scope_qname}.{name}"
                key = f"class::{qname}"
                _emit_memo(
                    CodeMemo(
                        key=key,
                        kind=CodeMemoKind.CLASS,
                        name=name,
                        qualified_name=qname,
                        language=parsed.language,
                        start_line=child.start_point[0],
                        end_line=child.end_point[0],
                    )
                )
                _emit_edge(CodeEdge(src_key=parent_key, kind=CodeEdgeKind.DEFINES, target_key=key))

                # Inheritance — language-specific child structure.
                for base in _iter_base_names(child, types, source, parsed.language):
                    _emit_edge(
                        CodeEdge(
                            src_key=key,
                            kind=CodeEdgeKind.INHERITS,
                            target_name=base,
                        )
                    )
                # Recurse into the body so we capture methods.
                walk(child, qname, key)
                continue

            # Function / method declaration
            if _matches(child, types["function"]):
                name = _identifier_of(child, types, source) or "<anonymous>"
                qname = f"{scope_qname}.{name}"
                key = f"function::{qname}"
                _emit_memo(
                    CodeMemo(
                        key=key,
                        kind=CodeMemoKind.FUNCTION,
                        name=name,
                        qualified_name=qname,
                        language=parsed.language,
                        start_line=child.start_point[0],
                        end_line=child.end_point[0],
                    )
                )
                _emit_edge(CodeEdge(src_key=parent_key, kind=CodeEdgeKind.DEFINES, target_key=key))

                # Inside the function body: capture call sites.
                _collect_calls(child, key, types, source, _emit_edge)
                # Continue walking — closures / nested defs.
                walk(child, qname, key)
                continue

            # Import statement
            if _matches(child, types["import_stmt"]):
                for imp_name in _iter_import_names(child, types, source, parsed.language):
                    key = f"import::{imp_name}"
                    _emit_memo(
                        CodeMemo(
                            key=key,
                            kind=CodeMemoKind.IMPORT,
                            name=imp_name,
                            qualified_name=imp_name,
                            language=parsed.language,
                            start_line=child.start_point[0],
                            end_line=child.end_point[0],
                        )
                    )
                    _emit_edge(
                        CodeEdge(src_key=module_key, kind=CodeEdgeKind.IMPORTS, target_key=key)
                    )
                continue

            # Recurse into any other named child to find nested defs.
            walk(child, scope_qname, parent_key)

    walk(parsed.tree.root_node, module_name, module_key)
    return ExtractResult(memos=tuple(memos), edges=tuple(edges))


# ---------------------------------------------------------------------------
# Language-specific helpers
# ---------------------------------------------------------------------------


def _iter_base_names(node: Any, types: dict[str, Any], source: bytes, language: str) -> list[str]:
    """Pull base-class identifiers out of a class declaration."""
    bases: list[str] = []
    if language == "python":
        # class_definition → superclasses (argument_list)
        for child in node.named_children:
            if child.type == "argument_list":
                bases.extend(
                    _node_text(arg, source)
                    for arg in child.named_children
                    if arg.type == "identifier"
                )
    elif language == "typescript":
        # class_declaration → class_heritage → extends_clause / implements_clause
        for child in node.named_children:
            if child.type in ("class_heritage", "extends_clause", "implements_clause"):
                for sub in child.named_children:
                    if sub.type in ("identifier", "type_identifier"):
                        bases.append(_node_text(sub, source))
                    else:
                        bases.extend(
                            _node_text(grand, source)
                            for grand in sub.named_children
                            if grand.type in ("identifier", "type_identifier")
                        )
    return bases


def _iter_import_names(node: Any, types: dict[str, Any], source: bytes, language: str) -> list[str]:
    """Extract the module/package name(s) from an import node."""
    names: list[str] = []
    if language == "python":
        # import_statement → dotted_name(s)
        # import_from_statement → first child is module name
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    names.append(_node_text(child, source))
                elif child.type == "aliased_import":
                    inner = _child_named(child, types, "name_field")
                    if inner is not None:
                        names.append(_node_text(inner, source))
        elif node.type == "import_from_statement":
            first = node.named_children[0] if node.named_children else None
            if first is not None and first.type in ("dotted_name", "relative_import"):
                names.append(_node_text(first, source))
    elif language == "typescript":
        # import_statement → string (the module source)
        for child in node.named_children:
            if child.type == "string":
                txt = _node_text(child, source).strip("\"'")
                names.append(txt)
    return names


def _collect_calls(
    node: Any,
    src_key: str,
    types: dict[str, Any],
    source: bytes,
    emit: Any,
) -> None:
    """Walk inside a function body, emit a CALLS edge for every call site."""
    call_type = types["call"]
    stack = [node]
    while stack:
        cur = stack.pop()
        if _matches(cur, call_type):
            callee = _call_target_name(cur, source)
            if callee:
                emit(
                    CodeEdge(
                        src_key=src_key,
                        kind=CodeEdgeKind.CALLS,
                        target_name=callee,
                    )
                )
        # Skip into nested function bodies — we DO want to record their
        # call sites, but they're handled by the outer walk. Treating
        # them as opaque here avoids double-counting.
        if cur is not node and cur.type in (
            "function_definition",
            "function_declaration",
            "method_definition",
            "arrow_function",
        ):
            continue
        stack.extend(cur.named_children)


def _call_target_name(call_node: Any, source: bytes) -> str | None:
    """Best-effort callee identifier from a call node."""
    # tree-sitter "call" / "call_expression" usually has the callee at
    # field "function" (python + js/ts both use this).
    try:
        fn = call_node.child_by_field_name("function")
    except Exception:
        fn = None
    target = (
        fn
        if fn is not None
        else (call_node.named_children[0] if call_node.named_children else None)
    )
    if target is None:
        return None
    if target.type in ("identifier", "type_identifier", "property_identifier"):
        return _node_text(target, source)
    if target.type in ("attribute", "member_expression"):
        # x.y.z — return the rightmost identifier
        for child in reversed(list(target.named_children)):
            if child.type in ("identifier", "property_identifier"):
                return _node_text(child, source)
    return None

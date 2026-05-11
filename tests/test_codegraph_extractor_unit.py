"""Unit tests for the codegraph parser + extractor (Phase D slice 5).

Skips when tree-sitter isn't installed. Runs both Python and
TypeScript fixtures, asserting the structural shape of the emitted
Memos + edges.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")
pytest.importorskip("tree_sitter_typescript")

from z3rno_core.codegraph import (
    CodeEdgeKind,
    CodeMemoKind,
    extract,
    parse_source,
)

# ---------------------------------------------------------------------------
# Python fixture
# ---------------------------------------------------------------------------


PY_SOURCE = """\
import os
from typing import List

class Foo(Bar):
    def hello(self):
        return os.getcwd()

def main():
    f = Foo()
    f.hello()
"""


def _memo_names(result, kind: CodeMemoKind) -> set[str]:
    return {m.qualified_name for m in result.memos if m.kind == kind}


def _edge_kinds(result) -> dict[CodeEdgeKind, int]:
    counts: dict[CodeEdgeKind, int] = {}
    for e in result.edges:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts


def test_python_extractor_emits_module_class_function() -> None:
    parsed = parse_source(PY_SOURCE, language="python")
    result = extract(parsed, module_name="mymod")
    assert _memo_names(result, CodeMemoKind.MODULE) == {"mymod"}
    assert _memo_names(result, CodeMemoKind.CLASS) == {"mymod.Foo"}
    assert _memo_names(result, CodeMemoKind.FUNCTION) == {
        "mymod.Foo.hello",
        "mymod.main",
    }


def test_python_extractor_emits_import_memos() -> None:
    parsed = parse_source(PY_SOURCE, language="python")
    result = extract(parsed, module_name="mymod")
    imports = _memo_names(result, CodeMemoKind.IMPORT)
    assert "os" in imports
    assert "typing" in imports


def test_python_extractor_emits_calls_edges() -> None:
    parsed = parse_source(PY_SOURCE, language="python")
    result = extract(parsed, module_name="mymod")
    counts = _edge_kinds(result)
    assert counts.get(CodeEdgeKind.CALLS, 0) >= 2  # getcwd, hello, Foo
    assert counts.get(CodeEdgeKind.INHERITS, 0) >= 1  # Foo extends Bar
    assert counts.get(CodeEdgeKind.IMPORTS, 0) >= 2


def test_python_extractor_inheritance_edge_targets_base_class() -> None:
    parsed = parse_source(PY_SOURCE, language="python")
    result = extract(parsed, module_name="mymod")
    inherits = [e for e in result.edges if e.kind == CodeEdgeKind.INHERITS]
    assert any(e.target_name == "Bar" for e in inherits)


# ---------------------------------------------------------------------------
# TypeScript fixture
# ---------------------------------------------------------------------------


TS_SOURCE = """\
import { useEffect } from 'react';
import * as utils from './utils';

class Widget extends Base {
    render(): void {
        utils.format();
    }
}

function main(): void {
    const w = new Widget();
    w.render();
}
"""


def test_typescript_extractor_emits_module_class_function() -> None:
    parsed = parse_source(TS_SOURCE, language="typescript")
    result = extract(parsed, module_name="widget")
    assert _memo_names(result, CodeMemoKind.MODULE) == {"widget"}
    assert _memo_names(result, CodeMemoKind.CLASS) == {"widget.Widget"}
    # Function set includes the method.
    fns = _memo_names(result, CodeMemoKind.FUNCTION)
    assert "widget.main" in fns
    assert "widget.Widget.render" in fns


def test_typescript_extractor_emits_imports() -> None:
    parsed = parse_source(TS_SOURCE, language="typescript")
    result = extract(parsed, module_name="widget")
    imports = _memo_names(result, CodeMemoKind.IMPORT)
    assert "react" in imports
    assert "./utils" in imports


def test_typescript_extractor_emits_inheritance() -> None:
    parsed = parse_source(TS_SOURCE, language="typescript")
    result = extract(parsed, module_name="widget")
    inherits = [e for e in result.edges if e.kind == CodeEdgeKind.INHERITS]
    assert any(e.target_name == "Base" for e in inherits)


def test_typescript_extractor_emits_calls() -> None:
    parsed = parse_source(TS_SOURCE, language="typescript")
    result = extract(parsed, module_name="widget")
    counts = _edge_kinds(result)
    assert counts.get(CodeEdgeKind.CALLS, 0) >= 1  # format() at minimum


# ---------------------------------------------------------------------------
# Acceptance criterion #5 — function-level call graph is queryable
# ---------------------------------------------------------------------------


def test_call_graph_meets_acceptance_bar_for_python() -> None:
    """The extractor produces a function-level call graph that can be
    walked deterministically — one Memo per function, plus CALLS edges
    that reference identifiers."""
    parsed = parse_source(PY_SOURCE, language="python")
    result = extract(parsed, module_name="mymod")

    # Function Memos exist.
    function_keys = {m.key for m in result.memos if m.kind == CodeMemoKind.FUNCTION}
    assert len(function_keys) >= 2
    calls = [e for e in result.edges if e.kind == CodeEdgeKind.CALLS]
    assert calls  # at least one
    for call in calls:
        assert call.src_key in function_keys


def test_call_graph_meets_acceptance_bar_for_typescript() -> None:
    parsed = parse_source(TS_SOURCE, language="typescript")
    result = extract(parsed, module_name="widget")
    function_keys = {m.key for m in result.memos if m.kind == CodeMemoKind.FUNCTION}
    assert len(function_keys) >= 2
    calls = [e for e in result.edges if e.kind == CodeEdgeKind.CALLS]
    for call in calls:
        assert call.src_key in function_keys

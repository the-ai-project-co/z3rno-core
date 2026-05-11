"""v0.19.5 — codegraph completeness tests.

Covers the three additions:
  1. TS new_expression → CALLS edge (extractor node-type table).
  2. Go + Rust appear in SUPPORTED_LANGUAGES.
  3. Connected-component clustering helper SQL shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from z3rno_core.codegraph.parser import SUPPORTED_LANGUAGES


def test_supported_languages_includes_go_and_rust() -> None:
    assert "go" in SUPPORTED_LANGUAGES
    assert "rust" in SUPPORTED_LANGUAGES
    # Back-compat: existing entries stay.
    assert "python" in SUPPORTED_LANGUAGES
    assert "typescript" in SUPPORTED_LANGUAGES


def test_ts_node_types_includes_new_expression() -> None:
    """``new X()`` parses as ``new_expression``; the extractor's TS
    node-type table must list it alongside ``call_expression`` so
    constructor invocations surface as CALLS edges."""
    from z3rno_core.codegraph.extractor import _TS_NODE_TYPES

    call_types = _TS_NODE_TYPES["call"]
    assert "new_expression" in call_types
    assert "call_expression" in call_types


def test_go_node_types_table_present() -> None:
    from z3rno_core.codegraph.extractor import _GO_NODE_TYPES, _types_for

    assert _types_for("go") is _GO_NODE_TYPES
    # Go's function-like nodes.
    assert "function_declaration" in _GO_NODE_TYPES["function"]
    assert "method_declaration" in _GO_NODE_TYPES["function"]


def test_rust_node_types_table_present() -> None:
    from z3rno_core.codegraph.extractor import _RUST_NODE_TYPES, _types_for

    assert _types_for("rust") is _RUST_NODE_TYPES
    # Rust's class-equivalents.
    assert "struct_item" in _RUST_NODE_TYPES["class"]
    assert "trait_item" in _RUST_NODE_TYPES["class"]
    # Macros count as calls (``println!`` etc).
    assert "macro_invocation" in _RUST_NODE_TYPES["call"]


@pytest.mark.asyncio
async def test_connected_components_clustering_groups_via_relationships() -> None:
    """The CTE query must SELECT both source_memory_id and
    target_memory_id from memory_relationships so the walk follows
    edges in either direction."""
    from z3rno_core.refine.summarize import _cluster_memos_by_components

    conn = MagicMock()
    result = MagicMock()
    a, b, c = uuid4(), uuid4(), uuid4()
    # Two components: {a, b} and {c} (the latter dropped because
    # below _MIN_CLUSTER_SIZE).
    result.fetchall = MagicMock(
        return_value=[
            (a, a),
            (a, b),
            (c, c),
        ]
    )
    conn.execute = AsyncMock(return_value=result)

    clusters = await _cluster_memos_by_components(conn, uuid4(), None, 10)
    assert len(clusters) == 1  # singleton skipped
    assert sorted(clusters[0]) == sorted([a, b])

    # SQL should reference memory_relationships AND walk both
    # source_memory_id and target_memory_id (bidirectional walk).
    args, _ = conn.execute.call_args
    sql = args[0].text if hasattr(args[0], "text") else str(args[0])
    assert "memory_relationships" in sql
    assert "source_memory_id" in sql
    assert "target_memory_id" in sql
    assert "RECURSIVE" in sql


def test_refine_options_exposes_cluster_strategy() -> None:
    from z3rno_core.refine.pipeline import RefineOptions

    # Default is back-compat memo_type clustering.
    assert RefineOptions().cluster_strategy == "memo_type"
    # Operators flip to graph-aware.
    custom = RefineOptions(cluster_strategy="connected_components")
    assert custom.cluster_strategy == "connected_components"

"""Unit tests for z3rno_core.engine.audit.compute_row_hash.

Tests the pure-function hash chain computation. No database needed.
"""

from __future__ import annotations

import hashlib
import json

from z3rno_core.engine.audit import compute_row_hash


class TestComputeRowHash:
    """Test the SHA-256 hash chain computation."""

    def test_returns_bytes(self) -> None:
        result = compute_row_hash(None, {"key": "value"})
        assert isinstance(result, bytes)

    def test_returns_32_bytes(self) -> None:
        result = compute_row_hash(None, {"key": "value"})
        assert len(result) == 32  # SHA-256 = 32 bytes

    def test_deterministic(self) -> None:
        data = {"operation": "store", "org_id": "abc-123"}
        h1 = compute_row_hash(None, data)
        h2 = compute_row_hash(None, data)
        assert h1 == h2

    def test_none_prev_hash(self) -> None:
        """First entry in chain has no previous hash."""
        data = {"a": 1}
        result = compute_row_hash(None, data)
        # Manually compute expected: SHA-256 of canonical JSON only
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).digest()
        assert result == expected

    def test_chained_hash_differs_from_unchained(self) -> None:
        data = {"operation": "recall"}
        h_no_chain = compute_row_hash(None, data)
        prev = b"\x00" * 32
        h_chained = compute_row_hash(prev, data)
        assert h_no_chain != h_chained

    def test_chained_hash_includes_prev(self) -> None:
        """Chained hash should incorporate prev_hash bytes."""
        data = {"x": "y"}
        prev = b"\xab" * 32
        result = compute_row_hash(prev, data)
        # Manually compute
        h = hashlib.sha256()
        h.update(prev)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        h.update(canonical.encode("utf-8"))
        assert result == h.digest()

    def test_different_data_different_hash(self) -> None:
        h1 = compute_row_hash(None, {"a": 1})
        h2 = compute_row_hash(None, {"a": 2})
        assert h1 != h2

    def test_canonical_json_sorted_keys(self) -> None:
        """Key order should not matter due to sort_keys=True."""
        h1 = compute_row_hash(None, {"b": 2, "a": 1})
        h2 = compute_row_hash(None, {"a": 1, "b": 2})
        assert h1 == h2

    def test_empty_dict(self) -> None:
        result = compute_row_hash(None, {})
        canonical = json.dumps({}, sort_keys=True, separators=(",", ":"), default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).digest()
        assert result == expected

    def test_chain_of_three(self) -> None:
        """Simulate a chain of three entries."""
        h1 = compute_row_hash(None, {"seq": 1})
        h2 = compute_row_hash(h1, {"seq": 2})
        h3 = compute_row_hash(h2, {"seq": 3})
        # Each hash should be unique
        assert len({h1, h2, h3}) == 3
        # Verify h3 depends on h2
        h3_alt = compute_row_hash(h1, {"seq": 3})
        assert h3 != h3_alt

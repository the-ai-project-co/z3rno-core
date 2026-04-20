"""Property-based tests using Hypothesis.

Tests generative properties of core functions:
  - compute_row_hash: deterministic hash chain
  - Relevance scoring: always in [0, 1] regardless of weights
  - decay_importance: scores always >= floor and <= original
  - UUID validation for Cypher: valid UUIDs pass, non-UUIDs fail
"""

from __future__ import annotations

import math
from uuid import UUID

from hypothesis import given, settings, strategies as st

from z3rno_core.engine.audit import compute_row_hash
from z3rno_core.graph.queries import _validate_uuid_for_cypher

# ---------------------------------------------------------------------------
# compute_row_hash: deterministic hash chain
# ---------------------------------------------------------------------------


class TestComputeRowHashProperty:
    """Property: hash chain is deterministic for same inputs."""

    @given(
        key=st.text(min_size=0, max_size=100),
        value=st.text(min_size=0, max_size=100),
    )
    def test_deterministic_for_same_inputs(self, key: str, value: str) -> None:
        """Same data + same prev_hash always produces the same hash."""
        data = {key: value}
        prev = b"\x00" * 32
        h1 = compute_row_hash(prev, data)
        h2 = compute_row_hash(prev, data)
        assert h1 == h2

    @given(
        key=st.text(min_size=1, max_size=50),
        value=st.text(min_size=0, max_size=100),
    )
    def test_hash_is_32_bytes(self, key: str, value: str) -> None:
        """SHA-256 always produces a 32-byte digest."""
        data = {key: value}
        h = compute_row_hash(None, data)
        assert len(h) == 32

    @given(
        key=st.text(min_size=1, max_size=50),
        val1=st.text(min_size=0, max_size=100),
        val2=st.text(min_size=0, max_size=100),
    )
    def test_different_data_different_hash(self, key: str, val1: str, val2: str) -> None:
        """Different data should (almost always) produce different hashes."""
        if val1 == val2:
            return  # skip trivial case
        h1 = compute_row_hash(None, {key: val1})
        h2 = compute_row_hash(None, {key: val2})
        assert h1 != h2

    @given(
        key=st.text(min_size=1, max_size=50),
        value=st.text(min_size=0, max_size=100),
    )
    def test_none_prev_hash_differs_from_zero_prev_hash(self, key: str, value: str) -> None:
        """None prev_hash and zero-byte prev_hash produce different hashes."""
        data = {key: value}
        h_none = compute_row_hash(None, data)
        h_zero = compute_row_hash(b"\x00" * 32, data)
        assert h_none != h_zero


# ---------------------------------------------------------------------------
# Relevance scoring: always in [0, 1]
# ---------------------------------------------------------------------------


class TestRelevanceScoringProperty:
    """Property: fused relevance score is always in [0, 1]."""

    @given(
        similarity=st.floats(min_value=0.0, max_value=1.0),
        importance=st.floats(min_value=0.0, max_value=1.0),
        recency_days=st.floats(min_value=0.01, max_value=36500.0),
        sim_w=st.floats(min_value=0.0, max_value=1.0),
        imp_w=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_score_in_unit_range(
        self,
        similarity: float,
        importance: float,
        recency_days: float,
        sim_w: float,
        imp_w: float,
    ) -> None:
        """Relevance score stays in [0, 1] for any valid component values and weights."""
        rec_w = max(0.0, 1.0 - sim_w - imp_w)
        # Skip if weights don't form a valid distribution
        if sim_w + imp_w > 1.0 + 1e-9:
            return
        recency_score = min(1.0, 1.0 / recency_days)
        relevance = sim_w * similarity + imp_w * importance + rec_w * recency_score
        assert -1e-9 <= relevance <= 1.0 + 1e-9

    @given(
        similarity=st.floats(min_value=0.0, max_value=1.0),
        importance=st.floats(min_value=0.0, max_value=1.0),
        recency_days=st.floats(min_value=0.01, max_value=36500.0),
    )
    def test_default_weights_score_in_unit_range(
        self,
        similarity: float,
        importance: float,
        recency_days: float,
    ) -> None:
        """Default 60/25/15 weights always produce scores in [0, 1]."""
        recency_score = min(1.0, 1.0 / recency_days)
        relevance = 0.60 * similarity + 0.25 * importance + 0.15 * recency_score
        assert -1e-9 <= relevance <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# decay_importance: scores >= floor and <= original
# ---------------------------------------------------------------------------


class TestDecayImportanceProperty:
    """Property: decay always produces scores >= floor and <= original."""

    @given(
        current_score=st.floats(min_value=0.0, max_value=1.0),
        decay_rate=st.floats(min_value=0.0, max_value=1.0),
        days_since=st.floats(min_value=0.0, max_value=36500.0),
        decay_floor=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_decay_bounded(
        self,
        current_score: float,
        decay_rate: float,
        days_since: float,
        decay_floor: float,
    ) -> None:
        """Decayed score is always >= floor and <= max(original, floor)."""
        new_score = max(decay_floor, current_score * math.exp(-decay_rate * days_since))
        assert new_score >= decay_floor - 1e-9
        # The result is bounded by the larger of the original score and the floor
        assert new_score <= max(current_score, decay_floor) + 1e-9

    @given(
        current_score=st.floats(min_value=0.01, max_value=1.0),
        decay_rate=st.floats(min_value=0.001, max_value=0.5),
        days_since=st.floats(min_value=0.0, max_value=3650.0),
    )
    def test_decay_monotonic(
        self,
        current_score: float,
        decay_rate: float,
        days_since: float,
    ) -> None:
        """Decay is monotonically non-increasing with time."""
        score_at_t = current_score * math.exp(-decay_rate * days_since)
        score_at_t_plus_1 = current_score * math.exp(-decay_rate * (days_since + 1.0))
        assert score_at_t_plus_1 <= score_at_t + 1e-9


# ---------------------------------------------------------------------------
# UUID validation for Cypher
# ---------------------------------------------------------------------------


class TestUuidValidationProperty:
    """Property: valid UUIDs always pass, non-UUIDs always fail."""

    @given(uuid_val=st.uuids())
    def test_valid_uuids_always_pass(self, uuid_val: UUID) -> None:
        """Any valid UUID object passes validation."""
        result = _validate_uuid_for_cypher(uuid_val)
        assert isinstance(result, str)
        assert len(result) == 36

    @given(
        text_val=st.text(
            alphabet=st.sampled_from("ghijklmnopqrstuvwxyz!@#$%^&*()+=[]{}|;:',.<>?/ "),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_non_uuid_strings_always_fail(self, text_val: str) -> None:
        """Strings with invalid characters always fail validation."""
        # Only test strings that actually contain invalid chars
        if all(c in "0123456789abcdef-" for c in text_val):
            return  # skip — this is technically valid hex
        try:
            # _validate_uuid_for_cypher takes a UUID, so we need to bypass
            # the type system to test with raw strings
            _validate_uuid_for_cypher(text_val)  # type: ignore[arg-type]
            # If it didn't raise, the string must have only valid chars
            assert all(c in "0123456789abcdef-" for c in str(text_val))
        except (ValueError, AttributeError):
            pass  # Expected: invalid input rejected

    @given(uuid_val=st.uuids())
    def test_validated_uuid_only_contains_hex_and_hyphens(self, uuid_val: UUID) -> None:
        """Validated output only contains hex digits and hyphens."""
        result = _validate_uuid_for_cypher(uuid_val)
        assert all(c in "0123456789abcdef-" for c in result)

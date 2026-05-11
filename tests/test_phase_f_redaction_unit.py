"""Unit tests for Phase F slice 2 — compliance-graded retrieval.

Covers:
  * Built-in default rules (email / SSN / credit card / phone).
  * YAML rules-file loading + role resolution + fallback_role.
  * Recursive scrubbing of metadata + graph_context.
  * The Phase F acceptance bar: zero PII leakage at the configured
    role on a synthetic PII corpus.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from z3rno_core.retrieval.base import StrategyResult
from z3rno_core.retrieval.redaction import (
    RedactionConfig,
    RedactionFilter,
    RedactionRule,
    _builtin_config,
    load_config,
    make_redaction_filter,
    reset_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(content: str = "", *, metadata: dict | None = None) -> StrategyResult:
    now = datetime.now(UTC)
    return StrategyResult(
        memory_id=uuid4(),
        content=content,
        summary=None,
        memory_type="semantic",
        importance_score=0.5,
        relevance_score=0.5,
        recall_count=0,
        created_at=now,
        valid_from=now,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


def test_builtin_redacts_email() -> None:
    f = RedactionFilter()  # built-in defaults
    out = f.apply(role=None, results=[_result("Contact: jane@example.com")])
    assert "jane@example.com" not in out[0].content
    assert "[REDACTED:EMAIL]" in out[0].content


def test_builtin_redacts_ssn() -> None:
    f = RedactionFilter()
    out = f.apply(role=None, results=[_result("SSN 123-45-6789 noted")])
    assert "123-45-6789" not in out[0].content
    assert "[REDACTED:SSN]" in out[0].content


def test_builtin_redacts_credit_card() -> None:
    f = RedactionFilter()
    out = f.apply(role=None, results=[_result("card 4111 1111 1111 1111 ok")])
    assert "4111 1111 1111 1111" not in out[0].content


def test_builtin_redacts_phone() -> None:
    f = RedactionFilter()
    out = f.apply(role=None, results=[_result("call +1 (415) 555-2671 please")])
    assert "(415) 555-2671" not in out[0].content


def test_filter_leaves_content_alone_when_no_pii() -> None:
    f = RedactionFilter()
    plain = "the user prefers dark mode and weekly digest emails"
    out = f.apply(role=None, results=[_result(plain)])
    assert out[0].content == plain


# ---------------------------------------------------------------------------
# Metadata + graph_context recursion
# ---------------------------------------------------------------------------


def test_redacts_metadata_recursively() -> None:
    f = RedactionFilter()
    meta = {
        "source": "leak@example.com",
        "nested": {"phone": "+1 (415) 555-2671", "safe": "ok"},
        "list": ["ssn 123-45-6789", "clean"],
    }
    out = f.apply(role=None, results=[_result("plain", metadata=meta)])
    md = out[0].metadata
    assert "leak@example.com" not in md["source"]
    assert "555-2671" not in md["nested"]["phone"]
    assert md["nested"]["safe"] == "ok"
    assert "123-45-6789" not in md["list"][0]
    assert md["list"][1] == "clean"


def test_redacts_graph_context_recursively() -> None:
    """The graph_context list of dicts also gets scrubbed."""
    f = RedactionFilter()
    now = datetime.now(UTC)
    r = StrategyResult(
        memory_id=uuid4(),
        content="plain",
        summary=None,
        memory_type="semantic",
        importance_score=0.5,
        relevance_score=0.5,
        recall_count=0,
        created_at=now,
        valid_from=now,
        metadata={},
        graph_context=[{"row": "ssn 123-45-6789"}],
    )
    out = f.apply(role=None, results=[r])
    assert "123-45-6789" not in out[0].graph_context[0]["row"]


# ---------------------------------------------------------------------------
# RedactionRule + RedactionConfig
# ---------------------------------------------------------------------------


def test_rule_from_strings_compiles_pattern() -> None:
    rule = RedactionRule.from_strings(pattern=r"foo\d+", replacement="[X]")
    assert rule.pattern.search("foo42") is not None
    assert rule.replacement == "[X]"


def test_config_rules_for_unknown_role_uses_fallback() -> None:
    cfg = RedactionConfig(
        defaults=(),
        by_role={
            "intern": (RedactionRule.from_strings(pattern="x", replacement="X"),),
        },
        fallback_role="intern",
    )
    rules = cfg.rules_for("janitor")
    assert len(rules) == 1
    assert rules[0].pattern.pattern == "x"


def test_config_rules_for_role_prepends_defaults() -> None:
    """Role-specific rules apply *before* defaults so role-specific
    replacements win on overlap."""
    cfg = RedactionConfig(
        defaults=(RedactionRule.from_strings(pattern="d", replacement="D"),),
        by_role={
            "intern": (RedactionRule.from_strings(pattern="i", replacement="I"),),
        },
    )
    rules = cfg.rules_for("intern")
    assert [r.replacement for r in rules] == ["I", "D"]


def test_config_no_role_returns_defaults_only() -> None:
    cfg = RedactionConfig(
        defaults=(RedactionRule.from_strings(pattern="d", replacement="D"),),
        by_role={},
    )
    assert len(cfg.rules_for(None)) == 1


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_load_config_missing_path_returns_builtins() -> None:
    cfg = load_config("/no/such/file.yaml")
    assert any(r.replacement == "[REDACTED:EMAIL]" for r in cfg.defaults)


def test_load_config_none_path_returns_builtins() -> None:
    cfg = load_config(None)
    assert len(cfg.defaults) >= 3


def test_load_config_parses_real_yaml(tmp_path: Path) -> None:
    yaml_text = """
defaults:
  - { pattern: 'CARDX\\\\d+', replacement: '[CARD]' }
roles:
  intern:
    - { pattern: 'SECRET-\\\\w+', replacement: '[SECRET]' }
  admin: []
fallback_role: intern
"""
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.fallback_role == "intern"
    assert "intern" in cfg.by_role
    assert "admin" in cfg.by_role
    assert cfg.by_role["admin"] == ()
    # Defaults compile correctly.
    assert any(r.replacement == "[CARD]" for r in cfg.defaults)


# ---------------------------------------------------------------------------
# make_redaction_filter cache
# ---------------------------------------------------------------------------


def test_make_filter_is_cached_per_path(tmp_path: Path) -> None:
    reset_cache()
    p = tmp_path / "r.yaml"
    p.write_text("defaults: []\nroles: {}\n")
    f1 = make_redaction_filter(rules_path=str(p))
    f2 = make_redaction_filter(rules_path=str(p))
    # Both filters share the same underlying config object (lru_cache hit).
    assert f1.config is f2.config


# ---------------------------------------------------------------------------
# Acceptance bar: zero PII leakage on a synthetic dataset
# ---------------------------------------------------------------------------

# Patterns we *expect* to be redacted. A leaked recall means the filter
# missed one of these — the test fails. Conservative: matches the
# built-in rule set's regexes.
_PII_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # credit card
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
)


_SYNTHETIC_CORPUS: tuple[str, ...] = (
    "User jane.doe@example.com asked about pricing.",
    "Followup: please verify SSN 123-45-6789 with the user.",
    "Card on file: 4111-1111-1111-1111. Refund processed.",
    "Call the user back on +1 (415) 555-2671 tomorrow.",
    "Escalation note — secondary contact bob+test@example.co.uk reachable.",
    "Plain note, no PII here. The cat sat on the mat.",
    "Mixed: SSN 555-66-7777 and phone 415-555-2671 in same note.",
    "Customer email leak@example.com complained about the dashboard.",
    "Lookup error for 8765-4321-2222-9999 — retried twice, still failing.",
    "Internal: jane.doe@example.com cc: bob@example.co.uk for the audit.",
    # Deliberate clean controls — these MUST survive redaction unchanged.
    "Q3 dashboard incident at 9am UTC; followups due Friday.",
    "Ticket T-117 is marked priority high.",
)


def test_zero_pii_leakage_at_configured_role() -> None:
    """Phase F acceptance #2: zero PII leakage on the synthetic corpus
    at the configured role.

    The corpus mixes emails / SSNs / credit cards / phones with clean
    control strings. The filter must scrub every PII match and leave
    every clean string byte-identical.
    """
    f = RedactionFilter(_builtin_config())
    results = [_result(content=line) for line in _SYNTHETIC_CORPUS]
    out = f.apply(role="intern", results=results)

    leaks: list[str] = []
    for line, r in zip(_SYNTHETIC_CORPUS, out, strict=True):
        for pat in _PII_PATTERNS:
            leaks.extend(
                f"{pat.pattern!r} → {match.group(0)!r} in {line!r}"
                for match in pat.finditer(r.content)
            )

    assert leaks == [], "PII leaked at configured role:\n  " + "\n  ".join(leaks)

    # Clean controls must survive unchanged.
    clean_lines = _SYNTHETIC_CORPUS[10:]
    for i, original in enumerate(clean_lines, start=10):
        assert out[i].content == original, (
            f"clean control mutated: {original!r} → {out[i].content!r}"
        )


def test_filter_does_not_raise_on_empty_results() -> None:
    f = RedactionFilter()
    assert f.apply(role="any", results=[]) == []


def test_filter_returns_new_instances_not_mutating_input() -> None:
    """StrategyResult is frozen — the filter must build new instances."""
    f = RedactionFilter()
    inp = _result("call 415-555-2671")
    out = f.apply(role=None, results=[inp])
    # Original input is untouched.
    assert inp.content == "call 415-555-2671"
    assert out[0] is not inp

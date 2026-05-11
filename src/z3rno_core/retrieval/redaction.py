"""Phase F slice 2 — compliance-graded retrieval.

Post-retrieval filter chain that applies role-aware redactions to
``StrategyResult.content`` / ``summary`` / metadata before the recall
response leaves the server. The Phase F acceptance bar is *zero* PII
leakage at the configured role on a synthetic PII benchmark.

Design
------

  * :class:`RetrievalFilter` — narrow ABC. ``apply(role, results)``
    returns a fresh list (StrategyResult is frozen). Multiple filters
    can be chained — the engine layer composes them.
  * :class:`RedactionFilter` — regex-based PII redactor driven by a
    YAML rules file. Per-role pattern lists; falls back to a
    built-in default set when no file is configured.
  * :class:`RedactionRule` — one (pattern, replacement) pair.

Rules file shape::

    # Optional 'defaults' apply to every role on top of role-specific rules.
    defaults:
      - { pattern: '[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}', replacement: '[EMAIL]' }
    roles:
      intern:
        - { pattern: '\\b\\d{3}-\\d{2}-\\d{4}\\b', replacement: '[SSN]' }
        - { pattern: '\\b\\d{16}\\b',              replacement: '[CARD]' }
      analyst:
        - { pattern: '\\b\\d{3}-\\d{2}-\\d{4}\\b', replacement: '[SSN]' }
      admin: []   # no role-specific redactions
    fallback_role: intern

Roles unknown to the file get the ``fallback_role`` rules — fail-safe
"redact more, not less" when configuration is incomplete.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from z3rno_core.retrieval.base import StrategyResult

# ---------------------------------------------------------------------------
# Built-in default rule set — applied when no rules file is configured.
# Conservative: matches the regulated-tenant defaults from the Phase F doc.
# ---------------------------------------------------------------------------

_DEFAULT_RULES: dict[str, tuple[str, str]] = {
    # Email
    "email": (
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "[REDACTED:EMAIL]",
    ),
    # US SSN: 3-2-4 digit pattern with word boundaries.
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED:SSN]"),
    # Credit card: 13-19 digit runs (with optional separators), word-bounded.
    "credit_card": (
        r"\b(?:\d[ -]?){13,19}\b",
        "[REDACTED:CARD]",
    ),
    # US phone with separators; a deliberate conservative match.
    "phone_us": (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[REDACTED:PHONE]",
    ),
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionRule:
    """One (compiled-pattern, replacement) pair."""

    pattern: re.Pattern[str]
    replacement: str
    name: str = ""

    @classmethod
    def from_strings(cls, *, pattern: str, replacement: str, name: str = "") -> RedactionRule:
        return cls(pattern=re.compile(pattern), replacement=replacement, name=name)


@dataclass(frozen=True)
class RedactionConfig:
    """Parsed rules-file contents."""

    defaults: tuple[RedactionRule, ...] = ()
    by_role: dict[str, tuple[RedactionRule, ...]] = field(default_factory=dict)
    fallback_role: str = ""

    def rules_for(self, role: str | None) -> tuple[RedactionRule, ...]:
        """Resolve the effective rule list for ``role``.

        Order: role-specific (or fallback) rules first, then defaults.
        Unknown roles use ``fallback_role`` so the default behavior is
        "redact more, not less". Empty role → defaults only.
        """
        if not role:
            return self.defaults
        role_rules = self.by_role.get(role)
        if role_rules is None and self.fallback_role:
            role_rules = self.by_role.get(self.fallback_role, ())
        if role_rules is None:
            role_rules = ()
        return (*role_rules, *self.defaults)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _builtin_config() -> RedactionConfig:
    """Conservative defaults used when no YAML rules file is supplied."""
    rules = tuple(
        RedactionRule.from_strings(pattern=p, replacement=r, name=name)
        for name, (p, r) in _DEFAULT_RULES.items()
    )
    return RedactionConfig(defaults=rules, by_role={}, fallback_role="")


def load_config(path: str | Path | None) -> RedactionConfig:
    """Parse a YAML rules file into a :class:`RedactionConfig`.

    ``path=None`` or a missing file returns the built-in defaults —
    never raises on missing file, so operators can flip
    ``RETRIEVAL_REDACTION_ENABLED=true`` without immediately needing a
    rules file on disk.
    """
    if path is None:
        return _builtin_config()
    p = Path(path)
    if not p.exists():
        return _builtin_config()

    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — optional dep
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to parse RETRIEVAL_REDACTION_RULES_PATH. "
            "Install with: pip install pyyaml"
        ) from exc

    raw = yaml.safe_load(p.read_text()) or {}
    defaults = tuple(
        RedactionRule.from_strings(
            pattern=r["pattern"],
            replacement=r.get("replacement", "[REDACTED]"),
            name=str(r.get("name", "")),
        )
        for r in (raw.get("defaults") or [])
        if isinstance(r, dict) and r.get("pattern")
    )
    by_role: dict[str, tuple[RedactionRule, ...]] = {}
    for role, rules in (raw.get("roles") or {}).items():
        compiled = tuple(
            RedactionRule.from_strings(
                pattern=r["pattern"],
                replacement=r.get("replacement", "[REDACTED]"),
                name=str(r.get("name", "")),
            )
            for r in (rules or [])
            if isinstance(r, dict) and r.get("pattern")
        )
        by_role[str(role)] = compiled
    fallback_role = str(raw.get("fallback_role", "") or "")
    return RedactionConfig(defaults=defaults, by_role=by_role, fallback_role=fallback_role)


@lru_cache(maxsize=8)
def _load_config_cached(path: str | None) -> RedactionConfig:
    """Process-cached config loader. Tests can call ``reset_cache()``."""
    return load_config(path)


def reset_cache() -> None:
    _load_config_cached.cache_clear()


# ---------------------------------------------------------------------------
# Filter ABC + RedactionFilter
# ---------------------------------------------------------------------------


class RetrievalFilter(ABC):
    """A post-retrieval filter.

    Implementations transform a list of :class:`StrategyResult`
    before the engine returns it to the caller. Filters MUST NOT
    raise — they return the (possibly empty) input on error.
    """

    @abstractmethod
    def apply(
        self,
        role: str | None,
        results: list[StrategyResult],
    ) -> list[StrategyResult]: ...


class RedactionFilter(RetrievalFilter):
    """Regex-based redactor — applies the role's rules to every
    string-valued field of every result."""

    def __init__(self, config: RedactionConfig | None = None) -> None:
        self._config = config or _builtin_config()

    @property
    def config(self) -> RedactionConfig:
        return self._config

    def apply(
        self,
        role: str | None,
        results: list[StrategyResult],
    ) -> list[StrategyResult]:
        rules = self._config.rules_for(role)
        if not rules:
            return results

        from z3rno_core.retrieval.base import StrategyResult  # noqa: PLC0415

        out: list[StrategyResult] = []
        for r in results:
            content = _scrub(r.content, rules)
            summary = _scrub(r.summary, rules) if r.summary else r.summary
            metadata = _scrub_obj(r.metadata, rules)
            graph_context = [_scrub_obj(c, rules) for c in r.graph_context]
            out.append(
                StrategyResult(
                    memory_id=r.memory_id,
                    content=content,
                    summary=summary,
                    memory_type=r.memory_type,
                    importance_score=r.importance_score,
                    relevance_score=r.relevance_score,
                    recall_count=r.recall_count,
                    created_at=r.created_at,
                    valid_from=r.valid_from,
                    metadata=metadata,
                    score_components=r.score_components,
                    graph_context=graph_context,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Scrub helpers — applied to strings, lists, and dicts recursively.
# ---------------------------------------------------------------------------


def _scrub(value: str, rules: tuple[RedactionRule, ...]) -> str:
    if not value:
        return value
    out = value
    for rule in rules:
        out = rule.pattern.sub(rule.replacement, out)
    return out


def _scrub_obj(value: Any, rules: tuple[RedactionRule, ...]) -> Any:
    if isinstance(value, str):
        return _scrub(value, rules)
    if isinstance(value, dict):
        return {k: _scrub_obj(v, rules) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_obj(v, rules) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_obj(v, rules) for v in value)
    return value


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def make_redaction_filter(
    *,
    rules_path: str | None = None,
) -> RedactionFilter:
    """Build a :class:`RedactionFilter` from a rules-file path (or defaults).

    Cached per-path so repeated calls in one process don't re-read the file.
    """
    return RedactionFilter(_load_config_cached(rules_path))

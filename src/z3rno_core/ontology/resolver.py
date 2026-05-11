"""OntologyResolver — exact + fuzzy lookup over a loaded OntologyIndex.

Two strategies, both case-folded:

  * ``exact`` — name must hit ``OntologyIndex.by_label`` verbatim.
  * ``fuzzy`` — rapidfuzz token-set ratio over every entry's labels;
    must clear the configured threshold to count.

rapidfuzz is lazy-imported. Operators who pick ``exact`` (or never
turn on the resolver at all) don't need the optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from z3rno_core.ontology.loader import OntologyEntry, OntologyIndex


MatchStrategy = Literal["exact", "fuzzy"]


@dataclass(frozen=True)
class ResolveMatch:
    """One resolver hit."""

    uri: str
    score: float  # 1.0 for exact; [threshold, 1.0] for fuzzy
    matched_label: str


class OntologyResolver:
    """Resolve free-form entity names to canonical ontology URIs."""

    def __init__(
        self,
        index: OntologyIndex,
        *,
        strategy: MatchStrategy = "fuzzy",
        fuzzy_threshold: float = 0.80,
    ) -> None:
        self.index = index
        self.strategy: MatchStrategy = strategy
        self.fuzzy_threshold = fuzzy_threshold

    def resolve(self, name: str, type_hint: str | None = None) -> ResolveMatch | None:
        """Return the best match, or None when nothing clears the bar.

        ``type_hint`` (e.g. ``"PERSON"`` / ``"ORG"``) currently biases
        only the fuzzy strategy — exact match ignores it because exact
        label collisions across types are rare and almost always the
        canonical entity.
        """
        if not name:
            return None
        folded = name.casefold().strip()
        if not folded:
            return None

        # --- exact ---
        uri = self.index.by_label.get(folded)
        if uri is not None:
            return ResolveMatch(uri=uri, score=1.0, matched_label=name)
        if self.strategy == "exact":
            return None

        # --- fuzzy ---
        return self._fuzzy_match(folded, type_hint)

    def _fuzzy_match(self, folded_name: str, type_hint: str | None) -> ResolveMatch | None:
        try:
            from rapidfuzz import fuzz  # noqa: PLC0415 — lazy import
        except ImportError as exc:
            raise ImportError(
                "rapidfuzz is required for ONTOLOGY_MATCHING_STRATEGY=fuzzy. "
                "Install with: pip install 'z3rno-core[ontology]'"
            ) from exc

        best: tuple[float, OntologyEntry, str] | None = None
        type_hint_folded = type_hint.casefold() if type_hint else None

        for entry in self.index.entries:
            for label in (entry.primary_label, *entry.aliases):
                score = fuzz.token_set_ratio(folded_name, label.casefold()) / 100.0
                # Type-hint bonus: nudge matches whose recorded type
                # matches the caller's hint. Small (+0.05) so a near-
                # miss on the right type wins over a perfect match on
                # the wrong type, but neither blocks reasonable hits.
                if (
                    type_hint_folded
                    and entry.type_hint
                    and type_hint_folded == entry.type_hint.casefold()
                ):
                    score = min(1.0, score + 0.05)
                if best is None or score > best[0]:
                    best = (score, entry, label)

        if best is None or best[0] < self.fuzzy_threshold:
            return None
        return ResolveMatch(uri=best[1].uri, score=best[0], matched_label=best[2])

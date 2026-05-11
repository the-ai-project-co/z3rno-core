"""OWL / TTL ontology loader (Phase D slice 4).

Reads an ontology file via rdflib, extracts every named individual /
class with its labels, and builds an index suitable for the resolver.

rdflib is lazy-imported — operators who never set
``ONTOLOGY_RESOLVER=rdflib`` pay zero cost. The :class:`OntologyIndex`
data structure is plain stdlib so it can flow through async tasks and
be cached without dragging rdflib along.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class OntologyEntry:
    """One indexed concept — canonical URI + every label seen for it."""

    uri: str
    primary_label: str
    aliases: tuple[str, ...]
    type_hint: str | None = None  # the RDF type as a coarse hint, when known


@dataclass(frozen=True)
class OntologyIndex:
    """Process-cached projection of an ontology file.

    ``by_label`` is a case-folded lookup; ``entries`` is the full
    catalogue the fuzzy resolver scans.
    """

    entries: tuple[OntologyEntry, ...]
    by_label: dict[str, str]  # case-folded label → uri

    def __len__(self) -> int:
        return len(self.entries)


# Common predicates carrying human-readable labels in OWL/RDFS/SKOS.
_LABEL_PREDICATES = (
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2008/05/skos-xl#literalForm",
)


def _localname(uri: str) -> str:
    """Fallback label = the URI's local name (after `#` or last `/`)."""
    for sep in ("#", "/"):
        if sep in uri:
            return uri.rsplit(sep, 1)[-1]
    return uri


@lru_cache(maxsize=8)
def load_ontology(path: str) -> OntologyIndex:
    """Parse an OWL / TTL / RDF file and return the indexed entries.

    Cached per-path so repeated calls inside one Celery worker (or one
    web server instance) parse the file only once. Cache is process-
    local; restarting the worker re-parses.

    Raises ``FileNotFoundError`` if the file is missing, and
    ``ImportError`` (with a clear message) if rdflib isn't installed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ontology file not found: {path}")

    try:
        from rdflib import Graph  # noqa: PLC0415 — lazy import
    except ImportError as exc:
        raise ImportError(
            "rdflib is required for ONTOLOGY_RESOLVER=rdflib. "
            "Install with: pip install 'z3rno-core[ontology]'"
        ) from exc

    graph = Graph()
    graph.parse(str(p))

    # Build (uri → set of labels) by collecting every triple whose
    # predicate is a known label predicate, plus the URI's own
    # local-name as a fallback.
    labels: dict[str, list[str]] = {}
    types: dict[str, str] = {}

    for subj, pred, obj in graph:
        sub_str = str(subj)
        pred_str = str(pred)
        if pred_str in _LABEL_PREDICATES:
            labels.setdefault(sub_str, []).append(str(obj))
        elif pred_str == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            # First type wins as the coarse hint — finer typing belongs
            # in the resolver, not here.
            types.setdefault(sub_str, _localname(str(obj)))

    entries: list[OntologyEntry] = []
    by_label: dict[str, str] = {}
    for uri, ls in labels.items():
        primary = ls[0]
        aliases = tuple(dict.fromkeys(ls[1:]))  # de-dup preserving order
        entries.append(
            OntologyEntry(
                uri=uri,
                primary_label=primary,
                aliases=aliases,
                type_hint=types.get(uri),
            )
        )
        for lbl in (primary, *aliases):
            by_label.setdefault(lbl.casefold(), uri)

    # Also index every URI that has no explicit label by its local name.
    for uri, type_hint in types.items():
        if uri not in labels:
            ln = _localname(uri)
            entries.append(
                OntologyEntry(
                    uri=uri,
                    primary_label=ln,
                    aliases=(),
                    type_hint=type_hint,
                )
            )
            by_label.setdefault(ln.casefold(), uri)

    return OntologyIndex(entries=tuple(entries), by_label=by_label)


def reset_cache() -> None:
    """Clear the per-process ontology cache (test/admin hook)."""
    load_ontology.cache_clear()

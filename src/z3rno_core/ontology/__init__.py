"""z3rno_core.ontology — OWL/TTL grounding for Phase D slice 4.

Lazy-loaded — importing this package does NOT pull rdflib until the
operator opts in via ``ONTOLOGY_RESOLVER=rdflib``. The ``ontology``
optional-dependency group ships the runtime requirements:

    pip install 'z3rno-core[ontology]'

Public API: :class:`OntologyResolver` (resolver + match strategy) and
:func:`load_ontology` (one-shot loader returning a label → URI index).
"""

from __future__ import annotations

from z3rno_core.ontology.loader import OntologyIndex, load_ontology
from z3rno_core.ontology.resolver import OntologyResolver

__all__ = ["OntologyIndex", "OntologyResolver", "load_ontology"]

"""Concrete retrieval strategies for Phase C.

Importing this package registers every strategy via the
``@register_strategy`` decorator. Phase C ships them incrementally:

  * C.1 — VECTOR, LEXICAL, AUTO (skeleton)
  * C.2 — GRAPH, TRIPLET
  * C.3 — AUTO (real classifier), TRACE
  * C.4 — TEMPORAL, ASK, CYPHER

Each module imports its dependencies lazily so a deployment that
hasn't installed an optional extra still imports cleanly.
"""

from z3rno_core.retrieval.strategies import (  # noqa: F401  (side-effect imports register strategies)
    ask,
    auto,
    code,
    cypher,
    graph,
    lexical,
    temporal,
    trace,
    triplet,
    vector,
)

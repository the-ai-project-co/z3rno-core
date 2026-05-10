"""z3rno_core.scrapers — web-search and discovery primitives (Phase B.2).

The URL loader (Phase B.1) handles ingesting *one known URL*. The
scrapers package adds the discovery primitive: given a query, find
relevant URLs, then hand them to the IngestPipeline.

Phase B.2 ships :class:`TavilyScraper`. Future providers (Brave Search,
SerpAPI, plain Google CSE) plug in behind the same
:class:`SearchProvider` interface.

Modules
-------

- ``base``    — :class:`SearchProvider` ABC + :class:`SearchResult` schema
- ``tavily``  — :class:`TavilyScraper` wrapping the tavily-python client

Phase B.2 also adds an opt-in Playwright fallback inside
:mod:`z3rno_core.loaders.url` for JS-rendered pages — that lives next
to the loader, not here, because it's a body-rendering concern, not a
discovery one.
"""

from __future__ import annotations

from z3rno_core.scrapers.base import SearchError, SearchProvider, SearchResult
from z3rno_core.scrapers.tavily import TavilyScraper

__all__ = [
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "TavilyScraper",
]

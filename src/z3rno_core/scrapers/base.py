"""Search-provider contract — the discovery half of Phase B.2.

Search providers turn a free-text query into a list of URLs +
metadata. The IngestPipeline never calls them directly; the
``POST /v1/ingest/search`` endpoint asks the provider for the top N
results, then enqueues a separate ``ingest_run`` task per URL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """One result returned by :meth:`SearchProvider.search`."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., min_length=1)
    title: str = ""
    snippet: str = ""
    raw_content: str = ""
    score: float | None = Field(default=None, description="Provider-specific relevance score.")


class SearchError(Exception):
    """Base exception for search-provider failures."""


class SearchProvider(ABC):
    """Discovery primitive: query → top-N URLs.

    Implementations are stateless; construction is import-safe.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logging."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Return up to ``max_results`` search results for ``query``."""

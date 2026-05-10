"""TavilyScraper — :class:`SearchProvider` backed by tavily-python.

The Tavily client is fully synchronous. We wrap calls in
``asyncio.to_thread`` so the scraper plays nicely with the rest of the
async pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from z3rno_core.scrapers.base import SearchError, SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class TavilyScraper(SearchProvider):
    """Tavily-backed web search.

    Construction validates the API key but does not make a network
    call. The first call to :meth:`search` exercises credentials.
    """

    def __init__(
        self,
        *,
        api_key: str,
        search_depth: str = "basic",
    ) -> None:
        if not api_key:
            raise ValueError("TavilyScraper requires a non-empty api_key")
        if search_depth not in {"basic", "advanced"}:
            raise ValueError(f"unknown search_depth: {search_depth!r}")
        self._api_key = api_key
        self._search_depth = search_depth
        self._client: Any = None  # lazy

    @property
    def name(self) -> str:
        return "tavily"

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient  # noqa: PLC0415 — lazy import keeps construction cheap

            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        max_results = max(1, min(20, max_results))
        client = self._get_client()

        def _do_search() -> dict[str, Any]:
            result: dict[str, Any] = client.search(
                query=query,
                search_depth=self._search_depth,
                max_results=max_results,
                include_raw_content=False,
            )
            return result

        try:
            response = await asyncio.to_thread(_do_search)
        except Exception as exc:
            raise SearchError(f"tavily search failed: {exc}") from exc

        raw_results = response.get("results") if isinstance(response, dict) else None
        if not raw_results:
            return []

        out: list[SearchResult] = []
        for item in raw_results:
            url = item.get("url") or ""
            if not url:
                continue
            out.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or "",
                    snippet=item.get("content") or item.get("snippet") or "",
                    raw_content=item.get("raw_content") or "",
                    score=_safe_float(item.get("score")),
                )
            )
        return out


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

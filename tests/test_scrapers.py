"""Unit tests for z3rno_core.scrapers (Phase B.2)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from z3rno_core.scrapers import SearchError, SearchProvider, SearchResult, TavilyScraper


class TestSearchResult:
    def test_construct_minimal(self) -> None:
        r = SearchResult(url="https://x.com")
        assert r.url == "https://x.com"
        assert r.title == ""
        assert r.score is None

    def test_empty_url_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017, PT011
            SearchResult(url="")


class TestTavilyScraper:
    def test_construct(self) -> None:
        s = TavilyScraper(api_key="tvly-fake")
        assert s.name == "tavily"
        assert isinstance(s, SearchProvider)

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty api_key"):
            TavilyScraper(api_key="")

    def test_unknown_search_depth_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown search_depth"):
            TavilyScraper(api_key="x", search_depth="medium")

    def test_search_empty_query_short_circuits(self) -> None:
        s = TavilyScraper(api_key="tvly-fake")
        results = asyncio.run(s.search("", max_results=5))
        assert results == []

    def test_search_round_trip(self) -> None:
        fake_client = MagicMock()
        fake_client.search.return_value = {
            "results": [
                {
                    "url": "https://a.com",
                    "title": "A",
                    "content": "snippet a",
                    "score": 0.9,
                },
                {
                    "url": "https://b.com",
                    "title": "B",
                    "content": "snippet b",
                    "score": 0.8,
                },
            ]
        }
        s = TavilyScraper(api_key="tvly-fake")
        with patch("tavily.TavilyClient", return_value=fake_client):
            results = asyncio.run(s.search("hello", max_results=2))
        assert len(results) == 2
        assert results[0].url == "https://a.com"
        assert results[0].title == "A"
        assert results[0].score == 0.9

    def test_search_empty_results(self) -> None:
        fake_client = MagicMock()
        fake_client.search.return_value = {"results": []}
        s = TavilyScraper(api_key="tvly-fake")
        with patch("tavily.TavilyClient", return_value=fake_client):
            results = asyncio.run(s.search("hello"))
        assert results == []

    def test_search_drops_url_less_results(self) -> None:
        fake_client = MagicMock()
        fake_client.search.return_value = {
            "results": [
                {"url": "", "title": "no url"},
                {"url": "https://x.com", "title": "ok"},
            ]
        }
        s = TavilyScraper(api_key="tvly-fake")
        with patch("tavily.TavilyClient", return_value=fake_client):
            results = asyncio.run(s.search("hello"))
        assert len(results) == 1
        assert results[0].url == "https://x.com"

    def test_search_max_results_clamped(self) -> None:
        fake_client = MagicMock()
        fake_client.search.return_value = {"results": []}
        s = TavilyScraper(api_key="tvly-fake")
        with patch("tavily.TavilyClient", return_value=fake_client):
            asyncio.run(s.search("hello", max_results=999))
        # max_results=20 cap should have been passed to tavily client
        kwargs = fake_client.search.call_args.kwargs
        assert kwargs["max_results"] == 20

    def test_search_provider_error_translated(self) -> None:
        fake_client = MagicMock()
        fake_client.search.side_effect = RuntimeError("API down")
        s = TavilyScraper(api_key="tvly-fake")
        with (
            patch("tavily.TavilyClient", return_value=fake_client),
            pytest.raises(SearchError, match="tavily search failed"),
        ):
            asyncio.run(s.search("hello"))

    def test_lazy_client_construction(self) -> None:
        # Construction must NOT instantiate the Tavily client.
        s = TavilyScraper(api_key="tvly-fake")
        assert s._client is None

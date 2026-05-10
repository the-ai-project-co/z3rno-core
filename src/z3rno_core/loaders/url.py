"""URL loader — HTTP(S) fetch + HTML main-content extraction.

Two seams sit in this module:

  * :func:`fetch_url` — the network side. Performs the HTTP fetch with
    httpx, enforces a scheme allowlist, a per-request timeout, and a
    response-size cap. Returns the raw bytes plus the canonical
    ``Content-Type`` header so the registry can route to the right
    body-side loader.
  * :class:`UrlLoader` — the body side. Implements the standard
    :class:`Loader` interface: take HTML / plain / Markdown bytes and
    produce a :class:`LoaderResult`. For HTML we use BeautifulSoup to
    drop ``<script>`` / ``<style>`` / ``<nav>`` / ``<aside>`` /
    ``<header>`` / ``<footer>`` and prefer the ``<article>`` /
    ``<main>`` element when present.

Phase B.2 will add a Playwright-rendered fallback when JS is required.
For Phase B.1 we only handle servers that return useful HTML for an
unauthenticated GET.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from z3rno_core.loaders.base import Loader, LoaderInputError, LoaderResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network side — fetch_url
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResult:
    """Bytes + canonical headers returned from :func:`fetch_url`."""

    url: str
    content: bytes
    content_type: str
    status_code: int


class UrlFetchError(LoaderInputError):
    """The HTTP fetch itself failed (timeout, non-2xx, oversize, scheme)."""


async def fetch_url(
    url: str,
    *,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    timeout_seconds: float = 15.0,
    max_bytes: int = 50 * 1024 * 1024,
    user_agent: str = "z3rno-ingest/0.4 (+https://z3rno.dev)",
) -> FetchResult:
    """Fetch ``url`` and return its body + canonical Content-Type.

    Hard limits enforced here so the IngestPipeline never has to deal
    with a runaway response. Raises :class:`UrlFetchError` (a subclass
    of :class:`LoaderInputError`) on every failure mode.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {s.lower() for s in allowed_schemes}:
        raise UrlFetchError(
            f"scheme {parsed.scheme!r} not allowed (allowed: {','.join(allowed_schemes)})"
        )
    if not parsed.netloc:
        raise UrlFetchError(f"URL missing host: {url!r}")

    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        ) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise UrlFetchError(f"timeout fetching {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise UrlFetchError(f"transport error fetching {url}: {exc}") from exc

    if response.status_code >= _HTTP_ERROR_THRESHOLD:
        raise UrlFetchError(
            f"HTTP {response.status_code} fetching {url}: {response.reason_phrase or ''}".strip()
        )

    body = response.content
    if len(body) > max_bytes:
        raise UrlFetchError(
            f"response from {url} is {len(body)} bytes, exceeds max_bytes={max_bytes}"
        )

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not content_type:
        # Best-effort default for unlabeled responses.
        content_type = "application/octet-stream"

    return FetchResult(
        url=str(response.url),
        content=body,
        content_type=content_type,
        status_code=response.status_code,
    )


# ---------------------------------------------------------------------------
# Body side — UrlLoader
# ---------------------------------------------------------------------------


# Tags whose contents are noise in an article view.
_DROP_TAGS = ("script", "style", "noscript", "nav", "aside", "header", "footer", "form")

# HTTP status codes >= this are treated as failures.
_HTTP_ERROR_THRESHOLD = 400


# ---------------------------------------------------------------------------
# Playwright fallback — Phase B.2 opt-in for JS-rendered pages
# ---------------------------------------------------------------------------


async def render_with_playwright(
    url: str,
    *,
    timeout_seconds: float = 30.0,
) -> str:
    """Render ``url`` in headless Chromium and return the resolved DOM.

    Imported lazily so the rest of the URL loader keeps working even
    when the optional ``[playwright]`` extra isn't installed. Raises
    :class:`UrlFetchError` if Playwright isn't importable, or if the
    browser/page step fails.
    """
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError as exc:
        raise UrlFetchError(
            "Playwright fallback requested but `playwright` is not installed. "
            "Install with `pip install 'z3rno-core[playwright]'` and run "
            "`playwright install chromium` once."
        ) from exc

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=int(timeout_seconds * 1000))
                await page.wait_for_load_state("networkidle", timeout=int(timeout_seconds * 1000))
                html = await page.content()
            finally:
                await browser.close()
        return str(html or "")
    except Exception as exc:
        raise UrlFetchError(f"playwright render failed for {url}: {exc}") from exc


class UrlLoader(Loader):
    """HTML / plain text / Markdown body loader.

    Routes to BeautifulSoup for ``text/html``; for ``text/plain`` and
    ``text/markdown`` it decodes UTF-8 and returns the body as-is.
    """

    @property
    def name(self) -> str:
        return "url"

    async def load(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> LoaderResult:
        if not content:
            return _empty_result(self.name, filename, mime_type or "text/html")

        normalized_mime = (mime_type or "text/html").split(";", 1)[0].strip().lower()

        if normalized_mime in {"text/plain", "text/markdown", "text/x-markdown"}:
            text = content.decode("utf-8", errors="replace")
            return LoaderResult(
                text=text,
                metadata=_meta(self.name, filename, normalized_mime, content, text),
            )

        # Default to HTML extraction for text/html and any unknown type
        # (we already know the content was fetched successfully).
        return await self._load_html(content, filename, normalized_mime)

    async def _load_html(
        self,
        content: bytes,
        filename: str | None,
        mime_type: str,
    ) -> LoaderResult:
        try:
            soup = BeautifulSoup(content, "html.parser")
        except Exception as exc:
            raise LoaderInputError(f"malformed HTML: {exc}") from exc

        # Strip noise tags entirely.
        for tag_name in _DROP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Prefer the article-shaped containers when present; fall back to body.
        primary = soup.find("article") or soup.find("main") or soup.body or soup

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # get_text with separators preserves block-level boundaries so the
        # paragraph chunker has something to work with.
        body_text = primary.get_text(separator="\n", strip=True)

        if title and not body_text.startswith(title):
            text = f"# {title}\n\n{body_text}".strip()
        else:
            text = body_text.strip()

        meta = _meta(self.name, filename, mime_type, content, text)
        if title:
            meta["title"] = title
        return LoaderResult(text=text, metadata=meta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_result(loader_name: str, filename: str | None, mime_type: str) -> LoaderResult:
    return LoaderResult(
        text="",
        metadata={
            "loader": loader_name,
            "mime_type": mime_type,
            "filename": filename,
            "byte_size": 0,
            "char_count": 0,
        },
    )


def _meta(
    loader_name: str,
    filename: str | None,
    mime_type: str,
    content: bytes,
    text: str,
) -> dict[str, object]:
    return {
        "loader": loader_name,
        "mime_type": mime_type,
        "filename": filename,
        "byte_size": len(content),
        "char_count": len(text),
    }

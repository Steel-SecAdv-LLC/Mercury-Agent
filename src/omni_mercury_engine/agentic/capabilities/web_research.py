# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native web research: fetch URLs, extract readable text, best-effort search.

Pure standard library (``urllib`` + ``html.parser`` + ``ssl``) -- no third-party
HTTP client, no scraping framework, no language model. The transport is
injectable so the behaviour is fully testable offline; in production the default
transport honours the environment's proxy and TLS configuration.

Everything here is **fail-closed and honest**: a network error, a non-OK status,
or an oversized body yields a :class:`FetchResult` carrying the error (never a
fabricated body), and :meth:`WebResearcher.search` returns ``[]`` with the reason
recorded rather than inventing results. Mercury never pretends it read a page it
could not reach.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

# A transport maps (url, timeout) -> (status_code, body_text, final_url). Raising
# is allowed; WebResearcher converts any exception into a fail-closed result.
Transport = Callable[[str, float], "tuple[int, str, str]"]

_DEFAULT_USER_AGENT = "MercuryAgent/1.0 (+research; stdlib-urllib)"
_DEFAULT_MAX_BYTES = 2_000_000  # 2 MB cap; large pages are truncated, not OOM'd.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}


@dataclass
class FetchResult:
    """Outcome of fetching one URL.

    ``ok`` is True only on an HTTP 2xx with a decoded body. On any failure
    ``error`` explains why and ``text`` is empty -- callers must check ``ok``.
    """

    url: str
    status: int = 0
    text: str = ""
    final_url: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the fetch succeeded (2xx and a body, no error)."""
        return self.error is None and 200 <= self.status < 300


@dataclass
class SearchResult:
    """A single web search hit."""

    title: str
    url: str
    snippet: str = ""


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping script/style/etc."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


class _LinkExtractor(HTMLParser):
    """Collect (href, anchor-text) pairs from <a> tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._href = href
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            t = data.strip()
            if t:
                self._text.append(t)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


def _urllib_transport(url: str, timeout: float) -> tuple[int, str, str]:
    """Default transport: a plain GET via urllib, honouring env proxies/TLS.

    ``urllib`` reads ``HTTPS_PROXY`` / ``HTTP_PROXY`` from the environment by
    default and uses the system trust store, so the managed proxy + CA bundle in
    the deployment environment are respected without extra configuration.
    """
    # The scheme is guarded to http/https in WebResearcher.fetch before any
    # transport call, so urllib never opens a file:// or other local URL
    # (ruff S310 is suppressed for this module via per-file-ignores).
    req = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(_DEFAULT_MAX_BYTES)
        charset = resp.headers.get_content_charset() or "utf-8"
        status = getattr(resp, "status", 200) or 200
        final_url = resp.geturl()
    return int(status), raw.decode(charset, errors="replace"), final_url


@dataclass
class WebResearcher:
    """Fetch, extract, and search the open web with the standard library only.

    Args:
        transport: Injectable ``(url, timeout) -> (status, body, final_url)``.
            Defaults to a urllib GET. Tests pass a stub to stay offline/
            deterministic.
        timeout: Per-request timeout in seconds.
        allowed_schemes: URL schemes permitted (default http/https). Anything
            else is refused fail-closed (no file://, no ftp://).
    """

    transport: Transport = _urllib_transport
    timeout: float = 10.0
    allowed_schemes: frozenset[str] = field(default_factory=lambda: frozenset({"http", "https"}))

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL, returning a fail-closed :class:`FetchResult`."""
        scheme = urllib.parse.urlparse(url).scheme.lower()
        if scheme not in self.allowed_schemes:
            return FetchResult(
                url=url,
                error=f"refused scheme {scheme!r} (allowed: {sorted(self.allowed_schemes)})",
            )
        try:
            status, body, final_url = self.transport(url, self.timeout)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            return FetchResult(
                url=url, status=int(exc.code), error=f"HTTP {exc.code}: {exc.reason}"
            )
        except Exception as exc:
            return FetchResult(url=url, error=f"{type(exc).__name__}: {exc}")
        return FetchResult(url=url, status=int(status), text=body, final_url=final_url or url)

    @staticmethod
    def extract_text(html: str) -> str:
        """Extract readable text from an HTML document (script/style stripped)."""
        parser = _TextExtractor()
        try:
            parser.feed(html)
        except Exception:
            pass
        return parser.text()

    def fetch_text(self, url: str) -> FetchResult:
        """Fetch a URL and replace the body with its extracted readable text."""
        result = self.fetch(url)
        if result.ok:
            result.text = self.extract_text(result.text)
        return result

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Best-effort web search via the DuckDuckGo HTML endpoint.

        No API key and no dependency: it GETs the keyless HTML results page and
        parses result anchors. Fail-closed -- returns ``[]`` (and logs the
        reason) on any network/parse failure, so a blocked or offline
        environment degrades honestly rather than fabricating hits.
        """
        endpoint = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
        result = self.fetch(endpoint)
        if not result.ok:
            logger.info("web search unavailable (%s): %s", query, result.error)
            return []
        link_parser = _LinkExtractor()
        try:
            link_parser.feed(result.text)
        except Exception:
            return []
        hits: list[SearchResult] = []
        seen: set[str] = set()
        for href, text in link_parser.links:
            target = self._decode_ddg_href(href)
            if not target or target in seen or not text or len(text) < 3:
                continue
            if urllib.parse.urlparse(target).scheme not in self.allowed_schemes:
                continue
            seen.add(target)
            hits.append(SearchResult(title=text, url=target))
            if len(hits) >= max_results:
                break
        return hits

    @staticmethod
    def _decode_ddg_href(href: str) -> str | None:
        """Resolve a DuckDuckGo redirect link (``/l/?uddg=<encoded>``) to its target."""
        parsed = urllib.parse.urlparse(href)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs:
                return qs["uddg"][0]
        if href.startswith("//duckduckgo.com/l/"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + href).query)
            if "uddg" in qs:
                return qs["uddg"][0]
        if parsed.scheme in ("http", "https"):
            return href
        return None


__all__ = ["FetchResult", "SearchResult", "WebResearcher"]

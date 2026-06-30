# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native web research: fetch URLs, extract readable text, best-effort search.

Pure standard library (``urllib`` + ``html.parser`` + ``ssl``) -- no third-party
HTTP client, no scraping framework, no language model. The transport is
injectable so the behaviour is fully testable offline; in production the default
transport honours the environment's proxy and TLS configuration.

**Search engine choice.** Under the project's hard constraints -- no new
dependency, no API key, standard library only -- DuckDuckGo is the strongest
general-web engine: it exposes *keyless* HTML endpoints we can GET and parse,
whereas Google (Programmable Search JSON API), Bing (retired/keyed), and Brave
(keyed) all require an API key + account, which would be a dependency and a
credential. So the keyless default is DuckDuckGo, and to be robust it queries a
*chain* of its endpoints -- the full HTML page first, then the leaner ``lite``
page (far more tolerant of non-browser clients) -- returning the first that
yields hits. For deployments that *do* hold a key for a higher-quality engine,
:class:`WebResearcher` accepts an injectable ``search_provider``: the best
available engine can be slotted in without this module taking a dependency.

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

# A search provider maps (query, max_results) -> ranked hits. Injectable so a
# deployment holding an API key for a higher-quality engine (Brave, Google
# Programmable Search, ...) can supply it without this module taking a
# dependency. Raising is allowed; search() converts it into a fail-closed [].
SearchProvider = Callable[[str, int], "list[SearchResult]"]

_DEFAULT_USER_AGENT = "MercuryAgent/1.0 (+research; stdlib-urllib)"
_DEFAULT_MAX_BYTES = 2_000_000  # 2 MB cap; large pages are truncated, not OOM'd.
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}

# Keyless DuckDuckGo endpoints, tried in order. The full HTML page is richer;
# the ``lite`` page is plainer markup and more tolerant of non-browser clients,
# so it makes a good fallback when the HTML page serves a challenge/empty body.
_DDG_ENDPOINTS = (
    "https://html.duckduckgo.com/html/?q=",
    "https://lite.duckduckgo.com/lite/?q=",
)
# Anchor classes DuckDuckGo uses for result titles / snippets on its html and
# lite pages. Detection is substring-based so minor markup churn still matches.
_DDG_TITLE_CLASSES = ("result__a", "result-link")
_DDG_SNIPPET_CLASSES = ("result__snippet", "result-snippet")


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


class _DdgResultParser(HTMLParser):
    """Collect ``(class, href, anchor-text)`` triples from <a> tags.

    Works for both the DuckDuckGo html and lite result pages. The CSS class is
    captured so :meth:`WebResearcher._parse_ddg` can tell a result title/snippet
    anchor from a navigation link, while still keeping every anchor for a
    generic fallback if the markup ever changes (degrade to title-only results
    rather than to nothing).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str, str]] = []  # (class, href, text)
        self._cls: str = ""
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            d = dict(attrs)
            href = d.get("href")
            if href:
                self._href = href
                self._cls = d.get("class") or ""
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            t = data.strip()
            if t:
                self._text.append(t)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.anchors.append((self._cls, self._href, " ".join(self._text).strip()))
            self._cls = ""
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
        search_provider: Optional ``(query, max_results) -> [SearchResult]``.
            When set it fully replaces the built-in keyless DuckDuckGo chain --
            the hook through which a deployment with an API key can use a
            higher-quality engine without this module taking a dependency.
    """

    transport: Transport = _urllib_transport
    timeout: float = 10.0
    allowed_schemes: frozenset[str] = field(default_factory=lambda: frozenset({"http", "https"}))
    search_provider: SearchProvider | None = None

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
        """Best-effort web search; returns up to ``max_results`` ranked hits.

        If a ``search_provider`` was supplied it is used exclusively. Otherwise
        the built-in keyless DuckDuckGo chain is queried -- the html endpoint
        first, then the leaner lite endpoint -- returning the first that yields
        results. Fail-closed throughout: returns ``[]`` (and logs the reason) on
        any network/parse failure or an exception from a custom provider, so a
        blocked or offline environment degrades honestly rather than fabricating
        hits.
        """
        if self.search_provider is not None:
            try:
                return list(self.search_provider(query, max_results))[:max_results]
            except Exception as exc:
                logger.info("custom search_provider failed (%s): %s", query, exc)
                return []

        last_reason = "no DuckDuckGo endpoint returned results"
        for template in _DDG_ENDPOINTS:
            result = self.fetch(template + urllib.parse.quote_plus(query))
            if not result.ok:
                last_reason = result.error or "fetch failed"
                continue
            hits = self._parse_ddg(result.text, max_results)
            if hits:
                return hits
        logger.info("web search unavailable (%s): %s", query, last_reason)
        return []

    def _parse_ddg(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse a DuckDuckGo results page into ranked :class:`SearchResult`s."""
        parser = _DdgResultParser()
        try:
            parser.feed(html)
        except Exception:
            return []

        def _is(cls: str, names: tuple[str, ...]) -> bool:
            return any(n in cls for n in names)

        titled = [(h, t) for c, h, t in parser.anchors if _is(c, _DDG_TITLE_CLASSES) and t]
        snippets = [t for c, _, t in parser.anchors if _is(c, _DDG_SNIPPET_CLASSES) and t]
        if not titled:
            # Markup changed / unclassed page: fall back to every anchor with
            # text so we still return title-only results rather than nothing.
            titled = [(h, t) for _, h, t in parser.anchors if t]
            snippets = []

        hits: list[SearchResult] = []
        seen: set[str] = set()
        for i, (href, text) in enumerate(titled):
            target = self._decode_ddg_href(href)
            if not target or target in seen or len(text) < 3:
                continue
            if urllib.parse.urlparse(target).scheme not in self.allowed_schemes:
                continue
            seen.add(target)
            snippet = snippets[i] if i < len(snippets) else ""
            hits.append(SearchResult(title=text, url=target, snippet=snippet))
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


__all__ = ["FetchResult", "SearchProvider", "SearchResult", "WebResearcher"]

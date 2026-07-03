# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native web research: fetch URLs, extract readable text, best-effort search.

HTML parsing and extraction are pure standard library (``html.parser``); no
scraping framework and no language model. All outbound HTTP is routed through
Mercury's :class:`~omni_mercury_engine.security.safe_http.SafeHTTPClient` -- the
single sanctioned, **SSRF-guarded** egress path (it validates the URL, resolves
and re-checks the IP to refuse private/link-local/IMDS targets, pins the
connection to the validated IP against DNS-rebinding, and refuses redirects).
This matters because search results are *untrusted* URLs: a hostile result that
points at ``http://169.254.169.254/`` (cloud metadata) or an internal host must
never be fetched. The transport is injectable so the behaviour is fully testable
offline; in production the default transport honours the environment's proxy and
TLS configuration via ``requests``.

**Search backend -- a provider ladder, not a single scrape.** Search is a
*ranked ladder of providers*, tried in order, each fail-closed. The recommended
rungs are robust and operator-owned; HTML scraping is the explicit last resort,
not the default:

1. **A keyed engine** (Brave, Google Programmable Search, ...), if the operator
   holds a key -- the highest-quality, contractually-stable option.
2. **A self-hosted SearXNG** (``MERCURY_SEARXNG_URL``) -- *keyless* and
   *self-hostable*, so it adds no SaaS dependency and runs fully under the
   operator's control. This is the preferred default for an offline-leaning
   deployment: pair it with the local Ollama reasoning backend and Mercury's
   open-web research needs no third-party credential at all.
3. **Keyless DuckDuckGo HTML scrape** -- a *best-effort fallback only*. Scrape
   endpoints rate-limit, serve challenges, and change markup without notice, so
   relying on them as a primary backend invites silent breakage; they are the
   bottom rung, enabled by default so a zero-config install still returns
   *something*, but never positioned as the recommended path. Disable with
   ``enable_ddg_fallback=False``.

Build the ladder explicitly via ``search_providers=[...]`` or from the
environment via :meth:`WebResearcher.from_env`. The legacy singular
``search_provider`` (a single ``(query, max_results) -> [SearchResult]``) still
fully replaces the built-in chain, for backward compatibility.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from omni_mercury_engine.agentic.capabilities.contract import Invariant, capability_contract

logger = logging.getLogger(__name__)

# A transport maps (url, timeout) -> (status_code, body_text, final_url). Raising
# is allowed; WebResearcher converts any exception into a fail-closed result.
Transport = Callable[[str, float], "tuple[int, str, str]"]

# A search provider maps (query, max_results) -> ranked hits. Injectable so a
# deployment holding an API key for a higher-quality engine (Brave, Google
# Programmable Search, ...) can supply it without this module taking a
# dependency. Raising is allowed; search() converts it into a fail-closed [].
SearchProvider = Callable[[str, int], "list[SearchResult]"]

# A content gate maps extracted page text -> (blocked, note). Injectable so the
# post-retrieval harm screen (spec §5.2) rides on every fetch without this
# transport module importing the ethics layer. Fetched web content is
# *untrusted*: a benign query can still return operational weapons procedure,
# and the verbatim synthesizer downstream would faithfully reproduce it. Any
# error in the gate is treated as "blocked" by fetch_text (fail-closed).
ContentGate = Callable[[str], "tuple[bool, str]"]

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
    # Post-retrieval harm screen (spec §5.2). ``screened`` records whether a
    # content gate ran at all (so a consumer can tell "gate said safe" from "no
    # gate configured"); ``harm_blocked`` is the gate's verdict; ``harm_note``
    # is its human-readable reason for audit. Defaults keep an ungated fetch
    # (no content_gate) fully backward-compatible.
    screened: bool = False
    harm_blocked: bool = False
    harm_note: str = ""

    @property
    def ok(self) -> bool:
        """Whether the fetch succeeded (2xx and a body, no error)."""
        return self.error is None and 200 <= self.status < 300

    @property
    def usable(self) -> bool:
        """Whether the content may be consumed: fetched OK *and* not harm-blocked."""
        return self.ok and not self.harm_blocked


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


def _host_is_loopback_literal(url: str) -> bool:
    """True when ``url``'s host is a loopback literal (localhost / 127.x / ::1)."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def _safe_http_transport(url: str, timeout: float) -> tuple[int, str, str]:
    """Default transport: a validated GET via Mercury's SSRF-guarded SafeHTTPClient.

    The open web is *untrusted*, so this uses the SSRF-guarded egress path --
    private/link-local/IMDS/loopback targets are refused. ``user_configured=True``
    is required and deliberate: the arbitrary open-web hosts a research fetch
    targets are, by design, NOT in :attr:`TrustedEndpoints.TRUSTED_DOMAINS`
    (that allowlist is for pinned dataset/API mirrors). Without this flag every
    real fetch would fail closed against the dataset allowlist AND -- for an
    https URL -- skip the resolve-and-recheck IP gate entirely. Setting it
    bypasses the *dataset* allowlist while activating the private-network / IMDS
    IP gate that actually protects the open-web path (it matches the
    ``operator_hosted=False`` JSON-opener policy above). ``allow_http=True``
    because the open web includes plain-``http``: pages (still IP-gated). Body is
    bounded to ``_DEFAULT_MAX_BYTES``; ``requests`` honours
    ``HTTPS_PROXY``/``HTTP_PROXY`` and the system trust store.
    """
    import requests

    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    with SafeHTTPClient.get(
        url,
        headers={"User-Agent": _DEFAULT_USER_AGENT},
        timeout=timeout,
        allow_http=True,
        user_configured=True,
        stream=True,
    ) as resp:
        status = int(resp.status_code)
        # Explicit DECODED-size cap (decompression-bomb mitigation). Stream the
        # already-decompressed body in bounded chunks and stop at
        # ``_DEFAULT_MAX_BYTES`` of *decoded* output. ``iter_content`` decodes the
        # Content-Encoding (gzip/deflate/br) incrementally via urllib3's
        # ``stream(..., decode_content=True)``, so a small compressed payload that
        # inflates to gigabytes never fully materialises: the decompressor yields
        # ~chunk-sized decoded blocks and we abort once the cap is reached. This
        # bounds the peak regardless of the urllib3 version's ``read(amt)``
        # semantics -- we no longer rely on ``read`` honouring ``amt`` as a
        # decoded bound (older urllib3 measured ``amt`` against the *compressed*
        # stream, which a gzip bomb defeats). The urllib3 floor pin in
        # pyproject is defence in depth; this cap is the primary control.
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= _DEFAULT_MAX_BYTES:
                break
        raw = b"".join(chunks)[:_DEFAULT_MAX_BYTES]
        # ``requests`` reports ``ISO-8859-1`` as its *no-charset-in-header*
        # sentinel for ``text/*`` -- trusting it mojibakes the UTF-8 majority of
        # the open web (and DDG's own scrape pages). Use the header charset only
        # when one was actually declared; otherwise sniff the real bytes
        # (``apparent_encoding``), defaulting to UTF-8.
        header_charset = requests.utils.get_encoding_from_headers(resp.headers)
        if header_charset and header_charset.lower() != "iso-8859-1":
            charset = header_charset
        else:
            resp._content = raw  # so apparent_encoding sees the body we just read
            charset = resp.apparent_encoding or "utf-8"
        final_url = resp.url or url
    return status, raw.decode(charset, errors="replace"), final_url


def default_content_gate(text: str, harm_classifier: Any | None = None) -> tuple[bool, str]:
    """Default post-retrieval harm screen: the two-axis weapons/mass-casualty gate.

    Returns ``(blocked, note)``. Blocks fetched content whose weapons
    disposition is anything other than plain ALLOW/ALLOW_LOG -- i.e. a page
    carrying operational production/weaponization/acquisition-evasion/
    enhancement/targeting content, even when the query that surfaced it was
    benign. ``harm_classifier`` (optional) wires the reasoning-backed semantic
    layer through to the gate. The ethics import is lazy so :mod:`web_research`
    stays importable without the cognitive layer; any failure fails closed to
    *blocked*.
    """
    try:
        from omni_mercury_engine.cognitive.ethical_bounding import assess_weapons_uplift

        verdict = assess_weapons_uplift(text, harm_classifier=harm_classifier)
        if verdict.blocks:
            return True, (
                f"weapons/mass-casualty content gate: {verdict.disposition.value} "
                f"(hazard={verdict.hazard_domain.value}, intent={verdict.intent_tier.value})"
            )
        return False, ""
    except Exception as exc:  # pragma: no cover - defensive fail-closed
        return True, f"content gate error (fail-closed): {exc}"


@dataclass
class WebResearcher:
    """Fetch, extract, and search the open web with the standard library only.

    Args:
        transport: Injectable ``(url, timeout) -> (status, body, final_url)``.
            Defaults to an SSRF-guarded SafeHTTPClient GET. Tests pass a stub to
            stay offline/deterministic.
        timeout: Per-request timeout in seconds.
        allowed_schemes: URL schemes permitted (default http/https). Anything
            else is refused fail-closed (no file://, no ftp://).
        search_providers: Ordered ladder of providers tried in turn (first with
            hits wins), each fail-closed. The recommended way to configure
            search: put a keyed engine and/or a self-hosted SearXNG ahead of the
            keyless DuckDuckGo fallback. See :func:`brave_provider`,
            :func:`searxng_provider`, and :meth:`from_env`.
        enable_ddg_fallback: Whether the keyless DuckDuckGo HTML scrape is the
            final rung when every configured provider yields nothing. Default
            ``True`` so a zero-config install still returns results; set
            ``False`` to refuse scraping entirely (provider-only).
        search_provider: Legacy single ``(query, max_results) -> [SearchResult]``.
            When set it fully replaces the built-in chain (backward-compatible);
            prefer ``search_providers`` for new code.
    """

    transport: Transport = _safe_http_transport
    timeout: float = 10.0
    allowed_schemes: frozenset[str] = field(default_factory=lambda: frozenset({"http", "https"}))
    search_provider: SearchProvider | None = None
    search_providers: tuple[SearchProvider, ...] = ()
    enable_ddg_fallback: bool = True
    content_gate: ContentGate | None = None

    @classmethod
    def from_env(cls, **kwargs: Any) -> WebResearcher:
        """Build a researcher whose search ladder is configured from the environment.

        Recognised variables (all optional):

        * ``BRAVE_API_KEY`` -- prepend a keyed Brave provider (highest priority).
        * ``MERCURY_SEARXNG_URL`` -- add a keyless self-hosted SearXNG provider.
        * ``MERCURY_SEARCH_DDG_FALLBACK`` -- ``"0"``/``"false"`` disables the
          DuckDuckGo scrape fallback (default enabled).

        The resulting ladder is ``[Brave?, SearXNG?]`` with the DuckDuckGo scrape
        as the (optional) final rung -- the recommended, provider-first ordering.
        Any keyword overrides are forwarded to the constructor.
        """
        import functools
        import os

        # Optional reasoning-backed harm classifier to thread through the content
        # gate (wired by the GeneralAssistant on the open-web surface). Popped
        # here because it is not a WebResearcher field -- it is bound into the
        # content-gate closure instead.
        harm_classifier = kwargs.pop("harm_classifier", None)

        providers: list[SearchProvider] = []
        brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
        if brave_key:
            providers.append(brave_provider(brave_key))
        searxng_url = os.environ.get("MERCURY_SEARXNG_URL", "").strip()
        if searxng_url:
            providers.append(searxng_provider(searxng_url))
        ddg = os.environ.get("MERCURY_SEARCH_DDG_FALLBACK", "1").strip().lower()
        enable_ddg = ddg not in {"0", "false", "no", "off"}
        kwargs.setdefault("search_providers", tuple(providers))
        kwargs.setdefault("enable_ddg_fallback", enable_ddg)
        # Wire the default post-retrieval harm screen unless the caller supplied
        # one. Keeps this module dependency-free (the ethics import is lazy,
        # inside the gate) while giving every from_env researcher the §5.2
        # content screen for free, with the reasoning classifier bound through.
        kwargs.setdefault(
            "content_gate", functools.partial(default_content_gate, harm_classifier=harm_classifier)
        )
        return cls(**kwargs)

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
            # Preserve the numeric status when the transport raised an HTTP error
            # carrying one (``requests.exceptions.HTTPError`` exposes it via
            # ``exc.response.status_code``); fail-closed to 0 otherwise.
            http_status = getattr(getattr(exc, "response", None), "status_code", 0) or 0
            return FetchResult(
                url=url, status=int(http_status), error=f"{type(exc).__name__}: {exc}"
            )
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

    @capability_contract(
        Invariant.FAIL_CLOSED,
        on_error=lambda exc, args, kwargs: FetchResult(
            url=str(args[1]) if len(args) > 1 else str(kwargs.get("url", "")),
            error=(
                "fail-closed: fetch aborted by an unexpected error "
                f"({type(exc).__name__}: {exc})"
            ),
        ),
    )
    def fetch_text(self, url: str) -> FetchResult:
        """Fetch a URL, extract readable text, and run the post-retrieval harm screen.

        When a :attr:`content_gate` is configured, the extracted text is
        classified before the result is returned so the harm verdict travels
        with the :class:`FetchResult` for *every* consumer (spec §5.2), not
        just callers that remember to screen. Fail-closed: a gate that raises
        marks the content blocked rather than passing it through.
        """
        result = self.fetch(url)
        if result.ok:
            result.text = self.extract_text(result.text)
            if self.content_gate is not None and result.text:
                result.screened = True
                try:
                    blocked, note = self.content_gate(result.text)
                except Exception as exc:
                    result.harm_blocked = True
                    result.harm_note = f"content gate error (fail-closed): {exc}"
                else:
                    result.harm_blocked = bool(blocked)
                    result.harm_note = str(note)
        return result

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Best-effort web search over the provider ladder; up to ``max_results`` hits.

        Resolution order:

        1. If the legacy singular ``search_provider`` is set, it is used
           *exclusively* (fully replaces the chain), fail-closed to ``[]`` -- an
           operator who pinned one engine does not want a silent fallback to a
           different one.
        2. Otherwise each provider in ``search_providers`` is tried in order; the
           first that returns hits wins. Each rung is fail-closed (an exception
           yields no hits and moves to the next rung).
        3. If no provider yielded hits and ``enable_ddg_fallback`` is set, the
           keyless DuckDuckGo HTML->lite scrape chain is the final rung.

        Fail-closed throughout: returns ``[]`` (and logs the reason) on any
        network/parse failure, so a blocked or offline environment degrades
        honestly rather than fabricating hits.
        """
        # (1) Legacy singular provider fully replaces the chain (back-compat).
        if self.search_provider is not None:
            try:
                return list(self.search_provider(query, max_results))[:max_results]
            except Exception as exc:
                logger.info("custom search_provider failed (%s): %s", query, exc)
                return []

        # (2) Provider ladder, each rung fail-closed.
        for i, provider in enumerate(self.search_providers):
            try:
                hits = list(provider(query, max_results))[:max_results]
            except Exception as exc:
                logger.info("search provider %d failed (%s): %s", i, query, exc)
                continue
            if hits:
                return hits

        # (3) Keyless DuckDuckGo scrape -- explicit best-effort last resort.
        if self.enable_ddg_fallback:
            return self._ddg_search(query, max_results)
        logger.info(
            "web search: no provider returned results and DDG fallback disabled (%s)", query
        )
        return []

    def _ddg_search(self, query: str, max_results: int) -> list[SearchResult]:
        """Keyless DuckDuckGo html->lite scrape chain (the fallback rung)."""
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
    def _is_ddg_host(netloc: str) -> bool:
        """Exact-match the DuckDuckGo host (or a subdomain of it).

        Never a substring containment, which a host like
        ``duckduckgo.com.evil.test`` or ``notduckduckgo.com`` would
        otherwise satisfy.
        """
        host = netloc.rsplit("@", 1)[-1].split(":", 1)[0].lower().rstrip(".")
        return host == "duckduckgo.com" or host.endswith(".duckduckgo.com")

    @classmethod
    def _decode_ddg_href(cls, href: str) -> str | None:
        """Resolve a DuckDuckGo redirect link (``/l/?uddg=<encoded>``) to its target."""
        parsed = urllib.parse.urlparse(href)
        if cls._is_ddg_host(parsed.netloc) and parsed.path.startswith("/l/"):
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs:
                return qs["uddg"][0]
        if href.startswith("//"):
            protocol_relative = urllib.parse.urlparse("https:" + href)
            if cls._is_ddg_host(protocol_relative.netloc) and protocol_relative.path.startswith(
                "/l/"
            ):
                qs = urllib.parse.parse_qs(protocol_relative.query)
                if "uddg" in qs:
                    return qs["uddg"][0]
        if parsed.scheme in ("http", "https"):
            return href
        return None


# ---------------------------------------------------------------------------
# Built-in search providers (the recommended rungs of the ladder). Each is a
# factory returning a ``SearchProvider`` -- a ``(query, max_results) ->
# [SearchResult]`` callable -- and is fail-closed (any error yields ``[]`` so the
# ladder moves to the next rung). Both route HTTP through the SSRF-guarded
# SafeHTTPClient and accept an injectable ``opener`` so they are testable offline.
# ---------------------------------------------------------------------------

# An opener maps a (url, headers) GET to the decoded response body. Injectable so
# a keyed/self-hosted provider is testable without network.
JsonOpener = Callable[[str, "dict[str, str]"], str]


def _make_json_opener(*, operator_hosted: bool) -> JsonOpener:
    """Build a JSON GET opener with the right SSRF policy for its host class.

    ``operator_hosted=True`` (a self-hosted SearXNG the deployer points wherever
    its own infra lives) trusts the operator's network: a localhost instance via
    ``loopback_only``, a LAN host via ``allow_private``. ``operator_hosted=False``
    (a *public* keyed cloud engine such as Brave) keeps the **strict default**
    SSRF policy -- private/RFC1918/IMDS all refused -- so a public provider host
    that DNS-rebinds to an internal address is still blocked. Either way the GET
    goes through the SSRF-guarded SafeHTTPClient (no raw ``urlopen``), which
    refuses redirects and re-checks the resolved IP.
    """

    def _opener(url: str, headers: dict[str, str]) -> str:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        merged = {"User-Agent": _DEFAULT_USER_AGENT, "Accept": "application/json"}
        merged.update(headers)
        kwargs: dict[str, Any] = {
            "headers": merged,
            "timeout": 15.0,
            "allow_http": True,
            "user_configured": True,
        }
        if operator_hosted:
            if _host_is_loopback_literal(url):
                kwargs["loopback_only"] = True
            else:
                kwargs["allow_private"] = True
        return SafeHTTPClient.get_text(url, **kwargs)

    return _opener


# SearXNG is operator-hosted (often localhost/LAN) -> trust the operator's infra.
# A keyed cloud engine (Brave) is a public host -> strict default SSRF policy.
_SEARXNG_JSON_OPENER = _make_json_opener(operator_hosted=True)
_PUBLIC_JSON_OPENER = _make_json_opener(operator_hosted=False)


def searxng_provider(
    base_url: str,
    *,
    opener: JsonOpener | None = None,
    extra_params: dict[str, str] | None = None,
) -> SearchProvider:
    """A keyless, self-hostable SearXNG provider (the recommended offline rung).

    Queries ``{base_url}/search?q=...&format=json`` -- a SearXNG instance the
    operator runs, so there is no API key and no third-party SaaS dependency.
    Pair it with the local Ollama reasoning backend for fully operator-owned,
    offline-leaning open-web research.

    Args:
        base_url: Base URL of the SearXNG instance (e.g. ``http://localhost:8080``).
        opener: Injectable ``(url, headers) -> body`` (defaults to a urllib GET).
        extra_params: Extra query parameters (e.g. ``{"language": "en"}``).
    """
    import json

    base = base_url.rstrip("/")
    get = opener or _SEARXNG_JSON_OPENER

    def _provider(query: str, max_results: int) -> list[SearchResult]:
        params = {"q": query, "format": "json"}
        if extra_params:
            params.update(extra_params)
        url = f"{base}/search?{urllib.parse.urlencode(params)}"
        try:
            data = json.loads(get(url, {}))
        except Exception as exc:
            logger.info("searxng provider failed (%s): %s", query, exc)
            return []
        out: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            link = str(item.get("url", "")).strip()
            if urllib.parse.urlparse(link).scheme not in {"http", "https"}:
                continue
            out.append(
                SearchResult(
                    title=str(item.get("title", "")).strip() or link,
                    url=link,
                    snippet=str(item.get("content", "")).strip(),
                )
            )
        return out

    return _provider


def brave_provider(api_key: str, *, opener: JsonOpener | None = None) -> SearchProvider:
    """A keyed Brave Search provider (the highest-quality, contractually-stable rung).

    Uses the operator's Brave Search API key. Fail-closed: any error yields
    ``[]`` so the ladder falls through to the next rung.

    Args:
        api_key: Brave Search API subscription token.
        opener: Injectable ``(url, headers) -> body``. Defaults to the strict
            public-host opener (Brave's host is public, so the default SSRF
            policy applies -- private/IMDS refused).
    """
    import json

    get = opener or _PUBLIC_JSON_OPENER

    def _provider(query: str, max_results: int) -> list[SearchResult]:
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
            {"q": query, "count": max_results}
        )
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
        try:
            data = json.loads(get(url, headers))
        except Exception as exc:
            logger.info("brave provider failed (%s): %s", query, exc)
            return []
        out: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            link = str(item.get("url", "")).strip()
            if urllib.parse.urlparse(link).scheme not in {"http", "https"}:
                continue
            out.append(
                SearchResult(
                    title=str(item.get("title", "")).strip() or link,
                    url=link,
                    snippet=str(item.get("description", "")).strip(),
                )
            )
        return out

    return _provider


__all__ = [
    "ContentGate",
    "FetchResult",
    "JsonOpener",
    "SearchProvider",
    "SearchResult",
    "WebResearcher",
    "brave_provider",
    "default_content_gate",
    "searxng_provider",
]

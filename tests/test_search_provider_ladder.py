# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the search provider ladder: provider-first resolution with the keyless DuckDuckGo scrape demoted to an explicit best-effort fallback, plus the keyless SearXNG and keyed Brave provider factories and env-driven wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.agentic.capabilities.web_research import (
    SearchResult,
    WebResearcher,
    brave_provider,
    searxng_provider,
)

if TYPE_CHECKING:
    import pytest

_DDG_HTML = """
<html><body>
<a class="result__a" href="https://example.org/a">Result A</a>
<a class="result__snippet" href="https://example.org/a">snippet a</a>
</body></html>
"""


def _ddg_transport(url: str, timeout: float) -> tuple[int, str, str]:
    return 200, _DDG_HTML, url


def _hit(title: str) -> list[SearchResult]:
    return [SearchResult(title=title, url=f"https://x/{title}", snippet="")]


class TestProviderLadder:
    def test_first_nonempty_rung_wins(self) -> None:
        r = WebResearcher(
            search_providers=(
                lambda q, n: [],
                lambda q, n: _hit("second"),
                lambda q, n: _hit("third"),
            )
        )
        hits = r.search("q")
        assert [h.title for h in hits] == ["second"]

    def test_failing_rung_continues_to_next(self) -> None:
        def _boom(q: str, n: int) -> list[SearchResult]:
            raise RuntimeError("rung down")

        r = WebResearcher(search_providers=(_boom, lambda q, n: _hit("ok")))
        assert [h.title for h in r.search("q")] == ["ok"]

    def test_falls_back_to_ddg_when_all_empty(self) -> None:
        r = WebResearcher(
            transport=_ddg_transport,
            search_providers=(lambda q, n: [],),
            enable_ddg_fallback=True,
        )
        hits = r.search("q")
        assert hits and hits[0].url == "https://example.org/a"

    def test_no_ddg_when_fallback_disabled(self) -> None:
        r = WebResearcher(
            transport=_ddg_transport,
            search_providers=(lambda q, n: [],),
            enable_ddg_fallback=False,
        )
        assert r.search("q") == []

    def test_legacy_singular_provider_takes_precedence(self) -> None:
        # The legacy singular hook fully replaces the chain even if a ladder and
        # DDG fallback are also configured (backward compatibility).
        r = WebResearcher(
            transport=_ddg_transport,
            search_provider=lambda q, n: _hit("legacy"),
            search_providers=(lambda q, n: _hit("ladder"),),
        )
        assert [h.title for h in r.search("q")] == ["legacy"]


class TestSearxngProvider:
    def test_parses_results(self) -> None:
        body = (
            '{"results": [{"title": "T1", "url": "https://a.test", "content": "c1"},'
            '{"title": "T2", "url": "https://b.test", "content": "c2"}]}'
        )

        def _opener(url: str, headers: dict[str, str]) -> str:
            assert "/search?" in url and "format=json" in url
            return body

        provider = searxng_provider("http://localhost:8080/", opener=_opener)
        hits = provider("quantum", 5)
        assert [h.url for h in hits] == ["https://a.test", "https://b.test"]
        assert hits[0].snippet == "c1"

    def test_failclosed_on_opener_error(self) -> None:
        def _opener(url: str, headers: dict[str, str]) -> str:
            raise RuntimeError("searxng down")

        provider = searxng_provider("http://localhost:8080", opener=_opener)
        assert provider("q", 5) == []

    def test_rejects_nonhttp_result_urls(self) -> None:
        body = '{"results": [{"title": "bad", "url": "javascript:alert(1)"}]}'
        provider = searxng_provider("http://h", opener=lambda u, h: body)
        assert provider("q", 5) == []


class TestBraveProvider:
    def test_sends_token_and_parses(self) -> None:
        captured: dict[str, str] = {}
        body = '{"web": {"results": [{"title": "B", "url": "https://b.test", "description": "d"}]}}'

        def _opener(url: str, headers: dict[str, str]) -> str:
            captured.update(headers)
            return body

        provider = brave_provider("secret-token", opener=_opener)
        hits = provider("q", 3)
        assert captured.get("X-Subscription-Token") == "secret-token"
        assert hits[0].url == "https://b.test" and hits[0].snippet == "d"

    def test_failclosed_on_error(self) -> None:
        provider = brave_provider("k", opener=lambda u, h: "not json")
        assert provider("q", 3) == []


class TestDefaultOpenerSSRFPolicy:
    """The built-in openers must apply the right SSRF policy per host class:
    SearXNG (operator-hosted) may reach localhost/LAN; a public keyed engine
    (Brave) stays on the strict default policy (no private/IMDS)."""

    def test_searxng_default_opener_trusts_operator_localhost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        captured: dict[str, object] = {}

        def _fake_get_text(url: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return '{"results": []}'

        monkeypatch.setattr(SafeHTTPClient, "get_text", _fake_get_text)
        searxng_provider("http://localhost:8080")("q", 5)
        assert captured.get("loopback_only") is True
        assert "allow_private" not in captured  # loopback wins, not blanket private

    def test_brave_default_opener_uses_strict_public_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        captured: dict[str, object] = {}

        def _fake_get_text(url: str, **kwargs: object) -> str:
            captured.update(kwargs)
            return '{"web": {"results": []}}'

        monkeypatch.setattr(SafeHTTPClient, "get_text", _fake_get_text)
        brave_provider("token")("q", 3)
        # Public host -> strict default SSRF: neither private nor loopback opened.
        assert "allow_private" not in captured
        assert "loopback_only" not in captured
        assert captured.get("user_configured") is True


class TestDefaultTransportEncoding:
    """The SafeHTTPClient-backed default transport must not mojibake a UTF-8
    page that is served as text/* with no charset (requests reports its
    ISO-8859-1 sentinel there)."""

    def test_charsetless_utf8_text_is_not_mojibaked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        from omni_mercury_engine.agentic.capabilities import web_research
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        text = "café — naïve résumé ☃"
        body = text.encode("utf-8")

        class _Raw:
            def read(self, amt: int = -1, decode_content: bool = True) -> bytes:
                return body

        class _Resp:
            status_code = 200
            url = "https://example.test/"
            headers = requests.structures.CaseInsensitiveDict({"Content-Type": "text/html"})
            encoding = "ISO-8859-1"  # requests' "no charset declared" sentinel
            raw = _Raw()
            _content = None

            @property
            def apparent_encoding(self) -> str:
                return "utf-8"

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: object) -> bool:
                return False

        monkeypatch.setattr(SafeHTTPClient, "get", staticmethod(lambda url, **kw: _Resp()))
        status, decoded, _final = web_research._safe_http_transport("https://example.test/", 5.0)
        assert status == 200
        assert decoded == text  # decoded as UTF-8, not Latin-1 mojibake


class TestFromEnv:
    def test_builds_ladder_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "bk")
        monkeypatch.setenv("MERCURY_SEARXNG_URL", "http://localhost:8080")
        monkeypatch.delenv("MERCURY_SEARCH_DDG_FALLBACK", raising=False)
        r = WebResearcher.from_env()
        # Brave first, then SearXNG; DDG fallback enabled by default.
        assert len(r.search_providers) == 2
        assert r.enable_ddg_fallback is True

    def test_env_can_disable_ddg_and_omit_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("MERCURY_SEARXNG_URL", raising=False)
        monkeypatch.setenv("MERCURY_SEARCH_DDG_FALLBACK", "0")
        r = WebResearcher.from_env()
        assert r.search_providers == ()
        assert r.enable_ddg_fallback is False
        # Provider-only with no providers and no fallback -> honest empty result.
        assert r.search("anything") == []

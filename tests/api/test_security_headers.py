# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The API must return browser-security headers on every surface.

Before :mod:`omni_mercury_engine.api.security_headers` existed, ``/health``,
``/login``, ``/dashboard``, ``/docs`` and ``/metrics`` all answered with no
``Content-Security-Policy``, ``X-Content-Type-Options``, ``X-Frame-Options``,
``Referrer-Policy`` or ``Strict-Transport-Security``, and no deployment layer
supplied them either. These tests pin the header set, the per-surface CSP
selection, and the two properties the strict frontend policy depends on:

* the shipped HTML carries no inline script, style or event-handler attribute
  (otherwise ``script-src 'self'`` would render the account UI inert), and
* only the two OpenAPI viewer paths are allowed to reach the widened policy.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.frontend import register_frontend
from omni_mercury_engine.api.security_headers import (
    API_CSP,
    DOCS_CSP,
    FRONTEND_CSP,
    SecurityHeadersMiddleware,
)

_ASSETS = (
    Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine" / "api" / "frontend_assets"
)

#: Headers every response must carry, with their required values.
_REQUIRED = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}


@pytest.fixture()
def client() -> TestClient:
    """A private app carrying the middleware plus the frontend page routes."""
    app = FastAPI(docs_url="/docs", redoc_url="/redoc")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    register_frontend(app)
    app.add_middleware(SecurityHeadersMiddleware)
    return TestClient(app)


class TestHeaderPresence:
    """Every surface carries the static header set."""

    @pytest.mark.parametrize("path", ["/health", "/login", "/dashboard", "/docs", "/openapi.json"])
    def test_static_headers_present(self, client: TestClient, path: str) -> None:
        response = client.get(path)
        assert response.status_code == 200, path
        for header, value in _REQUIRED.items():
            assert response.headers.get(header) == value, f"{path} missing {header}"
        assert "permissions-policy" in response.headers
        assert "content-security-policy" in response.headers

    def test_headers_survive_a_404(self, client: TestClient) -> None:
        """Error responses are browser-reachable too."""
        response = client.get("/definitely-not-a-route")
        assert response.status_code == 404
        for header, value in _REQUIRED.items():
            assert response.headers.get(header) == value

    def test_xss_protection_header_is_not_emitted(self, client: TestClient) -> None:
        """The deprecated filter header stays absent; CSP replaced it."""
        assert "x-xss-protection" not in client.get("/health").headers


class TestPolicySelection:
    """The right CSP reaches the right surface."""

    def test_json_gets_the_locked_api_policy(self, client: TestClient) -> None:
        csp = client.get("/health").headers["content-security-policy"]
        assert csp == API_CSP
        assert "default-src 'none'" in csp

    @pytest.mark.parametrize("path", ["/", "/login", "/register", "/dashboard"])
    def test_html_pages_get_the_strict_frontend_policy(self, client: TestClient, path: str) -> None:
        csp = client.get(path).headers["content-security-policy"]
        assert csp == FRONTEND_CSP
        assert "unsafe-inline" not in csp
        assert "unsafe-eval" not in csp

    def test_static_assets_get_the_frontend_policy(self, client: TestClient) -> None:
        assert client.get("/static/mercury.css").headers["content-security-policy"] == FRONTEND_CSP

    @pytest.mark.parametrize("path", ["/docs", "/redoc"])
    def test_only_the_viewers_get_the_widened_policy(self, client: TestClient, path: str) -> None:
        assert client.get(path).headers["content-security-policy"] == DOCS_CSP

    def test_widened_policy_still_forbids_framing_and_plugins(self) -> None:
        assert "frame-ancestors 'none'" in DOCS_CSP
        assert "object-src 'none'" in DOCS_CSP
        assert "base-uri 'none'" in DOCS_CSP

    def test_no_other_path_can_reach_the_docs_policy(self, client: TestClient) -> None:
        for path in ("/health", "/", "/login", "/dashboard", "/static/mercury.css"):
            assert client.get(path).headers["content-security-policy"] != DOCS_CSP


class TestHsts:
    """HSTS is pinned to requests that really arrived over TLS."""

    def test_absent_on_plain_http(self, client: TestClient) -> None:
        assert "strict-transport-security" not in client.get("/health").headers

    def test_present_on_https(self) -> None:
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        app.add_middleware(SecurityHeadersMiddleware)
        with TestClient(app, base_url="https://testserver") as tls_client:
            value = tls_client.get("/health").headers["strict-transport-security"]
        assert value == "max-age=31536000; includeSubDomains"

    def test_forwarded_proto_ignored_without_a_declared_proxy(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unproxied deployment cannot be tricked into pinning HSTS."""
        monkeypatch.delenv("MERCURY_TRUSTED_PROXY_HOPS", raising=False)
        response = client.get("/health", headers={"X-Forwarded-Proto": "https"})
        assert "strict-transport-security" not in response.headers

    def test_forwarded_proto_honoured_behind_a_declared_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_TRUSTED_PROXY_HOPS", "1")
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        app.add_middleware(SecurityHeadersMiddleware)
        with TestClient(app) as proxied:
            response = proxied.get("/health", headers={"X-Forwarded-Proto": "https"})
        assert response.headers["strict-transport-security"].startswith("max-age=31536000")

    def test_max_age_zero_suppresses_the_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_HSTS_MAX_AGE", "0")
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        app.add_middleware(SecurityHeadersMiddleware)
        with TestClient(app, base_url="https://testserver") as tls_client:
            assert "strict-transport-security" not in tls_client.get("/health").headers


class TestOptOut:
    """The documented opt-out actually opts out."""

    def test_disabled_emits_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERCURY_SECURITY_HEADERS", "false")
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "healthy"}

        app.add_middleware(SecurityHeadersMiddleware)
        with TestClient(app) as disabled:
            headers = disabled.get("/health").headers
        assert "content-security-policy" not in headers
        assert "x-frame-options" not in headers


class TestShippedAssetsSupportTheStrictPolicy:
    """``script-src 'self'`` is only enforceable if the pages obey it."""

    _INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>", re.IGNORECASE)
    _INLINE_STYLE_BLOCK = re.compile(r"<style[^>]*>", re.IGNORECASE)
    _STYLE_ATTR = re.compile(r"\sstyle\s*=", re.IGNORECASE)
    _EVENT_ATTR = re.compile(r"\son(?:click|load|error|submit|change|input|focus|blur)\s*=", re.I)

    @pytest.mark.parametrize("page", sorted(p.name for p in (_ASSETS / "pages").glob("*.html")))
    def test_page_has_no_inline_script_style_or_handler(self, page: str) -> None:
        html = (_ASSETS / "pages" / page).read_text(encoding="utf-8")
        assert not self._INLINE_SCRIPT.search(html), f"{page}: inline <script> block"
        assert not self._INLINE_STYLE_BLOCK.search(html), f"{page}: inline <style> block"
        assert not self._STYLE_ATTR.search(html), f"{page}: inline style= attribute"
        assert not self._EVENT_ATTR.search(html), f"{page}: inline event-handler attribute"

    def test_pages_load_only_same_origin_assets(self) -> None:
        """No page references a third-party origin: ``'self'`` suffices."""
        remote = re.compile(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', re.IGNORECASE)
        for page in (_ASSETS / "pages").glob("*.html"):
            html = page.read_text(encoding="utf-8")
            assert not remote.search(html), f"{page.name} loads a cross-origin asset"

    def test_stylesheet_fetches_nothing(self) -> None:
        """``mercury.css`` has no ``url()`` / ``@import``, so no font-src need."""
        css = (_ASSETS / "static" / "mercury.css").read_text(encoding="utf-8")
        assert "@import" not in css
        assert "url(" not in css

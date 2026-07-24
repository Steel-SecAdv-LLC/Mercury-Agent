# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the opt-in browser frontend.

Three layers:

* **Pages** — every route serves 200 with the markup its page script binds to
  (form ids, status regions), and the static assets (stylesheet, scripts, the
  vendored QR generator) resolve.
* **E2E (browserless)** — the full account journey the UI drives, executed
  through TestClient exactly as the page JS would: register → capture the
  emailed token → verify → login (cookie session + CSRF) → mint an API key
  (one-time reveal) → spend it on a metered route → the usage endpoint the
  dashboard reads reflects the charge.
* **Contract** — frontend-off (the default) leaves every existing route
  byte-identical, and the shipped HTML stays CSP-compatible (no inline
  handlers, no inline scripts, no external asset references).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import auth, frontend
from omni_mercury_engine.api.auth import APIKeyStore
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.quota import QuotaConfig, QuotaEnforcer, get_shared_quota_enforcer
from omni_mercury_engine.api.quota_middleware import QuotaMiddleware
from omni_mercury_engine.api.routes import accounts
from omni_mercury_engine.api.usage_ledger import InMemoryUsageLedger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class RecordingMailer:
    """Mailer that records messages so tests can read the emailed token."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record instead of delivering."""
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email."""
        match = re.search(r"token=([A-Za-z0-9_\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


#: What the ``site`` fixture yields.
Site = tuple[TestClient, RecordingMailer]


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch) -> Iterator[Site]:
    """A full frontend+API app: pages, accounts routes, metered route, quotas."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("MERCURY_QUOTA_ENABLED", "true")
    mailer = RecordingMailer()
    service = AuthService(InMemoryIdentityStore(), mailer)
    enforcer = QuotaEnforcer(
        InMemoryUsageLedger(),
        QuotaConfig(window_seconds=3600, max_requests=100, max_compute_ms=1e9),
    )
    # The quota middleware resolves principals through the process-wide
    # singletons (not FastAPI dependencies), so pin those too.
    monkeypatch.setattr(accounts, "_service", service)
    monkeypatch.setattr(auth, "_api_key_store", APIKeyStore())

    app = FastAPI()
    frontend.register_frontend(app)
    app.include_router(accounts.router)

    @app.post("/api/v1/detect/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(QuotaMiddleware, enforcer=enforcer)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    app.dependency_overrides[get_shared_quota_enforcer] = lambda: enforcer
    with TestClient(app) as client:
        yield client, mailer


# --------------------------------------------------------------------------- #
# pages + assets
# --------------------------------------------------------------------------- #
_PAGE_MARKERS = {
    "/": "Mercury Agent",
    "/register": 'id="register-form"',
    "/login": 'id="two-factor-step"',
    "/verify-email": 'id="verify-status"',
    "/reset-password": 'id="reset-request-form"',
    "/confirm-email-change": 'id="confirm-status"',
    "/dashboard": 'id="api-key-form"',
}


@pytest.mark.parametrize(("path", "marker"), sorted(_PAGE_MARKERS.items()))
def test_page_serves_expected_markup(site: Site, path: str, marker: str) -> None:
    """Each page route serves 200 HTML containing the markup its JS binds to."""
    client, _ = site
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert marker in response.text


def test_static_assets_resolve(site: Site) -> None:
    """The stylesheet, page scripts, and vendored QR generator all serve."""
    client, _ = site
    for path in (
        "/static/mercury.css",
        "/static/common.js",
        "/static/auth.js",
        "/static/dashboard.js",
        "/static/vendor/qrcode.js",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
    qr = client.get("/static/vendor/qrcode.js").text
    assert "MIT license" in qr and "Kazuhiko Arase" in qr  # license header retained


def test_login_page_wires_csrf_and_2fa_handling(site: Site) -> None:
    """The shared JS carries the CSRF echo and the two_factor_required branch."""
    client, _ = site
    common = client.get("/static/common.js").text
    assert "mercury_csrf" in common
    assert "X-CSRF-Token" in common
    auth_js = client.get("/static/auth.js").text
    assert "two_factor_required" in auth_js


def test_login_page_offers_resend_verification(site: Site) -> None:
    """A user who lost the verification email can resend it from /login."""
    client, mailer = site
    assert 'id="resend-button"' in client.get("/login").text
    assert "/resend-verification" in client.get("/static/auth.js").text
    # The endpoint the button drives works end-to-end: an unverified account
    # gets a fresh token, enumeration-safely (202 either way).
    client.post("/api/v1/auth/register", json={"email": "lost@b.com", "password": "a-strong-pw"})
    sent_before = len(mailer.sent)
    resp = client.post("/api/v1/auth/resend-verification", json={"email": "lost@b.com"})
    assert resp.status_code == 202
    assert len(mailer.sent) == sent_before + 1
    assert (
        client.post("/api/v1/auth/resend-verification", json={"email": "ghost@b.com"}).status_code
        == 202
    )  # unknown address: same answer, no email
    assert len(mailer.sent) == sent_before + 1


def test_frontend_js_covers_every_account_route() -> None:
    """Every account API route is reachable from the shipped UI scripts.

    Guards against the gap this test was born from (the UI shipping without
    a resend-verification path): if a new route lands in accounts.py without
    a frontend affordance, this fails until the UI (or an explicit exemption
    here) catches up.
    """
    router_paths = {
        route.path.removeprefix("/api/v1/auth")
        for route in accounts.router.routes
        if hasattr(route, "path")
    }
    js_dir = frontend._STATIC_DIR
    js_text = "".join(p.read_text(encoding="utf-8") for p in js_dir.glob("*.js"))
    missing = {
        path
        for path in router_paths
        # The JS calls DELETE /api-keys/{id} by string concatenation.
        if path.replace("/{key_id}", "/") not in js_text
    }
    assert not missing, f"account routes with no frontend affordance: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# browserless end-to-end journey
# --------------------------------------------------------------------------- #
def test_full_account_journey(site: Site) -> None:
    """register → verify → login → mint key → metered call → usage reflects it."""
    client, mailer = site

    # Register; the verification token arrives on the (recorded) email.
    assert (
        client.post(
            "/api/v1/auth/register",
            json={"email": "e2e@b.com", "password": "a-strong-pw"},
        ).status_code
        == 201
    )
    assert (
        client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()}).status_code
        == 200
    )

    # Login sets the session + readable CSRF cookie pair.
    login = client.post(
        "/api/v1/auth/login", json={"email": "e2e@b.com", "password": "a-strong-pw"}
    )
    assert login.status_code == 200
    assert accounts.CSRF_COOKIE in login.cookies  # the JS reads this cookie
    csrf = str(login.json()["csrf_token"])

    # Mint an API key exactly as the dashboard does (CSRF header echoed).
    created = client.post(
        "/api/v1/auth/api-keys", json={"name": "e2e"}, headers={"X-CSRF-Token": csrf}
    )
    assert created.status_code == 201
    raw_key = str(created.json()["api_key"])

    # A programmatic caller (no cookies) spends the key on a metered route.
    machine = TestClient(client.app)
    assert machine.post("/api/v1/detect/probe", headers={"X-API-Key": raw_key}).status_code == 200

    # The dashboard's usage read reflects that spend — for the browser session
    # and for the key itself (both charge the same owning account).
    usage = client.get("/api/v1/auth/usage")
    assert usage.status_code == 200
    assert usage.json()["requests_used"] >= 1
    by_key = machine.get("/api/v1/auth/usage", headers={"X-API-Key": raw_key})
    assert by_key.status_code == 200
    assert by_key.json()["requests_used"] >= 1


# --------------------------------------------------------------------------- #
# frontend-off contract + CSP hygiene
# --------------------------------------------------------------------------- #
def test_frontend_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env gate defaults off and parses strictly."""
    monkeypatch.delenv("MERCURY_FRONTEND_ENABLED", raising=False)
    assert frontend.frontend_enabled() is False
    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "true")
    assert frontend.frontend_enabled() is True
    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "1")
    assert frontend.frontend_enabled() is False  # only the documented literal


def test_frontend_off_leaves_existing_routes_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var the server app has no pages and no /static mount."""
    monkeypatch.delenv("MERCURY_FRONTEND_ENABLED", raising=False)
    from omni_mercury_engine.api import server

    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 404
        assert client.get("/login").status_code == 404
        assert client.get("/dashboard").status_code == 404
        assert client.get("/static/mercury.css").status_code == 404


_INLINE_HANDLER = re.compile(r"<[^>]+\son[a-z]+\s*=", re.IGNORECASE)
_INLINE_SCRIPT = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>", re.IGNORECASE)
_EXTERNAL_REF = re.compile(r'(?:src|href)\s*=\s*"(?!/|#)', re.IGNORECASE)


def _page_files() -> list[Path]:
    """Every shipped HTML page."""
    return sorted(frontend._PAGES_DIR.glob("*.html"))


def test_pages_are_csp_compatible_and_self_contained() -> None:
    """No inline handlers, no inline scripts, no external (CDN) references."""
    pages = _page_files()
    assert len(pages) == len(frontend._PAGE_ROUTES)
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert not _INLINE_HANDLER.search(html), f"inline event handler in {page.name}"
        assert not _INLINE_SCRIPT.search(html), f"inline <script> in {page.name}"
        assert not _EXTERNAL_REF.search(html), f"non-local asset reference in {page.name}"


def test_pages_have_accessible_status_regions() -> None:
    """Interactive pages announce results via aria-live regions; forms label inputs."""
    for name in ("register", "login", "verify-email", "reset-password", "dashboard"):
        html = (frontend._PAGES_DIR / f"{name}.html").read_text(encoding="utf-8")
        assert 'aria-live="polite"' in html, name
    for name in ("register", "login", "reset-password", "dashboard"):
        html = (frontend._PAGES_DIR / f"{name}.html").read_text(encoding="utf-8")
        assert "<label" in html, name

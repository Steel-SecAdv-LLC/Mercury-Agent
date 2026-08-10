# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Security response headers for every response the API emits.

The API previously returned no browser-security headers on any surface — not
on JSON (``/health``, ``/api/v1/*``), not on the account frontend's HTML
pages, not on ``/docs``. Nothing downstream compensated: the kustomize
ingress carried them in an ``nginx.ingress.kubernetes.io/configuration-
snippet``, an annotation group ingress-nginx has disabled by default since
v1.9 (``allow-snippet-annotations: false``), so a default controller rejects
the Ingress outright rather than applying it; the Helm ingress declared no
header annotations at all; and the Caddy edge used by the container platform
added none. Headers belong in the application for exactly this reason — it is
the one layer present in every deployment topology.

Three policies, selected by what the response actually is
-----------------------------------------------------------------
``Content-Security-Policy`` is not one-size-fits-all, so this middleware
selects between three fixed policies (:data:`API_CSP`, :data:`FRONTEND_CSP`,
:data:`DOCS_CSP`) rather than shipping the loosest one everywhere:

* **API responses** (JSON, plain text, anything that is not HTML) get
  ``default-src 'none'`` — an API response has no legitimate need to load a
  subresource, frame anything, or submit a form.
* **The account frontend** (``/``, ``/login``, ``/dashboard``, … and the
  ``/static`` mount) gets a strict same-origin policy with **no**
  ``'unsafe-inline'`` and **no** ``'unsafe-eval'``. That is achievable
  because the shipped pages carry zero inline ``<script>``/``<style>``
  blocks and zero inline event-handler attributes — every page loads its
  behaviour from ``/static/*.js`` and its presentation from
  ``/static/mercury.css``. ``tests/api/test_security_headers.py`` asserts
  that property against the shipped assets so the policy cannot silently
  become unenforceable.
* **The OpenAPI viewers** (``/docs``, ``/redoc``) get a narrowly widened
  policy. FastAPI serves Swagger UI and ReDoc from jsDelivr with an inline
  bootstrap script it generates itself; a policy without the CDN origin and
  without ``'unsafe-inline'`` for that bootstrap renders both viewers blank.
  The widening is scoped to those two paths and to ``script-src`` /
  ``style-src`` / ``img-src`` / ``worker-src`` only — ``frame-ancestors``,
  ``object-src`` and ``base-uri`` stay locked shut, and no other path can
  reach this policy.

``X-XSS-Protection`` is deliberately absent. The header is non-standard,
removed from every current browser, and its ``mode=block`` filter was itself
exploitable for cross-site info leaks in the browsers that still honour it;
``Content-Security-Policy`` is its replacement.

``Strict-Transport-Security`` is emitted only when the request actually
arrived over TLS — directly, or through the proxy tier the deployment has
declared via ``MERCURY_TRUSTED_PROXY_HOPS`` (the same declaration
:mod:`omni_mercury_engine.api.client_ip` uses to decide whether
``X-Forwarded-For`` may be believed). With no declared proxy the
``X-Forwarded-Proto`` header is attacker-controlled and is ignored, so a
plaintext deployment cannot be tricked into pinning HSTS on its own origin.

Configuration
-------------
``MERCURY_SECURITY_HEADERS``
    ``true`` (default) / ``false``. Set ``false`` only when a reverse proxy
    already emits an equivalent set and duplicate headers are undesirable.
``MERCURY_HSTS_MAX_AGE``
    Seconds for ``Strict-Transport-Security`` (default ``31536000``, one
    year). ``0`` suppresses the header entirely — the right setting while a
    certificate is still being provisioned, since HSTS is not revocable
    inside its own max-age.
``MERCURY_HSTS_INCLUDE_SUBDOMAINS``
    ``true`` (default) / ``false``. Turn off when sibling sub-domains of the
    API's registrable domain are not all HTTPS-capable.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from omni_mercury_engine.api.client_ip import trusted_proxy_hops

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.types import ASGIApp

__all__ = [
    "API_CSP",
    "DOCS_CSP",
    "DOCS_PATHS",
    "FRONTEND_CSP",
    "PERMISSIONS_POLICY",
    "STATIC_PREFIX",
    "SecurityHeadersMiddleware",
    "hsts_value",
    "security_headers_enabled",
]

#: Policy for every non-HTML response (JSON APIs, ``/metrics``,
#: ``/openapi.json``). An API payload never legitimately loads a subresource.
API_CSP = "default-src 'none'; " "frame-ancestors 'none'; " "base-uri 'none'; " "form-action 'none'"

#: Policy for the account frontend. No ``'unsafe-inline'``, no
#: ``'unsafe-eval'``, no third-party origin: the shipped pages load every
#: script and stylesheet from same-origin ``/static`` files.
FRONTEND_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'self'"
)

#: The one CDN FastAPI's bundled viewers load their bundles from.
_DOCS_CDN = "https://cdn.jsdelivr.net"

#: Origin of the default favicon FastAPI writes into both viewers' markup.
_DOCS_FAVICON_ORIGIN = "https://fastapi.tiangolo.com"

#: Policy for ``/docs`` and ``/redoc`` only. Widened exactly as far as
#: FastAPI's own generated markup requires and no further: the inline
#: bootstrap script/style FastAPI writes into the page, the jsDelivr bundles,
#: the ReDoc ``blob:`` web worker, and the default favicon. Both third-party
#: origins are named explicitly rather than allowed as a blanket ``https:``.
#: Framing, plugins and ``<base>`` stay forbidden.
DOCS_CSP = (
    "default-src 'self'; "
    f"script-src 'self' 'unsafe-inline' {_DOCS_CDN}; "
    f"style-src 'self' 'unsafe-inline' {_DOCS_CDN}; "
    f"img-src 'self' data: {_DOCS_CDN} {_DOCS_FAVICON_ORIGIN}; "
    f"font-src 'self' {_DOCS_CDN}; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'self'"
)

#: Paths served as an interactive OpenAPI viewer.
DOCS_PATHS: frozenset[str] = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})

#: Mount prefix for the frontend's static assets.
STATIC_PREFIX = "/static/"

#: Every browser feature the API and its frontend have no use for. Denying
#: them shrinks what an injected script could reach if one ever landed.
PERMISSIONS_POLICY = (
    "accelerometer=(), ambient-light-sensor=(), autoplay=(), battery=(), "
    "camera=(), display-capture=(), document-domain=(), encrypted-media=(), "
    "fullscreen=(), geolocation=(), gyroscope=(), magnetometer=(), "
    "microphone=(), midi=(), payment=(), picture-in-picture=(), "
    "publickey-credentials-get=(), screen-wake-lock=(), serial=(), usb=(), "
    "xr-spatial-tracking=()"
)

#: Headers applied to every response regardless of content type.
_STATIC_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", PERMISSIONS_POLICY),
)

_ENABLED_ENV = "MERCURY_SECURITY_HEADERS"
_HSTS_MAX_AGE_ENV = "MERCURY_HSTS_MAX_AGE"
_HSTS_SUBDOMAINS_ENV = "MERCURY_HSTS_INCLUDE_SUBDOMAINS"

_DEFAULT_HSTS_MAX_AGE = 31_536_000  # one year


def security_headers_enabled() -> bool:
    """Whether the middleware should attach headers (default: yes)."""
    return os.getenv(_ENABLED_ENV, "true").strip().lower() != "false"


def hsts_value() -> str | None:
    """Build the ``Strict-Transport-Security`` value, or ``None`` if disabled.

    Returns:
        ``"max-age=<n>; includeSubDomains"``, or ``None`` when the configured
        max-age is zero or unparseable. An unparseable value degrades to the
        one-year default rather than to "no HSTS": a typo in an operator's
        value must not silently drop transport pinning.
    """
    raw = os.getenv(_HSTS_MAX_AGE_ENV, str(_DEFAULT_HSTS_MAX_AGE)).strip()
    try:
        max_age = int(raw)
    except ValueError:
        max_age = _DEFAULT_HSTS_MAX_AGE
    if max_age <= 0:
        return None
    value = f"max-age={max_age}"
    if os.getenv(_HSTS_SUBDOMAINS_ENV, "true").strip().lower() != "false":
        value += "; includeSubDomains"
    return value


def _request_is_secure(request: Request) -> bool:
    """Whether this request reached the process over TLS.

    ``X-Forwarded-Proto`` is consulted only when the deployment has declared
    a trusted proxy tier (``MERCURY_TRUSTED_PROXY_HOPS >= 1``). Without that
    declaration the header is attacker-supplied, and honouring it would let a
    single crafted request pin HSTS on a plaintext origin — a self-inflicted
    denial of service that lasts for the whole max-age.
    """
    if request.url.scheme == "https":
        return True
    if trusted_proxy_hops() < 1:
        return False
    forwarded = request.headers.get("x-forwarded-proto", "")
    # A proxy chain appends, so the client-facing hop is the left-most entry.
    return forwarded.split(",")[0].strip().lower() == "https"


def _csp_for(path: str, content_type: str) -> str:
    """Select the policy for one response.

    Args:
        path: The request path (``request.url.path``).
        content_type: The response's ``Content-Type``, lower-cased.

    Returns:
        One of :data:`DOCS_CSP`, :data:`FRONTEND_CSP` or :data:`API_CSP`.
    """
    if path in DOCS_PATHS:
        return DOCS_CSP
    if path.startswith(STATIC_PREFIX) or content_type.startswith("text/html"):
        return FRONTEND_CSP
    return API_CSP


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the security header set to every response.

    Registered outermost so it also covers responses short-circuited by the
    inner middleware — the rate limiter's 429 and the quota layer's 503 never
    reach a route handler, and both are as much a browser-reachable response
    as a 200.

    Existing values are preserved: a handler that has deliberately set, say, a
    looser ``Content-Security-Policy`` on one response keeps it. Nothing in
    the shipped surface does that today; the rule exists so this middleware
    can never silently override a more specific decision made downstream.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Snapshot configuration once, at application construction."""
        super().__init__(app)
        self._enabled = security_headers_enabled()
        self._hsts = hsts_value()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Run the request, then decorate the response."""
        response = await call_next(request)
        if not self._enabled:
            return response

        for name, value in _STATIC_HEADERS:
            response.headers.setdefault(name, value)

        content_type = response.headers.get("content-type", "").lower()
        response.headers.setdefault(
            "Content-Security-Policy", _csp_for(request.url.path, content_type)
        )

        if self._hsts is not None and _request_is_secure(request):
            response.headers.setdefault("Strict-Transport-Security", self._hsts)

        return response

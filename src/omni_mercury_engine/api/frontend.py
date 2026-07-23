# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The optional browser frontend for the self-service account platform.

Serves the pages the auth emails already link to (``/verify-email``,
``/reset-password``, ``/confirm-email-change``) plus registration, login
(with an inline second step when the API answers ``two_factor_required``),
a landing page, and the account dashboard (profile, API keys with one-time
reveal, usage vs. limits, password/email change, 2FA lifecycle with
client-side QR rendering, data export, account deletion).

Everything is plain static assets — vanilla HTML/CSS/JS served straight from
the installed package (``frontend_assets/``), no build toolchain, no CDN, no
template engine. The JS talks to the existing ``/api/v1/auth`` routes with
``fetch``, reading the ``mercury_csrf`` cookie and echoing it as
``X-CSRF-Token`` on every state-changing call. The QR code for TOTP
enrollment is drawn client-side by the vendored ``qrcode-generator`` library
(``frontend_assets/static/vendor/qrcode.js``, MIT — Copyright (c) 2009
Kazuhiko Arase; license header retained in the file).

**Opt-in and inert by default**: nothing registers unless
``MERCURY_FRONTEND_ENABLED=true``, so an upgrade changes no existing
behaviour — ``/`` keeps returning 404 for a deployment that never asked for
a UI. Enabling it only *adds* page routes and the ``/static`` mount; the API
surface is untouched either way.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import FileResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI

__all__ = ["frontend_enabled", "register_frontend", "router"]

#: Root of the shipped assets (packaged; see ``[tool.setuptools.package-data]``).
_ASSETS_DIR = Path(__file__).parent / "frontend_assets"
_PAGES_DIR = _ASSETS_DIR / "pages"
_STATIC_DIR = _ASSETS_DIR / "static"

#: Browser path → page file. The three email-link paths must match the URLs
#: ``AuthService`` builds (``/verify-email?token=…`` etc.) exactly.
_PAGE_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/register": "register.html",
    "/login": "login.html",
    "/verify-email": "verify-email.html",
    "/reset-password": "reset-password.html",
    "/confirm-email-change": "confirm-email-change.html",
    "/dashboard": "dashboard.html",
}

router = APIRouter(tags=["Frontend"], include_in_schema=False)


def frontend_enabled() -> bool:
    """Whether the browser frontend should be mounted (default: off).

    Off by default so a solo self-hoster keeps byte-identical behaviour —
    the platform contract for every feature in this PR series. Set
    ``MERCURY_FRONTEND_ENABLED=true`` to serve the account UI.
    """
    return os.getenv("MERCURY_FRONTEND_ENABLED", "false").strip().lower() == "true"


def _page_response(page: str) -> FileResponse:
    """Serve one shipped HTML page."""
    return FileResponse(_PAGES_DIR / page, media_type="text/html")


def _make_page_handler(page: str) -> Callable[[], FileResponse]:
    """Build the route handler for ``page`` (early-bound, one per route)."""

    def handler() -> FileResponse:
        return _page_response(page)

    handler.__doc__ = f"Serve the {page} page."
    return handler


for _path, _page in _PAGE_ROUTES.items():
    router.add_api_route(_path, _make_page_handler(_page), methods=["GET"])


def register_frontend(app: FastAPI) -> None:
    """Attach the page routes and the ``/static`` asset mount to ``app``.

    Called by the server at import time when :func:`frontend_enabled` says
    so, and directly by tests on a private app instance (the enable gate is
    the caller's job, mirroring how the other routers register).

    Args:
        app: The FastAPI application to serve the frontend from.
    """
    from fastapi.staticfiles import StaticFiles

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="frontend-static")

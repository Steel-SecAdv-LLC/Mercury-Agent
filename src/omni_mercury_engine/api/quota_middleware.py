# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""HTTP wiring for per-account quotas on the metered routes.

The quota engine (:mod:`~omni_mercury_engine.api.quota`) is framework-free;
this middleware is the single place it meets HTTP. For every request under a
metered path prefix (default ``/api/v1/detect`` and ``/api/v1/batch``) it:

1. Resolves the **principal**: the session cookie's account, else the
   ``X-API-Key``'s owning user, else an anonymous principal keyed by the
   trusted-proxy-resolved client IP (so unauthenticated abuse is still
   bounded rather than unmetered).
2. **Reserves** a request slot atomically (hard request ceiling; HTTP 429
   with ``Retry-After`` and the usage counters on denial).
3. Runs the request, then **commits** the measured wall-clock compute cost
   onto the reserved ledger row — so the compute ceiling reflects reality,
   not estimates.

Enforcement is opt-in via ``MERCURY_QUOTA_ENABLED=true`` (a solo self-hoster
keeps byte-identical behaviour), and *fails open by policy*: an internal
metering error is logged and the request proceeds — detection availability
outranks accounting, and the global rate limiter still bounds volume.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from omni_mercury_engine.api.client_ip import resolve_client_ip

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request, Response
    from starlette.types import ASGIApp

    from omni_mercury_engine.api.quota import QuotaEnforcer

logger = logging.getLogger(__name__)

__all__ = ["QuotaMiddleware", "quota_enforcement_enabled"]

_DEFAULT_METERED_PREFIXES = "/api/v1/detect,/api/v1/batch"


def quota_enforcement_enabled() -> bool:
    """Whether quota enforcement is switched on for this deployment."""
    return os.getenv("MERCURY_QUOTA_ENABLED", "false").strip().lower() == "true"


class QuotaMiddleware(BaseHTTPMiddleware):
    """Reserve-run-commit quota enforcement over the metered path prefixes."""

    def __init__(self, app: ASGIApp, enforcer: QuotaEnforcer | None = None) -> None:
        """Wire the middleware; the enforcer builds lazily from the env.

        Args:
            app: The wrapped ASGI app.
            enforcer: Injected enforcer for tests; ``None`` builds one on
                first use (after env/test fixtures are in place).
        """
        super().__init__(app)
        self._enforcer = enforcer
        self._enforcer_lock = threading.Lock()
        prefixes = os.getenv("MERCURY_QUOTA_METERED_PREFIXES", _DEFAULT_METERED_PREFIXES)
        self._prefixes = tuple(p.strip() for p in prefixes.split(",") if p.strip())

    def _get_enforcer(self) -> QuotaEnforcer:
        """Build (once, lock-guarded) or return the quota enforcer."""
        if self._enforcer is None:
            with self._enforcer_lock:
                if self._enforcer is None:
                    from omni_mercury_engine.api.quota import build_quota_enforcer

                    self._enforcer = build_quota_enforcer()
        return self._enforcer

    def _resolve_principal(self, request: Request) -> tuple[str, str]:
        """Map the request to a ``(principal_id, tier)`` pair.

        Session cookie wins (a browser user), then API key (a programmatic
        caller, charged to the key's owning user), then the anonymous
        per-IP principal with the ``anon`` tier.
        """
        from omni_mercury_engine.api.routes.accounts import SESSION_COOKIE, get_auth_service

        raw_session = request.cookies.get(SESSION_COOKIE)
        if raw_session:
            try:
                account = get_auth_service().authenticate_session(raw_session)
            except Exception:  # pragma: no cover - principal resolution is best-effort
                account = None
            if account is not None:
                return account.id, account.tier

        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                from omni_mercury_engine.api.auth import get_api_key_store

                key_obj = get_api_key_store().get_by_key(api_key)
            except Exception:  # pragma: no cover - principal resolution is best-effort
                key_obj = None
            if key_obj is not None and key_obj.is_active and not key_obj.is_expired:
                return f"key:{key_obj.user_id}", "free"

        ip = resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )
        return f"anon:{ip}", "anon"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Enforce the quota around a metered request."""
        path = request.url.path
        if not quota_enforcement_enabled() or not path.startswith(self._prefixes):
            return await call_next(request)

        try:
            enforcer = self._get_enforcer()
            principal, tier = self._resolve_principal(request)
            decision = enforcer.reserve(principal, path, tier)
        except Exception:
            logger.exception("quota reservation failed; admitting request unmetered")
            return await call_next(request)

        if not decision.allowed:
            from fastapi import Response as FastAPIResponse

            retry_after = decision.retry_after_seconds or 60
            return FastAPIResponse(
                content=(
                    '{"error": "quota_exceeded", "message": "'
                    + (decision.reason or "quota exceeded")
                    + '"}'
                ),
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-Quota-Requests-Used": str(decision.request_count),
                    "X-Quota-Compute-Ms-Used": str(int(decision.compute_ms)),
                },
            )

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            if decision.event_id is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                try:
                    enforcer.commit(decision.event_id, elapsed_ms)
                except Exception:  # pragma: no cover - accounting must not break replies
                    logger.exception("quota commit failed for event %s", decision.event_id)
        return response

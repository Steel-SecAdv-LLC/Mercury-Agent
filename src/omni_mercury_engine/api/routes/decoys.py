# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The opt-in decoy surface (``MERCURY_DECEPTION_ENABLED``).

Two kinds of decoy, each indistinguishable from the real thing by
construction rather than by imitation:

* ``auth_challenge`` decoys are real routes. Each resolves the client
  exactly as the rate limiter does, records a tripwire, holds through the
  bounded tarpit, and raises the ``HTTPException`` Mercury's protected API
  routes raise for an anonymous caller (measured across the live app, and
  pinned by a differential test against a real protected route). Because they
  *are* routes, their trailing-slash redirects and wrong-method answers are a
  real route's too.
* ``not_found`` decoys are silent tripwires implemented as observe-only
  middleware: they record the touch and then pass the request on untouched,
  so the application's own router answers with its genuine 404 -- for every
  method (a route would answer ``OPTIONS`` or ``PROPFIND`` with a telltale
  405) and every path variant (a route would redirect its trailing-slash form
  with a telltale 307). The middleware sits innermost, beside the router, so
  it sees only requests the application itself answers: one the rate limiter
  or another outer layer refuses is neither recorded nor held, and is refused
  exactly as a real missing path is.

Registration refuses any decoy path the application already serves, checked
with the framework's own matcher, so a configured decoy can never shadow a
real route -- including one reached through a path parameter -- or record
legitimate traffic as probing. Fail closed: an internal failure in an
``auth_challenge`` decoy answers with the plain 404 of a missing path, and in
a silent tripwire simply lets the request through; neither ever answers 500.
Each survived failure is counted on the sensor and exported as
``mercury_deception_failures_total{kind}``, so a degraded sensor is visible
to an operator even though the attacker sees nothing different. Left
disabled, nothing is registered at all.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import APIRouter, HTTPException, Request, Response
from starlette.middleware import Middleware
from starlette.requests import HTTPConnection
from starlette.routing import Match

from omni_mercury_engine.api.client_ip import resolve_client_ip
from omni_mercury_engine.security.deception import (
    STAND_DOWN_HEADER,
    STAND_DOWN_NOTICE,
    DeceptionSensor,
    DecoyRoute,
    TarpitGate,
    TripwireEvent,
    configured_decoys,
    deception_enabled,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

    from omni_mercury_engine.security.secure_audit_logging import (
        AuditEventCategory,
        AuditEventSeverity,
    )

logger = logging.getLogger(__name__)


class AuditChain(Protocol):
    """The one call (by keyword) the tripwire sink makes on the tamper-evident chain.

    :class:`~omni_mercury_engine.security.secure_audit_logging.SecureAuditLogger`
    satisfies it.
    """

    def log(
        self,
        *,
        category: AuditEventCategory,
        action: str,
        outcome: str = ...,
        severity: AuditEventSeverity = ...,
        actor: str | None = ...,
        resource: str | None = ...,
        details: dict[str, Any] | None = ...,
        session_id: str | None = ...,
        client_ip: str | None = ...,
    ) -> str:
        """Append one event to the chain; return its event id."""
        ...


__all__ = [
    "SilentTripwireMiddleware",
    "build_decoy_router",
    "register_decoys",
    "secure_audit_sink",
]

#: The challenge Mercury's protected API routes return to a caller with no
#: valid credential (the ``_get_current_user`` dependencies of the models,
#: export, batch and assessment routers).
_AUTH_REQUIRED_DETAIL = "Authentication required"
_AUTH_CHALLENGE = "Bearer, ApiKey"

#: Headers whose presence marks a request as presenting credentials.
_CREDENTIAL_HEADERS = ("x-api-key", "authorization")

#: The browser-session cookie set by the account routes; equal to
#: ``omni_mercury_engine.api.routes.accounts.SESSION_COOKIE`` (pinned by test,
#: not imported, so the decoy surface never loads the account stack).
_SESSION_COOKIE = "mercury_session"


def _credential_presented(conn: HTTPConnection) -> bool:
    """Whether the request carries an API key, a bearer token or a session."""
    return any(h in conn.headers for h in _CREDENTIAL_HEADERS) or _SESSION_COOKIE in conn.cookies


def _client_ip(conn: HTTPConnection) -> str:
    """Resolve the client exactly as the rate limiter does."""
    return resolve_client_ip(
        conn.client.host if conn.client else None, conn.headers.get("x-forwarded-for")
    )


def _silent_key(path: str) -> str:
    """Normalise a path for silent-tripwire matching (trailing slash ignored)."""
    return path.rstrip("/") or "/"


def secure_audit_sink(
    secure_logger: AuditChain | None,
) -> Callable[[TripwireEvent], None]:
    """Build the tripwire sink over the process's tamper-evident audit chain.

    Args:
        secure_logger: The shared :class:`SecureAuditLogger` (from
            ``shared_secure_logger()``), or ``None`` for structured logging.

    Returns:
        A sink recording each event as a ``security_incident`` on the same
        hash chain that holds authentication evidence.
    """

    def sink(event: TripwireEvent) -> None:
        if secure_logger is not None:
            from omni_mercury_engine.security.secure_audit_logging import (
                AuditEventCategory,
                AuditEventSeverity,
            )

            secure_logger.log(
                category=AuditEventCategory.SECURITY_INCIDENT,
                action="deception.tripwire",
                outcome="failure",
                severity=AuditEventSeverity.SECURITY,
                actor=event.source_key,
                resource=event.path,
                details=event.audit_details(),
                client_ip=event.client_ip,
            )
        else:
            logger.warning(
                "deception tripwire source=%s decoy=%s method=%s hits=%d distinct=%d "
                "credential_presented=%s",
                event.source_key,
                event.path,
                event.method,
                event.hits_in_window,
                event.distinct_decoys,
                event.credential_presented,
            )

    return sink


def _decoy_handler(
    decoy: DecoyRoute, sensor: DeceptionSensor, gate: TarpitGate
) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        try:
            event = sensor.record(
                _client_ip(request),
                decoy.path,
                request.method,
                credential_presented=_credential_presented(request),
            )
            await gate.hold(event.delay_s)
        except Exception:
            logger.exception("decoy %s failed; answering as a missing path", decoy.path)
            sensor.note_failure("handler")
            raise HTTPException(status_code=404) from None
        headers = {"WWW-Authenticate": _AUTH_CHALLENGE}
        # Benevolent-first stand-down: while the source is still in the free
        # window, carry an explicit advisory to desist. A normal client reads
        # an ordinary 401; an agent parsing headers gets a clear chance to
        # stop before any containment delay applies. Header-only, so the
        # decoy's auth-challenge camouflage is unchanged for everything else.
        if event.stand_down_offered:
            headers[STAND_DOWN_HEADER] = STAND_DOWN_NOTICE
        raise HTTPException(
            status_code=401,
            detail=_AUTH_REQUIRED_DETAIL,
            headers=headers,
        )

    return handler


def build_decoy_router(
    sensor: DeceptionSensor,
    gate: TarpitGate,
    decoys: Iterable[DecoyRoute] | None = None,
) -> APIRouter:
    """Build the router serving the ``auth_challenge`` decoys.

    Args:
        sensor: The deception sensor.
        gate: The tarpit gate.
        decoys: Decoys to route (defaults to the sensor's); ``not_found``
            decoys are skipped -- they are served by
            :class:`SilentTripwireMiddleware`, never by a route.

    Returns:
        A router whose routes are excluded from the OpenAPI schema.
    """
    router = APIRouter()
    for decoy in sensor.decoys if decoys is None else decoys:
        if decoy.response != "auth_challenge":
            continue
        handler = _decoy_handler(decoy, sensor, gate)
        # One route per method, as Mercury's routers declare them: a 405 then
        # lists the first route's method, exactly as on a real path.
        for method in decoy.methods:
            router.add_api_route(
                decoy.path,
                handler,
                methods=[method],
                include_in_schema=False,
                response_model=None,
            )
    return router


class SilentTripwireMiddleware:
    """Observe-only ASGI middleware for ``not_found`` decoys.

    Records a touch of a silent-tripwire path (any method, trailing slash
    ignored), holds through the tarpit, then passes the request on unchanged.
    It never builds a response: the application answers, so the answer is the
    router's own 404. A recording failure is logged and the request passes
    through all the same.

    Args:
        app: The wrapped ASGI application.
        sensor: The deception sensor.
        gate: The tarpit gate.
        decoys: The silent tripwires to observe.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        sensor: DeceptionSensor,
        gate: TarpitGate,
        decoys: Iterable[DecoyRoute],
    ) -> None:
        """Index the silent tripwires by normalised path."""
        self.app = app
        self._sensor = sensor
        self._gate = gate
        self._by_key = {
            _silent_key(decoy.path): decoy for decoy in decoys if decoy.response == "not_found"
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Observe a matching request, then delegate to the application."""
        if scope["type"] == "http":
            decoy = self._by_key.get(_silent_key(scope.get("path", "")))
            if decoy is not None:
                await self._observe(scope, decoy)
        await self.app(scope, receive, send)

    async def _observe(self, scope: Scope, decoy: DecoyRoute) -> None:
        try:
            conn = HTTPConnection(scope)
            event = self._sensor.record(
                _client_ip(conn),
                decoy.path,
                str(scope.get("method", "GET")),
                credential_presented=_credential_presented(conn),
            )
            await self._gate.hold(event.delay_s)
        except Exception:
            logger.exception("silent tripwire %s failed to record; passing through", decoy.path)
            self._sensor.note_failure("tripwire")


def _served_by_app(app: FastAPI, path: str) -> bool:
    """Whether any route already registered on ``app`` matches ``path``.

    Uses each route's own ``matches`` (so path parameters, mounts and lazily
    included routers all count), for the path and its trailing-slash twin.
    """
    twin = path.rstrip("/") if path.endswith("/") else path + "/"
    for candidate in (path, twin):
        for method in ("GET", "POST"):
            scope: Scope = {
                "type": "http",
                "path": candidate,
                "root_path": "",
                "method": method,
                "headers": [],
                "query_string": b"",
            }
            for route in app.router.routes:
                match, _ = route.matches(scope)
                if match is not Match.NONE:
                    return True
    return False


def register_decoys(
    app: FastAPI,
    *,
    enabled: bool | None = None,
    sensor: DeceptionSensor | None = None,
    gate: TarpitGate | None = None,
) -> DeceptionSensor | None:
    """Register the decoy surface on ``app`` when enabled.

    Call after every real route is registered, so collisions can be checked
    against all of them.

    Args:
        app: The application.
        enabled: Override for :func:`deception_enabled` (``None`` reads the
            environment).
        sensor: Sensor to use; defaults to :func:`configured_decoys` (the
            shipped decoys plus this deployment's private ones) over the shared
            tamper-evident audit chain, with cross-worker tripwire state when
            ``MERCURY_KEYSTORE_PATH`` is set.
        gate: Tarpit gate; defaults to the sensor policy's concurrency cap.

    Returns:
        The sensor (also exposed as ``app.state.deception_sensor`` for
        consumers that grade a source; the active decoys are
        ``app.state.deception_decoys``), or ``None`` when disabled -- in which
        case nothing is registered.

    Raises:
        RuntimeError: ``app`` has already started (its middleware stack is
            built), so a silent tripwire could no longer be installed.
    """
    if not (deception_enabled() if enabled is None else enabled):
        return None
    if app.middleware_stack is not None:
        raise RuntimeError("Cannot add middleware after an application has started")
    if sensor is None:
        from omni_mercury_engine.api import platform_metrics
        from omni_mercury_engine.api.auth_audit import shared_secure_logger
        from omni_mercury_engine.security.deception_store import build_tripwire_store

        sensor = DeceptionSensor(
            decoys=configured_decoys(),
            audit_sink=secure_audit_sink(shared_secure_logger()),
            store=build_tripwire_store(),
            failure_hook=platform_metrics.record_deception_failure,
        )
    if gate is None:
        gate = TarpitGate(sensor.policy.max_concurrent)
    active: list[DecoyRoute] = []
    for decoy in sensor.decoys:
        if _served_by_app(app, decoy.path):
            logger.error(
                "decoy %s is served by a real route; refusing it so legitimate "
                "traffic is never recorded as probing",
                decoy.path,
            )
        else:
            active.append(decoy)
    app.include_router(build_decoy_router(sensor, gate, active))
    silent = [decoy for decoy in active if decoy.response == "not_found"]
    if silent:
        # Innermost user middleware, not add_middleware's outermost slot: the
        # tripwire must observe only what the application answers, behind the
        # rate limiter and every other outer layer.
        app.user_middleware.append(
            Middleware(SilentTripwireMiddleware, sensor=sensor, gate=gate, decoys=silent)
        )
    app.state.deception_sensor = sensor
    app.state.deception_decoys = tuple(active)
    return sensor

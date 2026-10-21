# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The opt-in decoy surface (``MERCURY_DECEPTION_ENABLED``).

Each decoy resolves the client exactly as the rate limiter does, records a
tripwire, holds the response through the bounded tarpit, and then answers by
raising the same ``HTTPException`` a real route raises, so the application's
own exception handling renders a decoy and a real route identically:

* ``auth_challenge`` decoys answer as Mercury's API data surface (models,
  export, batch, assessment) answers an anonymous caller -- measured across
  the live application, not assumed -- and a differential test pins the decoy
  to a real protected route, so a change to the real challenge fails CI
  instead of quietly fingerprinting every decoy;
* ``not_found`` decoys raise the router's own 404.

Fail closed: any internal failure answers with the plain 404 of a path that
does not exist. A decoy never answers 500, never leaks, and never takes the
request down with it. Left disabled, no decoy route is registered at all.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, Response

from omni_mercury_engine.api.client_ip import resolve_client_ip
from omni_mercury_engine.security.deception import (
    DeceptionSensor,
    DecoyRoute,
    TarpitGate,
    TripwireEvent,
    deception_enabled,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI

    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

logger = logging.getLogger(__name__)

__all__ = ["build_decoy_router", "register_decoys", "secure_audit_sink"]

#: The challenge Mercury's protected API routes return to a caller with no
#: valid credential (the ``_get_current_user`` dependencies of the models,
#: export, batch and assessment routers).
_AUTH_REQUIRED_DETAIL = "Authentication required"
_AUTH_CHALLENGE = "Bearer, ApiKey"

#: Headers whose presence marks a request as presenting credentials.
_CREDENTIAL_HEADERS = ("x-api-key", "authorization")


def secure_audit_sink(
    secure_logger: SecureAuditLogger | None,
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
            client_ip = resolve_client_ip(
                request.client.host if request.client else None,
                request.headers.get("x-forwarded-for"),
            )
            event = sensor.record(
                client_ip,
                decoy.path,
                request.method,
                credential_presented=any(h in request.headers for h in _CREDENTIAL_HEADERS),
            )
            await gate.hold(event.delay_s)
        except Exception:
            logger.exception("decoy %s failed; answering as a missing path", decoy.path)
            raise HTTPException(status_code=404) from None
        if decoy.response == "auth_challenge":
            raise HTTPException(
                status_code=401,
                detail=_AUTH_REQUIRED_DETAIL,
                headers={"WWW-Authenticate": _AUTH_CHALLENGE},
            )
        raise HTTPException(status_code=404)

    return handler


def build_decoy_router(sensor: DeceptionSensor, gate: TarpitGate) -> APIRouter:
    """Build the router serving every decoy of ``sensor``.

    Args:
        sensor: The deception sensor.
        gate: The tarpit gate.

    Returns:
        A router whose routes are excluded from the OpenAPI schema.
    """
    router = APIRouter()
    for decoy in sensor.decoys:
        router.add_api_route(
            decoy.path,
            _decoy_handler(decoy, sensor, gate),
            methods=list(decoy.methods),
            include_in_schema=False,
            response_model=None,
        )
    return router


def register_decoys(
    app: FastAPI,
    *,
    enabled: bool | None = None,
    sensor: DeceptionSensor | None = None,
    gate: TarpitGate | None = None,
) -> DeceptionSensor | None:
    """Register the decoy surface on ``app`` when enabled.

    Args:
        app: The application.
        enabled: Override for :func:`deception_enabled` (``None`` reads the
            environment).
        sensor: Sensor to use; defaults to the shipped decoys over the shared
            tamper-evident audit chain.
        gate: Tarpit gate; defaults to the sensor policy's concurrency cap.

    Returns:
        The sensor (also exposed as ``app.state.deception_sensor`` for
        consumers that grade a source), or ``None`` when disabled -- in which
        case nothing is registered.
    """
    if not (deception_enabled() if enabled is None else enabled):
        return None
    if sensor is None:
        from omni_mercury_engine.api.auth_audit import shared_secure_logger

        sensor = DeceptionSensor(audit_sink=secure_audit_sink(shared_secure_logger()))
    if gate is None:
        gate = TarpitGate(sensor.policy.max_concurrent)
    app.include_router(build_decoy_router(sensor, gate))
    app.state.deception_sensor = sensor
    return sensor

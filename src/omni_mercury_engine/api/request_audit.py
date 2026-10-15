# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-request audit trail for the HTTP API.

``record_audit_log`` (``api/routes/export.py``) had no caller, so
``GET /export/audit-logs`` and the export statistics always reported an empty
audit trail. :func:`audit_request` is called by ``CorrelationIDMiddleware`` once
per finished request and writes one entry to:

* the export store (what ``/export/audit-logs`` serves), and
* the tamper-evident :class:`~omni_mercury_engine.security.secure_audit_logging.SecureAuditLogger`
  shared with the auth auditor, when ``MERCURY_AUDIT_LOG_DIR`` is set.

What is recorded: the authenticated principal (``request.state.user``, set by
the auth dependencies while the route ran; ``anonymous`` otherwise), the
method and route *template* (never the raw path), path parameters, status,
duration, the trusted-proxy-resolved client IP, the correlation id and the
*names* of query parameters -- never their values or the body, which can carry
credentials or personal data.

What is not: probe and documentation endpoints, requests that matched no
route (scanner noise; still counted by ``http_requests_total`` as
``__unmatched__``), and rate-limit rejections (429) -- an abusive client must
not be able to flush the bounded audit window with rejected requests; those
are metered, not audited. Auditing is best-effort: a sink failure is logged
and never changes the response.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["audit_request", "audit_target"]

#: Route templates that are never audited (probes, scrape target, API docs).
_UNAUDITED_ROUTES = frozenset(
    {"/health", "/health/live", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}
)
_RATE_LIMITED = 429


def audit_target(request: Any) -> tuple[str, str, str] | None:
    """Return ``(action, resource_type, resource_id)``, or ``None`` if not audited.

    ``action`` is ``"<METHOD> <route template>"``; ``resource_type`` is the
    first literal path segment after any ``api`` / version prefix;
    ``resource_id`` lists the path parameters as ``name=value`` pairs.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not isinstance(template, str) or template in _UNAUDITED_ROUTES:
        return None
    literals = [seg for seg in template.strip("/").split("/") if seg and not seg.startswith("{")]
    while literals and (literals[0] == "api" or _is_version(literals[0])):
        literals.pop(0)
    resource_type = literals[0] if literals else "root"
    params = request.scope.get("path_params") or {}
    resource_id = ",".join(f"{name}={params[name]}" for name in sorted(params))
    return f"{request.method} {template}", resource_type, resource_id


def _is_version(segment: str) -> bool:
    return len(segment) > 1 and segment[0] == "v" and segment[1:].isdigit()


async def audit_request(request: Any, status_code: int, duration_ms: float) -> None:
    """Record one finished API request (never raises)."""
    if status_code == _RATE_LIMITED:
        return
    try:
        target = audit_target(request)
        if target is None:
            return
        action, resource_type, resource_id = target

        from omni_mercury_engine.api.auth_audit import shared_secure_logger
        from omni_mercury_engine.api.client_ip import resolve_client_ip
        from omni_mercury_engine.api.routes.export import record_audit_log

        user = getattr(request.state, "user", None)
        user_id = str(getattr(user, "id", None) or "anonymous")
        user_meta = getattr(user, "metadata", None) or {}
        client_ip = resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )
        correlation_id = getattr(request.state, "correlation_id", None)
        metadata = {
            "correlation_id": correlation_id,
            "auth_method": user_meta.get("auth_method") if isinstance(user_meta, dict) else None,
            "query_params": sorted(request.query_params.keys()),
        }
        await record_audit_log(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request=request,
            response_status=status_code,
            duration_ms=duration_ms,
            metadata=metadata,
            client_ip=client_ip,
        )

        secure = shared_secure_logger()
        if secure is not None:
            from omni_mercury_engine.security.secure_audit_logging import (
                AuditEventCategory,
                AuditEventSeverity,
            )

            secure.log(
                category=(
                    AuditEventCategory.DATA_ACCESS
                    if request.method in ("GET", "HEAD", "OPTIONS")
                    else AuditEventCategory.DATA_MODIFICATION
                ),
                action=action,
                outcome="success" if status_code < 400 else "failure",
                severity=(
                    AuditEventSeverity.WARNING
                    if status_code in (401, 403) or status_code >= 500
                    else AuditEventSeverity.INFO
                ),
                actor=user_id,
                resource=f"{resource_type}:{resource_id}" if resource_id else resource_type,
                details={
                    "status": status_code,
                    "duration_ms": round(duration_ms, 3),
                    "query_params": metadata["query_params"],
                },
                session_id=correlation_id,
                client_ip=client_ip,
            )
    except Exception:  # pragma: no cover - auditing must never break a request
        logger.exception("request audit failed")

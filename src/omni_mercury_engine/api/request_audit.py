# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request audit trail for the HTTP API.

``record_audit_log`` (``api/routes/export.py``) had no caller, so
``GET /export/audit-logs`` and the export statistics always reported an empty
audit trail. :func:`audit_request` is called by ``CorrelationIDMiddleware`` once
per finished request and writes one entry to:

* the export store (what ``/export/audit-logs`` serves), and
* the tamper-evident :class:`~omni_mercury_engine.security.secure_audit_logging.SecureAuditLogger`
  shared with the auth auditor, when ``MERCURY_AUDIT_LOG_DIR`` is set.

What is recorded: the authenticated principal (``request.state.user``, set by
the API-key / JWT dependencies, or the session-cookie actor published with
:func:`set_audit_actor`; ``anonymous`` otherwise), the method and route
*template* (never the raw path), path parameters, status, duration, the
trusted-proxy-resolved client IP, the correlation id and the *names* of query
parameters -- never their values or the body, which can carry credentials or
personal data. Every caller-controlled string is clipped
(:data:`MAX_FIELD_CHARS`, :data:`MAX_PARAM_CHARS`, :data:`MAX_QUERY_NAMES`)
so one request cannot make an entry arbitrarily large.

What is not:

* probe and documentation endpoints, and requests that matched no route
  (scanner noise; still counted by ``http_requests_total`` as
  ``__unmatched__``);
* rate-limit rejections (429) -- an abusive client must not be able to flush
  the bounded audit window with rejected requests; those are metered, not
  audited;
* **unauthenticated client errors** (4xx with no principal). The secure sink
  is the same hash chain that holds login / 2FA evidence and it rotates by
  size, so letting anyone append to it -- with path parameters of their
  choosing -- would let an anonymous flood rotate that evidence into
  retention. These requests are still metered by ``http_requests_total`` and
  tallied here in a fixed-size counter (:func:`anonymous_rejection_counts`);
  authentication *failures* themselves are recorded by the auth auditor.

Auditing is best-effort: a sink failure is logged and never changes the
response.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ANONYMOUS",
    "MAX_FIELD_CHARS",
    "MAX_PARAM_CHARS",
    "MAX_QUERY_NAMES",
    "anonymous_rejection_counts",
    "audit_request",
    "audit_target",
    "clip",
    "set_audit_actor",
]

#: Actor recorded when no principal authenticated the request.
ANONYMOUS = "anonymous"
#: Cap for a free-text field (user agent, route template, resource id).
MAX_FIELD_CHARS = 256
#: Cap for one path-parameter value or one query-parameter name.
MAX_PARAM_CHARS = 64
#: At most this many query-parameter names are kept per entry.
MAX_QUERY_NAMES = 16
#: Distinct ``(action, status)`` keys tracked for anonymous rejections; any
#: further keys share the :data:`_OVERFLOW_KEY` bucket.
_MAX_REJECTION_KEYS = 256
_OVERFLOW_KEY = ("__other__", 0)

#: Route templates that are never audited (probes, scrape target, API docs).
_UNAUDITED_ROUTES = frozenset(
    {"/health", "/health/live", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}
)
_RATE_LIMITED = 429

_rejections: Counter[tuple[str, int]] = Counter()
_rejections_lock = threading.Lock()


def clip(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    """Return ``value`` cut to ``limit`` characters (marked with ``...``).

    Args:
        value: The string to bound.
        limit: Maximum length of the result.

    Returns:
        ``value`` unchanged when short enough, else its prefix ending in
        ``"..."`` so a reader can tell it was clipped.
    """
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def set_audit_actor(request: Any, actor_id: str, auth_method: str) -> None:
    """Publish the request's principal for auditing without a :class:`User`.

    Session-cookie routes authenticate an account rather than an API
    credential, so they do not populate ``request.state.user`` (whose
    permission set other code consults). They call this instead, so their
    actions are attributed to the account rather than ``anonymous``.

    Args:
        request: The Starlette request.
        actor_id: The authenticated account id.
        auth_method: How it authenticated (e.g. ``"session"``).
    """
    request.state.audit_actor = (str(actor_id), auth_method)


def anonymous_rejection_counts() -> dict[str, int]:
    """Snapshot of unauthenticated 4xx responses by ``"<action> -> <status>"``.

    Returns:
        A copy of the bounded tally (at most ``_MAX_REJECTION_KEYS`` + 1 keys).
    """
    with _rejections_lock:
        return {f"{action} -> {status}": n for (action, status), n in _rejections.items()}


def _count_anonymous_rejection(action: str, status_code: int) -> None:
    key = (action, status_code)
    with _rejections_lock:
        if key not in _rejections and len(_rejections) >= _MAX_REJECTION_KEYS:
            key = _OVERFLOW_KEY
        _rejections[key] += 1


def audit_target(request: Any) -> tuple[str, str, str] | None:
    """Return ``(action, resource_type, resource_id)``, or ``None`` if not audited.

    ``action`` is ``"<METHOD> <route template>"``; ``resource_type`` is the
    first literal path segment after any ``api`` / version prefix;
    ``resource_id`` lists the path parameters as ``name=value`` pairs, each
    value clipped to :data:`MAX_PARAM_CHARS` and the whole to
    :data:`MAX_FIELD_CHARS`.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not isinstance(template, str) or template in _UNAUDITED_ROUTES:
        return None
    literals = [seg for seg in template.strip("/").split("/") if seg and not seg.startswith("{")]
    while literals and (literals[0] == "api" or _is_version(literals[0])):
        literals.pop(0)
    resource_type = clip(literals[0], MAX_PARAM_CHARS) if literals else "root"
    params = request.scope.get("path_params") or {}
    resource_id = ",".join(
        f"{clip(str(name), MAX_PARAM_CHARS)}={clip(str(params[name]), MAX_PARAM_CHARS)}"
        for name in sorted(params)
    )
    return f"{request.method} {clip(template)}", resource_type, clip(resource_id)


def _is_version(segment: str) -> bool:
    return len(segment) > 1 and segment[0] == "v" and segment[1:].isdigit()


def _principal(request: Any) -> tuple[str | None, str | None]:
    """``(actor id, auth method)`` for the request; ``(None, None)`` if anonymous."""
    user = getattr(request.state, "user", None)
    user_id = getattr(user, "id", None)
    if user_id:
        meta = getattr(user, "metadata", None) or {}
        method = meta.get("auth_method") if isinstance(meta, dict) else None
        return clip(str(user_id)), method
    actor = getattr(request.state, "audit_actor", None)
    if isinstance(actor, tuple) and len(actor) == 2 and actor[0]:
        return clip(str(actor[0])), str(actor[1])
    return None, None


def _query_names(request: Any) -> tuple[list[str], int]:
    """Sorted, clipped query-parameter names and how many were dropped."""
    names = sorted({str(name) for name in request.query_params})
    kept = [clip(name, MAX_PARAM_CHARS) for name in names[:MAX_QUERY_NAMES]]
    return kept, max(len(names) - MAX_QUERY_NAMES, 0)


async def audit_request(request: Any, status_code: int, duration_ms: float) -> None:
    """Record one finished API request (never raises)."""
    if status_code == _RATE_LIMITED:
        return
    try:
        target = audit_target(request)
        if target is None:
            return
        action, resource_type, resource_id = target

        principal, auth_method = _principal(request)
        if principal is None and 400 <= status_code < 500:
            _count_anonymous_rejection(action, status_code)
            return
        user_id = principal or ANONYMOUS

        from omni_mercury_engine.api.auth_audit import shared_secure_logger
        from omni_mercury_engine.api.client_ip import resolve_client_ip
        from omni_mercury_engine.api.routes.export import record_audit_log

        client_ip = resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )
        correlation_id = getattr(request.state, "correlation_id", None)
        if isinstance(correlation_id, str):
            correlation_id = clip(correlation_id, MAX_PARAM_CHARS)
        query_names, query_dropped = _query_names(request)
        metadata: dict[str, Any] = {
            "correlation_id": correlation_id,
            "auth_method": auth_method,
            "query_params": query_names,
        }
        if query_dropped:
            metadata["query_params_dropped"] = query_dropped
        route_template = action.split(" ", 1)[1]
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
            request_path=route_template,
        )

        secure = shared_secure_logger()
        if secure is not None:
            from omni_mercury_engine.security.secure_audit_logging import (
                AuditEventCategory,
                AuditEventSeverity,
            )

            details: dict[str, Any] = {
                "status": status_code,
                "duration_ms": round(duration_ms, 3),
                "query_params": query_names,
            }
            if query_dropped:
                details["query_params_dropped"] = query_dropped
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
                details=details,
                session_id=correlation_id,
                client_ip=client_ip,
            )
    except Exception:  # pragma: no cover - auditing must never break a request
        logger.exception("request audit failed")

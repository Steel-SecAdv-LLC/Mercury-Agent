# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit logging for authentication and account-lifecycle events.

Every security-relevant account event — login success/failure, registration,
password change/reset, 2FA enable/disable, recovery-code use, email change,
deletion — flows through :class:`AuthAuditor` so there is exactly one place
that decides *what* is recorded and *where*:

* With ``MERCURY_AUDIT_LOG_DIR`` set, events go to the tamper-evident
  :class:`~omni_mercury_engine.security.secure_audit_logging.SecureAuditLogger`
  (hash-chained, HMAC-signed JSONL) rooted at that directory — the compliance
  posture for a public deployment.
* Without it, events fall back to structured stdlib logging. A solo
  self-hoster gets a searchable trail with no surprise directories appearing
  on disk; the fallback is explicit, not an accident.

Events carry the **account id**, never the email — the audit trail must not
become a PII store (the SecureAuditLogger additionally masks anything that
slips through). Auditing is best-effort by design: a failure to record must
never turn into a failed login, so every sink call is wrapped and logged.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

logger = logging.getLogger(__name__)

__all__ = ["AUDIT_DIR_ENV", "AuthAuditor", "build_auth_auditor"]

AUDIT_DIR_ENV = "MERCURY_AUDIT_LOG_DIR"


class AuthAuditor:
    """Records auth events to the secure audit trail (or structured logs)."""

    def __init__(self, secure_logger: SecureAuditLogger | None = None) -> None:
        """Wire the auditor to its sink.

        Args:
            secure_logger: Tamper-evident sink; ``None`` selects the
                structured-logging fallback.
        """
        self._secure = secure_logger

    def record(
        self,
        action: str,
        *,
        outcome: str,
        account_id: str | None = None,
        client_ip: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record one auth event, never raising into the caller.

        Args:
            action: Stable event name (``login``, ``password_change``, ...).
            outcome: ``"success"`` or ``"failure"``.
            account_id: The acted-on account (omitted when unknown, e.g. a
                login attempt against a nonexistent email).
            client_ip: Trusted-proxy-resolved client address, when the route
                layer has it.
            details: Small, non-PII extras (e.g. the failure reason).
        """
        try:
            if self._secure is not None:
                self._secure.log_authentication(
                    action=action,
                    actor=account_id or "unknown",
                    outcome=outcome,
                    details=details,
                    client_ip=client_ip,
                )
            else:
                logger.info(
                    "auth-audit action=%s outcome=%s account=%s ip=%s details=%s",
                    action,
                    outcome,
                    account_id or "unknown",
                    client_ip or "-",
                    details or {},
                )
        except Exception:  # pragma: no cover - the sink must never break auth
            logger.exception("auth audit sink failed for action=%s", action)


def build_auth_auditor() -> AuthAuditor:
    """Construct the auditor from the environment.

    Returns:
        An :class:`AuthAuditor` over a :class:`SecureAuditLogger` rooted at
        ``MERCURY_AUDIT_LOG_DIR`` when that is set, else the logging fallback.
    """
    audit_dir = os.getenv(AUDIT_DIR_ENV, "").strip()
    if not audit_dir:
        return AuthAuditor(None)
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

    return AuthAuditor(SecureAuditLogger(log_dir=audit_dir))

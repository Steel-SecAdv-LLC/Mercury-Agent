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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIT_DIR_ENV",
    "AUDIT_MAX_FILES_ENV",
    "AUDIT_RETENTION_DAYS_ENV",
    "AUDIT_ROTATE_SIZE_MB_ENV",
    "AuthAuditor",
    "audit_retention_days",
    "build_auth_auditor",
    "prune_rotated_audit_segments",
]

AUDIT_DIR_ENV = "MERCURY_AUDIT_LOG_DIR"
#: Rotate the active audit log once it reaches this many megabytes.
AUDIT_ROTATE_SIZE_MB_ENV = "MERCURY_AUDIT_ROTATE_SIZE_MB"
#: Keep at most this many rotated segments (count-based cap; oldest deleted).
AUDIT_MAX_FILES_ENV = "MERCURY_AUDIT_MAX_FILES"
#: Additionally delete rotated segments older than this many days (time-based
#: retention for compliance). Unset/0 keeps only the count-based cap.
AUDIT_RETENTION_DAYS_ENV = "MERCURY_AUDIT_RETENTION_DAYS"

#: Defaults mirror :class:`SecureAuditLogger` (100 MB × 10 files ≈ 1 GB cap).
_DEFAULT_ROTATE_SIZE_MB = 100.0
_DEFAULT_MAX_FILES = 10


def _positive_float_env(name: str, default: float) -> float:
    """Read a positive float from ``name`` (default on unset/malformed/≤0)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive int from ``name`` (default on unset/malformed/≤0)."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def audit_retention_days() -> float:
    """Time-based audit retention in days; ``0.0`` (default) disables it."""
    raw = os.getenv(AUDIT_RETENTION_DAYS_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def prune_rotated_audit_segments(audit_dir: str | os.PathLike[str], older_than_epoch: float) -> int:
    """Delete rotated audit segments (+ their ``.sha256``) older than a cutoff.

    Only whole *rotated* segments (``audit_*.jsonl``) are removed — never the
    active ``audit.jsonl``, and never individual lines — so the hash chain of
    every retained segment stays intact and independently verifiable. Each
    segment's SHA-256 sidecar is removed alongside it. Failures on one segment
    are logged and skipped so retention never takes the sweep (or the API)
    down.

    Args:
        audit_dir: The audit log directory.
        older_than_epoch: UNIX seconds; segments last modified before this are
            deleted.

    Returns:
        The number of segment files deleted.
    """
    directory = Path(audit_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for segment in directory.glob("audit_*.jsonl"):
        try:
            if segment.stat().st_mtime >= older_than_epoch:
                continue
            segment.unlink(missing_ok=True)
            segment.with_suffix(".sha256").unlink(missing_ok=True)
            removed += 1
        except OSError:
            logger.exception("failed to prune audit segment %s", segment)
    return removed


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

    Rotation retention is operator-configurable: ``MERCURY_AUDIT_ROTATE_SIZE_MB``
    sets the per-file rotation threshold and ``MERCURY_AUDIT_MAX_FILES`` the
    count-based cap on retained rotated segments (both default to the
    :class:`SecureAuditLogger` values, ≈1 GB total). Time-based retention is a
    separate, sweep-driven concern (see :func:`prune_rotated_audit_segments` and
    ``MERCURY_AUDIT_RETENTION_DAYS``).

    Returns:
        An :class:`AuthAuditor` over a :class:`SecureAuditLogger` rooted at
        ``MERCURY_AUDIT_LOG_DIR`` when that is set, else the logging fallback.
    """
    audit_dir = os.getenv(AUDIT_DIR_ENV, "").strip()
    if not audit_dir:
        return AuthAuditor(None)
    from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

    return AuthAuditor(
        SecureAuditLogger(
            log_dir=audit_dir,
            rotate_size_mb=_positive_float_env(AUDIT_ROTATE_SIZE_MB_ENV, _DEFAULT_ROTATE_SIZE_MB),
            max_rotated_files=_positive_int_env(AUDIT_MAX_FILES_ENV, _DEFAULT_MAX_FILES),
        )
    )

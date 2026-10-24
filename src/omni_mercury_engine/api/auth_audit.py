# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
import threading
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
    "shared_secure_logger",
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

    Only whole files are removed, never individual lines, so the hash chain of
    every retained file stays intact and independently verifiable:

    * rotated segments (``audit_*.jsonl``, every writer's) and their SHA-256
      sidecars;
    * a finished worker's log (``audit-<pid>.jsonl``), only once its writer no
      longer runs (its ``.audit-<pid>.lock`` is free, or without one its pid is
      gone) -- a live worker's active log is never a target however old --
      together with that stale lock file;
    * crash fragments a writer set aside on restart (``*.torn-<usec>``).

    The primary's active ``audit.jsonl`` is never removed. Failures on one file
    are logged and skipped so retention never takes the sweep (or the API) down.

    Args:
        audit_dir: The audit log directory.
        older_than_epoch: UNIX seconds; files last modified before this are
            deleted.

    Returns:
        The number of log files deleted (sidecars and lock files not counted).
    """
    from omni_mercury_engine.security.secure_audit_logging import (
        _WORKER_LOG_RE,
        _worker_lock_path,
        _writer_is_live,
    )

    directory = Path(audit_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in sorted(directory.iterdir()):
        name = path.name
        worker = _WORKER_LOG_RE.fullmatch(name)
        if not (name.startswith("audit_") and name.endswith(".jsonl")) and worker is None:
            if ".torn-" not in name:
                continue
        try:
            if not path.is_file() or path.stat().st_mtime >= older_than_epoch:
                continue
            if worker is not None:
                pid = int(worker.group("pid"))
                if _writer_is_live(directory, pid):
                    continue
                _worker_lock_path(directory, pid).unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            if name.endswith(".jsonl"):
                path.with_suffix(".sha256").unlink(missing_ok=True)
            removed += 1
        except OSError:
            logger.exception("failed to prune audit file %s", path)
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


_shared_loggers: dict[Path, SecureAuditLogger] = {}
_shared_loggers_lock = threading.Lock()


def shared_secure_logger() -> SecureAuditLogger | None:
    """The process's one tamper-evident sink for ``MERCURY_AUDIT_LOG_DIR``.

    Every auditor in the process (auth events, per-request API audit) must
    append to the *same* :class:`SecureAuditLogger`: two loggers on one
    directory would each run their own hash chain, and the second would be
    pushed onto a per-process file by the directory's writer lock. Returns
    ``None`` when ``MERCURY_AUDIT_LOG_DIR`` is unset. Rotation settings are
    read when a directory's logger is first built.
    """
    audit_dir = os.getenv(AUDIT_DIR_ENV, "").strip()
    if not audit_dir:
        return None
    key = Path(audit_dir).resolve()
    with _shared_loggers_lock:
        secure = _shared_loggers.get(key)
        if secure is None:
            from omni_mercury_engine.security.secure_audit_logging import SecureAuditLogger

            secure = SecureAuditLogger(
                log_dir=key,
                rotate_size_mb=_positive_float_env(
                    AUDIT_ROTATE_SIZE_MB_ENV, _DEFAULT_ROTATE_SIZE_MB
                ),
                max_rotated_files=_positive_int_env(AUDIT_MAX_FILES_ENV, _DEFAULT_MAX_FILES),
            )
            _shared_loggers[key] = secure
        return secure


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
    return AuthAuditor(shared_secure_logger())

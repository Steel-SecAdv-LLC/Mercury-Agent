# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Durable, append-only audit log for harm-gate decisions.

The harm-policy spec calls for a persisted refusal/decision audit trail (domain,
intent, signals, disposition). Before this module those decisions were only
``logger.info`` -- lost when the process exits. Here they are written durably:

* **Primary sink** -- an append-only JSON-Lines file (one decision per line,
  flushed + ``fsync``\\ed) at ``$MERCURY_GATE_AUDIT_LOG`` or, by default,
  ``<repo>/artifacts/audit/gate_decisions.jsonl``. Simple, dependency-free, and
  survives process exit.
* **Tamper-evident sink (opt-in)** -- when ``MERCURY_GATE_AUDIT_SECURELOG=1`` the
  decision is *also* forwarded to the hash-chained, PII-masking
  :class:`~omni_mercury_engine.security.secure_audit_logging.SecureAuditLogger`,
  giving integrity-verifiable durability where an operator wants it.

Every write is fail-safe: an audit failure is logged and swallowed, never raised
into the gate -- auditing must not be able to break the control it records.
Disable entirely with ``MERCURY_GATE_AUDIT_DISABLED=1``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MAX_FIELD_CHARS = 800  # cap persisted free-text so a full procedure is not stored verbatim


def _default_log_path() -> Path:
    """Resolve the JSONL sink path (env override, else repo artifacts/audit/)."""
    env = os.environ.get("MERCURY_GATE_AUDIT_LOG", "").strip()
    if env:
        return Path(env)
    # .../src/omni_mercury_engine/cognitive/gate_audit.py -> repo root is 3 up.
    return Path(__file__).resolve().parents[3] / "artifacts" / "audit" / "gate_decisions.jsonl"


def _truncate(value: Any) -> Any:
    """Cap long strings so the audit log records the decision, not a full payload."""
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return value[:_MAX_FIELD_CHARS] + f"...[+{len(value) - _MAX_FIELD_CHARS} chars]"
    return value


def record_gate_decision(
    *,
    decision: str,
    source: str,
    disposition: str,
    hazard_domain: str = "none",
    intent: str = "mechanism",
    signals: tuple[str, ...] | list[str] = (),
    reason: str = "",
    query: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Durably record one harm-gate decision (fail-safe; never raises).

    Args:
        decision: The acted-on outcome -- e.g. ``"refused"``, ``"escalated"``,
            ``"allow_provenance"``, ``"redacted"``, ``"approved"``.
        source: Which control emitted it (``"weapons_gate"``, ``"aggregate_gate"``,
            ``"assistant"``, ``"escalation_broker"``).
        disposition: The gate disposition value.
        hazard_domain: Axis-A hazard domain value.
        intent: Axis-B operational-intent value.
        signals: Matched audit-signal labels.
        reason: Human-readable reason string.
        query: The action/query text (capped when persisted).
        extra: Any additional structured fields.
    """
    if os.environ.get("MERCURY_GATE_AUDIT_DISABLED") == "1":
        return
    record = {
        "ts": time.time(),
        "decision": decision,
        "source": source,
        "disposition": disposition,
        "hazard_domain": hazard_domain,
        "intent": intent,
        "signals": list(signals),
        "reason": _truncate(reason),
    }
    if query is not None:
        record["query"] = _truncate(query)
    if extra:
        record.update({k: _truncate(v) for k, v in extra.items()})

    _write_jsonl(record)
    _forward_secure(record)


def _write_jsonl(record: dict[str, Any]) -> None:
    """Append one record to the durable JSONL sink (flushed + fsynced)."""
    try:
        path = _default_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with _LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:  # pragma: no cover - durability is best-effort, never fatal
        logger.warning("gate audit: durable JSONL write failed (%s); decision=%s", exc, record)


def _forward_secure(record: dict[str, Any]) -> None:
    """Best-effort forward to the hash-chained SecureAuditLogger (opt-in)."""
    if os.environ.get("MERCURY_GATE_AUDIT_SECURELOG") != "1":
        return
    try:
        from omni_mercury_engine.security.secure_audit_logging import get_audit_logger

        get_audit_logger().log_security_incident(
            action=f"harm_gate:{record.get('decision', 'decision')}",
            details=record,
            resource=record.get("source", "harm_gate"),
        )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.info("gate audit: secure-log forward unavailable (%s)", exc)


__all__ = ["record_gate_decision"]

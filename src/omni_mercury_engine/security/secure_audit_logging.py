# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Secure Audit Logging System.

Production-grade secure audit logging providing:
- Cryptographically signed audit logs
- Tamper-evident log chains (hash-linked)
- Secure log rotation with integrity verification
- PII masking and sanitization
- Compliance-ready audit trail (SOC2, HIPAA, GDPR)
- Constant-time comparisons for sensitive operations
- Rate limiting for audit log access
- Encrypted log storage option

This addresses the security audit finding: "No Audit Log Integrity"

On-disk layout (one directory, possibly several writer processes):

* ``audit.jsonl`` -- the active log of the *primary* writer, the one process
  holding the exclusive ``.audit.lock`` flock.
* ``audit-<pid>.jsonl`` -- the active log of each further process on the same
  directory (e.g. ``mercury serve --workers 4``); each holds
  ``.audit-<pid>.lock`` for its lifetime so other processes can tell a live
  writer from an orphan.
* ``audit_<YYYYmmdd_HHMMSS>.<usec>[-<n>].jsonl`` -- a rotated segment of the
  primary; ``..._<pid>.jsonl`` for a worker's. ``-<n>`` disambiguates
  rotations in the same microsecond: a rotation never overwrites a segment.
  Legacy second-resolution names (``audit_<stamp>.jsonl`` /
  ``audit_<stamp>_<pid>.jsonl``) are still recognised. Each segment has a
  ``.sha256`` sidecar written at rotation.

Every writer resumes, rotates and prunes **only its own** files (the
primary: unsuffixed segments; a worker: its ``_<pid>`` segments), so one
writer's retention can never delete another's evidence and a chain never
resumes from a different writer's segment. Files of writers that no longer
run (their ``.audit-<pid>.lock`` is free, or -- for files from before the
lock existed -- their pid is gone) are *orphans*: the primary keeps the newest
``max_rotated_files`` of them (active logs and segments pooled, by mtime) and
deletes the rest, so a fleet that restarts with fresh pids does not grow the
directory forever. Orphans are never renamed or rewritten, so each stays
verifiable.

:func:`verify_audit_directory` verifies every writer's chain read-only,
including the links *between* consecutive segments. What a hash chain can
and cannot prove: an edit, insertion, reordering or interior deletion breaks
content hashes, ``previous_hash`` links or the sequence numbers (which carry
across restarts and rotation). Removing records from the *head* (the oldest
segment, or its first lines) or the *tail* (the newest lines) leaves a
shorter chain that is still internally valid; that can only be detected
against an external anchor (a periodically exported last ``event_hash`` /
``sequence_number``), which this module does not keep.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: no advisory locks, single writer assumed
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Held (flock, exclusive) by the one process that writes ``audit.jsonl``.
_PRIMARY_LOCK_NAME = ".audit.lock"
#: Read this much from the end of a log to find its last record.
_TAIL_BLOCK = 64 * 1024


def _last_line(path: Path) -> bytes:
    """Return the last non-empty line of ``path`` without reading the whole file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        buf = b""
        pos = end
        while pos > 0:
            step = min(_TAIL_BLOCK, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            stripped = buf.rstrip(b"\n")
            if b"\n" in stripped:
                return stripped.rsplit(b"\n", 1)[1]
        return buf.rstrip(b"\n")


#: Rotated segment names, current and legacy (see the module docstring).
_SEGMENT_RE = re.compile(
    r"audit_(?P<stamp>\d{8}_\d{6})(?:\.(?P<usec>\d{6}))?(?:-(?P<dup>\d+))?"
    r"(?:_(?P<pid>\d+))?\.jsonl"
)
#: Active log of a non-primary writer.
_WORKER_LOG_RE = re.compile(r"audit-(?P<pid>\d+)\.jsonl")
#: Preview length of a dropped torn tail in the log message.
_TORN_PREVIEW_BYTES = 200


def _segment_info(path: Path) -> tuple[int | None, tuple[str, int, int]] | None:
    """``(writer pid or None for the primary, sort key)`` of a rotated segment.

    Returns ``None`` when ``path`` is not a rotated-segment name.
    """
    match = _SEGMENT_RE.fullmatch(path.name)
    if match is None:
        return None
    pid = int(match["pid"]) if match["pid"] else None
    usec = int(match["usec"]) if match["usec"] else -1
    return pid, (match["stamp"], usec, int(match["dup"] or 0))


def _writer_segments(log_dir: Path, pid: int | None) -> list[Path]:
    """Rotated segments of one writer (``pid=None``: the primary), oldest first."""
    found: list[tuple[tuple[str, int, int], float, Path]] = []
    for path in log_dir.glob("audit_*.jsonl"):
        info = _segment_info(path)
        if info is None or info[0] != pid:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        found.append((info[1], mtime, path))
    found.sort(key=lambda item: (item[0], item[1]))
    return [path for _key, _mtime, path in found]


def _worker_lock_path(log_dir: Path, pid: int) -> Path:
    return log_dir / f".audit-{pid}.lock"


def _pid_running(pid: int) -> bool:
    """Best-effort liveness of ``pid`` (errs towards *running*).

    Never probes on Windows, where ``os.kill`` terminates the target process.
    """
    if pid == os.getpid() or os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError, OverflowError):
        return True
    return True


def _writer_is_live(log_dir: Path, pid: int) -> bool:
    """Whether the writer that owns ``pid``'s files still runs.

    A live writer holds ``.audit-<pid>.lock``; if that lock can be taken the
    writer is gone. Without a lock file (files written before per-writer
    locks existed, or a platform without ``fcntl``) fall back to a pid check,
    which errs towards *live* so nothing is deleted on doubt.
    """
    if pid == os.getpid():
        return True
    lock_path = _worker_lock_path(log_dir, pid)
    if fcntl is None or not lock_path.exists():
        return _pid_running(pid)
    try:
        with open(lock_path, "a") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return True


def _rename_no_clobber(src: Path, dst: Path) -> bool:
    """Rename ``src`` to ``dst`` unless ``dst`` exists; ``True`` on success.

    ``Path.rename`` silently replaces an existing target on POSIX. A hard
    link is created atomically or fails with ``FileExistsError``, so no
    concurrent actor can slip a file in between check and rename. On a
    filesystem without hard links, fall back to check-then-rename (the
    caller is the only writer naming files in its own namespace).
    """
    try:
        os.link(src, dst)
    except FileExistsError:
        return False
    except OSError:
        if dst.exists():
            return False
        src.rename(dst)
        return True
    try:
        src.unlink()
    except OSError:
        # Both names now share one inode: appending to src would also change
        # the "rotated" segment. Undo the link and let the caller fail loudly.
        dst.unlink(missing_ok=True)
        raise
    return True


def _file_sha3(path: Path) -> str:
    """SHA3-256 hex digest of a file, streamed."""
    sha3 = hashlib.sha3_256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            sha3.update(chunk)
    return sha3.hexdigest()


def _remove_segment(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".sha256").unlink(missing_ok=True)


class AuditEventSeverity(StrEnum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"  # Security-specific events


class AuditEventCategory(StrEnum):
    """Categories of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CRYPTOGRAPHIC = "cryptographic"
    ANOMALY_DETECTION = "anomaly_detection"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class AuditEvent:
    """Immutable audit event record."""

    event_id: str
    timestamp: float
    category: AuditEventCategory
    severity: AuditEventSeverity
    action: str
    actor: str | None
    resource: str | None
    outcome: str  # 'success', 'failure', 'partial'
    details: dict[str, Any]

    # Integrity fields
    sequence_number: int
    previous_hash: str
    event_hash: str

    # Metadata
    session_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S.%f%z", time.gmtime(self.timestamp))[
                :26
            ]
            + "Z",
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action,
            "actor": self.actor,
            "resource": self.resource,
            "outcome": self.outcome,
            "details": self.details,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)


class PIIMasker:
    """Masks Personally Identifiable Information in audit logs.

    Supports common PII patterns and custom patterns.
    """

    # Common PII patterns
    # Note: Patterns are ordered from most specific to least specific
    DEFAULT_PATTERNS = {
        "email": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "***@***.***",
        ),
        "phone": (
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "***-***-****",
        ),
        "ssn": (
            r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
            "***-**-****",
        ),
        "credit_card": (
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "****-****-****-****",
        ),
        # More restrictive IP pattern: only valid octets 0-255
        "ip_address": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "***.***.***.***",
        ),
        # More specific API key patterns - only match known formats
        # AWS access key pattern (starts with AKIA)
        "aws_access_key": (
            r"\bAKIA[0-9A-Z]{16}\b",
            "[REDACTED_AWS_KEY]",
        ),
        # Generic API key with prefix markers (api_key=, apikey:, etc.)
        "api_key_prefixed": (
            r"(?:api[_-]?key|apikey|secret[_-]?key|auth[_-]?token)[=:]\s*['\"]?([A-Za-z0-9_-]{20,64})['\"]?",
            "[REDACTED_API_KEY]",
        ),
        "jwt_token": (
            r"\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b",
            "[REDACTED_JWT]",
        ),
        # Bearer tokens
        "bearer_token": (
            r"[Bb]earer\s+[A-Za-z0-9_-]{20,}",
            "Bearer [REDACTED]",
        ),
    }

    def __init__(
        self,
        enabled: bool = True,
        custom_patterns: dict[str, tuple[str, str]] | None = None,
    ):
        """Initialize PII masker.

        Args:
            enabled: Enable PII masking
            custom_patterns: Additional patterns to mask
        """
        self.enabled = enabled
        self.patterns = self.DEFAULT_PATTERNS.copy()

        # Validate and add custom patterns
        if custom_patterns:
            self._validate_patterns(custom_patterns)
            self.patterns.update(custom_patterns)

        # Compile patterns
        import re

        self._compiled: dict[str, tuple[Any, str]] = {}
        failed_patterns: list[str] = []

        for name, (pattern, replacement) in self.patterns.items():
            try:
                self._compiled[name] = (re.compile(pattern), replacement)
            except re.error as e:
                failed_patterns.append(f"{name}: {e}")
                logger.error(f"CRITICAL: Invalid PII pattern '{name}': {e}")

        # Fail fast if any patterns are invalid - PII leakage risk
        if failed_patterns:
            raise ValueError(
                f"Invalid PII masking patterns detected - refusing to start "
                f"(PII leakage risk). Fix these patterns: {failed_patterns}"
            )

    def _validate_patterns(self, patterns: dict[str, tuple[str, str]]) -> None:
        """Validate custom patterns before adding."""
        import re

        for name, (pattern, _replacement) in patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"Custom PII pattern '{name}' is invalid: {e}. "
                    f"All custom patterns must be valid regular expressions."
                )

    def mask(self, data: Any) -> Any:
        """Mask PII in data recursively."""
        if not self.enabled:
            return data

        if isinstance(data, str):
            return self._mask_string(data)
        elif isinstance(data, dict):
            return {k: self.mask(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.mask(item) for item in data]
        return data

    def _mask_string(self, text: str) -> str:
        """Mask PII patterns in string.

        On top of the heuristic patterns (which require prefix markers and
        a 20-character floor), the canonical redaction primitives run so a
        credentialed URL in an audit detail is caught STRUCTURALLY — any
        length, no prefix marker needed — and any configured env-held
        credential is caught by value (the only defence for path-segment
        keys like FIRMS).
        """
        # Lazy import — the codebase-wide pattern; redaction is stdlib-only.
        from omni_mercury_engine.security.redaction import (
            redact_env_secrets,
            redact_text,
        )

        result = text
        for _, (pattern, replacement) in self._compiled.items():
            result = pattern.sub(replacement, result)
        return redact_env_secrets(redact_text(result))


class SecureHashChain:
    """Cryptographically secure hash chain for audit log integrity.

    Each event is linked to the previous via SHA-256 hash, making tampering detectable.
    """

    def __init__(self, hmac_key: bytes | None = None):
        """Initialize hash chain.

        Args:
            hmac_key: HMAC key for signed hashes. When omitted, the key is
                derived from AMA HD Key Management (purpose ``audit_sign``) so
                the chain is verifiable across workers/replicas/restarts — see
                :meth:`_default_hmac_key`.
        """
        self.hmac_key = hmac_key if hmac_key is not None else self._default_hmac_key()
        self._previous_hash = self._compute_genesis_hash()
        self._sequence_number = 0
        self._lock = threading.Lock()

    @staticmethod
    def _default_hmac_key() -> bytes:
        """Derive the audit-chain HMAC key from AMA HD key management.

        The chain's tamper-evidence is only useful if the key survives the
        process that wrote the log. The previous default —
        ``secrets.token_bytes(32)`` — was a fresh per-process key, so an audit
        log could never be verified after a restart or by another worker, even
        though this module's contract is cross-process integrity. Sourcing the
        key from :class:`~omni_mercury_engine.api.auth.AuthKeyManager`'s
        ``audit_sign`` purpose makes it stable, rotatable, and identical
        fleet-wide whenever ``AMA_MASTER_SEED`` is set.

        Falls back to a per-process random key only if HD derivation is
        unavailable (keeps minimal environments starting), and warns when the
        derived key is itself ephemeral (no ``AMA_MASTER_SEED``) so the
        cross-process gap is visible rather than silent.
        """
        try:
            from omni_mercury_engine.api.auth import get_auth_key_manager

            km = get_auth_key_manager()
            key = km.get_active_key_material("audit_sign")
            if km.seed_is_ephemeral:
                logger.warning(
                    "Audit hash-chain HMAC key derived from an EPHEMERAL per-process "
                    "HD master seed: the chain will not verify across workers, "
                    "replicas, or restarts. Set AMA_MASTER_SEED "
                    "(`openssl rand -hex 64`) for a stable, rotatable audit key."
                )
            return key
        except Exception as exc:
            logger.warning(
                "Could not derive the audit HMAC key from AMA HD key management "
                "(%s); falling back to a per-process random key.",
                exc,
            )
            return secrets.token_bytes(32)

    def _compute_genesis_hash(self) -> str:
        """Compute genesis block hash."""
        genesis_data = f"MERCURY_AUDIT_GENESIS:{time.time()}"
        return self._hmac_hash(genesis_data.encode())

    def _hmac_hash(self, data: bytes) -> str:
        """Compute HMAC-SHA3-256 hash for AMA Cryptography alignment."""
        return hmac.new(self.hmac_key, data, hashlib.sha3_256).hexdigest()

    def compute_event_hash(self, event_data: dict[str, Any]) -> tuple[str, str, int]:
        """Compute hash for event and link to chain.

        Args:
            event_data: Event data to hash

        Returns:
            Tuple of (event_hash, previous_hash, sequence_number)
        """
        with self._lock:
            previous = self._previous_hash
            seq = self._sequence_number

            # Hash includes: previous_hash + sequence + event_data
            hash_input = json.dumps(
                {
                    "previous_hash": previous,
                    "sequence_number": seq,
                    "event_data": event_data,
                },
                sort_keys=True,
            ).encode()

            event_hash = self._hmac_hash(hash_input)

            # Update chain state
            self._previous_hash = event_hash
            self._sequence_number += 1

            return event_hash, previous, seq

    def resume(self, previous_hash: str, next_sequence: int) -> None:
        """Continue an existing chain: link the next event to ``previous_hash``."""
        with self._lock:
            self._previous_hash = previous_hash
            self._sequence_number = next_sequence

    def _recompute_event_hash(self, event: AuditEvent) -> str:
        """Recompute an event's HMAC hash from its persisted content.

        Mirrors :meth:`compute_event_hash` exactly (same ``event_data`` shape,
        same ``sort_keys`` JSON envelope), so an untampered event recomputes to
        its stored ``event_hash`` and any edit to the hashed content does not.
        """
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "category": event.category.value,
            "action": event.action,
            "outcome": event.outcome,
            "actor": event.actor,
            "resource": event.resource,
            "details": event.details,
        }
        hash_input = json.dumps(
            {
                "previous_hash": event.previous_hash,
                "sequence_number": event.sequence_number,
                "event_data": event_data,
            },
            sort_keys=True,
        ).encode()
        return self._hmac_hash(hash_input)

    def verify_chain(self, events: list[AuditEvent]) -> tuple[bool, list[int]]:
        """Verify integrity of the event chain (content *and* linkage).

        Two constant-time checks per event, either of which flags the index:

        * **Content** -- the stored ``event_hash`` must equal a fresh HMAC recompute
          over the event's persisted content (:meth:`_recompute_event_hash`). This
          catches an edit to a hashed field (``action`` / ``details`` / ...) that
          left the hash columns untouched -- undetectable by linkage alone, and
          unforgeable without the HMAC key. Recompute requires the *same* key the
          log was written with, so cross-process verification needs a stable
          configured key (``AMA_MASTER_SEED`` / an explicit ``hmac_key``), not the
          per-process ephemeral default.
        * **Linkage** -- each event's ``previous_hash`` must equal the prior
          event's ``event_hash``, so a deletion or reordering breaks the chain.
        * **Sequence** -- each event's ``sequence_number`` must be the prior
          event's plus one. Sequence numbers carry across restarts (the chain
          resumes from disk) and rotation, so within a file a gap means
          records are missing and a repeat means a replay. The number is part
          of the HMAC input, so this mostly restates linkage in a form that
          says *how many* records are gone (:func:`sequence_gaps`).

        Head and tail truncation leave a shorter but internally valid chain
        and cannot be detected here (see the module docstring).

        Uses :func:`hmac.compare_digest` to avoid leaking hash information via
        timing.

        Args:
            events: List of events to verify.

        Returns:
            Tuple of ``(is_valid, sorted_list_of_invalid_indices)``.
        """
        invalid_indices: set[int] = set()

        for i, event in enumerate(events):
            # (1) Content integrity: recompute the event's own hash.
            expected = self._recompute_event_hash(event)
            if not hmac.compare_digest(expected.encode("utf-8"), event.event_hash.encode("utf-8")):
                invalid_indices.add(i)

            # (2) Linkage: this event's back-pointer must match the prior hash.
            # Skip for the first event (its previous_hash is the genesis hash).
            if i == 0:
                continue
            if not hmac.compare_digest(
                event.previous_hash.encode("utf-8"), events[i - 1].event_hash.encode("utf-8")
            ):
                invalid_indices.add(i)

            # (3) Sequence contiguity.
            if event.sequence_number != events[i - 1].sequence_number + 1:
                invalid_indices.add(i)

        return len(invalid_indices) == 0, sorted(invalid_indices)

    def get_chain_state(self) -> dict[str, Any]:
        """Get current chain state."""
        with self._lock:
            return {
                "previous_hash": self._previous_hash,
                "sequence_number": self._sequence_number,
            }


def sequence_gaps(events: list[AuditEvent]) -> list[str]:
    """Describe every break in ``sequence_number`` contiguity.

    Args:
        events: Events in file order.

    Returns:
        One human-readable line per break (empty when contiguous).
    """
    gaps: list[str] = []
    for prev, nxt in pairwise(events):
        expected = prev.sequence_number + 1
        if nxt.sequence_number > expected:
            gaps.append(
                f"sequence {prev.sequence_number} -> {nxt.sequence_number} "
                f"({nxt.sequence_number - expected} record(s) missing)"
            )
        elif nxt.sequence_number != expected:
            gaps.append(
                f"sequence {prev.sequence_number} -> {nxt.sequence_number} "
                "(repeated or out of order)"
            )
    return gaps


def load_audit_events(path: Path) -> list[AuditEvent]:
    """Parse one audit JSONL file into events (read-only).

    Args:
        path: The log or segment file.

    Returns:
        The events in file order.

    Raises:
        ValueError: A line is not a complete audit record (the message names
            the 1-based line number).
        OSError: The file cannot be read.
    """
    events: list[AuditEvent] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                events.append(
                    AuditEvent(
                        event_id=data["event_id"],
                        timestamp=data["timestamp"],
                        category=AuditEventCategory(data["category"]),
                        severity=AuditEventSeverity(data["severity"]),
                        action=data["action"],
                        actor=data.get("actor"),
                        resource=data.get("resource"),
                        outcome=data["outcome"],
                        details=data["details"],
                        sequence_number=data["sequence_number"],
                        previous_hash=data["previous_hash"],
                        event_hash=data["event_hash"],
                        session_id=data.get("session_id"),
                        client_ip=data.get("client_ip"),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: {exc}") from exc
    return events


def verify_log_file(path: Path, hash_chain: SecureHashChain) -> tuple[bool, str]:
    """Verify one audit file's chain without opening it for writing.

    Args:
        path: The log or segment file.
        hash_chain: Supplies the HMAC key the file was written with.

    Returns:
        ``(is_valid, message)``.
    """
    ok, message, _events = _verify_file(path, hash_chain)
    return ok, message


def _verify_file(path: Path, hash_chain: SecureHashChain) -> tuple[bool, str, list[AuditEvent]]:
    if not path.exists():
        return False, "Log file does not exist", []
    try:
        events = load_audit_events(path)
    except (OSError, ValueError) as e:
        return False, f"Failed to parse log file: {e}", []
    is_valid, invalid_indices = hash_chain.verify_chain(events)
    if is_valid:
        return True, f"Log integrity verified ({len(events)} events)", events
    message = f"Chain broken at indices: {invalid_indices}"
    gaps = sequence_gaps(events)
    if gaps:
        message += "; " + "; ".join(gaps)
    return False, message, events


def _verify_writer_chain(
    files: list[Path], hash_chain: SecureHashChain, label: str
) -> dict[str, Any]:
    """Verify one writer's files (oldest first) and the links between them."""
    report: dict[str, Any] = {"writer": label, "ok": True, "files": [], "notes": []}
    last: AuditEvent | None = None
    last_name = ""
    for path in files:
        ok, message, events = _verify_file(path, hash_chain)
        entry: dict[str, Any] = {"file": path.name, "ok": ok, "message": message}
        sidecar = path.with_suffix(".sha256")
        if path.name.startswith("audit_") and sidecar.exists():
            if not hmac.compare_digest(_file_sha3(path), sidecar.read_text().strip()):
                entry["ok"] = False
                entry["message"] += "; content differs from its .sha256 sidecar"
        if events and last is not None:
            first = events[0]
            linked = hmac.compare_digest(
                first.previous_hash.encode(), last.event_hash.encode()
            ) and (first.sequence_number == last.sequence_number + 1)
            if not linked:
                entry["ok"] = False
                entry["message"] += (
                    f"; does not continue {last_name} (sequence {last.sequence_number} -> "
                    f"{first.sequence_number}, previous_hash mismatch or gap)"
                )
        elif events and last is None and events[0].sequence_number != 0:
            report["notes"].append(
                f"{path.name} starts at sequence {events[0].sequence_number}: earlier "
                "records were rotated out or removed (head truncation is not detectable "
                "without an external anchor)"
            )
        if events:
            last, last_name = events[-1], path.name
        report["ok"] = report["ok"] and entry["ok"]
        report["files"].append(entry)
    return report


def verify_audit_directory(
    log_dir: str | Path, hmac_key: bytes | None = None
) -> tuple[bool, list[dict[str, Any]]]:
    """Verify every writer's chain in ``log_dir`` strictly read-only.

    Unlike constructing a :class:`SecureAuditLogger` (which takes the writer
    lock, may switch to a per-process file and starts a flush thread), this
    opens nothing for writing, so it is safe while the server runs. Each
    writer's rotated segments are verified oldest first, then its active
    log, including the ``previous_hash`` / ``sequence_number`` link from
    each file to the one before it.

    Args:
        log_dir: The audit directory (``MERCURY_AUDIT_LOG_DIR``).
        hmac_key: The chain key; derived like the writer's when omitted.

    Returns:
        ``(all_valid, reports)`` with one report per writer chain (the
        primary first). ``all_valid`` is ``False`` when no audit file exists.
    """
    directory = Path(log_dir)
    hash_chain = SecureHashChain(hmac_key)
    chains: dict[int | None, list[Path]] = {}
    for path in directory.glob("audit_*.jsonl"):
        info = _segment_info(path)
        if info is not None:
            chains.setdefault(info[0], [])
    for path in directory.glob("audit-*.jsonl"):
        match = _WORKER_LOG_RE.fullmatch(path.name)
        if match is not None:
            chains.setdefault(int(match["pid"]), [])
    if (directory / "audit.jsonl").exists():
        chains.setdefault(None, [])
    reports: list[dict[str, Any]] = []
    for pid in sorted(chains, key=lambda p: -1 if p is None else p):
        files = _writer_segments(directory, pid)
        active = directory / ("audit.jsonl" if pid is None else f"audit-{pid}.jsonl")
        if active.exists():
            files.append(active)
        label = "primary" if pid is None else f"pid {pid}"
        reports.append(_verify_writer_chain(files, hash_chain, label))
    if not reports:
        return False, [{"writer": "none", "ok": False, "files": [], "notes": ["no audit files"]}]
    return all(r["ok"] for r in reports), reports


class SecureAuditLogger:
    """Production-grade secure audit logging system.

    Features:
    - Cryptographically signed log entries
    - Hash-linked event chain for tamper detection
    - PII masking
    - Async batch writing
    - Log rotation with integrity verification
    - Compliance-ready format (JSON)
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        hmac_key: bytes | None = None,
        mask_pii: bool = True,
        max_buffer_size: int = 100,
        flush_interval: float = 5.0,
        rotate_size_mb: float = 100.0,
        max_rotated_files: int = 10,
    ):
        """Initialize secure audit logger.

        Args:
            log_dir: Directory for audit logs
            hmac_key: HMAC key for signing (generated if None)
            mask_pii: Enable PII masking
            max_buffer_size: Buffer size before forced flush
            flush_interval: Seconds between automatic flushes
            rotate_size_mb: Max file size before rotation
            max_rotated_files: Maximum rotated files to keep
        """
        self.log_dir = Path(log_dir) if log_dir else Path("./audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.hash_chain = SecureHashChain(hmac_key)
        self.pii_masker = PIIMasker(enabled=mask_pii)

        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self.rotate_size_mb = rotate_size_mb
        self.max_rotated_files = max_rotated_files

        # Event buffer
        self._buffer: list[AuditEvent] = []
        self._buffer_lock = threading.Lock()

        # Current log file. One process owns audit.jsonl (an exclusive flock on
        # .audit.lock, held for the logger's lifetime); a second process on the
        # same directory -- e.g. `mercury serve --workers 4` -- gets its own
        # audit-<pid>.jsonl. Two writers interleaving two chains in one file
        # would make every later verification fail.
        self._primary_lock: IO[str] | None = self._claim_primary_log()
        self._writer_lock: IO[str] | None = None
        #: ``None`` for the primary; this process's pid for a worker writer.
        self._writer_pid: int | None
        if self._primary_lock is not None or fcntl is None:
            self._current_log_path = self.log_dir / "audit.jsonl"
            self._rotated_tag = ""
            self._writer_pid = None
        else:
            self._writer_pid = os.getpid()
            self._current_log_path = self.log_dir / f"audit-{self._writer_pid}.jsonl"
            self._rotated_tag = f"_{self._writer_pid}"
            self._writer_lock = self._claim_worker_lock(self._writer_pid)
            logger.critical(
                "Another process already writes %s; this process audits to %s so the "
                "two hash chains stay separate and verifiable.",
                self.log_dir / "audit.jsonl",
                self._current_log_path,
            )
        self._file_lock = threading.Lock()
        self._resume_chain_from_disk()
        if self._writer_pid is None:
            # Restarted fleets leave the previous workers' files behind.
            try:
                self._cleanup_orphans()
            except OSError as exc:  # pragma: no cover - best-effort housekeeping
                logger.warning("Orphaned audit log cleanup failed: %s", exc)

        # Flush timer
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_flush_thread()

        # Event hooks
        self._event_hooks: list[Callable[[AuditEvent], None]] = []

        logger.info(f"SecureAuditLogger initialized: {self.log_dir}")

    def _claim_primary_log(self) -> IO[str] | None:
        """Take the directory's writer lock; ``None`` if another process holds it."""
        if fcntl is None:
            return None
        handle = open(self.log_dir / _PRIMARY_LOCK_NAME, "a")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        return handle

    def _claim_worker_lock(self, pid: int) -> IO[str] | None:
        """Hold ``.audit-<pid>.lock`` so other processes see this writer as live."""
        if fcntl is None:
            return None
        try:
            handle = open(_worker_lock_path(self.log_dir, pid), "a")
        except OSError as exc:
            logger.warning("Cannot create the audit writer lock for pid %d: %s", pid, exc)
            return None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Same pid live in another pid namespace on a shared volume: keep
            # going (the files are still distinct per process here).
            logger.warning("Audit writer lock for pid %d is held by another process", pid)
            handle.close()
            return None
        return handle

    def _repair_torn_tail(self, path: Path) -> None:
        """Cut an unterminated final line (a write interrupted by a crash).

        Records are written as whole newline-terminated lines, so a file that
        does not end in a newline ends in a fragment of a record that never
        completed. Appending after it would glue the next record onto the
        fragment and make the whole file unparseable. Only the bytes after the
        last newline are removed; they are logged (length, SHA3-256 and a
        preview) and preserved next to the log as ``<name>.torn-<usec>`` for
        forensics. The caller must own ``path`` (writer lock held).
        """
        with open(path, "rb+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            cut = 0
            pos = size
            while pos > 0:
                step = min(_TAIL_BLOCK, pos)
                pos -= step
                f.seek(pos)
                idx = f.read(step).rfind(b"\n")
                if idx != -1:
                    cut = pos + idx + 1
                    break
            f.seek(cut)
            fragment = f.read(size - cut)
            f.truncate(cut)
            f.flush()
            os.fsync(f.fileno())
        keep = path.with_name(f"{path.name}.torn-{time.time_ns() // 1000}")
        try:
            keep.write_bytes(fragment)
        except OSError as exc:  # pragma: no cover - best-effort preservation
            logger.warning("Could not preserve torn audit fragment to %s: %s", keep, exc)
        logger.critical(
            "Audit log %s ended in an unterminated %d-byte fragment at offset %d "
            "(sha3-256 %s, preview %r, saved to %s); it was cut so the chain resumes "
            "from the last complete record.",
            path,
            len(fragment),
            cut,
            hashlib.sha3_256(fragment).hexdigest(),
            fragment[:_TORN_PREVIEW_BYTES],
            keep.name,
        )

    def _resume_chain_from_disk(self) -> None:
        """Link the first new event to the last one already on disk.

        The chain used to restart at a fresh genesis on every process start
        while appending to the same file, so a log that had been through a
        single restart failed :meth:`verify_log_integrity` exactly as a
        tampered one does. Resumes from the active log, or -- right after a
        rotation -- from the newest rotated segment of *this* writer (never
        another writer's). A torn final line in the active log is cut first
        (:meth:`_repair_torn_tail`).
        """
        candidates = [self._current_log_path]
        if self._current_log_path.exists() and self._current_log_path.stat().st_size > 0:
            try:
                self._repair_torn_tail(self._current_log_path)
            except OSError as exc:
                logger.critical("Could not repair the tail of %s (%s)", self._current_log_path, exc)
        if not self._current_log_path.exists() or self._current_log_path.stat().st_size == 0:
            candidates = _writer_segments(self.log_dir, self._writer_pid)[-1:]
        for path in candidates:
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                record = json.loads(_last_line(path))
                self.hash_chain.resume(
                    str(record["event_hash"]), int(record["sequence_number"]) + 1
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.critical(
                    "Audit log %s does not end in a readable event (%s); new events "
                    "start a fresh chain and verification will report the break.",
                    path,
                    exc,
                )

    def _start_flush_thread(self) -> None:
        """Start background flush thread."""
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="AuditFlushThread"
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background flush loop.

        Blocks on the stop event rather than an unconditional ``sleep`` so that
        :meth:`shutdown` (and the audit-logger reconfigure path that calls it)
        returns promptly instead of waiting up to ``flush_interval`` for an
        in-progress sleep to elapse. ``Event.wait`` returns ``True`` the instant
        the event is set, ending the loop; it returns ``False`` on timeout, which
        preserves the original "wait one interval, then flush" cadence. The final
        flush on shutdown is issued explicitly by :meth:`shutdown`.
        """
        while not self._stop_event.wait(self.flush_interval):
            self.flush()

    def log(
        self,
        category: AuditEventCategory,
        action: str,
        outcome: str = "success",
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        actor: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
        session_id: str | None = None,
        client_ip: str | None = None,
    ) -> str:
        """Log an audit event.

        Args:
            category: Event category
            action: Action performed
            outcome: 'success', 'failure', or 'partial'
            severity: Event severity
            actor: User/system performing action
            resource: Resource affected
            details: Additional details
            session_id: Session identifier
            client_ip: Client IP address

        Returns:
            Event ID
        """
        # Generate event ID
        event_id = f"AE-{secrets.token_hex(8)}"

        # Mask PII in details
        masked_details = self.pii_masker.mask(details or {})

        # Prepare event data for hashing
        event_data = {
            "event_id": event_id,
            "timestamp": time.time(),
            "category": category.value,
            "action": action,
            "outcome": outcome,
            "actor": actor,
            "resource": resource,
            "details": masked_details,
        }

        # Compute hash and link to chain
        event_hash, previous_hash, seq = self.hash_chain.compute_event_hash(event_data)

        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=event_data["timestamp"],
            category=category,
            severity=severity,
            action=action,
            actor=actor,
            resource=resource,
            outcome=outcome,
            details=masked_details,
            sequence_number=seq,
            previous_hash=previous_hash,
            event_hash=event_hash,
            session_id=session_id,
            client_ip=client_ip,
        )

        # Add to buffer
        with self._buffer_lock:
            self._buffer.append(event)

            if len(self._buffer) >= self.max_buffer_size:
                self._flush_buffer()

        # Invoke hooks
        for hook in self._event_hooks:
            try:
                hook(event)
            except Exception as e:
                logger.warning(f"Audit hook error: {e}")

        return event_id

    def log_authentication(
        self,
        action: str,
        actor: str,
        outcome: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log authentication event."""
        severity = AuditEventSeverity.INFO if outcome == "success" else AuditEventSeverity.WARNING
        return self.log(
            category=AuditEventCategory.AUTHENTICATION,
            action=action,
            outcome=outcome,
            severity=severity,
            actor=actor,
            details=details,
            **kwargs,
        )

    def log_data_access(
        self,
        resource: str,
        actor: str,
        action: str = "read",
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log data access event."""
        return self.log(
            category=AuditEventCategory.DATA_ACCESS,
            action=action,
            resource=resource,
            actor=actor,
            details=details,
            **kwargs,
        )

    def log_anomaly_detection(
        self,
        action: str,
        details: dict[str, Any],
        severity: AuditEventSeverity = AuditEventSeverity.INFO,
        **kwargs: Any,
    ) -> str:
        """Log anomaly detection event."""
        return self.log(
            category=AuditEventCategory.ANOMALY_DETECTION,
            action=action,
            severity=severity,
            details=details,
            **kwargs,
        )

    def log_security_incident(
        self,
        action: str,
        details: dict[str, Any],
        actor: str | None = None,
        resource: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Log security incident."""
        return self.log(
            category=AuditEventCategory.SECURITY_INCIDENT,
            action=action,
            severity=AuditEventSeverity.SECURITY,
            outcome="failure",
            actor=actor,
            resource=resource,
            details=details,
            **kwargs,
        )

    def log_cryptographic(
        self,
        action: str,
        outcome: str,
        algorithm: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log cryptographic operation."""
        return self.log(
            category=AuditEventCategory.CRYPTOGRAPHIC,
            action=action,
            outcome=outcome,
            details={"algorithm": algorithm, **(details or {})},
            **kwargs,
        )

    def flush(self) -> None:
        """Flush buffered events to disk."""
        with self._buffer_lock:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Internal buffer flush (caller must hold lock)."""
        if not self._buffer:
            return

        events_to_write = self._buffer.copy()
        self._buffer.clear()

        self._write_events(events_to_write)

    def _write_events(self, events: list[AuditEvent]) -> None:
        """Write events to log file with robust error handling.

        SECURITY: Audit log write failures are critical - they indicate
        potential data loss in the audit trail. This method will raise
        exceptions on failure to alert operators.
        """
        with self._file_lock:
            try:
                # Check for rotation
                if self._current_log_path.exists():
                    size_mb = self._current_log_path.stat().st_size / (1024 * 1024)
                    if size_mb >= self.rotate_size_mb:
                        self._rotate_log()

                # Write events with fsync for durability
                with open(self._current_log_path, "a") as f:
                    for event in events:
                        f.write(event.to_json() + "\n")
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data reaches disk

            except OSError as e:
                # Critical error - audit events could be lost
                logger.critical(
                    f"AUDIT LOG WRITE FAILURE: {e}. "
                    f"{len(events)} audit events may be lost. "
                    f"Check disk space and file permissions."
                )
                # Re-raise to alert callers
                raise RuntimeError(f"Failed to write audit log (potential data loss): {e}") from e

    def _rotate_log(self) -> None:
        """Rotate the active log to a uniquely named segment.

        Names carry microseconds plus a ``-<n>`` counter on collision, and
        the rename refuses to replace an existing file
        (:func:`_rename_no_clobber`), so two rotations in the same second (a
        small ``rotate_size_mb`` or a burst) can never destroy a segment.
        """
        now = time.time()
        base = f"audit_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}"
        base += f".{int(now * 1_000_000) % 1_000_000:06d}"
        dup = 0
        while True:
            suffix = f"-{dup}" if dup else ""
            rotated_path = self.log_dir / f"{base}{suffix}{self._rotated_tag}.jsonl"
            if _rename_no_clobber(self._current_log_path, rotated_path):
                break
            dup += 1

        # Generate integrity hash for rotated file
        file_hash = self._compute_file_hash(rotated_path)
        hash_file = rotated_path.with_suffix(".sha256")
        hash_file.write_text(file_hash)

        # Cleanup old rotated files
        self._cleanup_old_logs()

        logger.info(f"Rotated audit log: {rotated_path} (hash: {file_hash[:16]}...)")

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA3-256 hash of file for AMA Cryptography alignment."""
        return _file_sha3(path)

    def _cleanup_old_logs(self) -> None:
        """Keep this writer's newest ``max_rotated_files`` segments.

        Retention is per writer: only this writer's own segments are counted
        and deleted (a worker's ``_<pid>`` segments, or the primary's
        unsuffixed ones), so one busy writer cannot prune another's evidence.
        The primary additionally caps orphaned writers' files
        (:meth:`_cleanup_orphans`).
        """
        own = _writer_segments(self.log_dir, self._writer_pid)
        for oldest in own[: max(len(own) - max(self.max_rotated_files, 0), 0)]:
            _remove_segment(oldest)
        if self._writer_pid is None:
            self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        """Cap the files left by writers that no longer run.

        Every ``mercury serve --workers N`` restart leaves the previous
        workers' ``audit-<pid>.jsonl`` and ``audit_*_<pid>.jsonl`` behind
        under pids that will never write again, so per-writer retention never
        sees them. The primary pools all orphaned files (active logs and
        segments) and keeps the newest ``max_rotated_files`` by mtime; older
        ones are deleted with their sidecars, and a dead writer's lock file is
        removed once none of its files remain. Orphans are never renamed or
        rewritten, so the kept ones remain verifiable. A writer is judged by
        :func:`_writer_is_live`, which errs towards keeping files.
        """
        by_pid: dict[int, list[Path]] = {}
        for path in self.log_dir.glob("audit_*.jsonl"):
            info = _segment_info(path)
            if info is not None and info[0] is not None:
                by_pid.setdefault(info[0], []).append(path)
        for path in self.log_dir.glob("audit-*.jsonl"):
            match = _WORKER_LOG_RE.fullmatch(path.name)
            if match is not None:
                by_pid.setdefault(int(match["pid"]), []).append(path)
        orphaned: list[tuple[float, Path]] = []
        dead: list[int] = []
        for pid, paths in by_pid.items():
            if _writer_is_live(self.log_dir, pid):
                continue
            dead.append(pid)
            for path in paths:
                try:
                    orphaned.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        orphaned.sort(key=lambda item: (item[0], item[1].name))
        excess = max(len(orphaned) - max(self.max_rotated_files, 0), 0)
        for _mtime, path in orphaned[:excess]:
            logger.info("Removing orphaned audit file %s (writer no longer running)", path)
            _remove_segment(path)
        for pid in dead:
            if not any(p.exists() for p in by_pid[pid]):
                _worker_lock_path(self.log_dir, pid).unlink(missing_ok=True)

    def add_hook(self, hook: Callable[[AuditEvent], None]) -> None:
        """Add event hook for real-time processing."""
        self._event_hooks.append(hook)

    def verify_log_integrity(self, path: Path | None = None) -> tuple[bool, str]:
        """Verify integrity of log file.

        Args:
            path: Path to log file (current if None)

        Returns:
            Tuple of (is_valid, message)
        """
        return verify_log_file(path or self._current_log_path, self.hash_chain)

    def get_recent_events(
        self,
        count: int = 100,
        category: AuditEventCategory | None = None,
        severity: AuditEventSeverity | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent events from buffer and disk.

        Args:
            count: Maximum events to return
            category: Filter by category
            severity: Filter by severity

        Returns:
            List of event dictionaries
        """
        events = []

        # Get from buffer
        with self._buffer_lock:
            events.extend([e.to_dict() for e in self._buffer])

        # Get from disk if needed
        if len(events) < count and self._current_log_path.exists():
            try:
                with open(self._current_log_path) as f:
                    disk_events = []
                    for line_num, line in enumerate(f, 1):
                        try:
                            disk_events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                            continue
                    events = disk_events + events
            except OSError as e:
                logger.error(f"Failed to read audit log from disk: {e}")

        # Apply filters
        if category:
            events = [e for e in events if e["category"] == category.value]
        if severity:
            events = [e for e in events if e["severity"] == severity.value]

        # Return most recent
        return events[-count:]

    def shutdown(self) -> None:
        """Shutdown logger and flush remaining events."""
        self._stop_event.set()
        self.flush()

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)

        if self._primary_lock is not None:
            self._primary_lock.close()  # releases the flock
            self._primary_lock = None
        if self._writer_lock is not None:
            self._writer_lock.close()
            self._writer_lock = None

        logger.info("SecureAuditLogger shutdown complete")


# Global instance
_audit_logger: SecureAuditLogger | None = None
_audit_logger_lock = threading.Lock()


def get_audit_logger() -> SecureAuditLogger:
    """Get or create global audit logger instance (thread-safe).

    Lock-guarded double-checked construction: two threads racing the first
    call must not each build a logger — the loser's instance would keep an
    orphan flush thread alive and split the hash chain across two files.
    """
    global _audit_logger
    if _audit_logger is None:
        with _audit_logger_lock:
            if _audit_logger is None:
                _audit_logger = SecureAuditLogger()
    return _audit_logger


def configure_audit_logger(**kwargs: Any) -> SecureAuditLogger:
    """Configure and return global audit logger (shutting the previous one down)."""
    global _audit_logger
    with _audit_logger_lock:
        if _audit_logger is not None:
            _audit_logger.shutdown()
        _audit_logger = SecureAuditLogger(**kwargs)
    return _audit_logger


# Exports
__all__ = [
    "AuditEvent",
    "AuditEventCategory",
    "AuditEventSeverity",
    "PIIMasker",
    "SecureAuditLogger",
    "SecureHashChain",
    "configure_audit_logger",
    "get_audit_logger",
    "load_audit_events",
    "sequence_gaps",
    "verify_audit_directory",
    "verify_log_file",
]

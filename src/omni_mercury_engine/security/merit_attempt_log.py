# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restart-safe record of fusion merit tests, so alpha spending survives a restart.

:meth:`IntelligenceFusionEngine.load_trained_weights` halves its significance
level for every retry against the same held-out set
(:func:`~omni_mercury_engine.security.intelligence_fusion.merit_alpha_for_attempt`),
which bounds the chance of admitting weights no better than evidence scoring
at ``2 * MERIT_ALPHA`` over unlimited retries. That bound holds only if every
attempt is counted. Kept on the engine instance, the count restarted at 1 in
every new process, so restarting the server bought a fresh ``MERIT_ALPHA``.
This log keeps it on disk.

Format: one JSON record per line, ``{schema, index, prev_digest, entry,
record_digest}``, a SHA3-256 chain over canonical JSON (the adjudication
ledger's conventions). ``entry`` is ``{kind, fingerprint, checkpoint,
checkpoint_sha256, attempt, recorded_at}`` for a reservation and adds
``p_value``, ``alpha``, ``admitted`` and accuracies for a result.

Rules:

* **Reserve before measuring.** The engine records the attempt after the
  checkpoint has loaded and before any case is scored. A run killed mid-test
  has still spent its attempt, so crashing a test cannot buy a free retry.
* **Fail closed.** A log that does not verify (an edited, reordered or
  truncated-mid-chain record) refuses every reservation; the engine then
  refuses the weights instead of guessing the count.
* **One writer at a time.** Appends hold an exclusive ``flock`` on
  ``<log>.lock`` across verification and the write (POSIX); each record is
  one ``O_APPEND`` write followed by ``fsync``.
* **Deleting the file resets the count.** The log is evidence against
  accidental restarts, not against an operator with write access to it;
  pin the directory's permissions accordingly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

try:  # POSIX advisory locks; without fcntl the log is single-writer.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

SCHEMA = "mercury.fusion.merit-attempts/v1"
GENESIS_DIGEST = "0" * 64
#: Environment variable naming the log; unset keeps the per-instance count.
MERIT_ATTEMPT_LOG_ENV = "MERCURY_MERIT_ATTEMPT_LOG"


class MeritLogError(RuntimeError):
    """The attempt log does not verify, so no attempt count can be trusted."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(payload: Any) -> str:
    return hashlib.sha3_256(_canonical(payload)).hexdigest()


class MeritAttemptLog:
    """Append-only, hash-chained log of merit-test reservations and results."""

    def __init__(self, path: str | Path) -> None:
        """Open (without creating) the log at ``path``."""
        self.path = Path(path)

    # -- reading ------------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        """Every record, verified.

        Raises:
            MeritLogError: A line is unparseable, out of order or re-sealed.
        """
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        prev = GENESIS_DIGEST
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise MeritLogError(f"{self.path}: the last record is incomplete")
        for index, line in enumerate(raw.decode("utf-8").splitlines()):
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise MeritLogError(f"{self.path}: line {index + 1} is not JSON") from exc
            if not isinstance(record, dict) or record.get("schema") != SCHEMA:
                raise MeritLogError(f"{self.path}: line {index + 1} is not a {SCHEMA} record")
            body = {k: v for k, v in record.items() if k != "record_digest"}
            if (
                record.get("index") != index
                or record.get("prev_digest") != prev
                or record.get("record_digest") != _digest(body)
            ):
                raise MeritLogError(f"{self.path}: the chain breaks at record {index}")
            prev = record["record_digest"]
            records.append(record)
        return records

    def attempts(self, fingerprint: str) -> int:
        """Number of reservations recorded against ``fingerprint``."""
        return sum(
            1
            for r in self.records()
            if r["entry"].get("kind") == "reservation"
            and r["entry"].get("fingerprint") == fingerprint
        )

    # -- writing ------------------------------------------------------------

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:  # pragma: no cover - non-POSIX platforms
            yield
            return
        fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _append(self, records: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "index": len(records),
            "prev_digest": records[-1]["record_digest"] if records else GENESIS_DIGEST,
            "entry": entry,
        }
        record["record_digest"] = _digest(record)
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        return record

    def reserve(self, fingerprint: str, checkpoint: str | Path) -> int:
        """Record the next attempt against ``fingerprint`` and return its 1-based number.

        Raises:
            MeritLogError: The log does not verify.
        """
        path = Path(checkpoint)
        checkpoint_sha256 = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        with self._lock():
            records = self.records()
            attempt = 1 + sum(
                1
                for r in records
                if r["entry"].get("kind") == "reservation"
                and r["entry"].get("fingerprint") == fingerprint
            )
            self._append(
                records,
                {
                    "kind": "reservation",
                    "fingerprint": fingerprint,
                    "checkpoint": str(path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "attempt": attempt,
                    "recorded_at": time.time(),
                },
            )
        return attempt

    def record_result(self, fingerprint: str, attempt: int, result: dict[str, Any]) -> None:
        """Append the outcome of a reserved attempt (``p_value``, ``alpha``, ``admitted``...).

        Raises:
            MeritLogError: The log does not verify.
        """
        with self._lock():
            records = self.records()
            self._append(
                records,
                {
                    "kind": "result",
                    "fingerprint": fingerprint,
                    "attempt": attempt,
                    "recorded_at": time.time(),
                    **result,
                },
            )


def log_from_env(environ: dict[str, str] | None = None) -> MeritAttemptLog | None:
    """The log named by :data:`MERIT_ATTEMPT_LOG_ENV`, or ``None`` when unset."""
    value = (os.environ if environ is None else environ).get(MERIT_ATTEMPT_LOG_ENV, "").strip()
    return MeritAttemptLog(Path(value).expanduser()) if value else None


__all__ = [
    "GENESIS_DIGEST",
    "MERIT_ATTEMPT_LOG_ENV",
    "SCHEMA",
    "MeritAttemptLog",
    "MeritLogError",
    "log_from_env",
]

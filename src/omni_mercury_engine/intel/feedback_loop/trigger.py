# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Signed, audited authorization for a gated retrain run.

A learning loop that retrains on ingested audit + red-team data is a
data-poisoning surface. Human-gating alone is not enough: the *authorization to
retrain* must itself be unforgeable and auditable. A :class:`RetrainTrigger` is
an HMAC-SHA256 signature (keyed by ``MERCURY_RETRAIN_TRIGGER_SECRET``) over a
payload that **binds the exact queue snapshot** the retrain is authorized for --
so a signature cannot be replayed against a different (e.g. later-poisoned) queue
state, and every verification is durably audited.

Two independent replay defenses, so "single-use authorization" is a real
guarantee rather than a slogan:

* **Queue-snapshot binding** (primary) -- the signed payload includes the queue
  ``snapshot_hash``; a trigger only authorizes the *exact* queue state it was
  signed against, and any later enqueue changes the hash and invalidates it.
* **Nonce ledger** (:class:`NonceLedger`) -- the retrain pipeline durably records
  each consumed ``nonce`` and refuses a second run that reuses one, so even a
  replay against the *same* queue state is rejected. This is what makes the
  ``nonce`` field's "prevents signature reuse" contract enforceable rather than
  decorative.

Fail-closed throughout: no configured secret means no authorization is possible
(:func:`verify_trigger` returns ``False``), signatures are compared in constant
time, and a mismatch is audited as a rejected authorization.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

SECRET_ENV = "MERCURY_RETRAIN_TRIGGER_SECRET"  # noqa: S105 - env var name, not a secret

_NONCE_LOCK = threading.Lock()


def secret_from_env() -> str | None:
    """Return the configured retrain secret, or ``None`` if unset/empty."""
    secret = os.environ.get(SECRET_ENV, "").strip()
    return secret or None


@dataclass(frozen=True)
class RetrainTrigger:
    """A signed authorization to run a gated retrain against a bound queue state.

    Attributes:
        queue_hash: The :meth:`DurableLabeledQueue.snapshot_hash` this trigger
            authorizes (binds the signature to an exact queue state).
        corpus_version: The base corpus version being augmented.
        requested_by: The operator/service that requested the retrain.
        n_examples: How many labeled examples the queue held at signing time.
            Cryptographically bound (tampering breaks the signature) and asserted
            against the live queue count when the trigger is consumed.
        nonce: A caller-supplied single-use token. Recorded in the
            :class:`NonceLedger` when the trigger is consumed so a replay that
            reuses it is refused (see the module docstring's replay defenses).
        signature: HMAC-SHA256 hex of the canonical payload.
    """

    queue_hash: str
    corpus_version: str
    requested_by: str
    n_examples: int
    nonce: str
    signature: str = ""

    def canonical_payload(self) -> str:
        """The exact bytes signed/verified (deterministic JSON, no signature)."""
        return json.dumps(
            {
                "queue_hash": self.queue_hash,
                "corpus_version": self.corpus_version,
                "requested_by": self.requested_by,
                "n_examples": self.n_examples,
                "nonce": self.nonce,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def fingerprint(self) -> str:
        """Short, non-secret id of this trigger (for artifacts/audit)."""
        return (
            hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:16]
            if self.signature
            else ""
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping (includes the signature)."""
        return {
            "queue_hash": self.queue_hash,
            "corpus_version": self.corpus_version,
            "requested_by": self.requested_by,
            "n_examples": self.n_examples,
            "nonce": self.nonce,
            "signature": self.signature,
            "fingerprint": self.fingerprint(),
        }


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_trigger(
    *,
    queue_hash: str,
    corpus_version: str,
    requested_by: str,
    n_examples: int,
    nonce: str,
    secret: str | None = None,
) -> RetrainTrigger:
    """Create a signed :class:`RetrainTrigger`.

    Args:
        queue_hash: The queue snapshot hash to bind.
        corpus_version: The base corpus version.
        requested_by: Requesting operator/service id.
        n_examples: Labeled-example count at signing time.
        nonce: Uniqueness token.
        secret: HMAC key; defaults to ``MERCURY_RETRAIN_TRIGGER_SECRET``.

    Raises:
        RuntimeError: if no secret is configured (a retrain cannot be authorized
            without one -- fail-closed).
    """
    key = secret if secret is not None else secret_from_env()
    if not key:
        raise RuntimeError(
            f"cannot sign a retrain trigger: {SECRET_ENV} is not set. A gated "
            "retrain requires a signing secret so the authorization is unforgeable."
        )
    unsigned = RetrainTrigger(
        queue_hash=queue_hash,
        corpus_version=corpus_version,
        requested_by=requested_by,
        n_examples=n_examples,
        nonce=nonce,
    )
    signature = _sign(unsigned.canonical_payload(), key)
    return RetrainTrigger(
        queue_hash=queue_hash,
        corpus_version=corpus_version,
        requested_by=requested_by,
        n_examples=n_examples,
        nonce=nonce,
        signature=signature,
    )


def verify_trigger(
    trigger: RetrainTrigger,
    *,
    secret: str | None = None,
    expected_queue_hash: str | None = None,
    audit: bool = True,
) -> bool:
    """Verify a trigger's signature (and optional queue binding); audited.

    Args:
        trigger: The trigger to verify.
        secret: HMAC key; defaults to ``MERCURY_RETRAIN_TRIGGER_SECRET``.
        expected_queue_hash: If given, the trigger's ``queue_hash`` must equal it
            (the queue must not have changed since signing) or verification fails.
        audit: Whether to durably record the verification outcome.

    Returns:
        ``True`` only if a secret is configured, the HMAC matches in constant
        time, and (when supplied) the queue binding holds. Fail-closed otherwise.
    """
    key = secret if secret is not None else secret_from_env()
    reason: str
    # ``reason_code`` is a fixed, enumerated category per branch; it carries no
    # data flow from any source and is the *only* thing written to the operator
    # log. The composed ``reason`` string (which names ``SECRET_ENV`` and embeds
    # queue-hash prefixes) is recorded solely in the structured audit sink below.
    reason_code: str
    ok = False
    if not key:
        reason_code = "secret_unconfigured"
        reason = f"{SECRET_ENV} not configured; authorization impossible (fail-closed)"
    elif not trigger.signature:
        reason_code = "no_signature"
        reason = "trigger carries no signature"
    else:
        expected = _sign(trigger.canonical_payload(), key)
        sig_ok = hmac.compare_digest(expected, trigger.signature)
        if not sig_ok:
            reason_code = "signature_mismatch"
            reason = "HMAC signature mismatch"
        elif expected_queue_hash is not None and trigger.queue_hash != expected_queue_hash:
            reason_code = "queue_hash_mismatch"
            reason = (
                "queue hash mismatch: trigger bound to "
                f"{trigger.queue_hash[:12]} but queue is {expected_queue_hash[:12]} "
                "(queue changed since signing)"
            )
        else:
            ok = True
            reason_code = "valid"
            reason = "signature and queue binding valid"

    if audit:
        record_gate_decision(
            decision="retrain_trigger_verified" if ok else "retrain_trigger_rejected",
            source="feedback_loop:trigger",
            disposition="approved" if ok else "hard_refuse",
            signals=("closed_loop", "signed_trigger"),
            reason=reason,
            extra={
                "reason_code": reason_code,
                "requested_by": trigger.requested_by,
                "fingerprint": trigger.fingerprint(),
                "queue_hash": trigger.queue_hash[:16],
            },
        )
    if not ok:
        # Log only the enumerated ``reason_code`` -- never the composed ``reason``.
        # ``reason`` interpolates the ``SECRET_ENV`` name (a secret-flagged token)
        # and queue-hash prefixes; CodeQL's clear-text-logging query follows that
        # flow into the logger. A fixed category string keeps the operator log
        # actionable while the full detail lives only in the audit record above --
        # so no secret-derived data can reach clear-text logs by construction.
        logger.warning("retrain trigger rejected (%s)", reason_code)
    return ok


class NonceLedger:
    r"""A durable, append-only ledger of consumed retrain-trigger nonces.

    Makes a signed trigger genuinely single-use: :meth:`consume` records a nonce
    the first time it is seen and refuses it thereafter, so a trigger cannot be
    replayed to authorize a second retrain even against an unchanged queue. The
    sink is a JSON-Lines file (flushed + ``fsync``\ ed) so a consumed nonce
    survives process exit; the default location is co-located with the staging
    registry the retrain writes to.

    The check-then-append is atomic **across processes**, not just threads: an
    in-process :class:`threading.Lock` alone would let two concurrent *processes*
    (two retrain workers, a CI matrix job racing an operator) both observe the
    nonce as absent and both append it, silently consuming it twice and defeating
    single-use authorization. So the critical section is additionally guarded by
    an exclusive advisory file lock (:func:`fcntl.flock` on a sidecar ``.lock``
    file) held for the whole read-modify-write; the kernel releases it if the
    holder dies, so a crashed worker cannot wedge the ledger with a stale lock.
    """

    DEFAULT_NAME = "consumed_nonces.jsonl"

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open (or lazily create) the ledger at ``path`` (a file, or a dir)."""
        p = Path(path)
        # A directory argument is a convenience: use the conventional filename.
        self.path = p / self.DEFAULT_NAME if (p.is_dir() or not p.suffix) else p
        # Sidecar lock file for the inter-process advisory lock. Kept distinct
        # from the data file so locking never truncates/appends the ledger itself.
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        r"""Hold an exclusive advisory lock over the ledger's critical section.

        Combined with :data:`_NONCE_LOCK` (threads) this makes check-then-append
        atomic across both threads and processes. The lock is advisory
        (``flock``): every accessor goes through this method, so cooperating
        writers serialize; the kernel drops the lock on ``close``/process death,
        so there is no stale-lock recovery to get wrong.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _consumed_unlocked(self) -> set[str]:
        if not self.path.is_file():
            return set()
        seen: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(str(json.loads(line)["nonce"]))
            except (ValueError, KeyError, TypeError):  # tolerate a corrupt line
                continue
        return seen

    def is_consumed(self, nonce: str) -> bool:
        """Whether ``nonce`` has already been consumed."""
        with _NONCE_LOCK, self._interprocess_lock():
            return nonce in self._consumed_unlocked()

    def consume(self, nonce: str, *, queue_hash: str = "", requested_by: str = "") -> bool:
        r"""Durably record ``nonce`` as consumed. Returns ``False`` on replay.

        The first call for a nonce returns ``True`` (newly consumed -> proceed);
        any later call with the same nonce returns ``False`` (replay -> refuse).
        The write is flushed + ``fsync``\ ed so a consumed nonce is never lost.

        The whole check-then-append runs under both the in-process lock and the
        inter-process advisory lock, so two concurrent *processes* can never both
        see the nonce as absent and both append it -- exactly one wins.
        """
        with _NONCE_LOCK, self._interprocess_lock():
            if nonce in self._consumed_unlocked():
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "nonce": nonce,
                "queue_hash": queue_hash,
                "requested_by": requested_by,
                "ts": time.time(),
            }
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return True

    def clear(self) -> None:
        """Delete the ledger (staging/test/demo reset convenience, not a loop op)."""
        with _NONCE_LOCK, self._interprocess_lock():
            if self.path.is_file():
                self.path.unlink()
        # Best-effort removal of the sidecar lock file (never inside the lock we
        # are holding on it): its absence is harmless -- it is recreated on demand.
        with contextlib.suppress(FileNotFoundError):
            self.lock_path.unlink()


__all__ = [
    "SECRET_ENV",
    "NonceLedger",
    "RetrainTrigger",
    "secret_from_env",
    "sign_trigger",
    "verify_trigger",
]

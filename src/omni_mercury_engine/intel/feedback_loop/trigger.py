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

Fail-closed throughout: no configured secret means no authorization is possible
(:func:`verify_trigger` returns ``False``), signatures are compared in constant
time, and a mismatch is audited as a rejected authorization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

logger = logging.getLogger(__name__)

SECRET_ENV = "MERCURY_RETRAIN_TRIGGER_SECRET"  # noqa: S105 - env var name, not a secret


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
        nonce: A caller-supplied uniqueness token (prevents signature reuse).
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
    ok = False
    if not key:
        reason = f"{SECRET_ENV} not configured; authorization impossible (fail-closed)"
    elif not trigger.signature:
        reason = "trigger carries no signature"
    else:
        expected = _sign(trigger.canonical_payload(), key)
        sig_ok = hmac.compare_digest(expected, trigger.signature)
        if not sig_ok:
            reason = "HMAC signature mismatch"
        elif expected_queue_hash is not None and trigger.queue_hash != expected_queue_hash:
            reason = (
                "queue hash mismatch: trigger bound to "
                f"{trigger.queue_hash[:12]} but queue is {expected_queue_hash[:12]} "
                "(queue changed since signing)"
            )
        else:
            ok = True
            reason = "signature and queue binding valid"

    if audit:
        record_gate_decision(
            decision="retrain_trigger_verified" if ok else "retrain_trigger_rejected",
            source="feedback_loop:trigger",
            disposition="approved" if ok else "hard_refuse",
            signals=("closed_loop", "signed_trigger"),
            reason=reason,
            extra={
                "requested_by": trigger.requested_by,
                "fingerprint": trigger.fingerprint(),
                "queue_hash": trigger.queue_hash[:16],
            },
        )
    if not ok:
        logger.warning("retrain trigger rejected: %s", reason)
    return ok


__all__ = [
    "SECRET_ENV",
    "RetrainTrigger",
    "secret_from_env",
    "sign_trigger",
    "verify_trigger",
]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The accept-gated staged retrain pipeline.

This is where the closed loop's safety guarantees compose. A retrain runs only
when **all** of these hold, checked in order and each fail-closed:

1. **Signed trigger** -- the run carries a valid :class:`RetrainTrigger` whose
   HMAC verifies and whose bound queue hash still matches the live queue
   (:mod:`.trigger`). No signature, no run.
2. **Human verification** -- ``human_verified`` is ``True``. Human-in-the-loop
   sign-off is required *before* any model update; the pipeline refuses without
   it even with a valid trigger.
3. **Regression gate** -- the candidate refit on ``base corpus + queue`` must
   pass the OOF/adversarial merge-blocker (:mod:`.regression_gate`). A regressing
   or poisoned candidate is refused and the baseline stands.

Only then is the candidate written to the **staging** registry (:mod:`.rollback`)
-- never straight to production -- and a retrain **artifact** (metrics, trigger
fingerprint, verifier, gate verdict) is emitted for the runbook/demo. Every
decision is durably audited. Promotion staging -> production and the one-command
rollback are separate, deliberate operator steps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision
from omni_mercury_engine.intel.feedback_loop.regression_gate import (
    RegressionVerdict,
    evaluate_candidate,
    fit_candidate_weights,
)
from omni_mercury_engine.intel.feedback_loop.rollback import ModelEntry, ModelRegistry
from omni_mercury_engine.intel.feedback_loop.trigger import RetrainTrigger, verify_trigger

if TYPE_CHECKING:
    from omni_mercury_engine.intel.feedback_loop.queue import DurableLabeledQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainResult:
    """The outcome of a staged retrain attempt."""

    accepted: bool
    reason: str
    trigger_verified: bool
    human_verified: bool
    verdict: RegressionVerdict | None = None
    version: str | None = None
    artifact_path: str | None = None
    model_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "trigger_verified": self.trigger_verified,
            "human_verified": self.human_verified,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "model_path": self.model_path,
            "verdict": self.verdict.as_dict() if self.verdict else None,
        }


def _refuse(
    reason: str,
    *,
    trigger_verified: bool,
    human_verified: bool,
    verdict: RegressionVerdict | None = None,
) -> RetrainResult:
    """Build and audit a refusal (the baseline model stands)."""
    record_gate_decision(
        decision="retrain_refused",
        source="feedback_loop:retrain",
        disposition="hard_refuse",
        signals=("closed_loop", "gated_retrain"),
        reason=reason,
    )
    logger.warning("staged retrain refused: %s", reason)
    return RetrainResult(
        accepted=False,
        reason=reason,
        trigger_verified=trigger_verified,
        human_verified=human_verified,
        verdict=verdict,
    )


def staged_refit(
    queue: DurableLabeledQueue,
    trigger: RetrainTrigger,
    *,
    human_verified: bool,
    staging_dir: str | Path,
    secret: str | None = None,
    corpus_version: str = "current",
    base_rows: list[dict[str, Any]] | None = None,
) -> RetrainResult:
    """Run the accept-gated staged retrain (see module docstring for the gates).

    Args:
        queue: The durable labeled queue to retrain on.
        trigger: The signed authorization for this run.
        human_verified: Whether a human has verified/approved the retrain. Must
            be ``True`` or the run is refused.
        staging_dir: Directory for the staged model artifact + registry.
        secret: HMAC key for trigger verification (defaults to env).
        corpus_version: The base corpus version stamped into the artifact.
        base_rows: Base corpus rows (loaded from the authoritative corpus if None).

    Returns:
        A :class:`RetrainResult`. Accepted only when the trigger verifies, a human
        verified, and the candidate passes the regression gate.
    """
    staging = Path(staging_dir)

    # Gate 1: signed trigger, bound to the live queue snapshot.
    live_hash = queue.snapshot_hash()
    if not verify_trigger(trigger, secret=secret, expected_queue_hash=live_hash):
        return _refuse(
            "signed trigger did not verify (bad signature or queue changed since signing)",
            trigger_verified=False,
            human_verified=human_verified,
        )

    # Gate 2: human verification before any model update.
    if not human_verified:
        return _refuse(
            "human verification required before a model update; none provided",
            trigger_verified=True,
            human_verified=False,
        )

    examples = queue.pending()
    if not examples:
        return _refuse(
            "labeled queue is empty; nothing to retrain on",
            trigger_verified=True,
            human_verified=True,
        )

    # Gate 3: OOF/adversarial regression gate (the merge-blocker).
    verdict = evaluate_candidate(examples, base_rows=base_rows)
    if not verdict.accepted:
        return _refuse(
            "candidate model failed the OOF/adversarial regression gate: "
            + "; ".join(verdict.violations),
            trigger_verified=True,
            human_verified=True,
            verdict=verdict,
        )

    # Accepted: write the staged candidate + artifact, register it.
    version = f"{corpus_version}+{trigger.queue_hash[:12]}"
    staging.mkdir(parents=True, exist_ok=True)
    weights = fit_candidate_weights(examples, base_rows=base_rows)
    model_path = staging / f"candidate_{version}.json"
    model_path.write_text(
        json.dumps(
            {
                "version": version,
                "weights": [float(w) for w in weights],
                "feature_order": ["n_offensive", "n_allow", "hazard_weight"],
                "n_examples": len(examples),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = {
        "version": version,
        "accepted": True,
        "corpus_version": corpus_version,
        "n_labeled_examples": len(examples),
        "trigger": trigger.as_dict(),
        "requested_by": trigger.requested_by,
        "human_verified": True,
        "regression_verdict": verdict.as_dict(),
        "model_path": str(model_path),
        "note": "STAGED ONLY -- promotion to production is a separate operator step.",
    }
    artifact_path = staging / f"retrain_{version}.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    registry = ModelRegistry(staging)
    registry.register(
        ModelEntry(
            version=version,
            model_path=str(model_path),
            kind="candidate",
            metrics=verdict.candidate.as_dict(),
        )
    )

    record_gate_decision(
        decision="retrain_accepted",
        source="feedback_loop:retrain",
        disposition="approved",
        signals=("closed_loop", "gated_retrain", "staged"),
        reason=f"staged candidate {version} passed all gates ({len(examples)} examples)",
        extra={"version": version, "requested_by": trigger.requested_by},
    )
    logger.info("staged retrain accepted: %s (%d examples)", version, len(examples))
    return RetrainResult(
        accepted=True,
        reason="candidate passed trigger, human, and regression gates; staged",
        trigger_verified=True,
        human_verified=True,
        verdict=verdict,
        version=version,
        artifact_path=str(artifact_path),
        model_path=str(model_path),
    )


__all__ = ["RetrainResult", "staged_refit"]

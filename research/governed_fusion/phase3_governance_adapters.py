# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gate-backed Phase 3 governance policies for the live engine surfaces.

This module is the bridge between the engine-owned governance seam
(:mod:`omni_mercury_engine.governance.self_improvement`) and the Phase 2
governed promotion gate. The engine packages (`agentic.orchestration`,
`ml.online_learning`) depend only on the seam; the concrete policies that turn
a live proposal into Phase 2 promotion-gate evidence live here, in the research
tier, and are injected at composition time. The dependency therefore points
research → engine, matching the rest of the codebase and keeping the engine
wheel free of the research tree.

Both policies are **fail-closed for autonomous application**: routing a live
proposal through the gate never authorises an autonomous mutation. A proposal
that clears the gate is *queued for human approval* (the gate's ``promote``
result is human-review gated by design); a proposal that fails — including one
with no held-out-replay candidate evidence — is *withheld*. The live operating
point or model therefore moves only through an out-of-band, evidence-backed,
human-approved promotion, which is the entire contract of governed recursive
self-improvement.

An optional ``evidence_provider`` supplies the held-out-replay candidate record
for a proposal in contexts that can produce one (a governed offline promotion
run). With no provider — the unattended live loop — there is no candidate
evidence, so the gate rejects and the change is withheld.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.governance.self_improvement import (
    GovernanceOutcome,
    GovernanceReview,
)
from research.governed_fusion.phase3_governance import (
    Phase3Action,
    Phase3Decision,
    route_drift_recalibration,
    route_reflexion_threshold,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from omni_mercury_engine.governance.self_improvement import (
        ProposedRecalibration,
        ProposedThresholdChange,
    )

    ThresholdEvidenceProvider = Callable[[ProposedThresholdChange], Mapping[str, object] | None]
    RecalibrationEvidenceProvider = Callable[[ProposedRecalibration], Mapping[str, object] | None]

# Phase 3 actions that mean "the gate cleared the candidate" (still human-review
# gated). Every other terminal action is a withhold.
_QUEUE_ACTIONS = frozenset(
    {
        Phase3Action.QUEUE_REFLEXION_CANDIDATE.value,
        Phase3Action.QUEUE_RECALIBRATION_CANDIDATE.value,
        Phase3Action.QUEUE_DORMANT_REVIVAL_CANDIDATE.value,
    }
)


def _review_from_decision(decision: Phase3Decision) -> GovernanceReview:
    """Map a Phase 3 routing decision to a fail-closed governance review.

    A queued (gate-cleared) candidate is never applied autonomously — it awaits
    human approval. Every other disposition is withheld. ``applied`` is
    therefore always ``False`` for a gate-backed live policy.
    """
    from dataclasses import asdict

    record = asdict(decision)
    if decision.action in _QUEUE_ACTIONS:
        reasons = [
            "promotion gate cleared the candidate; queued for human approval "
            "(no autonomous application)",
            *decision.reasons,
        ]
        return GovernanceReview(
            applied=False,
            outcome=GovernanceOutcome.QUEUED.value,
            reasons=reasons,
            record=record,
        )
    reasons = list(decision.reasons) or [f"promotion gate returned {decision.action}"]
    return GovernanceReview(
        applied=False,
        outcome=GovernanceOutcome.WITHHELD.value,
        reasons=reasons,
        record=record,
    )


class PromotionGateThresholdGovernance:
    """Route Reflexion threshold proposals through the Phase 2 promotion gate.

    Implements
    :class:`omni_mercury_engine.governance.self_improvement.ThresholdGovernance`.
    """

    def __init__(
        self,
        *,
        manifest: Mapping[str, object],
        ledger: Mapping[str, object],
        evidence_provider: ThresholdEvidenceProvider | None = None,
    ) -> None:
        """Initialise the gate-backed threshold-governance policy.

        Args:
            manifest: The Phase 1 provenance manifest the gate reads the
                external-label fitness bucket from.
            ledger: The marginal-ablation ledger the gate checks for regression.
            evidence_provider: Optional callable returning a held-out-replay
                candidate record for a proposed change. Without it, a live
                proposal has no candidate evidence and is withheld (fail-closed).
        """
        self._manifest = manifest
        self._ledger = ledger
        self._evidence_provider = evidence_provider

    def review_threshold_change(self, change: ProposedThresholdChange) -> GovernanceReview:
        """Route ``change`` through the gate and return a fail-closed review."""
        candidate_record = (
            self._evidence_provider(change) if self._evidence_provider is not None else None
        )
        decision = route_reflexion_threshold(
            {
                "recommendation": change.recommendation,
                "current_threshold": change.current_threshold,
                "suggested_threshold": change.suggested_threshold,
                "reasoning": change.reasoning,
            },
            candidate_record=candidate_record,
            manifest=self._manifest,
            ledger=self._ledger,
        )
        return _review_from_decision(decision)


class PromotionGateRecalibrationGovernance:
    """Route drift/performance recalibration proposals through the Phase 2 gate.

    Implements
    :class:`omni_mercury_engine.governance.self_improvement.RecalibrationGovernance`.
    A performance-degradation (non-drift) trigger is not a governed
    drift-recalibration surface, so it is withheld; only a high/critical drift
    proposal backed by held-out-replay evidence can clear the gate (and is then
    queued for human approval).
    """

    def __init__(
        self,
        *,
        manifest: Mapping[str, object],
        ledger: Mapping[str, object],
        evidence_provider: RecalibrationEvidenceProvider | None = None,
    ) -> None:
        """Initialise the gate-backed recalibration-governance policy.

        Args:
            manifest: The Phase 1 provenance manifest.
            ledger: The marginal-ablation ledger.
            evidence_provider: Optional callable returning a held-out-replay
                candidate record for a proposed recalibration. Without it the
                proposal is withheld (fail-closed).
        """
        self._manifest = manifest
        self._ledger = ledger
        self._evidence_provider = evidence_provider

    def review_recalibration(self, proposal: ProposedRecalibration) -> GovernanceReview:
        """Route ``proposal`` through the gate and return a fail-closed review."""
        candidate_record = (
            self._evidence_provider(proposal) if self._evidence_provider is not None else None
        )
        decision = route_drift_recalibration(
            [
                {
                    "is_drift": proposal.is_drift,
                    "severity": proposal.severity,
                    "message": proposal.reasoning,
                }
            ],
            candidate_record=candidate_record,
            manifest=self._manifest,
            ledger=self._ledger,
        )
        return _review_from_decision(decision)

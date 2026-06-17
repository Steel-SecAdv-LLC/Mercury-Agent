# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Engine-owned governance seam for autonomous self-improvement surfaces.

Phase 3 of governed recursive self-improvement enforces one rule at the place a
change would actually take effect: **no surface may mutate Mercury's live
behaviour autonomously.** Reflexion threshold adaptation
(:mod:`omni_mercury_engine.agentic.orchestration`) and drift-/performance-
triggered recalibration (:mod:`omni_mercury_engine.ml.online_learning`) no
longer apply their own recommendations; they hand each proposed change to a
governance policy that decides — fail closed — whether it may be applied.

This module defines the *interface* on the engine side so the production
packages never import the research-tier promotion gate. The concrete
gate-backed policy that turns a proposal into Phase 2 promotion-gate evidence
lives in ``research/governed_fusion/phase3_governance.py``
(:class:`PromotionGateThresholdGovernance`,
:class:`PromotionGateRecalibrationGovernance`) and is injected at composition
time. The dependency therefore points research → engine only, matching the rest
of the codebase and keeping the engine wheel free of the research tree.

Two built-in policies cover the two honest stances:

* :class:`FailClosedSelfImprovementGovernance` — the default. Every actionable
  autonomous change is *withheld*. A live operating point or model only moves
  through an evidence-backed, human-approved promotion. This is what a
  production / self-improving deployment installs (implicitly, by default).
* :class:`MeasurementGovernance` — an explicit, named opt-in for deliberate
  measurement / held-out-replay harnesses where applying the adaptation *is*
  the measurement. It is never the default; installing it by name keeps the
  measurement intent auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "FAIL_CLOSED_GOVERNANCE",
    "FailClosedSelfImprovementGovernance",
    "GovernanceOutcome",
    "GovernanceReview",
    "MeasurementGovernance",
    "ProposedRecalibration",
    "ProposedThresholdChange",
    "RecalibrationGovernance",
    "ThresholdGovernance",
    "default_self_improvement_governance",
]


class GovernanceOutcome(StrEnum):
    """Terminal disposition of a governed self-improvement proposal."""

    MAINTAIN = "maintain"
    """No actionable change was proposed; nothing to govern."""

    NOT_REQUESTED = "not_requested"
    """An actionable change exists but the caller did not request application."""

    WITHHELD = "withheld"
    """Governance refused autonomous application (fail-closed)."""

    QUEUED = "queued_for_approval"
    """Routed to the promotion gate; awaits evidence-backed human approval."""

    APPLIED = "applied"
    """Governance authorised application (measurement / approved promotion)."""


@dataclass(frozen=True)
class ProposedThresholdChange:
    """A Reflexion-proposed operating-threshold change awaiting governance.

    Attributes:
        surface: Stable surface identifier (``"reflexion_threshold"``).
        recommendation: ``"increase"`` or ``"decrease"`` (never ``"maintain"`` —
            a maintain recommendation is not a proposal and is not routed).
        current_threshold: The live operating point before the change.
        suggested_threshold: The threshold Reflexion recommends moving to.
        reasoning: The critic's recorded, evidence-grounded justification.
        evidence: Supporting counts (false positives/negatives, observations).
    """

    surface: str
    recommendation: str
    current_threshold: float
    suggested_threshold: float
    reasoning: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedRecalibration:
    """A drift-/performance-triggered recalibration awaiting governance.

    Attributes:
        surface: Stable surface identifier (``"drift_recalibration"``).
        trigger: The triggering condition (``"drift_detected"`` /
            ``"performance_degradation"`` / ``"manual"``).
        severity: Drift severity (``none``/``low``/``medium``/``high``/
            ``critical``); ``"none"`` for non-drift triggers.
        is_drift: Whether the trigger is a genuine drift detection.
        reasoning: Human-readable justification for the proposal.
        evidence: Supporting drift statistics / performance deltas.
    """

    surface: str
    trigger: str
    severity: str
    is_drift: bool
    reasoning: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceReview:
    """Outcome of a governance review over a single proposal.

    Attributes:
        applied: Whether the caller is authorised to apply the change now. A
            production policy returns ``False`` for every autonomous proposal.
        outcome: The :class:`GovernanceOutcome` value, for records and tests.
        reasons: Human-readable justification(s) for the disposition.
        record: An optional routed-decision record (e.g. a Phase 3 promotion
            routing) preserved for the append-only audit trail.
    """

    applied: bool
    outcome: str
    reasons: list[str] = field(default_factory=list)
    record: Mapping[str, object] | None = None


@runtime_checkable
class ThresholdGovernance(Protocol):
    """Reviews Reflexion operating-threshold changes before they take effect."""

    def review_threshold_change(self, change: ProposedThresholdChange) -> GovernanceReview:
        """Return whether ``change`` may be applied, fail-closed by default."""
        ...


@runtime_checkable
class RecalibrationGovernance(Protocol):
    """Reviews drift-/performance-triggered recalibration before it retrains."""

    def review_recalibration(self, proposal: ProposedRecalibration) -> GovernanceReview:
        """Return whether ``proposal`` may retrain the model, fail-closed."""
        ...


class FailClosedSelfImprovementGovernance:
    """Default policy: withhold every autonomous self-improvement change.

    Implements both governance protocols. No threshold move and no retraining is
    ever authorised autonomously; each proposal is recorded as *withheld* with
    the fail-closed reason. Under this policy a live operating point or model
    changes only through an evidence-backed, human-approved promotion executed
    out of band — which is the entire point of governed self-improvement.
    """

    _REASON = (
        "withheld: governed self-improvement forbids autonomous mutation; a "
        "change requires Phase 2 promotion-gate evidence and human approval"
    )

    def review_threshold_change(self, change: ProposedThresholdChange) -> GovernanceReview:
        """Withhold the threshold change (fail-closed)."""
        return GovernanceReview(
            applied=False,
            outcome=GovernanceOutcome.WITHHELD.value,
            reasons=[self._REASON],
        )

    def review_recalibration(self, proposal: ProposedRecalibration) -> GovernanceReview:
        """Withhold the recalibration (fail-closed)."""
        return GovernanceReview(
            applied=False,
            outcome=GovernanceOutcome.WITHHELD.value,
            reasons=[self._REASON],
        )


class MeasurementGovernance:
    """Explicit measurement policy: authorise the change to measure its effect.

    Used only by deliberate measurement / held-out-replay harnesses (the
    orchestration and online-learning validation benchmarks, and the behavioural
    characterisation tests) where applying the adaptation *is* the measurement.
    It is never the production default and must be installed by name, so that
    every context that adapts autonomously is auditable as a measurement.
    """

    def review_threshold_change(self, change: ProposedThresholdChange) -> GovernanceReview:
        """Authorise the threshold change for measurement."""
        return GovernanceReview(
            applied=True,
            outcome=GovernanceOutcome.APPLIED.value,
            reasons=["measurement context: adaptation applied to measure its effect"],
        )

    def review_recalibration(self, proposal: ProposedRecalibration) -> GovernanceReview:
        """Authorise the recalibration for measurement."""
        return GovernanceReview(
            applied=True,
            outcome=GovernanceOutcome.APPLIED.value,
            reasons=["measurement context: recalibration applied to measure its effect"],
        )


#: Process-wide stateless fail-closed policy (safe to share; holds no state).
FAIL_CLOSED_GOVERNANCE = FailClosedSelfImprovementGovernance()


def default_self_improvement_governance() -> FailClosedSelfImprovementGovernance:
    """Return the default (fail-closed) self-improvement governance policy."""
    return FAIL_CLOSED_GOVERNANCE

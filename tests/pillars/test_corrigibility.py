# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar: corrigibility — the system can be stopped, and cannot widen itself.

Corrigibility is not a disposition, it is a set of things the code must be
unable to do. Four of them:

* **A tripwire halts, and the halt sticks.** Once tripped, every dispatch is
  refused; nothing in the system can un-trip it. A stop that the stopped thing
  can undo is not a stop.
* **It cannot raise its own capability ceiling.** The ceiling is a frozen
  object; a subagent that somehow exceeds its autonomy cap trips the governor
  rather than being allowed to run.
* **No autonomous change to a live boundary.** The default self-improvement
  governance withholds every threshold move and every recalibration.
* **Promotion requires a human.** The permissive policy exists only for
  measurement harnesses, must be installed by name, and says so.
"""

from __future__ import annotations

import dataclasses

import pytest

from omni_mercury_engine.agentic.subagents.base import SubAgentResult
from omni_mercury_engine.agentic.subagents.governor import (
    AutonomyGovernor,
    CapabilityCeiling,
    GovernorTripped,
)
from omni_mercury_engine.governance.self_improvement import (
    FailClosedSelfImprovementGovernance,
    GovernanceOutcome,
    MeasurementGovernance,
    ProposedRecalibration,
    ProposedThresholdChange,
    default_self_improvement_governance,
)


def _result(status: str = "completed", *, autonomy: float = 0.5) -> SubAgentResult:
    return SubAgentResult(
        subagent_id="sa-1",
        specialty="generalist",
        task_id="t-1",
        status=status,
        autonomy_ceiling=autonomy,
    )


class TestTripwireHalts:
    def test_a_tripped_governor_refuses_every_dispatch(self) -> None:
        governor = AutonomyGovernor()
        governor.authorize(1, 0)  # healthy before the trip
        governor.release(1)

        governor.trip("operator kill")
        assert governor.is_tripped is True
        assert governor.is_halted is True
        with pytest.raises(GovernorTripped, match="operator kill"):
            governor.authorize(1, 0)

    def test_the_trip_is_irreversible_by_resume(self) -> None:
        """A safety stop that an automatic resume undoes is not a safety stop."""
        governor = AutonomyGovernor()
        governor.trip("tripwire")
        governor.resume()
        assert governor.is_tripped is True
        with pytest.raises(GovernorTripped):
            governor.authorize(1, 0)

    def test_no_public_untrip_or_reset_api_exists(self) -> None:
        exposed = {name for name in dir(AutonomyGovernor) if not name.startswith("_")}
        for forbidden in ("untrip", "reset", "clear", "unhalt", "override"):
            assert forbidden not in exposed, forbidden

    def test_a_degenerate_failure_rate_fires_the_tripwire(self) -> None:
        ceiling = CapabilityCeiling(max_failure_rate=0.5, tripwire_min_observations=4)
        governor = AutonomyGovernor(ceiling=ceiling)
        governor.observe_results([_result("failed") for _ in range(4)])
        assert governor.is_tripped is True
        with pytest.raises(GovernorTripped):
            governor.authorize(1, 0)

    def test_correct_refusals_are_not_counted_as_failures(self) -> None:
        """A gate that refuses is working; it must not trip the fleet."""
        ceiling = CapabilityCeiling(max_failure_rate=0.5, tripwire_min_observations=4)
        governor = AutonomyGovernor(ceiling=ceiling)
        governor.observe_results([_result("blocked") for _ in range(20)])
        assert governor.is_tripped is False

    def test_pause_halts_without_tripping(self) -> None:
        governor = AutonomyGovernor()
        governor.pause()
        assert governor.is_halted is True
        assert governor.is_tripped is False
        with pytest.raises(GovernorTripped, match="paused"):
            governor.authorize(1, 0)
        governor.resume()
        governor.authorize(1, 0)  # a pause is reversible; a trip is not
        governor.release(1)


class TestCannotRaiseItsOwnCeiling:
    def test_the_ceiling_is_frozen(self) -> None:
        ceiling = CapabilityCeiling()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ceiling.max_replicas = 10_000  # type: ignore[misc]

    def test_an_out_of_range_ceiling_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_autonomy"):
            CapabilityCeiling(max_autonomy=99.0)
        with pytest.raises(ValueError, match=">= 1"):
            CapabilityCeiling(max_replicas=0)
        with pytest.raises(ValueError, match="max_recursion_depth"):
            CapabilityCeiling(max_recursion_depth=-1)

    def test_exceeding_the_replica_ceiling_is_refused(self) -> None:
        governor = AutonomyGovernor(ceiling=CapabilityCeiling(max_replicas=4))
        with pytest.raises(GovernorTripped, match="exceeds per-dispatch ceiling"):
            governor.authorize(5, 0)

    def test_exceeding_the_recursion_ceiling_is_refused(self) -> None:
        governor = AutonomyGovernor(ceiling=CapabilityCeiling(max_recursion_depth=1))
        with pytest.raises(GovernorTripped, match="recursion depth"):
            governor.authorize(1, 2)

    def test_a_refused_dispatch_reserves_nothing(self) -> None:
        """Fail-closed means the failed attempt leaves no capacity consumed."""
        governor = AutonomyGovernor(ceiling=CapabilityCeiling(max_replicas=4))
        with pytest.raises(GovernorTripped):
            governor.authorize(5, 0)
        assert governor.snapshot().active == 0

    def test_an_autonomy_breach_trips_rather_than_being_tolerated(self) -> None:
        governor = AutonomyGovernor()
        with pytest.raises(GovernorTripped):
            governor.check_autonomy(governor.ceiling.max_autonomy + 0.5)
            governor.authorize(1, 0)
        assert governor.is_tripped is True

    def test_an_observed_over_autonomous_subagent_trips_the_fleet(self) -> None:
        governor = AutonomyGovernor()
        governor.observe_results([_result(autonomy=governor.ceiling.max_autonomy + 1.0)])
        assert governor.is_tripped is True


class TestNoAutonomousChangeToALiveBoundary:
    def test_the_default_governance_is_the_fail_closed_one(self) -> None:
        assert isinstance(
            default_self_improvement_governance(), FailClosedSelfImprovementGovernance
        )

    def test_a_threshold_move_is_withheld(self) -> None:
        review = default_self_improvement_governance().review_threshold_change(
            ProposedThresholdChange(
                surface="reflexion_threshold",
                recommendation="decrease",
                current_threshold=0.5,
                suggested_threshold=0.4,
                reasoning="false-negative rate rose over 1000 observations",
            )
        )
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value
        assert any("human approval" in reason for reason in review.reasons)

    def test_a_recalibration_is_withheld(self) -> None:
        review = default_self_improvement_governance().review_recalibration(
            ProposedRecalibration(
                surface="drift_recalibration",
                trigger="drift_detected",
                severity="high",
                is_drift=True,
                reasoning="population shift over 5000 samples",
            )
        )
        assert review.applied is False
        assert review.outcome == GovernanceOutcome.WITHHELD.value

    def test_the_withholding_reason_names_the_evidence_and_the_approval(self) -> None:
        """A refusal that does not say what would unblock it is not corrigible."""
        review = default_self_improvement_governance().review_recalibration(
            ProposedRecalibration(
                surface="drift_recalibration",
                trigger="drift_detected",
                severity="critical",
                is_drift=True,
            )
        )
        blob = " ".join(review.reasons).lower()
        assert "evidence" in blob or "promotion-gate" in blob
        assert "human approval" in blob

    def test_the_fail_closed_policy_holds_no_state_to_be_flipped(self) -> None:
        """A stateless policy cannot be nudged permissive over a long run."""
        policy = FailClosedSelfImprovementGovernance()
        for _ in range(50):
            assert (
                policy.review_recalibration(
                    ProposedRecalibration(
                        surface="drift_recalibration",
                        trigger="drift_detected",
                        severity="critical",
                        is_drift=True,
                    )
                ).applied
                is False
            )


class TestPromotionRequiresAHuman:
    def test_the_permissive_policy_is_not_the_default(self) -> None:
        assert not isinstance(default_self_improvement_governance(), MeasurementGovernance)

    def test_the_permissive_policy_labels_itself_a_measurement(self) -> None:
        """It must be installed by name and be auditable as a measurement."""
        review = MeasurementGovernance().review_threshold_change(
            ProposedThresholdChange(
                surface="reflexion_threshold",
                recommendation="decrease",
                current_threshold=0.5,
                suggested_threshold=0.4,
                reasoning="measurement harness",
            )
        )
        assert review.applied is True
        assert review.outcome == GovernanceOutcome.APPLIED.value
        assert any("measurement" in reason for reason in review.reasons)

    def test_the_governed_promotion_gate_is_documented(self) -> None:
        from pathlib import Path

        doc = Path(__file__).resolve().parents[2] / "docs" / "GOVERNED_PROMOTION_GATE.md"
        assert doc.is_file()
        body = doc.read_text(encoding="utf-8").lower()
        assert "human" in body

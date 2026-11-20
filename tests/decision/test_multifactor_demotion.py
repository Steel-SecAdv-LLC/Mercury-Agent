# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every applicable demotion overlay is reported, with exactly one state demotion.

Regression: ``_apply_demotions`` used to return at the first matching overlay,
so a verdict demoted for a skipped detector silently hid a simultaneous
symbolic-floor conflict and drift -- the record's reasons under-reported the
evidence against the verdict. The gate must collect *all* applicable demotion
reasons, demote the state once, and treat the deferral as resolvable by
re-input only when every matched overlay is.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    Disposition,
    ResponseAction,
    ThreeState,
)

SKIPPED = {"spectral": {"reason": "timeout", "detail": "5s budget", "trained_group": True}}
SYMBOLIC_CONFLICT = {"satisfaction": 0.10}
DRIFT_HIGH = {"is_drift": True, "severity": "HIGH"}
GOSNN_CONTESTS_ACT = {
    "ethical_gate_passed": True,
    "detection": {"anomaly_prob": 0.05, "demote_act_below": 0.2, "demote_clear_above": 0.8},
}

SKIPPED_REASON = "partial evidence: trained detector(s) did not report"
SYMBOLIC_REASON = "neuro-symbolic satisfaction"
DRIFT_REASON = "distribution drift"
GOSNN_REASON = "GOSNN fused-state detection head"


def _result(**over: Any) -> dict[str, Any]:
    """A grounded-positive result (conformal singleton {1}), overridable."""
    base: dict[str, Any] = {
        "anomaly_prob": 0.92,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "severity": 0.3,
        "conformal": {"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
    }
    base.update(over)
    return base


@pytest.fixture
def responder() -> DecisionAbstentionResponder:
    return DecisionAbstentionResponder()


class TestAllApplicableDemotionsAreReported:
    """Simultaneous overlays each leave their reason; the state demotes once."""

    def test_three_simultaneous_demotions_all_in_reasons(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                skipped_detectors=SKIPPED,
                symbolic_consistency=SYMBOLIC_CONFLICT,
                drift_detection=DRIFT_HIGH,
            )
        )
        reasons = list(rec.reasons)
        assert any(SKIPPED_REASON in r for r in reasons)
        assert any(SYMBOLIC_REASON in r for r in reasons)
        assert any(DRIFT_REASON in r for r in reasons)
        # Exactly one demotion of state: the grounding reason, then one entry
        # per matched overlay -- no duplicates from repeated demotions.
        assert "conformal singleton" in reasons[0]
        assert len(reasons) == 4
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER
        assert rec.abstained is True
        assert rec.decision_label is None
        assert rec.decision_confidence is None

    def test_gosnn_disagreement_joins_the_other_overlays(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                skipped_detectors=SKIPPED,
                symbolic_consistency=SYMBOLIC_CONFLICT,
                drift_detection=DRIFT_HIGH,
                gosnn_metadata=GOSNN_CONTESTS_ACT,
            )
        )
        reasons = list(rec.reasons)
        for fragment in (SKIPPED_REASON, SYMBOLIC_REASON, DRIFT_REASON, GOSNN_REASON):
            assert any(fragment in r for r in reasons), fragment
        assert len(reasons) == 5
        assert rec.state is ThreeState.UNAVAILABLE

    def test_resolvable_is_the_and_of_matched_overlays(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        # skipped (resolvable) + symbolic (not resolvable) -> not resolvable:
        # the deferral routes to a human, not a data request.
        rec = responder.decide(
            _result(skipped_detectors=SKIPPED, symbolic_consistency=SYMBOLIC_CONFLICT)
        )
        assert rec.response.action is ResponseAction.ESCALATE_TO_HUMAN

    def test_all_resolvable_overlays_keep_request_input(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        # skipped + drift are both resolvable by re-input / recalibration.
        rec = responder.decide(_result(skipped_detectors=SKIPPED, drift_detection=DRIFT_HIGH))
        reasons = list(rec.reasons)
        assert any(SKIPPED_REASON in r for r in reasons)
        assert any(DRIFT_REASON in r for r in reasons)
        assert rec.response.action is ResponseAction.REQUEST_INPUT


class TestSingleReasonCasesUnchanged:
    """Each overlay alone demotes exactly as it always has (wording included)."""

    def test_skipped_detector_alone(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(skipped_detectors=SKIPPED))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER
        assert rec.response.action is ResponseAction.REQUEST_INPUT
        assert len(rec.reasons) == 2
        assert rec.reasons[1] == (
            "partial evidence: trained detector(s) did not report this run "
            "(spectral); a verdict on part of the evidence is not grounded"
        )

    def test_symbolic_floor_alone(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(symbolic_consistency=SYMBOLIC_CONFLICT))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.response.action is ResponseAction.ESCALATE_TO_HUMAN
        assert len(rec.reasons) == 2
        assert rec.reasons[1] == (
            "neuro-symbolic satisfaction 0.10 below floor 0.5: the neural "
            "verdict and the symbolic rules disagree"
        )

    def test_drift_alone(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(drift_detection=DRIFT_HIGH))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.response.action is ResponseAction.REQUEST_INPUT
        assert len(rec.reasons) == 2
        assert rec.reasons[1] == (
            "distribution drift (HIGH) detected: the calibration may no longer hold"
        )

    def test_gosnn_alone(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(gosnn_metadata=GOSNN_CONTESTS_ACT))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.response.action is ResponseAction.ESCALATE_TO_HUMAN
        assert len(rec.reasons) == 2
        assert GOSNN_REASON in rec.reasons[1]

    def test_no_overlay_stays_grounded(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result())
        assert rec.state is ThreeState.GROUNDED
        assert rec.disposition is Disposition.ACT
        assert len(rec.reasons) == 1

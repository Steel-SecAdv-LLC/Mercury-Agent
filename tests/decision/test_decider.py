# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The abstention gate maps a detection certificate onto the three-state contract.

Pure-Python tier (no torch / no engine required): the decider is a deterministic
function of a result dict, so these run in every environment and pin the
calibration-grounded "don't-know" gate behaviour exactly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionPolicy,
    Disposition,
    ResponseAction,
    ThreeState,
)


def _result(**over: Any) -> dict[str, Any]:
    """A minimal detect_with_fusion-style result, overridable per test."""
    base: dict[str, Any] = {
        "anomaly_prob": 0.5,
        "is_anomaly": False,
        "threshold_used": 0.5,
        "severity": 0.0,
    }
    base.update(over)
    return base


def _conformal(labels: list[int], coverage: float = 0.9) -> dict[str, Any]:
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": coverage,
    }


@pytest.fixture
def responder() -> DecisionAbstentionResponder:
    return DecisionAbstentionResponder()


class TestConformalGroundsTheDecision:
    """The conformal label set is authoritative when present."""

    def test_singleton_anomaly_is_grounded_act(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(anomaly_prob=0.92, is_anomaly=True, severity=0.8, conformal=_conformal([1]))
        )
        assert rec.state is ThreeState.GROUNDED
        assert rec.disposition is Disposition.ACT
        assert rec.decision_label == 1
        assert rec.abstained is False
        # Confidence is the distribution-free coverage level, not a raw score.
        assert rec.decision_confidence == pytest.approx(0.9)
        assert rec.calibrated is True

    def test_singleton_normal_is_grounded_clear(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(_result(anomaly_prob=0.05, conformal=_conformal([0])))
        assert rec.state is ThreeState.GROUNDED
        assert rec.disposition is Disposition.CLEAR
        assert rec.decision_label == 0
        assert rec.response.action is ResponseAction.MONITOR
        assert rec.response.notify is False

    def test_ambiguous_set_is_unavailable_defer(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(_result(anomaly_prob=0.55, conformal=_conformal([0, 1])))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER
        assert rec.decision_label is None
        assert rec.decision_confidence is None
        assert rec.abstained is True
        # A calibrated ambiguity routes to a human, not a data request.
        assert rec.response.action is ResponseAction.ESCALATE_TO_HUMAN

    def test_empty_set_is_undecidable_hold(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(anomaly_prob=0.4, conformal=_conformal([])))
        assert rec.state is ThreeState.UNDECIDABLE
        assert rec.disposition is Disposition.HOLD
        assert rec.response.action is ResponseAction.HOLD
        assert rec.response.fail_closed is True
        assert rec.response.requires_human is True

    def test_singleton_label_overrides_raw_is_anomaly(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        # Raw threshold flag says normal, but the calibrated singleton says
        # anomaly -- the certificate wins and carries the coverage guarantee.
        rec = responder.decide(
            _result(anomaly_prob=0.49, is_anomaly=False, conformal=_conformal([1]))
        )
        assert rec.decision_label == 1
        assert rec.disposition is Disposition.ACT


class TestEthicalFailClosed:
    """A refused ethical boundary forces a fail-closed hold, over any score."""

    def test_ethical_block_beats_calibrated_anomaly(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.99,
                is_anomaly=True,
                severity=0.95,
                conformal=_conformal([1]),
                gosnn_metadata={
                    "ethical_gate_passed": False,
                    "sigma_immutable_score": 0.1,
                    "sigma_immutable_threshold": 0.93,
                },
            )
        )
        assert rec.state is ThreeState.UNDECIDABLE
        assert rec.disposition is Disposition.HOLD
        assert rec.response.fail_closed is True

    def test_ethical_pass_allows_grounding(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.92,
                is_anomaly=True,
                conformal=_conformal([1]),
                gosnn_metadata={"ethical_gate_passed": True, "sigma_immutable_score": 0.97},
            )
        )
        assert rec.state is ThreeState.GROUNDED
        # No "absent verdict" caveat when the gate actually ran and passed.
        assert not any("ethical-gate verdict is absent" in c for c in rec.caveats)


class TestDemotionOverlays:
    """Disagreement / drift can only weaken a grounded verdict toward abstain."""

    def test_symbolic_disagreement_demotes_to_defer(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.92,
                is_anomaly=True,
                conformal=_conformal([1]),
                symbolic_consistency={"satisfaction": 0.2},
            )
        )
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER
        # Neural-vs-symbolic conflict -> human adjudication.
        assert rec.response.action is ResponseAction.ESCALATE_TO_HUMAN

    def test_high_symbolic_agreement_keeps_grounded(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.92,
                is_anomaly=True,
                conformal=_conformal([1]),
                symbolic_consistency={"satisfaction": 0.95},
            )
        )
        assert rec.state is ThreeState.GROUNDED

    def test_drift_demotes_to_request_input(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.92,
                is_anomaly=True,
                conformal=_conformal([1]),
                drift_detection={"is_drift": True, "severity": "HIGH", "p_value": 0.001},
            )
        )
        assert rec.state is ThreeState.UNAVAILABLE
        # Drift is resolved by recalibration -> a data request.
        assert rec.response.action is ResponseAction.REQUEST_INPUT

    def test_low_drift_severity_does_not_demote(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(
            _result(
                anomaly_prob=0.92,
                is_anomaly=True,
                conformal=_conformal([1]),
                drift_detection={"is_drift": True, "severity": "LOW", "p_value": 0.2},
            )
        )
        assert rec.state is ThreeState.GROUNDED

    def test_overlays_demote_grounded_normal_too(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        # Disagreement on a "normal" verdict is still disagreement: defer.
        rec = responder.decide(
            _result(
                anomaly_prob=0.05,
                conformal=_conformal([0]),
                symbolic_consistency={"satisfaction": 0.1},
            )
        )
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER


class TestUncalibratedFallback:
    """With no certificate, the threshold band decides -- honestly flagged."""

    def test_near_threshold_band_defers(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(anomaly_prob=0.52, is_anomaly=True, threshold_used=0.5))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER
        assert rec.calibrated is False

    def test_far_above_grounds_but_flags_uncalibrated(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(_result(anomaly_prob=0.95, is_anomaly=True, threshold_used=0.5))
        assert rec.state is ThreeState.GROUNDED
        assert rec.disposition is Disposition.ACT
        assert rec.calibrated is False
        assert any("no conformal coverage certificate" in c for c in rec.caveats)

    def test_require_calibrated_for_act_demotes_uncalibrated_positive(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(require_calibrated_for_act=True)
        )
        rec = responder.decide(_result(anomaly_prob=0.95, is_anomaly=True, threshold_used=0.5))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER


class TestPolicyKnobs:
    """Policy changes the gate's behaviour deterministically."""

    def test_fail_open_on_atypical_makes_empty_set_defer(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(fail_closed_on_atypical=False)
        )
        rec = responder.decide(_result(anomaly_prob=0.4, conformal=_conformal([])))
        assert rec.state is ThreeState.UNAVAILABLE
        assert rec.disposition is Disposition.DEFER

    def test_wider_band_defers_more(self) -> None:
        wide = DecisionAbstentionResponder(policy=DecisionPolicy(indecision_margin=0.2))
        narrow = DecisionAbstentionResponder(policy=DecisionPolicy(indecision_margin=0.01))
        res = _result(anomaly_prob=0.62, is_anomaly=True, threshold_used=0.5)
        assert wide.decide(res).state is ThreeState.UNAVAILABLE
        assert narrow.decide(res).state is ThreeState.GROUNDED


class TestRecordContract:
    """The emitted record is deterministic, serialisable and self-explaining."""

    def test_determinism(self, responder: DecisionAbstentionResponder) -> None:
        res = _result(anomaly_prob=0.7, is_anomaly=True, conformal=_conformal([1]))
        assert responder.decide(res).to_dict() == responder.decide(res).to_dict()

    def test_to_dict_is_json_serialisable(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(anomaly_prob=0.92, conformal=_conformal([1])))
        text = json.dumps(rec.to_dict())
        assert '"state": "grounded"' in text

    def test_explain_mentions_state_and_response(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(_result(anomaly_prob=0.92, severity=0.9, conformal=_conformal([1])))
        text = rec.explain()
        assert "GROUNDED" in text
        assert "Response:" in text

    @pytest.mark.parametrize(
        ("result", "marker"),
        [
            (_result(anomaly_prob=0.55, conformal=_conformal([0, 1])), "UNAVAILABLE"),
            (_result(anomaly_prob=0.4, conformal=_conformal([])), "UNDECIDABLE"),
        ],
    )
    def test_explain_covers_each_abstention_state(
        self, responder: DecisionAbstentionResponder, result: dict[str, Any], marker: str
    ) -> None:
        text = responder.decide(result).explain()
        assert marker in text
        assert "Response:" in text

    def test_signals_carry_provenance(self, responder: DecisionAbstentionResponder) -> None:
        rec = responder.decide(_result(anomaly_prob=0.92, conformal=_conformal([1])))
        assert rec.signals["coverage"] == pytest.approx(0.9)
        assert "policy" in rec.signals  # the active thresholds travel with the record


class TestRecordRoundTrip:
    """``DecisionRecord.from_dict`` is the exact inverse of ``to_dict``."""

    @pytest.mark.parametrize(
        "result",
        [
            _result(anomaly_prob=0.95, is_anomaly=True, severity=0.9, conformal=_conformal([1])),
            _result(anomaly_prob=0.02, conformal=_conformal([0])),
            _result(anomaly_prob=0.55, conformal=_conformal([0, 1])),
            _result(anomaly_prob=0.40, conformal=_conformal([])),
            _result(anomaly_prob=0.52, is_anomaly=True),  # uncalibrated band -> defer
            _result(anomaly_prob=0.95, is_anomaly=True),  # uncalibrated grounded w/ caveats
        ],
    )
    def test_from_dict_is_inverse_of_to_dict(
        self, responder: DecisionAbstentionResponder, result: dict[str, Any]
    ) -> None:
        rec = responder.decide(result, domain="security")
        restored = type(rec).from_dict(rec.to_dict())
        # The reconstructed record serialises identically (a true round-trip)...
        assert restored.to_dict() == rec.to_dict()
        # ...with the enums and nested plan rebuilt as live objects, not strings.
        assert restored.state is rec.state
        assert restored.disposition is rec.disposition
        assert restored.response.action is rec.response.action
        assert restored.reasons == rec.reasons

    def test_from_dict_survives_a_json_text_round_trip(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        rec = responder.decide(_result(anomaly_prob=0.92, conformal=_conformal([1])))
        restored = type(rec).from_dict(json.loads(json.dumps(rec.to_dict())))
        assert restored.to_dict() == rec.to_dict()


class TestUnknownSignalsAreNoOps:
    """The gate reads only the signals it understands; new result keys are inert.

    The post-#278 governed-fusion substrate adds ``result["info_geometry_
    certificate"]`` (per-detector component price level-sets).  By its own
    contract it certifies a *component's* boundary, "NOT the fused/gated
    verdict", and in the single-sample serve path the gate runs in its adaptive
    threshold collapses onto the point (price == threshold), so it carries no
    information that could soundly refine the fused decision.  The gate must
    therefore treat it -- and any other unmodelled key -- as an exact no-op.
    """

    def test_info_geometry_certificate_does_not_change_the_decision(
        self, responder: DecisionAbstentionResponder
    ) -> None:
        base = _result(anomaly_prob=0.92, is_anomaly=True, severity=0.7, conformal=_conformal([1]))
        certificate = {
            "statistical": {
                "model": "information_geometry_mahalanobis",
                "certifies": "info_geometry component price level-set; NOT the fused/gated verdict",
                "price": [3.4],
                "threshold_price": 3.0,
                "component_verdict": [True],
                "certified_l2_radius": [0.0],
            }
        }
        with_cert = responder.decide({**base, "info_geometry_certificate": certificate})
        assert with_cert.to_dict() == responder.decide(base).to_dict()

    @pytest.mark.parametrize(
        "result",
        [
            _result(anomaly_prob=0.92, conformal=_conformal([1])),  # grounded
            _result(anomaly_prob=0.55, conformal=_conformal([0, 1])),  # unavailable
            _result(anomaly_prob=0.40, conformal=_conformal([])),  # undecidable
        ],
    )
    def test_arbitrary_unknown_keys_are_ignored(
        self, responder: DecisionAbstentionResponder, result: dict[str, Any]
    ) -> None:
        noisy = responder.decide(
            {**result, "info_geometry_certificate": {"d": {"price": [9.9]}}, "future_key": [1, 2]}
        )
        assert noisy.to_dict() == responder.decide(result).to_dict()

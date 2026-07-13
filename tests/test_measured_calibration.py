# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for measured calibration routing (issue #1).

- Golden-ratio PHI confidence is gone; confidence is a transparent monotone prior
  until a calibrator is fitted, then routed through it.
- epistemic/aleatoric are flagged measured vs placeholder transparently.
- The decider's uncalibrated fallback routes through an attached calibrator.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.states import Disposition


class TestUncertaintyHonesty:
    def test_no_golden_ratio_constant(self) -> None:
        # The magic golden-ratio confidence driver is removed.
        assert not hasattr(UncertaintyQuantifier, "PHI")

    def test_scalar_input_flags_unmeasured(self) -> None:
        q = UncertaintyQuantifier(enable_aci=False, seed=0)
        est = q.estimate_uncertainty(np.array([0.8]))
        assert est.epistemic_measured is False
        assert est.aleatoric_measured is False
        assert est.confidence_calibrated is False
        assert "not measured" in est.explanation
        assert 0.0 < est.confidence < 1.0

    def test_ensemble_gives_measured_epistemic(self) -> None:
        q = UncertaintyQuantifier(enable_aci=False, seed=0)
        members = np.array([0.2, 0.4, 0.6, 0.9])  # disagreeing ensemble
        est = q.estimate_uncertainty(np.array([0.5]), ensemble_predictions=members)
        assert est.epistemic_measured is True
        # epistemic is the measured variance across members
        assert np.isclose(est.epistemic, float(np.var(members.reshape(-1, 1), axis=0).mean()))

    def test_confidence_calibrator_changes_provenance(self) -> None:
        q = UncertaintyQuantifier(enable_aci=False, seed=0)
        # Accumulate a calibration history where confidence overstates accuracy.
        rng = np.random.default_rng(0)
        for _ in range(400):
            conf = float(rng.uniform(0.5, 0.99))
            correct = rng.uniform(0, 1) < conf**2  # systematically over-confident
            q.update_with_outcome(prediction=0.8, confidence=conf, true_value=bool(correct))
        report = q.fit_confidence_calibrator(min_samples=50)
        assert report is not None
        est = q.estimate_uncertainty(np.array([0.8]))
        assert est.confidence_calibrated is True


class _FakeCalibrator:
    """Minimal calibrator stub matching the CalibratedConfidence contract."""

    is_calibrated = True

    def transform_one(self, score: float) -> float:
        return 0.123  # a fixed, recognizable calibrated value


class TestDeciderCalibratorRouting:
    @staticmethod
    def _uncalibrated_positive_result() -> dict[str, float | bool]:
        # No conformal certificate -> the threshold-band fallback path.
        return {"anomaly_prob": 0.95, "threshold_used": 0.5, "is_anomaly": True, "severity": 0.6}

    def test_margin_heuristic_without_calibrator(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(require_calibrated_for_act=False)
        )
        rec = responder.decide(self._uncalibrated_positive_result())
        # 0.5 + |0.95 - 0.5| = 0.95 (clipped to 1.0)
        assert rec.decision_confidence is not None
        assert abs(rec.decision_confidence - min(1.0, 0.5 + 0.45)) < 1e-9
        assert any("uncalibrated margin heuristic" in r for r in rec.reasons)

    def test_calibrated_confidence_with_calibrator(self) -> None:
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(require_calibrated_for_act=False),
            confidence_calibrator=_FakeCalibrator(),
        )
        rec = responder.decide(self._uncalibrated_positive_result())
        assert rec.decision_confidence == 0.123
        assert any("fitted score calibrator" in r for r in rec.reasons)

    def test_calibrated_confidence_is_flipped_for_clear_verdict(self) -> None:
        """A confidently-benign CLEAR call must report high confidence, not the
        raw calibrated P(anomaly) -- which would read as near-zero confidence
        for the most confident possible CLEAR verdict."""
        responder = DecisionAbstentionResponder(
            policy=DecisionPolicy(require_calibrated_for_act=False),
            confidence_calibrator=_FakeCalibrator(),
        )
        # anomaly_prob well below threshold - margin -> label=0 (CLEAR).
        result = {
            "anomaly_prob": 0.02,
            "threshold_used": 0.5,
            "is_anomaly": False,
            "severity": 0.0,
        }
        rec = responder.decide(result)
        assert rec.disposition is Disposition.CLEAR
        # _FakeCalibrator.transform_one always returns 0.123 (P(anomaly));
        # confidence in a CLEAR verdict is 1 - P(anomaly) = 0.877.
        assert rec.decision_confidence is not None
        assert abs(rec.decision_confidence - 0.877) < 1e-9

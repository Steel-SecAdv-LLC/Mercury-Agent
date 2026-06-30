# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the single confidence-calibration routing point."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.confidence import CalibratedConfidence, ConfidenceReport


def _miscalibrated_data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that rank well but are systematically over-confident.

    The true P(y=1|s) is s**2, so the raw score s is miscalibrated (too high in
    the mid-range) while preserving rank order -- exactly what a calibrator
    should be able to fix without changing AUROC.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < s**2).astype(int)
    return s, y


class TestCalibratedConfidence:
    def test_identity_until_fit(self) -> None:
        cc = CalibratedConfidence()
        assert not cc.is_calibrated
        out = cc.transform(np.array([0.2, 0.9]))
        assert np.allclose(out, [0.2, 0.9])  # identity passthrough

    def test_fit_improves_calibration_and_reports(self) -> None:
        s, y = _miscalibrated_data()
        cc = CalibratedConfidence(method="auto", seed=0)
        report = cc.fit(s, y)
        assert isinstance(report, ConfidenceReport)
        assert report.held_out is True
        assert cc.is_calibrated
        # The accept-gate guarantees no held-out regression.
        assert report.brier_cal <= report.brier_raw + 1e-9
        assert report.ece_cal <= report.ece_raw + cc.ece_tol
        # On genuinely miscalibrated data, calibration should help.
        assert report.brier_improvement >= 0.0

    def test_transform_outputs_in_unit_interval(self) -> None:
        s, y = _miscalibrated_data()
        cc = CalibratedConfidence(method="isotonic", seed=1)
        cc.fit(s, y)
        p = cc.transform(np.linspace(0, 1, 50))
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_degenerate_data_stays_identity(self) -> None:
        cc = CalibratedConfidence()
        # single class
        report = cc.fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 0, 0]))
        assert not cc.is_calibrated
        assert not report.accepted
        assert "uncalibrated" in report.note or "single-class" in report.note

    def test_accept_gate_rejects_when_no_improvement(self) -> None:
        # Already well-calibrated data: calibration cannot meaningfully help and
        # may regress on the held-out split -> identity fallback is acceptable.
        rng = np.random.default_rng(3)
        s = rng.uniform(0, 1, 400)
        y = (rng.uniform(0, 1, 400) < s).astype(int)  # P(y|s) = s, already calibrated
        cc = CalibratedConfidence(method="auto", seed=3)
        report = cc.fit(s, y)
        # Whether accepted or not, the routing point never regresses held-out Brier.
        assert report.brier_cal <= report.brier_raw + 1e-9
        if not cc.is_calibrated:
            assert np.allclose(cc.transform(s[:5]), np.clip(s[:5], 0, 1))

    def test_no_holdout_split_never_accepts_in_sample(self) -> None:
        # n >= 8 and two classes (so it passes the degenerate gate), but the
        # minority class has a single sample -> the stratified split cannot
        # leave a positive on the eval side, so no genuine held-out split is
        # possible. The routing point must NOT fit-and-accept in-sample (that
        # would reward overfitting); it must stay identity and say so.
        s = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.95])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        cc = CalibratedConfidence(method="auto", seed=0)
        report = cc.fit(s, y)
        assert report.held_out is False
        assert report.accepted is False
        assert not cc.is_calibrated
        # Identity passthrough, and metrics reported are the raw (in-sample) ones
        # with no claimed improvement.
        assert report.brier_cal == report.brier_raw
        assert report.ece_cal == report.ece_raw
        assert np.allclose(cc.transform(s[:5]), np.clip(s[:5], 0, 1))

    def test_report_to_dict_is_json_safe(self) -> None:
        import json

        s, y = _miscalibrated_data()
        cc = CalibratedConfidence(seed=0)
        report = cc.fit(s, y)
        json.dumps(report.to_dict())  # must not raise


class TestCrossValidatedHonesty:
    """The deployed (refit-on-all) map is what is measured, the verdict is
    reproducible, and small-n noise cannot be accepted as calibration."""

    def test_eval_protocol_is_cross_validated(self) -> None:
        s, y = _miscalibrated_data()
        cc = CalibratedConfidence(method="auto", seed=0)
        report = cc.fit(s, y)
        # Metrics are an out-of-fold estimate of the deployed map, not a discarded
        # holdout: protocol is CV and at least two folds ran.
        assert report.eval_protocol == "cv_oof"
        assert report.n_folds >= 2
        assert report.held_out is True
        # The acceptance is backed by a bootstrap CI on the Brier improvement.
        d = report.to_dict()
        assert "brier_delta_ci" in d and len(d["brier_delta_ci"]) == 2
        if report.accepted:
            # Accepted only when the whole one-sided CI sits below zero.
            assert report.brier_delta_ci_high < 0.0
            assert report.accepted_significant is True

    def test_verdict_is_deterministic_by_default(self) -> None:
        # Same data, default (fixed) seed -> identical accept/reject + metrics,
        # so this calibration *contract* does not flip run to run.
        s, y = _miscalibrated_data()
        r1 = CalibratedConfidence(method="auto").fit(s, y)
        r2 = CalibratedConfidence(method="auto").fit(s, y)
        assert r1.accepted == r2.accepted
        assert r1.brier_cal == r2.brier_cal
        assert r1.ece_cal == r2.ece_cal
        assert r1.brier_delta_ci_high == r2.brier_delta_ci_high

    def test_no_signal_small_n_abstains(self) -> None:
        # Already well-calibrated data (P(y|s)=s) at small n: there is no
        # miscalibration to exploit, so the bootstrap gate must abstain rather
        # than manufacture an "improvement" from fold noise.
        for seed in range(5):
            rng = np.random.default_rng(seed)
            s = rng.uniform(0, 1, 30)
            y = (rng.uniform(0, 1, 30) < s).astype(int)
            cc = CalibratedConfidence(method="isotonic", seed=seed)
            report = cc.fit(s, y)
            assert report.accepted is False
            assert not cc.is_calibrated
            # Never claims an improvement it cannot support.
            assert report.brier_cal == report.brier_raw
            assert report.ece_cal == report.ece_raw

    def test_acceptance_implies_significance(self) -> None:
        # Soundness invariant: whenever the gate accepts, the improvement is
        # statistically backed (whole one-sided Brier-delta CI below zero), so an
        # accept is never a coin flip.
        s, y = _miscalibrated_data(n=500, seed=11)
        cc = CalibratedConfidence(method="auto", seed=11)
        report = cc.fit(s, y)
        if report.accepted:
            assert report.brier_delta_ci_high < 0.0
            assert report.brier_cal < report.brier_raw

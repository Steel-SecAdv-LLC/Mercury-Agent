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
        cc.fit(s, y)
        json.dumps(cc.report.to_dict())  # must not raise

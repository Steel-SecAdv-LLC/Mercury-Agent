# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for wiring Mercury's calibrators onto medical scores."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.medical.clinical_calibration import (
    SUPPORTED_METHODS,
    BayesianBinningCalibrator,
    calibrate_and_evaluate,
    compare_calibrators,
    conformal_coverage_report,
    fit_calibrator,
)
from omni_mercury_engine.medical.reference_cohorts import (
    ReferenceCohort,
    split_cohort,
    synthetic_calibrated_cohort,
)


def _overconfident_cohort(n: int = 4000, seed: int = 0) -> tuple[ReferenceCohort, ReferenceCohort]:
    """Return calibration/test splits of an over-confident synthetic cohort."""
    cohort = synthetic_calibrated_cohort(n=n, seed=seed, miscalibration=2.2)
    return split_cohort(cohort, seed=seed)


def test_all_methods_fit_and_transform_shapes() -> None:
    """Every supported calibrator fits and maps scores into [0, 1]."""
    cal, test = _overconfident_cohort()
    for method in SUPPORTED_METHODS:
        calibrator = fit_calibrator(method, cal.scores, cal.labels)
        out = calibrator.transform(test.scores)
        assert out.shape == test.scores.shape
        assert np.all((out >= 0.0) & (out <= 1.0))


def test_bayesian_binning_reduces_ece() -> None:
    """Beta-Binomial histogram calibration lowers ECE on an over-confident score."""
    cal, test = _overconfident_cohort()
    comp = calibrate_and_evaluate(
        cal.scores, cal.labels, test.scores, test.labels, method="bayesian"
    )
    assert comp.fitted
    assert comp.ece_reduction > 0.0


def test_isotonic_and_venn_abers_reduce_ece() -> None:
    """Isotonic and Venn-Abers both improve calibration on the over-confident score."""
    cal, test = _overconfident_cohort()
    for method in ("isotonic", "venn_abers"):
        comp = calibrate_and_evaluate(
            cal.scores, cal.labels, test.scores, test.labels, method=method
        )
        assert comp.fitted, method
        assert comp.report_calibrated.ece <= comp.report_uncalibrated.ece + 1e-9, method


def test_calibration_preserves_ranking() -> None:
    """Monotone calibrators leave AUROC (rank discrimination) essentially intact."""
    cal, test = _overconfident_cohort()
    comp = calibrate_and_evaluate(
        cal.scores, cal.labels, test.scores, test.labels, method="isotonic"
    )
    assert comp.report_calibrated.auroc == pytest.approx(comp.report_uncalibrated.auroc, abs=0.02)


def test_conformal_coverage_meets_target() -> None:
    """Empirical conformal coverage is at least the target (minus sampling slack)."""
    cohort = synthetic_calibrated_cohort(n=6000, seed=5, miscalibration=1.0)
    cal, test = split_cohort(cohort, seed=5)
    report = conformal_coverage_report(
        cal.scores, cal.labels, test.scores, test.labels, coverage=0.9
    )
    assert report["available"] is True
    assert report["empirical_coverage"] >= 0.9 - 0.05
    assert 0.0 <= report["average_set_size"] <= 2.0


def test_compare_calibrators_picks_fitted_best() -> None:
    """The sweep returns a fitted best method that beats or matches baseline ECE."""
    cal, test = _overconfident_cohort()
    result = compare_calibrators(cal.scores, cal.labels, test.scores, test.labels)
    assert result["best_method"] in SUPPORTED_METHODS
    assert result["best_ece"] <= result["baseline_ece"] + 1e-9
    assert result["comparisons"][result["best_method"]]["fitted"] is True


def test_single_class_calibration_degrades_to_identity() -> None:
    """A single-class calibration split degrades to identity without crashing."""
    scores_cal = np.array([0.2, 0.3, 0.4, 0.5])
    labels_cal = np.array([0, 0, 0, 0])  # single class
    calibrator = fit_calibrator("isotonic", scores_cal, labels_cal)
    assert calibrator.fitted is False
    out = calibrator.transform(np.array([0.1, 0.9]))
    assert np.allclose(out, [0.1, 0.9])


def test_bayesian_binning_shrinks_sparse_bins_to_prior() -> None:
    """An empty bin calibrates to the Beta(1,1) prior mean 0.5."""
    cal = BayesianBinningCalibrator(n_bins=10)
    # Only populate the lowest bin; all others empty -> prior 0.5.
    cal.fit(np.array([0.05, 0.05, 0.05]), np.array([1.0, 0.0, 0.0]))
    mid = cal.transform(np.array([0.55]))[0]
    assert mid == pytest.approx(0.5)

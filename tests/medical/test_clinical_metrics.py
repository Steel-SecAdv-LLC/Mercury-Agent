# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the clinical discrimination + calibration metric engine."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.medical.clinical_metrics import (
    ClinicalMetricReport,
    bootstrap_auroc_ci,
    confusion_at_threshold,
    evaluate_clinical_scores,
    npv,
    ppv,
    reliability_curve,
    sensitivity,
    specificity,
    youden_threshold,
)


def test_confusion_at_threshold_hand_example() -> None:
    """Confusion counts match a hand-worked example."""
    y = [1, 1, 0, 0]
    s = [0.9, 0.4, 0.6, 0.1]
    tp, fp, tn, fn = confusion_at_threshold(y, s, 0.5)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_sensitivity_specificity_ppv_npv() -> None:
    """Sensitivity/specificity/PPV/NPV match the confusion definitions."""
    y = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.2, 0.7, 0.1, 0.05]
    thr = 0.5
    # positives >=0.5: 0.9,0.8 -> tp=2, fn=1 ; negatives >=0.5: 0.7 -> fp=1, tn=2
    assert sensitivity(y, s, thr) == pytest.approx(2 / 3)
    assert specificity(y, s, thr) == pytest.approx(2 / 3)
    assert ppv(y, s, thr) == pytest.approx(2 / 3)
    assert npv(y, s, thr) == pytest.approx(2 / 3)


def test_perfect_separation_auroc_one() -> None:
    """A perfectly separating score scores AUROC 1.0 and clean sens/spec."""
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    report = evaluate_clinical_scores(y, s, bootstrap=False, n_bins=5)
    assert report.auroc == pytest.approx(1.0)
    assert report.sensitivity == pytest.approx(1.0)
    assert report.specificity == pytest.approx(1.0)


def test_random_score_auroc_near_half() -> None:
    """A score independent of the label yields AUROC near 0.5."""
    rng = np.random.RandomState(0)
    y = (rng.uniform(size=4000) < 0.4).astype(int)
    s = rng.uniform(size=4000)  # independent of y
    report = evaluate_clinical_scores(y, s, bootstrap=False)
    assert report.auroc == pytest.approx(0.5, abs=0.05)


def test_youden_threshold_separates() -> None:
    """Youden's-J threshold lies between the two classes when separable."""
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    thr = youden_threshold(y, s)
    assert 0.2 < thr < 0.8


def test_reliability_curve_counts_and_rates() -> None:
    """Reliability bins partition the samples and report the right rates."""
    # 10 samples at p=0.15 with 0/10 positive, 10 at p=0.85 with 9/10 positive.
    y = [0] * 10 + [1] * 9 + [0]
    s = [0.15] * 10 + [0.85] * 10
    bins = reliability_curve(y, s, n_bins=10)
    total = sum(b.count for b in bins)
    assert total == 20
    low = next(b for b in bins if b.lower <= 0.15 < b.upper)
    high = next(b for b in bins if b.lower <= 0.85 <= b.upper)
    assert low.empirical_rate == pytest.approx(0.0)
    assert high.empirical_rate == pytest.approx(0.9)


def test_ece_detects_miscalibration() -> None:
    """A miscalibrated (over-confident) score has larger ECE than a calibrated one."""
    rng = np.random.RandomState(1)
    q = rng.beta(2, 2, size=5000)
    y = (rng.uniform(size=5000) < q).astype(int)
    logit = np.log(q / (1 - q))
    over = 1.0 / (1.0 + np.exp(-2.0 * logit))  # over-confident
    calibrated = evaluate_clinical_scores(y, q, bootstrap=False)
    miscalibrated = evaluate_clinical_scores(y, over, bootstrap=False)
    assert miscalibrated.ece > calibrated.ece
    assert calibrated.ece < 0.05


def test_bootstrap_ci_brackets_point_auroc() -> None:
    """The bootstrap CI contains the point AUROC for a signal-bearing score."""
    rng = np.random.RandomState(2)
    y = (rng.uniform(size=800) < 0.5).astype(int)
    s = np.clip(0.5 + 0.25 * (y * 2 - 1) + rng.normal(0, 0.2, size=800), 0, 1)
    report = evaluate_clinical_scores(y, s, seed=3)
    lo, hi = bootstrap_auroc_ci(y, s, seed=3)
    assert lo <= report.auroc <= hi
    assert lo > 0.5  # signal is real


def test_report_is_json_serialisable() -> None:
    """The report round-trips through its ``to_dict`` mapping."""
    y = [0, 1, 0, 1, 1, 0, 1, 0]
    s = [0.2, 0.7, 0.3, 0.8, 0.6, 0.1, 0.9, 0.4]
    report = evaluate_clinical_scores(y, s, bootstrap=False, n_bins=4)
    assert isinstance(report, ClinicalMetricReport)
    d = report.to_dict()
    assert set(d) >= {"auroc", "sensitivity", "specificity", "brier", "ece", "reliability"}
    assert isinstance(d["reliability"], list)


def test_out_of_range_scores_do_not_crash() -> None:
    """Scores outside [0, 1] are clipped for probability metrics, not rejected."""
    y = [0, 1, 0, 1]
    s = [-0.5, 1.5, 0.2, 0.9]  # deliberately out of range
    report = evaluate_clinical_scores(y, s, bootstrap=False, n_bins=4)
    assert 0.0 <= report.brier <= 1.0
    assert 0.0 <= report.ece <= 1.0

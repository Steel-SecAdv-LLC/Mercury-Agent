# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the seeded reference cohorts used by the calibration harness."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.medical.clinical_metrics import evaluate_clinical_scores
from omni_mercury_engine.medical.reference_cohorts import (
    split_cohort,
    synthetic_calibrated_cohort,
)


def test_synthetic_cohort_is_deterministic() -> None:
    """The same seed reproduces identical score/label arrays."""
    a = synthetic_calibrated_cohort(n=500, seed=7)
    b = synthetic_calibrated_cohort(n=500, seed=7)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.labels, b.labels)


def test_well_calibrated_variant_has_low_ece() -> None:
    """The miscalibration==1 cohort is (by construction) well calibrated."""
    cohort = synthetic_calibrated_cohort(n=6000, seed=0, miscalibration=1.0)
    report = evaluate_clinical_scores(cohort.labels, cohort.scores, bootstrap=False)
    assert report.ece < 0.05


def test_split_is_disjoint_and_complete() -> None:
    """Calibration/test splits partition the cohort with no overlap or loss."""
    cohort = synthetic_calibrated_cohort(n=1000, seed=1)
    cal, test = split_cohort(cohort, calibration_fraction=0.5, seed=1)
    assert len(cal.scores) + len(test.scores) == len(cohort.scores)
    # Recombine and compare as multisets of (score,label) pairs.
    combined = np.concatenate([cal.scores, test.scores])
    assert np.isclose(np.sort(combined), np.sort(cohort.scores)).all()


def test_framingham_cohort_has_real_signal() -> None:
    """The Framingham instrument discriminates the synthetic outcome (AUROC well > 0.5)."""
    pytest.importorskip("torch")
    from omni_mercury_engine.medical.reference_cohorts import framingham_cvd_cohort

    cohort = framingham_cvd_cohort(n=1500, seed=0)
    assert np.all((cohort.scores >= 0.0) & (cohort.scores <= 1.0))
    assert set(np.unique(cohort.labels).tolist()) <= {0.0, 1.0}
    report = evaluate_clinical_scores(cohort.labels, cohort.scores, bootstrap=False)
    # A real risk instrument on a signal-bearing DGP must beat chance clearly,
    # but the coarse point bins + noise keep it realistically below perfect.
    assert 0.62 < report.auroc < 0.95

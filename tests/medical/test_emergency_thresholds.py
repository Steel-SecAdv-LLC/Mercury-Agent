# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for emergency-routing threshold validation."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.medical.emergency_thresholds import (
    evaluate_threshold,
    news2_score,
    nihss_stroke_risk,
    recommend_threshold,
    sweep_threshold,
    validate_threshold,
)


def test_news2_normal_patient_scores_zero() -> None:
    """A fully normal vitals set scores NEWS2 == 0."""
    vitals = {
        "respiratory_rate": 16,
        "spo2": 98,
        "on_oxygen": False,
        "temperature_c": 36.8,
        "systolic_bp": 120,
        "heart_rate": 72,
        "consciousness": "A",
    }
    assert news2_score(vitals) == 0


def test_news2_deranged_patient_crosses_high_threshold() -> None:
    """A clearly deteriorating patient scores in the high-risk (>=7) band."""
    vitals = {
        "respiratory_rate": 26,  # +3
        "spo2": 91,  # +3
        "on_oxygen": True,  # +2
        "temperature_c": 39.2,  # +2
        "systolic_bp": 95,  # +2
        "heart_rate": 125,  # +2
        "consciousness": "V",  # +3
    }
    assert news2_score(vitals) >= 7


def test_news2_missing_param_contributes_zero() -> None:
    """A missing parameter contributes 0 (unassessed, not deranged)."""
    assert news2_score({"respiratory_rate": 26}) == 3  # only the RR band


def test_nihss_mapping_matches_production() -> None:
    """nihss_stroke_risk reproduces the production _interpret_nihss mapping."""
    pytest.importorskip("torch")
    from omni_mercury_engine.medical.critical_care.neurocritical_care import NIHSSCalculator

    calc = NIHSSCalculator()
    for score in range(0, 43):
        prod = calc._interpret_nihss(score)["stroke_risk"]
        assert nihss_stroke_risk(score) == pytest.approx(prod), score


def test_evaluate_threshold_confusion() -> None:
    """Operating-point sensitivity/specificity match a hand example."""
    scores = [0.1, 0.5, 0.7, 0.9]
    outcomes = [0, 0, 1, 1]
    pt = evaluate_threshold(scores, outcomes, 0.6)
    assert pt.sensitivity == pytest.approx(1.0)
    assert pt.specificity == pytest.approx(1.0)
    assert pt.n_positive == 2


def test_sensitivity_monotonic_in_threshold() -> None:
    """Raising the threshold never increases sensitivity."""
    rng = np.random.RandomState(0)
    scores = rng.uniform(size=1000)
    outcomes = (rng.uniform(size=1000) < scores).astype(int)
    grid = list(np.linspace(0, 1, 21))
    points = sweep_threshold(scores, outcomes, grid)
    sens = [p.sensitivity for p in points]
    assert all(sens[i] >= sens[i + 1] - 1e-9 for i in range(len(sens) - 1))


def test_recommend_criteria() -> None:
    """F2 favors recall; sensitivity_floor respects the floor."""
    rng = np.random.RandomState(1)
    scores = rng.uniform(size=2000)
    outcomes = (rng.uniform(size=2000) < scores).astype(int)
    points = sweep_threshold(scores, outcomes, list(np.linspace(0, 1, 41)))
    floored = recommend_threshold(points, criterion="sensitivity_floor", min_sensitivity=0.95)
    assert floored.sensitivity >= 0.95 or all(p.sensitivity < 0.95 for p in points)
    f2 = recommend_threshold(points, criterion="f2")
    youden = recommend_threshold(points, criterion="youden")
    # The recall-weighted optimum sits at or below the Youden operating point.
    assert f2.threshold <= youden.threshold + 1e-9


def test_validate_threshold_report_shape() -> None:
    """validate_threshold returns a current point, a sweep, and advisory optima."""
    rng = np.random.RandomState(2)
    scores = rng.uniform(size=1500)
    outcomes = (rng.uniform(size=1500) < scores).astype(int)
    report = validate_threshold(
        instrument="demo",
        literature_anchor="demo>=0.5",
        scores=scores,
        outcomes=outcomes,
        current_threshold=0.5,
        grid=list(np.linspace(0, 1, 21)),
        dgp_doc="uniform score; outcome ~ Bernoulli(score)",
    )
    d = report.to_dict()
    assert set(d) >= {"current", "sweep", "recommended_f2", "recommended_youden"}
    assert d["current"]["threshold"] == 0.5
    assert "ADVISORY" in d["advisory_note"]

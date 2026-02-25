# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Fusion and decorrelator tests for the Anomaly Math Arrest."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from omni_mercury_engine.detectors.math_arrest.arrest import (
    AnomalyMathArrest,
)
from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
from omni_mercury_engine.detectors.math_arrest.fusion import (
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def make_normal_signal(n: int = 500, seed: int = 42) -> npt.NDArray[np.float64]:
    """Sinusoidal signal with mild noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    return (np.sin(t) + rng.normal(0, 0.05, n)).astype(np.float64)


def _make_probe_result(
    name: str,
    scores: npt.NDArray[np.float64],
    confidence: float = 0.9,
    fit_quality: float = 0.8,
) -> ProbeResult:
    """Convenience factory for test ProbeResults."""
    return ProbeResult(
        probe_name=name,
        deviation_scores=scores,
        confidence=confidence,
        trajectory_fit_quality=fit_quality,
        anomaly_geometry="test",
    )


# ===================================================================
# Decorrelator: no redundancy
# ===================================================================


class TestNoRedundancyLeavesWeightsUnchanged:
    """Independent probes: all multipliers should be 1.0."""

    def test_no_redundancy_leaves_weights_unchanged(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 100
        # Build 3 orthogonal probe score columns
        col_a = rng.normal(0, 1, n_samples)
        col_b = rng.normal(10, 1, n_samples)
        col_c = rng.normal(-5, 3, n_samples)
        score_matrix = np.column_stack([col_a, col_b, col_c]).astype(np.float64)
        probe_names = ["probe_a", "probe_b", "probe_c"]
        fit_qualities = {"probe_a": 0.9, "probe_b": 0.8, "probe_c": 0.7}

        dec = CorrelationAwareDecorrelator()
        multipliers = dec.calibrate(score_matrix, probe_names, fit_qualities)

        assert dec.is_calibrated
        for name in probe_names:
            assert multipliers[name] == pytest.approx(1.0)


# ===================================================================
# Decorrelator: correlated pair
# ===================================================================


class TestCorrelatedPairReducesLowerQualityProbe:
    """Two highly-correlated probes: lower-quality one gets 0.5 multiplier."""

    def test_correlated_pair_reduces_lower_quality_probe(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 200
        base = rng.normal(0, 1, n_samples)
        col_a = base + rng.normal(0, 0.05, n_samples)  # nearly identical
        col_b = base + rng.normal(0, 0.05, n_samples)
        col_c = rng.normal(0, 1, n_samples)  # independent
        score_matrix = np.column_stack([col_a, col_b, col_c]).astype(np.float64)
        probe_names = ["probe_a", "probe_b", "probe_c"]
        fit_qualities = {"probe_a": 0.9, "probe_b": 0.5, "probe_c": 0.8}

        dec = CorrelationAwareDecorrelator()
        multipliers = dec.calibrate(score_matrix, probe_names, fit_qualities)

        # probe_a is the better probe in the correlated pair
        assert multipliers["probe_a"] == pytest.approx(1.0)
        assert multipliers["probe_b"] == pytest.approx(0.5)
        assert multipliers["probe_c"] == pytest.approx(1.0)


# ===================================================================
# Decorrelator: cluster of three
# ===================================================================


class TestCorrelatedClusterOfThree:
    """Cluster of three correlated probes: best stays 1.0, others get 1/3."""

    def test_correlated_cluster_of_three_distributes_weight(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 200
        base = rng.normal(0, 1, n_samples)
        col_a = base + rng.normal(0, 0.02, n_samples)
        col_b = base + rng.normal(0, 0.02, n_samples)
        col_c = base + rng.normal(0, 0.02, n_samples)
        score_matrix = np.column_stack([col_a, col_b, col_c]).astype(np.float64)
        probe_names = ["probe_a", "probe_b", "probe_c"]
        fit_qualities = {"probe_a": 0.5, "probe_b": 0.9, "probe_c": 0.7}

        dec = CorrelationAwareDecorrelator()
        multipliers = dec.calibrate(score_matrix, probe_names, fit_qualities)

        # probe_b has highest fit quality
        assert multipliers["probe_b"] == pytest.approx(1.0)
        assert multipliers["probe_a"] == pytest.approx(1.0 / 3.0)
        assert multipliers["probe_c"] == pytest.approx(1.0 / 3.0)


# ===================================================================
# Decorrelator: effective probe count
# ===================================================================


class TestEffectiveProbeCount:
    """effective_probe_count = sum(weight_multipliers)."""

    def test_effective_probe_count_reports_correctly(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 200
        base = rng.normal(0, 1, n_samples)
        # Correlated pair + one independent
        col_a = base + rng.normal(0, 0.05, n_samples)
        col_b = base + rng.normal(0, 0.05, n_samples)
        col_c = rng.normal(5, 2, n_samples)  # independent
        score_matrix = np.column_stack([col_a, col_b, col_c]).astype(np.float64)
        probe_names = ["probe_a", "probe_b", "probe_c"]
        fit_qualities = {"probe_a": 0.9, "probe_b": 0.5, "probe_c": 0.7}

        dec = CorrelationAwareDecorrelator()
        dec.calibrate(score_matrix, probe_names, fit_qualities)

        # probe_a = 1.0, probe_b = 0.5, probe_c = 1.0 → 2.5
        assert dec.effective_probe_count == pytest.approx(2.5)


# ===================================================================
# Fusion: fail-open on uncalibrated decorrelator
# ===================================================================


class TestFailOpenWhenDecorrelatorNotCalibrated:
    """Uncalibrated decorrelator: fuse() proceeds with unmodified weights."""

    def test_fail_open_when_decorrelator_not_calibrated(self) -> None:
        n_samples = 50
        rng = np.random.default_rng(42)
        results = [
            _make_probe_result("probe_a", rng.uniform(0, 1, n_samples).astype(np.float64)),
            _make_probe_result("probe_b", rng.uniform(0, 1, n_samples).astype(np.float64)),
        ]

        fusion = PhiWeightedFusion(n_probes=2)
        uncalibrated = CorrelationAwareDecorrelator()
        assert not uncalibrated.is_calibrated

        # Should NOT raise
        scores = fusion.fuse(results, decorrelator=uncalibrated)
        assert scores.shape == (n_samples,)
        assert not np.all(scores == 0.0)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ===================================================================
# Arrest: auto-calibrate on fit
# ===================================================================


class TestCalibrateDecorrelatorCalledAutomaticallyOnFit:
    """fit() with n_samples >= 50 must auto-calibrate the decorrelator."""

    def test_calibrate_decorrelator_called_automatically_on_fit(self) -> None:
        arrest = AnomalyMathArrest()
        data = make_normal_signal(100)
        arrest.fit(data)

        assert arrest._decorrelator.is_calibrated
        report = arrest.get_correlation_report()
        assert len(report["weight_multipliers"]) > 0

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the InSAR-style subsidence / sinkhole-precursor detector.

Covers: Theil-Sen velocity recovery on known slopes, the quadratic-vs-linear
F-test/BIC acceleration gate (both directions), fixed-radius clustering, the
sinkhole-precursor criteria, and every fail-loud contract (epoch count,
all-NaN points, shape mismatches, coherence filtering).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.geological.subsidence_detector import (
    SEVERITY_BOUNDS_MM_YR,
    SubsidenceDetector,
    SubsidencePredictionResult,
    SubsidenceSeverity,
)

RNG = np.random.default_rng(20260709)


def _epochs(n: int = 30, years: float = 2.5) -> np.ndarray:
    """Regular acquisition epochs over `years` years."""
    return np.linspace(0.0, years, n)


def _linear_series(t: np.ndarray, v_mm_yr: float, noise_mm: float = 0.5) -> np.ndarray:
    """Linear LOS motion with Gaussian noise."""
    return v_mm_yr * t + RNG.normal(0.0, noise_mm, size=t.shape)


def _accelerating_series(
    t: np.ndarray, v_mm_yr: float, a_mm_yr2: float, noise_mm: float = 0.5
) -> np.ndarray:
    """Quadratic LOS motion d = v*t + a/2*t^2 with Gaussian noise."""
    return v_mm_yr * t + 0.5 * a_mm_yr2 * t**2 + RNG.normal(0.0, noise_mm, size=t.shape)


class TestTheilSenVelocity:
    """Robust velocity estimation."""

    def test_recovers_known_slope(self) -> None:
        t = _epochs()
        det = SubsidenceDetector()
        result = det.analyze(_linear_series(t, -25.0), t)
        k = result.point_kinematics[0]
        assert k.velocity_mm_yr == pytest.approx(-25.0, abs=1.5)
        assert k.velocity_ci_mm_yr[0] <= k.velocity_mm_yr <= k.velocity_ci_mm_yr[1]

    def test_robust_to_outliers(self) -> None:
        """Theil-Sen must shrug off gross outliers that would wreck OLS."""
        t = _epochs()
        y = _linear_series(t, -25.0, noise_mm=0.2)
        y[5] += 80.0
        y[17] -= 60.0
        det = SubsidenceDetector()
        k = det.analyze(y, t).point_kinematics[0]
        assert k.velocity_mm_yr == pytest.approx(-25.0, abs=2.0)

    def test_stable_point_classified_stable(self) -> None:
        t = _epochs()
        result = SubsidenceDetector().analyze(_linear_series(t, 0.0, noise_mm=0.3), t)
        assert result.severity == SubsidenceSeverity.STABLE.value
        assert not result.anomaly_detected

    def test_severity_class_boundaries(self) -> None:
        """Severity classes follow the documented mm/yr bounds."""
        t = _epochs(40, 4.0)
        det = SubsidenceDetector()
        for v, expected in [
            (-5.0, SubsidenceSeverity.SLOW),
            (-30.0, SubsidenceSeverity.MODERATE),
            (-100.0, SubsidenceSeverity.FAST),
            (-300.0, SubsidenceSeverity.EXTREME),
        ]:
            result = det.analyze(_linear_series(t, v, noise_mm=0.2), t)
            assert result.severity == expected.value, f"v={v}"
        assert SEVERITY_BOUNDS_MM_YR == (2.0, 10.0, 50.0, 150.0)


class TestAccelerationGate:
    """Quadratic-vs-linear nested model comparison."""

    def test_flags_true_acceleration(self) -> None:
        t = _epochs(40, 2.0)
        y = _accelerating_series(t, -10.0, -40.0, noise_mm=0.4)
        k = SubsidenceDetector().analyze(y, t).point_kinematics[0]
        assert k.accelerating
        assert k.f_pvalue < 0.01
        assert k.delta_bic > 2.0
        assert k.acceleration_mm_yr2 < 0.0

    def test_linear_motion_not_flagged(self) -> None:
        t = _epochs(40, 2.0)
        y = _linear_series(t, -30.0, noise_mm=0.4)
        k = SubsidenceDetector().analyze(y, t).point_kinematics[0]
        assert not k.accelerating

    def test_decelerating_motion_not_flagged(self) -> None:
        """A slowing trend is significant quadratic but must NOT be flagged
        as accelerating (the quadratic term opposes the trend)."""
        t = _epochs(40, 2.0)
        y = _accelerating_series(t, -40.0, +30.0, noise_mm=0.3)
        k = SubsidenceDetector().analyze(y, t).point_kinematics[0]
        assert not k.accelerating

    def test_acceleration_magnitude_recovered(self) -> None:
        t = _epochs(60, 3.0)
        y = _accelerating_series(t, -5.0, -24.0, noise_mm=0.2)
        k = SubsidenceDetector().analyze(y, t).point_kinematics[0]
        assert k.acceleration_mm_yr2 == pytest.approx(-24.0, rel=0.15)


class TestClusteringAndSinkhole:
    """Spatial density clustering and sinkhole-precursor screening."""

    def _bowl_scene(
        self, n_bowl: int = 6, n_bg: int = 30, bowl_extent: float = 60.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A localized accelerating bowl inside a stable background scene."""
        t = _epochs(36, 3.0)
        series = []
        coords = []
        # Background: stable scatter over a 4 km x 4 km scene.
        for _ in range(n_bg):
            series.append(_linear_series(t, RNG.normal(0.0, 0.5), noise_mm=0.3))
            coords.append(RNG.uniform(0.0, 4000.0, size=2))
        # Bowl: tight cluster, subsiding and accelerating.
        centre = np.array([2000.0, 2000.0])
        for _ in range(n_bowl):
            series.append(_accelerating_series(t, -20.0, -30.0, noise_mm=0.3))
            coords.append(centre + RNG.uniform(-bowl_extent, bowl_extent, size=2))
        return np.array(series), t, np.array(coords)

    def test_sinkhole_precursor_detected(self) -> None:
        disp, t, coords = self._bowl_scene()
        result = SubsidenceDetector().analyze(disp, t, coordinates_m=coords)
        assert result.sinkhole_precursor_detected
        assert len(result.clusters) >= 1
        cluster = next(c for c in result.clusters if c.sinkhole_precursor)
        assert len(cluster.point_indices) >= 4
        assert cluster.extent_m <= 500.0
        assert cluster.median_velocity_mm_yr < -10.0
        assert all(cluster.criteria.values())
        assert result.anomaly_detected
        assert result.confidence > 0.5

    def test_scattered_accelerating_points_do_not_cluster(self) -> None:
        """Accelerating points spread over km must not form a bowl."""
        t = _epochs(36, 3.0)
        series = [_accelerating_series(t, -20.0, -30.0, noise_mm=0.3) for _ in range(6)]
        coords = np.array([[i * 900.0, i * 900.0] for i in range(6)])
        result = SubsidenceDetector().analyze(np.array(series), t, coordinates_m=coords)
        assert not result.sinkhole_precursor_detected
        assert result.clusters == []

    def test_no_coordinates_skips_clustering_with_note(self) -> None:
        t = _epochs(36, 3.0)
        series = np.array([_accelerating_series(t, -20.0, -30.0, noise_mm=0.3) for _ in range(5)])
        result = SubsidenceDetector().analyze(series, t)
        assert result.clusters == []
        assert not result.sinkhole_precursor_detected
        assert any("coordinates_m not supplied" in n for n in result.notes)

    def test_basin_wide_subsidence_not_sinkhole(self) -> None:
        """A uniform accelerating basin fails the below-scene-median bowl
        criterion even when points are locally dense."""
        t = _epochs(36, 3.0)
        series = np.array([_accelerating_series(t, -20.0, -30.0, noise_mm=0.3) for _ in range(8)])
        coords = RNG.uniform(0.0, 150.0, size=(8, 2))
        result = SubsidenceDetector().analyze(series, t, coordinates_m=coords)
        # Every point moves together -> cluster median == scene median.
        assert len(result.clusters) == 1
        assert not result.clusters[0].criteria["below_scene_median"]
        assert not result.sinkhole_precursor_detected


class TestFailLoud:
    """Fail-loud input contracts."""

    def test_too_few_epochs_raises(self) -> None:
        t = _epochs(7)
        with pytest.raises(ValueError, match="need >= 8"):
            SubsidenceDetector().analyze(_linear_series(t, -5.0), t)

    def test_all_nan_point_raises(self) -> None:
        t = _epochs(20)
        disp = np.vstack([_linear_series(t, -5.0), np.full_like(t, np.nan)])
        with pytest.raises(ValueError, match="only NaN"):
            SubsidenceDetector().analyze(disp, t)

    def test_mostly_nan_point_raises(self) -> None:
        t = _epochs(20)
        y = _linear_series(t, -5.0)
        y[5:] = np.nan  # 5 valid epochs < min_epochs
        with pytest.raises(ValueError, match="valid epochs"):
            SubsidenceDetector().analyze(y, t)

    def test_non_increasing_epochs_raise(self) -> None:
        t = _epochs(20)
        t[10] = t[9]
        with pytest.raises(ValueError, match="strictly increasing"):
            SubsidenceDetector().analyze(_linear_series(_epochs(20), -5.0), t)

    def test_epoch_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            SubsidenceDetector().analyze(np.zeros((2, 20)), _epochs(19))

    def test_coordinates_shape_mismatch_raises(self) -> None:
        t = _epochs(20)
        disp = np.vstack([_linear_series(t, -5.0), _linear_series(t, -5.0)])
        with pytest.raises(ValueError, match="coordinates_m shape"):
            SubsidenceDetector().analyze(disp, t, coordinates_m=np.zeros((3, 2)))

    def test_all_low_coherence_raises(self) -> None:
        t = _epochs(20)
        disp = np.vstack([_linear_series(t, -5.0), _linear_series(t, -5.0)])
        with pytest.raises(ValueError, match="coherence_min"):
            SubsidenceDetector().analyze(disp, t, coherence=np.array([0.1, 0.2]))

    def test_low_coherence_points_dropped_with_note(self) -> None:
        t = _epochs(20)
        disp = np.vstack([_linear_series(t, -5.0), _linear_series(t, -25.0)])
        result = SubsidenceDetector().analyze(disp, t, coherence=np.array([0.9, 0.1]))
        assert result.n_points == 1
        assert any("dropped below coherence_min" in n for n in result.notes)

    def test_min_epochs_floor_enforced(self) -> None:
        with pytest.raises(ValueError, match="min_epochs"):
            SubsidenceDetector(min_epochs=4)

    def test_nan_gaps_tolerated_when_enough_valid(self) -> None:
        t = _epochs(30)
        y = _linear_series(t, -25.0)
        y[[3, 7, 11]] = np.nan
        result = SubsidenceDetector().analyze(y, t)
        assert isinstance(result, SubsidencePredictionResult)
        assert result.point_kinematics[0].n_valid_epochs == 27

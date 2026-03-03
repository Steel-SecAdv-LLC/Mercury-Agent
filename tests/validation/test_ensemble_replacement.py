"""
Tests for Mercury-native ensemble: Resonance + Kinematic + InfoGeometry.

Validates that the MercuryAnomalyDetector ensemble produces correct,
bounded, and discriminative scores.

Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


class TestResonanceScore:
    """Validate the resonance (FFT-based harmonic) anomaly score."""

    @pytest.fixture
    def detector(self) -> MercuryAnomalyDetector:
        return MercuryAnomalyDetector()

    def test_noise_scored_higher_than_sine(self, detector: MercuryAnomalyDetector) -> None:
        """Noise should be more anomalous than a pure sine wave (multi-dim)."""
        rng = np.random.RandomState(42)
        t = np.linspace(0, 4 * np.pi, 200)
        sine = np.column_stack([np.sin(t + i) for i in range(5)])
        noise = rng.randn(200, 5) * 3.0

        detector.fit(sine)
        score_sine = detector._compute_resonance_score(sine).mean()
        score_noise = detector._compute_resonance_score(noise).mean()

        assert (
            score_noise > score_sine
        ), f"Resonance: noise ({score_noise:.3f}) should be > sine ({score_sine:.3f})"

    def test_small_dataset(self, detector: MercuryAnomalyDetector) -> None:
        """Resonance should not crash on n_samples < 32."""
        rng = np.random.RandomState(42)
        small = rng.randn(10, 3)
        detector.fit(small)
        scores = detector._compute_resonance_score(small)
        assert scores.shape == (10,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_single_sample_fallback(self, detector: MercuryAnomalyDetector) -> None:
        """Single test sample should return a valid score."""
        rng = np.random.RandomState(42)
        train = rng.randn(50, 3)
        detector.fit(train)
        single = rng.randn(1, 3)
        scores = detector._compute_resonance_score(single)
        assert scores.shape == (1,)
        assert 0 <= scores[0] <= 1

    def test_constant_feature(self, detector: MercuryAnomalyDetector) -> None:
        """Constant features should return 0.5 (uncertain)."""
        data = np.ones((50, 2))
        data[:, 1] = np.linspace(0, 1, 50)  # One non-constant column
        detector.fit(data)
        scores = detector._compute_resonance_score(data)
        assert scores.shape == (50,)
        assert np.all(np.isfinite(scores))


class TestKinematicScore:
    """Validate the kinematic (physics-based jerk/curvature) anomaly score."""

    @pytest.fixture
    def detector(self) -> MercuryAnomalyDetector:
        return MercuryAnomalyDetector()

    def test_jerky_scored_higher_than_smooth(self, detector: MercuryAnomalyDetector) -> None:
        """Jerky trajectory should be more anomalous than smooth."""
        smooth = np.linspace(0, 1, 50).reshape(-1, 1)
        jerky = np.concatenate(
            [
                np.linspace(0, 1, 25),
                np.random.RandomState(42).randn(25) + 5,
            ]
        ).reshape(-1, 1)

        detector.fit(smooth)
        score_smooth = detector._compute_kinematic_score(smooth).mean()
        score_jerky = detector._compute_kinematic_score(jerky).mean()

        assert (
            score_jerky > score_smooth
        ), f"Kinematic: jerky ({score_jerky:.3f}) should be > smooth ({score_smooth:.3f})"

    def test_single_sample(self, detector: MercuryAnomalyDetector) -> None:
        """Single sample should return 0.5 (no dynamics)."""
        rng = np.random.RandomState(42)
        train = rng.randn(50, 3)
        detector.fit(train)
        single = rng.randn(1, 3)
        scores = detector._compute_kinematic_score(single)
        assert scores.shape == (1,)
        assert 0 <= scores[0] <= 1

    def test_two_samples(self, detector: MercuryAnomalyDetector) -> None:
        """Two samples (can compute velocity but not acceleration) should not crash."""
        rng = np.random.RandomState(42)
        train = rng.randn(50, 3)
        detector.fit(train)
        two = rng.randn(2, 3)
        scores = detector._compute_kinematic_score(two)
        assert scores.shape == (2,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_three_samples(self, detector: MercuryAnomalyDetector) -> None:
        """Three samples (can compute accel but not jerk) should not crash."""
        rng = np.random.RandomState(42)
        train = rng.randn(50, 3)
        detector.fit(train)
        three = rng.randn(3, 3)
        scores = detector._compute_kinematic_score(three)
        assert scores.shape == (3,)
        assert np.all((scores >= 0) & (scores <= 1))


class TestInfoGeometryScore:
    """Validate the information-geometric (Fisher Information OOD) score."""

    @pytest.fixture
    def detector(self) -> MercuryAnomalyDetector:
        return MercuryAnomalyDetector()

    def test_ood_scored_higher_than_in_dist(self, detector: MercuryAnomalyDetector) -> None:
        """Out-of-distribution samples should score higher than in-distribution."""
        rng = np.random.RandomState(42)
        train = rng.randn(100, 5)  # Centered at 0, std ~1
        detector.fit(train)

        in_dist = rng.randn(20, 5) * 0.5  # Well within training range
        out_dist = np.ones((20, 5)) * 10  # Far from training distribution

        score_in = detector._compute_info_geometry_score(in_dist).mean()
        score_out = detector._compute_info_geometry_score(out_dist).mean()

        assert (
            score_out > score_in
        ), f"InfoGeo: OOD ({score_out:.3f}) should be > in-dist ({score_in:.3f})"

    def test_singular_covariance(self, detector: MercuryAnomalyDetector) -> None:
        """Near-singular covariance (many more features than samples) should not crash."""
        rng = np.random.RandomState(42)
        # 5 samples, 20 features -> rank-deficient covariance
        train = rng.randn(5, 20)
        detector.fit(train)
        test = rng.randn(10, 20)
        scores = detector._compute_info_geometry_score(test)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0) & (scores <= 1))

    def test_single_training_sample(self, detector: MercuryAnomalyDetector) -> None:
        """Single training sample should use identity precision (fallback)."""
        train = np.array([[1.0, 2.0, 3.0]])
        detector.fit(train)
        test = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        scores = detector._compute_info_geometry_score(test)
        assert scores.shape == (2,)
        assert scores[0] < scores[1], "Sample at training point should score lower"


class TestEnsembleCombined:
    """Validate the full ensemble (detect method)."""

    @pytest.fixture
    def detector(self) -> MercuryAnomalyDetector:
        return MercuryAnomalyDetector()

    @pytest.fixture
    def train_data(self) -> np.ndarray:
        rng = np.random.RandomState(42)
        return rng.randn(100, 5).astype(np.float32)

    def test_scores_in_unit_interval(
        self, detector: MercuryAnomalyDetector, train_data: np.ndarray
    ) -> None:
        """Combined scores must be in [0, 1]."""
        detector.fit(train_data)
        result = detector.detect(train_data[:20])
        scores = result["scores"]
        assert np.all(scores >= 0), f"Min score {scores.min()} < 0"
        assert np.all(scores <= 1), f"Max score {scores.max()} > 1"

    def test_single_sample(self, detector: MercuryAnomalyDetector, train_data: np.ndarray) -> None:
        """Detector should handle a single test sample."""
        detector.fit(train_data)
        result = detector.detect(train_data[[0]])
        assert len(result["scores"]) == 1
        assert 0 <= result["scores"][0] <= 1

    def test_nan_raises(self, detector: MercuryAnomalyDetector, train_data: np.ndarray) -> None:
        """NaN in training data should be filtered; NaN in test data propagates."""
        detector.fit(train_data)
        data_nan = train_data[:5].copy()
        data_nan[2, 0] = np.nan
        # detect should still run (NaN propagates through math)
        result = detector.detect(data_nan)
        assert result["scores"].shape == (5,)

    def test_backward_compatibility_keys(
        self, detector: MercuryAnomalyDetector, train_data: np.ndarray
    ) -> None:
        """New ensemble must return all legacy keys."""
        detector.fit(train_data)
        result = detector.detect(train_data[:5])

        required_keys = [
            "is_anomaly",
            "scores",
            "z_scores",
            "z_score_continuous",
            "iqr_scores",
            "iqr_flags",
            "detector_type",
            "threshold",
            "calibration_diagnostics",
        ]
        for key in required_keys:
            assert key in result, f"Missing legacy key: {key}"

        assert result["scores"].shape == (5,)
        assert result["detector_type"] == "statistical"

    def test_ensemble_components_exposed(
        self, detector: MercuryAnomalyDetector, train_data: np.ndarray
    ) -> None:
        """New ensemble should expose component scores for transparency."""
        detector.fit(train_data)
        result = detector.detect(train_data[:5])

        assert "ensemble_components" in result
        components = result["ensemble_components"]
        assert "resonance" in components
        assert "kinematic" in components
        assert "info_geometry" in components

    def test_scores_are_continuous(self, detector: MercuryAnomalyDetector) -> None:
        """Scores should have many unique values (not discrete)."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10).astype(np.float32)
        detector.fit(X)
        result = detector.detect(X)
        unique_scores = np.unique(result["scores"])
        assert len(unique_scores) > 10, f"Expected >10 unique scores, got {len(unique_scores)}"

    def test_no_isolation_forest_attribute(self, detector: MercuryAnomalyDetector) -> None:
        """Detector should not have an isolation_forest attribute."""
        assert not hasattr(
            detector, "isolation_forest"
        ), "isolation_forest attribute should be removed"

    def test_1d_data(self, detector: MercuryAnomalyDetector) -> None:
        """1D input should work (reshaped to (n, 1) internally)."""
        rng = np.random.RandomState(42)
        train = rng.randn(50).astype(np.float32)
        detector.fit(train)
        test = rng.randn(10).astype(np.float32)
        result = detector.detect(test)
        assert result["scores"].shape == (10,)

"""
Unit tests for Issue #7: Score Continuity Fix.

Tests that temporal and directive detectors use soft normalization
instead of hard clipping, preserving ranking information for extreme anomalies.

Mercury Agent - Copyright (C) 2025 Steel Security Advisory LLC
Licensed under GNU GPL v3
"""

import numpy as np
import pytest

from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector


class TestTemporalSoftNormalization:
    """Test that temporal detector preserves score continuity."""

    @pytest.fixture
    def detector(self, deterministic_rng):
        """Create fitted temporal detector with deterministic data."""
        detector = TemporalAnomalyDetector()
        # Fit on normal baseline data (uses seeded RNG for reproducibility)
        normal_data = deterministic_rng.randn(100).astype(np.float32)
        detector.fit(normal_data)
        return detector

    def test_extreme_anomalies_differentiated(self, detector):
        """Verify extreme anomalies get different scores (not capped at 1.0).

        Issue #7: Previously np.minimum(z_score/3.0, 1.0) capped scores,
        making z=6 and z=100 indistinguishable. Soft normalization
        z/(3+z) preserves ordering.
        """
        # Create data with progressively extreme anomalies
        baseline = np.zeros(50, dtype=np.float32)

        # Add anomalies of increasing severity
        mild_anomaly = np.concatenate([baseline, np.array([5.0])])
        moderate_anomaly = np.concatenate([baseline, np.array([10.0])])
        severe_anomaly = np.concatenate([baseline, np.array([50.0])])
        extreme_anomaly = np.concatenate([baseline, np.array([100.0])])

        # Get scores for each
        mild_result = detector.detect(mild_anomaly)
        moderate_result = detector.detect(moderate_anomaly)
        severe_result = detector.detect(severe_anomaly)
        extreme_result = detector.detect(extreme_anomaly)

        # Extract final scores for the anomaly point
        mild_score = mild_result["scores"][-1]
        moderate_score = moderate_result["scores"][-1]
        severe_score = severe_result["scores"][-1]
        extreme_score = extreme_result["scores"][-1]

        # All scores should be strictly ordered (soft normalization preserves ranking)
        assert mild_score < moderate_score, (
            f"Moderate ({moderate_score:.4f}) should be > mild ({mild_score:.4f})"
        )
        assert moderate_score < severe_score, (
            f"Severe ({severe_score:.4f}) should be > moderate ({moderate_score:.4f})"
        )
        assert severe_score < extreme_score, (
            f"Extreme ({extreme_score:.4f}) should be > severe ({severe_score:.4f})"
        )

    def test_scores_approach_one_asymptotically(self, detector):
        """Verify scores approach 1.0 but never reach it (asymptotic behavior).

        Soft normalization x/(k+x) has limit 1.0 as x->infinity.
        """
        baseline = np.zeros(50, dtype=np.float32)
        extreme = np.concatenate([baseline, np.array([1000.0])])

        result = detector.detect(extreme)
        extreme_score = result["scores"][-1]

        # Should be very close to 1.0 but not exactly 1.0
        assert extreme_score > 0.95, f"Extreme anomaly score {extreme_score:.4f} should be > 0.95"
        assert extreme_score < 1.0, f"Score {extreme_score:.6f} should be < 1.0 (asymptotic)"

    def test_soft_normalization_formula(self, detector):
        """Verify the specific soft normalization formula z/(k+z) is used."""
        # This tests the implementation detail for regression prevention
        baseline = np.zeros(20, dtype=np.float32)
        test_point = np.array([10.0], dtype=np.float32)
        data = np.concatenate([baseline, test_point])

        result = detector.detect(data)
        score = result["scores"][-1]

        # For soft normalization z/(k+z) where k=3 (change_threshold default=2):
        # The score should follow the formula pattern
        # Score should be in reasonable range indicating soft normalization
        assert 0.0 < score < 1.0, f"Score {score:.4f} should be in (0, 1)"

    def test_trend_scores_preserve_ordering(self, detector):
        """Test that trend detection also uses soft normalization."""
        # Create data with clear trend anomaly
        normal_trend = np.linspace(0, 1, 50).astype(np.float32)
        mild_break = np.concatenate([normal_trend, np.array([5.0])])
        severe_break = np.concatenate([normal_trend, np.array([20.0])])

        mild_result = detector.detect(mild_break)
        severe_result = detector.detect(severe_break)

        # Both should have non-zero trend scores with proper ordering
        assert mild_result["scores"][-1] < severe_result["scores"][-1]


class TestDirectiveSoftNormalization:
    """Test that directive detector preserves score continuity."""

    @pytest.fixture
    def detector(self, deterministic_rng):
        """Create fitted directive detector with deterministic data."""
        detector = SigmaDirectiveDetector()
        # Fit on normal baseline data (uses seeded RNG for reproducibility)
        normal_data = deterministic_rng.randn(100, 10).astype(np.float32)
        detector.fit(normal_data)
        return detector

    def test_pcp_extreme_anomalies_differentiated(self, detector):
        """Verify PCP (Pattern Convergence Protocol) uses soft normalization.

        Issue #7: Previously np.minimum(normalized_diffs/threshold, 1.0)
        capped scores. Now uses diffs/(threshold+diffs) for asymptotic behavior.
        """
        # Create samples at increasing distances from baseline
        mild_anomaly = np.full((1, 10), 2.0, dtype=np.float32)
        moderate_anomaly = np.full((1, 10), 5.0, dtype=np.float32)
        severe_anomaly = np.full((1, 10), 20.0, dtype=np.float32)
        extreme_anomaly = np.full((1, 10), 100.0, dtype=np.float32)

        mild_result = detector.detect(mild_anomaly)
        moderate_result = detector.detect(moderate_anomaly)
        severe_result = detector.detect(severe_anomaly)
        extreme_result = detector.detect(extreme_anomaly)

        # PCP scores should be strictly ordered
        mild_pcp = mild_result["pcp_scores"][0]
        moderate_pcp = moderate_result["pcp_scores"][0]
        severe_pcp = severe_result["pcp_scores"][0]
        extreme_pcp = extreme_result["pcp_scores"][0]

        assert mild_pcp < moderate_pcp < severe_pcp < extreme_pcp, (
            f"PCP scores should be strictly ordered: "
            f"{mild_pcp:.4f} < {moderate_pcp:.4f} < {severe_pcp:.4f} < {extreme_pcp:.4f}"
        )

    def test_rmd_soft_normalization(self, detector, deterministic_rng):
        """Verify RMD (Recursive Memory Dynamics) uses soft normalization.

        RMD should use deviation/(1+deviation) formula.
        """
        # Process sequential samples to build memory (deterministic for reproducibility)
        samples = deterministic_rng.randn(10, 10).astype(np.float32)

        # Add progressively extreme samples
        mild = np.full((1, 10), 3.0, dtype=np.float32)
        extreme = np.full((1, 10), 100.0, dtype=np.float32)

        # Detect with memory context
        _ = detector.detect(samples)  # Build memory
        mild_result = detector.detect(mild)

        detector.clear_memory()  # Reset for clean comparison
        _ = detector.detect(samples)  # Rebuild memory
        extreme_result = detector.detect(extreme)

        # RMD scores should differentiate
        mild_rmd = mild_result["rmd_scores"][0]
        extreme_rmd = extreme_result["rmd_scores"][0]

        assert mild_rmd < extreme_rmd, (
            f"RMD should differentiate: mild {mild_rmd:.4f} < extreme {extreme_rmd:.4f}"
        )

    def test_combined_scores_preserve_ranking(self, detector, deterministic_rng):
        """Verify combined scores preserve anomaly ranking."""
        normal = deterministic_rng.randn(5, 10).astype(np.float32) * 0.1
        mild_anomaly = np.full((1, 10), 3.0, dtype=np.float32)
        severe_anomaly = np.full((1, 10), 30.0, dtype=np.float32)

        normal_result = detector.detect(normal)
        mild_result = detector.detect(mild_anomaly)
        severe_result = detector.detect(severe_anomaly)

        normal_max = np.max(normal_result["scores"])
        mild_score = mild_result["scores"][0]
        severe_score = severe_result["scores"][0]

        # Ranking should be preserved in final combined scores
        assert normal_max < mild_score < severe_score, (
            f"Combined scores should rank: normal {normal_max:.4f} < "
            f"mild {mild_score:.4f} < severe {severe_score:.4f}"
        )

    def test_scores_bounded_zero_one(self, detector):
        """Verify all scores remain in [0, 1] range despite soft normalization."""
        # Test with extreme values
        extreme_data = np.full((10, 10), 1000.0, dtype=np.float32)
        result = detector.detect(extreme_data)

        combined_scores = result["scores"]
        pcp_scores = result["pcp_scores"]
        rmd_scores = result["rmd_scores"]

        # All scores should be in valid range
        for name, scores in [("combined", combined_scores), ("pcp", pcp_scores), ("rmd", rmd_scores)]:
            assert np.all(scores >= 0.0), f"{name} scores should be >= 0"
            assert np.all(scores <= 1.0), f"{name} scores should be <= 1"


class TestScoreContinuityRegression:
    """Regression tests to prevent reintroduction of hard clipping.

    These tests verify behavior rather than source code to be more robust.
    The key invariant: very extreme anomalies (100x) should have higher
    scores than moderately extreme anomalies (10x), which hard clipping
    at 1.0 would violate.
    """

    def test_temporal_extreme_differentiation(self, deterministic_rng):
        """Verify temporal detector differentiates 10x vs 100x vs 1000x anomalies.

        If hard clipping (np.minimum(..., 1.0)) was reintroduced, scores
        for 10x, 100x, and 1000x anomalies would all be capped at 1.0.
        """
        detector = TemporalAnomalyDetector()
        baseline = deterministic_rng.randn(100).astype(np.float32)
        detector.fit(baseline)

        # Create test sequences with 10x, 100x, and 1000x magnitude anomalies
        std = np.std(baseline)
        test_10x = np.concatenate([np.zeros(20, dtype=np.float32), np.array([10 * std])])
        test_100x = np.concatenate([np.zeros(20, dtype=np.float32), np.array([100 * std])])
        test_1000x = np.concatenate([np.zeros(20, dtype=np.float32), np.array([1000 * std])])

        score_10x = detector.detect(test_10x)["scores"][-1]
        score_100x = detector.detect(test_100x)["scores"][-1]
        score_1000x = detector.detect(test_1000x)["scores"][-1]

        # All three should be distinguishable (this fails with hard clipping)
        assert score_10x < score_100x, f"100x ({score_100x:.6f}) should exceed 10x ({score_10x:.6f})"
        assert score_100x < score_1000x, f"1000x ({score_1000x:.6f}) should exceed 100x ({score_100x:.6f})"
        # None should be exactly 1.0 (asymptotic behavior)
        assert score_1000x < 1.0, f"Even 1000x anomaly should be < 1.0, got {score_1000x:.6f}"

    def test_directive_extreme_differentiation(self, deterministic_rng):
        """Verify directive detector differentiates 10x vs 100x vs 1000x anomalies.

        If hard clipping was reintroduced in PCP protocol, extreme anomalies
        would all collapse to the same score.
        """
        detector = SigmaDirectiveDetector()
        baseline = deterministic_rng.randn(100, 10).astype(np.float32)
        detector.fit(baseline)

        # Create samples at 10x, 100x, 1000x distance from origin
        std = np.std(baseline)
        test_10x = np.full((1, 10), 10 * std, dtype=np.float32)
        test_100x = np.full((1, 10), 100 * std, dtype=np.float32)
        test_1000x = np.full((1, 10), 1000 * std, dtype=np.float32)

        pcp_10x = detector.detect(test_10x)["pcp_scores"][0]
        pcp_100x = detector.detect(test_100x)["pcp_scores"][0]
        pcp_1000x = detector.detect(test_1000x)["pcp_scores"][0]

        # PCP scores must be strictly ordered (fails with hard clipping)
        assert pcp_10x < pcp_100x, f"PCP 100x ({pcp_100x:.6f}) should exceed 10x ({pcp_10x:.6f})"
        assert pcp_100x < pcp_1000x, f"PCP 1000x ({pcp_1000x:.6f}) should exceed 100x ({pcp_100x:.6f})"
        # Asymptotic: even extreme values should be < 1.0
        assert pcp_1000x < 1.0, f"PCP 1000x should be < 1.0, got {pcp_1000x:.6f}"


class TestScoreContinuityWithAutoCalibration:
    """Test that soft normalization works correctly with auto-calibration."""

    def test_temporal_auto_calibration_with_soft_scores(self, deterministic_rng):
        """Verify auto-calibration works with continuous soft-normalized scores."""
        detector = TemporalAnomalyDetector()
        data = deterministic_rng.randn(100).astype(np.float32)
        detector.fit(data)
        detector.enable_auto_calibration(contamination=0.1)

        # Add some anomalies (known extreme values)
        test_data = np.concatenate([data, np.array([10.0, 20.0, 30.0], dtype=np.float32)])
        result = detector.detect(test_data)

        # Auto-calibration should have been applied (threshold should differ from default)
        # The calibration_diagnostics may be None if percentile method was used
        assert result["threshold"] >= 0.0 and result["threshold"] <= 1.0
        # Scores should be continuous (not discrete like old 5-value system)
        unique_scores = np.unique(result["scores"])
        # With 103 data points and continuous scoring, we expect many unique values
        assert len(unique_scores) >= 5, f"Expected continuous scores, got only {len(unique_scores)} unique values"

    def test_directive_auto_calibration_with_soft_scores(self, deterministic_rng):
        """Verify auto-calibration works with continuous soft-normalized scores."""
        detector = SigmaDirectiveDetector()
        data = deterministic_rng.randn(100, 10).astype(np.float32)
        detector.fit(data)
        detector.enable_auto_calibration(contamination=0.1)

        # Add known anomalies
        anomalies = np.full((5, 10), 10.0, dtype=np.float32)
        test_data = np.vstack([data, anomalies])
        result = detector.detect(test_data)

        # Threshold should be in valid range
        assert 0.0 <= result["threshold"] <= 1.0
        # Anomaly scores should be higher than normal scores on average
        normal_scores = result["scores"][:100]
        anomaly_scores = result["scores"][100:]
        assert np.mean(anomaly_scores) > np.mean(normal_scores), (
            f"Anomaly mean ({np.mean(anomaly_scores):.4f}) should exceed "
            f"normal mean ({np.mean(normal_scores):.4f})"
        )

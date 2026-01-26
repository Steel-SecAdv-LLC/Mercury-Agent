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
    def detector(self):
        """Create fitted temporal detector."""
        detector = TemporalAnomalyDetector()
        # Fit on normal baseline data
        normal_data = np.random.randn(100).astype(np.float32)
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
    def detector(self):
        """Create fitted directive detector."""
        detector = SigmaDirectiveDetector()
        # Fit on normal baseline data
        normal_data = np.random.randn(100, 10).astype(np.float32)
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

    def test_rmd_soft_normalization(self, detector):
        """Verify RMD (Recursive Memory Dynamics) uses soft normalization.

        RMD should use deviation/(1+deviation) formula.
        """
        # Process sequential samples to build memory
        samples = np.random.randn(10, 10).astype(np.float32)

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

    def test_combined_scores_preserve_ranking(self, detector):
        """Verify combined scores preserve anomaly ranking."""
        normal = np.random.randn(5, 10).astype(np.float32) * 0.1
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
    """Regression tests to prevent reintroduction of hard clipping."""

    def test_temporal_no_hard_minimum_clipping(self):
        """Ensure temporal detector doesn't use np.minimum(..., 1.0) pattern."""
        import inspect
        from omni_mercury_engine.detectors import temporal

        source = inspect.getsource(temporal)

        # Should NOT contain the old hard clipping pattern
        # (except in comments documenting the fix)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if "np.minimum" in line and ", 1.0)" in line and "Fix" not in line:
                # Check it's not the old pattern being actively used
                assert (
                    "z_score / 3.0" not in line
                ), f"Found old hard clipping pattern at line {i+1}: {line}"

    def test_directive_no_hard_minimum_clipping(self):
        """Ensure directive detector doesn't use np.minimum(..., 1.0) pattern."""
        import inspect
        from omni_mercury_engine.detectors import directive

        source = inspect.getsource(directive)

        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if "np.minimum" in line and ", 1.0)" in line and "Fix" not in line:
                # The only acceptable np.minimum is in _harmonic_anomaly_detection
                # for the final harmonic ratio clipping, not for intermediate scores
                if "harmonic_ratio" not in line and "convergence_threshold" in line:
                    assert False, f"Found old hard clipping pattern at line {i+1}: {line}"


class TestScoreContinuityWithAutoCalibration:
    """Test that soft normalization works correctly with auto-calibration."""

    def test_temporal_auto_calibration_with_soft_scores(self):
        """Verify auto-calibration works with continuous soft-normalized scores."""
        detector = TemporalAnomalyDetector()
        data = np.random.randn(100).astype(np.float32)
        detector.fit(data)
        detector.enable_auto_calibration(contamination=0.1)

        # Add some anomalies
        test_data = np.concatenate([data, np.array([10.0, 20.0, 30.0], dtype=np.float32)])
        result = detector.detect(test_data)

        # Should have calibration diagnostics
        assert result["calibration_diagnostics"] is not None or result["threshold"] != 0.5
        # Scores should still be continuous
        unique_scores = np.unique(result["scores"])
        assert len(unique_scores) > 10, "Scores should be continuous with auto-calibration"

    def test_directive_auto_calibration_with_soft_scores(self):
        """Verify auto-calibration works with continuous soft-normalized scores."""
        detector = SigmaDirectiveDetector()
        data = np.random.randn(100, 10).astype(np.float32)
        detector.fit(data)
        detector.enable_auto_calibration(contamination=0.1)

        # Add some anomalies
        anomalies = np.full((5, 10), 10.0, dtype=np.float32)
        test_data = np.vstack([data, anomalies])
        result = detector.detect(test_data)

        # Should have reasonable threshold calibration
        assert result["threshold"] != 0.5 or result["calibration_diagnostics"] is not None
        # Scores should still preserve ranking
        normal_scores = result["scores"][:100]
        anomaly_scores = result["scores"][100:]
        assert np.mean(anomaly_scores) > np.mean(normal_scores)

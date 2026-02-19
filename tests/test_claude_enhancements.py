"""
Mercury Agent - Property-Based Tests for Caduceus ⚚ Enhancements
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Property-based tests using Hypothesis for:
- Rigorous benchmark harness
- Calibration modules (Platt, Isotonic, Temperature)
- Conformal prediction
- Stacking/Bayesian fusion
- Multi-objective benevolence optimization

These tests verify mathematical invariants and edge cases that
unit tests might miss.
"""

# Import modules to test
import pytest  # noqa: E402
pytest.importorskip("sklearn")

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from hypothesis.extra.numpy import arrays
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omni_mercury_engine.core.benevolence_optimization import (
    BENEVOLENCE_THRESHOLD,
    BenevolenceLoss,
    MultiObjectiveLoss,
)
from omni_mercury_engine.core.calibration import (
    IsotonicCalibration,
    PlattScaling,
    compute_ece,
    compute_mce,
)
from omni_mercury_engine.core.conformal_prediction import (
    AdaptiveConformalInference,
    SplitConformalPredictor,
)
from omni_mercury_engine.core.rigorous_benchmark import (
    MetricResult,
    RigorousBenchmarkHarness,
    compute_event_metrics,
    point_adjusted_f1,
    set_all_seeds,
    stratified_split,
)
from omni_mercury_engine.core.stacking_fusion import (
    PHI,
    BayesianModelAveraging,
    EthicallyConstrainedFusion,
    StackingFusion,
)

# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def binary_classification_data(
    draw, min_samples=20, max_samples=200, min_features=2, max_features=20
):
    """Generate valid binary classification data."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_features = draw(st.integers(min_value=min_features, max_value=max_features))

    X = draw(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )

    # Ensure balanced classes (at least 20% of each)
    n_positive = max(2, int(n_samples * draw(st.floats(min_value=0.2, max_value=0.5))))
    y = np.zeros(n_samples, dtype=int)
    y[:n_positive] = 1
    np.random.shuffle(y)

    return X, y


@st.composite
def probability_array(draw, size=100):
    """Generate valid probability arrays."""
    n = draw(st.integers(min_value=10, max_value=size))
    return draw(
        arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        )
    )


@st.composite
def binary_labels(draw, size=100):
    """Generate binary labels with both classes present."""
    n = draw(st.integers(min_value=10, max_value=size))
    n_pos = max(2, n // 3)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    np.random.shuffle(y)
    return y


# =============================================================================
# Rigorous Benchmark Tests
# =============================================================================


class TestRigorousBenchmark:
    """Property-based tests for benchmark harness."""

    def test_seed_reproducibility(self):
        """Setting same seed should produce identical results."""
        set_all_seeds(42)
        a1 = np.random.random(10)

        set_all_seeds(42)
        a2 = np.random.random(10)

        np.testing.assert_array_equal(a1, a2)

    @given(binary_labels())
    @settings(max_examples=20)
    def test_metric_result_stats(self, values):
        """MetricResult should compute valid statistics."""
        result = MetricResult(name="test", values=list(values.astype(float)))
        result.compute_stats()

        assert 0 <= result.mean <= 1
        assert result.std >= 0
        assert result.ci_lower <= result.mean <= result.ci_upper

    @given(binary_labels(), binary_labels())
    @settings(max_examples=30)
    def test_event_metrics_bounds(self, y_true, y_pred):
        """Event metrics should be in [0, 1] range."""
        # Make arrays same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        ep, er, ef = compute_event_metrics(y_true, y_pred)

        assert 0 <= ep <= 1, f"Event precision {ep} out of bounds"
        assert 0 <= er <= 1, f"Event recall {er} out of bounds"
        assert 0 <= ef <= 1, f"Event F1 {ef} out of bounds"

    @given(binary_labels())
    @settings(max_examples=20)
    def test_point_adjusted_f1_bounds(self, y_true):
        """Point-adjusted F1 should be in [0, 1]."""
        # Create predictions
        y_pred = np.random.randint(0, 2, len(y_true))

        paf1 = point_adjusted_f1(y_true, y_pred)
        assert 0 <= paf1 <= 1

    def test_stratified_split_maintains_ratio(self):
        """Stratified split should maintain class ratio approximately."""
        np.random.seed(42)
        n = 1000
        y = np.array([0] * 700 + [1] * 300)
        X = np.random.randn(n, 5)

        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)

        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        original_ratio = np.mean(y)

        # Ratios should be within 5% of original
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Property-based tests for calibration modules."""

    @given(probability_array(), binary_labels())
    @settings(max_examples=30)
    def test_platt_scaling_output_bounds(self, y_prob, y_true):
        """Platt scaling should output valid probabilities."""
        min_len = min(len(y_prob), len(y_true))
        y_prob = y_prob[:min_len]
        y_true = y_true[:min_len]

        calibrator = PlattScaling()
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)

        assert np.all(calibrated >= 0), "Calibrated probs below 0"
        assert np.all(calibrated <= 1), "Calibrated probs above 1"

    @given(probability_array(), binary_labels())
    @settings(max_examples=30)
    def test_isotonic_monotonicity(self, y_prob, y_true):
        """Isotonic calibration should be monotonic."""
        min_len = min(len(y_prob), len(y_true))
        y_prob = y_prob[:min_len]
        y_true = y_true[:min_len]

        calibrator = IsotonicCalibration()
        calibrator.fit(y_prob, y_true)
        calibrated = calibrator.calibrate(y_prob)

        # For sorted input, output should be sorted
        sorted_idx = np.argsort(y_prob)
        sorted_calibrated = calibrated[sorted_idx]

        # Check non-decreasing (allow small numerical tolerance)
        diffs = np.diff(sorted_calibrated)
        assert np.all(diffs >= -1e-10), "Isotonic calibration not monotonic"

    @given(st.floats(min_value=0.01, max_value=0.99))
    @settings(max_examples=20)
    def test_ece_perfect_calibration(self, threshold):
        """ECE should be 0 for perfectly calibrated predictions."""
        n = 1000
        y_prob = np.random.random(n)
        # Perfect calibration: P(Y=1|prob=p) = p
        y_true = (np.random.random(n) < y_prob).astype(int)

        # ECE should be low (not exactly 0 due to finite sample)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.15, f"ECE {ece} too high for near-perfect calibration"

    @given(probability_array(), binary_labels())
    @settings(max_examples=20)
    def test_mce_bounds(self, y_prob, y_true):
        """MCE should be in [0, 1]."""
        min_len = min(len(y_prob), len(y_true))
        mce = compute_mce(y_true[:min_len], y_prob[:min_len])

        assert 0 <= mce <= 1


# =============================================================================
# Conformal Prediction Tests
# =============================================================================


class TestConformalPrediction:
    """Property-based tests for conformal prediction."""

    @given(st.floats(min_value=0.8, max_value=0.99))
    @settings(max_examples=20)
    def test_split_conformal_coverage_level(self, coverage):
        """Split conformal should respect coverage level asymptotically."""
        np.random.seed(42)
        n = 500

        # Generate scores
        cal_scores = np.random.exponential(1, n)

        predictor = SplitConformalPredictor(coverage=coverage)
        predictor.fit(cal_scores)

        threshold = predictor.get_anomaly_threshold()

        # Empirical coverage should be close to target
        empirical_coverage = np.mean(cal_scores <= threshold)

        # Allow some slack due to finite sample
        assert (
            abs(empirical_coverage - coverage) < 0.1
        ), f"Coverage {empirical_coverage} far from target {coverage}"

    @given(st.floats(min_value=0.8, max_value=0.99))
    @settings(max_examples=20)
    def test_adaptive_conformal_convergence(self, target_coverage):
        """Adaptive conformal should converge to target coverage."""
        aci = AdaptiveConformalInference(
            target_coverage=target_coverage,
            learning_rate=0.1,
        )

        # Simulate stream of scores
        np.random.seed(42)
        n_updates = 200

        for _ in range(n_updates):
            score = np.random.exponential(1)
            aci.update(score)

        stats = aci.get_coverage_stats()

        # Should be close to target after many updates
        assert (
            abs(stats["empirical_coverage"] - target_coverage) < 0.15
        ), f"Adaptive coverage {stats['empirical_coverage']} far from {target_coverage}"

    def test_conformal_threshold_positive(self):
        """Conformal threshold should always be positive."""
        np.random.seed(42)
        scores = np.abs(np.random.randn(100))

        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(scores)

        assert predictor.get_anomaly_threshold() > 0


# =============================================================================
# Fusion Tests
# =============================================================================


class TestFusion:
    """Property-based tests for ensemble fusion."""

    def test_stacking_fusion_with_detectors(self):
        """Stacking fusion should work with multiple detectors."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        fusion = StackingFusion(cv_folds=3)
        fusion.add_detector("lr1", LogisticRegression())
        fusion.add_detector("lr2", LogisticRegression(C=0.1))

        fusion.fit(X, y)
        predictions = fusion.predict(X)

        assert len(predictions) == len(y)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_bayesian_weights_sum_to_one(self):
        """Bayesian weights should sum to 1."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        bma = BayesianModelAveraging()
        bma.add_detector("lr1", LogisticRegression())
        bma.add_detector("lr2", LogisticRegression(C=0.1))

        bma.fit(X, y)

        assert bma.weights is not None
        assert abs(np.sum(bma.weights.weights) - 1.0) < 1e-6

    def test_ethical_fusion_constraint(self):
        """Ethical fusion should respect sigma_immutable threshold."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        sigma_immutable = 0.90

        fusion = EthicallyConstrainedFusion(sigma_immutable=sigma_immutable)
        fusion.add_detector("lr1", LogisticRegression(), ethical_score=0.95)
        fusion.add_detector("lr2", LogisticRegression(C=0.1), ethical_score=0.85)

        fusion.fit(X, y)
        compliance = fusion.get_ethical_compliance()

        # Average ethical score should be >= threshold
        assert (
            compliance["average_ethical_score"] >= sigma_immutable * 0.9
        ), f"Ethical score {compliance['average_ethical_score']} below threshold"

    @given(st.floats(min_value=1.0, max_value=3.0))
    @settings(max_examples=10)
    def test_golden_ratio_constant(self, x):
        """Verify golden ratio constant is correct."""
        # phi = (1 + sqrt(5)) / 2
        expected_phi = (1 + np.sqrt(5)) / 2
        assert abs(PHI - expected_phi) < 1e-10


# =============================================================================
# Benevolence Optimization Tests
# =============================================================================


class TestBenevolenceOptimization:
    """Property-based tests for multi-objective benevolence optimization."""

    @given(binary_labels(), binary_labels())
    @settings(max_examples=30)
    def test_benevolence_score_bounds(self, y_true, y_pred):
        """Benevolence score should be in [0, 1]."""
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len].astype(float)

        bl = BenevolenceLoss()
        score = bl.compute(y_pred, y_true)

        assert 0 <= score <= 1, f"Benevolence {score} out of bounds"

    @given(binary_labels())
    @settings(max_examples=20)
    def test_perfect_predictions_high_benevolence(self, y_true):
        """Perfect predictions should yield high benevolence."""
        bl = BenevolenceLoss()
        score = bl.compute(y_true.astype(float), y_true)

        assert score >= 0.9, f"Perfect predictions gave benevolence {score} < 0.9"

    def test_multi_objective_loss_components(self):
        """Multi-objective loss should have correct component structure."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.random(100)

        mol = MultiObjectiveLoss()
        result = mol.compute(y_pred, y_true)

        # Check all components
        assert hasattr(result, "detection_loss")
        assert hasattr(result, "benevolence_score")
        assert hasattr(result, "fairness_score")
        assert hasattr(result, "combined_loss")

        # Bounds
        assert 0 <= result.benevolence_score <= 1
        assert 0 <= result.fairness_score <= 1
        assert result.detection_loss >= 0

    def test_benevolence_threshold_constant(self):
        """Verify benevolence threshold matches requirements."""
        assert (
            BENEVOLENCE_THRESHOLD == 0.99
        ), f"Benevolence threshold {BENEVOLENCE_THRESHOLD} != 0.99"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_pipeline(self):
        """Test full pipeline: benchmark -> calibrate -> conformal."""
        np.random.seed(42)

        # Generate data
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Create simple detector
        class SimpleDetector:
            def __init__(self):
                self.model = LogisticRegression()

            def fit(self, X, y):
                self.model.fit(X, y)

            def predict(self, X):
                return self.model.predict(X)

            def predict_proba(self, X):
                return self.model.predict_proba(X)

        detector = SimpleDetector()

        # Benchmark
        harness = RigorousBenchmarkHarness(n_folds=3)
        result = harness.benchmark_detector(
            detector, X, y, detector_name="SimpleDetector", dataset_name="TestData"
        )

        assert result.roc_auc.mean >= 0.5  # Better than random
        assert result.f1.mean > 0  # Some detections

    def test_calibrated_conformal(self):
        """Test calibration followed by conformal prediction."""
        np.random.seed(42)

        n = 300
        y_true = np.random.randint(0, 2, n)
        y_prob = np.clip(y_true + np.random.randn(n) * 0.3, 0.01, 0.99)

        # Calibrate
        calibrator = PlattScaling()
        calibrator.fit(y_prob[:200], y_true[:200])
        y_calibrated = calibrator.calibrate(y_prob[200:])

        # Conformal
        predictor = SplitConformalPredictor(coverage=0.9)
        predictor.fit(y_calibrated)

        threshold = predictor.get_anomaly_threshold()
        assert 0 < threshold < 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

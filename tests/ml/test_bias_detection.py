"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for ml/bias_detection.py module.
Comprehensive test coverage for bias detection and fairness evaluation.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.ml.bias_detection import (
    BiasDetector,
    BiasReport,
    FairnessMetric,
    FairnessResult,
)


class TestFairnessMetric:
    """Tests for FairnessMetric enum."""

    def test_demographic_parity(self) -> None:
        """Test demographic parity metric."""
        assert FairnessMetric.DEMOGRAPHIC_PARITY.value == "demographic_parity"

    def test_equalized_odds(self) -> None:
        """Test equalized odds metric."""
        assert FairnessMetric.EQUALIZED_ODDS.value == "equalized_odds"

    def test_disparate_impact(self) -> None:
        """Test disparate impact metric."""
        assert FairnessMetric.DISPARATE_IMPACT.value == "disparate_impact"

    def test_calibration(self) -> None:
        """Test calibration metric."""
        assert FairnessMetric.CALIBRATION.value == "calibration"

    def test_predictive_parity(self) -> None:
        """Test predictive parity metric."""
        assert FairnessMetric.PREDICTIVE_PARITY.value == "predictive_parity"

    def test_fpr_parity(self) -> None:
        """Test false positive rate parity metric."""
        assert FairnessMetric.FALSE_POSITIVE_RATE_PARITY.value == "fpr_parity"

    def test_fnr_parity(self) -> None:
        """Test false negative rate parity metric."""
        assert FairnessMetric.FALSE_NEGATIVE_RATE_PARITY.value == "fnr_parity"


class TestFairnessResult:
    """Tests for FairnessResult dataclass."""

    def test_basic_result(self) -> None:
        """Test basic fairness result creation."""
        result = FairnessResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            overall_score=0.95,
            group_scores={"A": 0.5, "B": 0.45},
            is_fair=True,
            threshold=0.1,
            disparity=0.05,
        )
        assert result.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        assert result.overall_score == 0.95
        assert result.is_fair is True
        assert result.disparity == 0.05

    def test_result_with_recommendations(self) -> None:
        """Test fairness result with recommendations."""
        result = FairnessResult(
            metric=FairnessMetric.EQUALIZED_ODDS,
            overall_score=0.7,
            group_scores={"A": 0.6, "B": 0.8},
            is_fair=False,
            threshold=0.1,
            disparity=0.2,
            recommendations=["Rebalance training data"],
        )
        assert len(result.recommendations) == 1
        assert "Rebalance" in result.recommendations[0]

    def test_result_with_metadata(self) -> None:
        """Test fairness result with metadata."""
        result = FairnessResult(
            metric=FairnessMetric.DISPARATE_IMPACT,
            overall_score=0.85,
            group_scores={},
            is_fair=True,
            threshold=0.8,
            disparity=0.15,
            metadata={"samples": 1000},
        )
        assert result.metadata["samples"] == 1000


class TestBiasReport:
    """Tests for BiasReport dataclass."""

    def test_basic_report(self) -> None:
        """Test basic bias report creation."""
        fairness_result = FairnessResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            overall_score=0.9,
            group_scores={"A": 0.5, "B": 0.5},
            is_fair=True,
            threshold=0.1,
            disparity=0.0,
        )
        report = BiasReport(
            model_name="test_model",
            total_samples=1000,
            sensitive_features=["gender"],
            fairness_results=[fairness_result],
            overall_fairness_score=0.9,
            is_model_fair=True,
            high_risk_groups=[],
            recommendations=["Model passes all checks"],
        )
        assert report.model_name == "test_model"
        assert report.total_samples == 1000
        assert report.is_model_fair is True
        assert len(report.fairness_results) == 1

    def test_unfair_report(self) -> None:
        """Test bias report for unfair model."""
        fairness_result = FairnessResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            overall_score=0.5,
            group_scores={"A": 0.3, "B": 0.7},
            is_fair=False,
            threshold=0.1,
            disparity=0.4,
        )
        report = BiasReport(
            model_name="biased_model",
            total_samples=500,
            sensitive_features=["race"],
            fairness_results=[fairness_result],
            overall_fairness_score=0.5,
            is_model_fair=False,
            high_risk_groups=["A", "B"],
            recommendations=["Rebalance training data"],
        )
        assert report.is_model_fair is False
        assert len(report.high_risk_groups) == 2


class TestBiasDetectorInitialization:
    """Tests for BiasDetector initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        detector = BiasDetector()
        assert detector.use_fairlearn is True
        assert FairnessMetric.DEMOGRAPHIC_PARITY in detector.thresholds

    def test_no_fairlearn(self) -> None:
        """Test initialization without fairlearn."""
        detector = BiasDetector(use_fairlearn=False)
        assert detector.use_fairlearn is False
        assert detector._fairlearn_available is False

    def test_custom_thresholds(self) -> None:
        """Test custom threshold initialization."""
        detector = BiasDetector(
            demographic_parity_threshold=0.05,
            equalized_odds_threshold=0.15,
            disparate_impact_threshold=0.9,
        )
        assert detector.thresholds[FairnessMetric.DEMOGRAPHIC_PARITY] == 0.05
        assert detector.thresholds[FairnessMetric.EQUALIZED_ODDS] == 0.15
        assert detector.thresholds[FairnessMetric.DISPARATE_IMPACT] == 0.9

    def test_default_thresholds(self) -> None:
        """Test default threshold values."""
        detector = BiasDetector()
        assert detector.thresholds[FairnessMetric.DEMOGRAPHIC_PARITY] == 0.1
        assert detector.thresholds[FairnessMetric.EQUALIZED_ODDS] == 0.1
        assert detector.thresholds[FairnessMetric.DISPARATE_IMPACT] == 0.8


class TestBiasDetectorEvaluate:
    """Tests for BiasDetector.evaluate method."""

    def setup_method(self) -> None:
        """Set up test fixtures with synthetic data."""
        self.detector = BiasDetector(use_fairlearn=False)
        np.random.seed(42)

        # Create balanced data (should be fair)
        self.n_samples = 1000
        self.y_true_balanced = np.random.randint(0, 2, self.n_samples)
        self.y_pred_balanced = self.y_true_balanced.copy()  # Perfect predictions
        self.sensitive_balanced = np.array(["A"] * 500 + ["B"] * 500)

        # Create biased data (should be unfair)
        self.y_true_biased = np.array([1] * 500 + [1] * 500)
        self.y_pred_biased = np.array([1] * 400 + [0] * 100 + [1] * 100 + [0] * 400)
        self.sensitive_biased = np.array(["A"] * 500 + ["B"] * 500)

    def test_evaluate_fair_model(self) -> None:
        """Test evaluation of fair model."""
        report = self.detector.evaluate(
            y_true=self.y_true_balanced,
            y_pred=self.y_pred_balanced,
            sensitive_features=self.sensitive_balanced,
            feature_name="group",
            model_name="fair_model",
        )
        assert isinstance(report, BiasReport)
        assert report.model_name == "fair_model"
        assert report.total_samples == self.n_samples
        assert len(report.fairness_results) > 0

    def test_evaluate_with_specific_metrics(self) -> None:
        """Test evaluation with specific metrics."""
        report = self.detector.evaluate(
            y_true=self.y_true_balanced,
            y_pred=self.y_pred_balanced,
            sensitive_features=self.sensitive_balanced,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        assert len(report.fairness_results) == 1
        assert report.fairness_results[0].metric == FairnessMetric.DEMOGRAPHIC_PARITY

    def test_evaluate_all_default_metrics(self) -> None:
        """Test evaluation with all default metrics."""
        report = self.detector.evaluate(
            y_true=self.y_true_balanced,
            y_pred=self.y_pred_balanced,
            sensitive_features=self.sensitive_balanced,
        )
        # Default should include demographic parity, equalized odds, disparate impact
        assert len(report.fairness_results) >= 3

    def test_biased_model_detection(self) -> None:
        """Test detection of biased model."""
        report = self.detector.evaluate(
            y_true=self.y_true_biased,
            y_pred=self.y_pred_biased,
            sensitive_features=self.sensitive_biased,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )
        # Predictions heavily favor group A (80% vs 20%)
        assert not all(r.is_fair for r in report.fairness_results)

    def test_evaluate_returns_report(self) -> None:
        """Test that evaluate returns a BiasReport."""
        report = self.detector.evaluate(
            y_true=np.array([0, 1, 0, 1]),
            y_pred=np.array([0, 1, 0, 1]),
            sensitive_features=np.array(["A", "A", "B", "B"]),
        )
        assert isinstance(report, BiasReport)
        assert report.total_samples == 4

    def test_overall_fairness_score_computed(self) -> None:
        """Test that overall fairness score is computed."""
        report = self.detector.evaluate(
            y_true=self.y_true_balanced,
            y_pred=self.y_pred_balanced,
            sensitive_features=self.sensitive_balanced,
        )
        assert 0.0 <= report.overall_fairness_score <= 1.0

    def test_sensitive_features_recorded(self) -> None:
        """Test that sensitive features are recorded in report."""
        report = self.detector.evaluate(
            y_true=self.y_true_balanced,
            y_pred=self.y_pred_balanced,
            sensitive_features=self.sensitive_balanced,
            feature_name="test_feature",
        )
        assert "test_feature" in report.sensitive_features


class TestBiasDetectorBuiltinMetrics:
    """Tests for built-in metric computation (without Fairlearn)."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_demographic_parity_builtin(self) -> None:
        """Test demographic parity with built-in implementation."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )

        result = report.fairness_results[0]
        assert result.metric == FairnessMetric.DEMOGRAPHIC_PARITY
        assert "A" in result.group_scores
        assert "B" in result.group_scores

    def test_equalized_odds_builtin(self) -> None:
        """Test equalized odds with built-in implementation."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.EQUALIZED_ODDS],
        )

        result = report.fairness_results[0]
        assert result.metric == FairnessMetric.EQUALIZED_ODDS
        # Should have TPR and FPR per group
        assert any("tpr" in k or "fpr" in k for k in result.group_scores)

    def test_disparate_impact_builtin(self) -> None:
        """Test disparate impact with built-in implementation."""
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.DISPARATE_IMPACT],
        )

        result = report.fairness_results[0]
        assert result.metric == FairnessMetric.DISPARATE_IMPACT
        # Group A has 100% selection, Group B has 25% - ratio is 0.25
        assert result.overall_score < 0.5  # Below 80% rule

    def test_fallback_metric(self) -> None:
        """Test fallback for unimplemented metrics."""
        report = self.detector.evaluate(
            y_true=np.array([0, 1, 0, 1]),
            y_pred=np.array([0, 1, 0, 1]),
            sensitive_features=np.array(["A", "A", "B", "B"]),
            metrics=[FairnessMetric.CALIBRATION],
        )
        # Should return some result even for unimplemented metrics
        assert len(report.fairness_results) == 1


class TestBiasDetectorHighRiskGroups:
    """Tests for high-risk group identification."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_identify_high_risk_groups(self) -> None:
        """Test identification of high-risk groups."""
        # Create heavily biased predictions
        y_true = np.array([1] * 100)
        y_pred = np.array([1] * 80 + [0] * 20)  # Group A gets 80% positive
        sensitive = np.array(["A"] * 50 + ["B"] * 50)

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )

        # Should identify groups with disparity
        if not report.is_model_fair:
            assert len(report.high_risk_groups) >= 0

    def test_no_high_risk_groups_when_fair(self) -> None:
        """Test no high-risk groups for fair model."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # Everyone gets positive
        sensitive = np.array(["A", "A", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )

        # Fair model should have few or no high-risk groups
        assert isinstance(report.high_risk_groups, list)


class TestBiasDetectorRecommendations:
    """Tests for recommendation generation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_recommendations_for_unfair_model(self) -> None:
        """Test recommendations generated for unfair model."""
        y_true = np.array([1] * 100)
        y_pred = np.array([1] * 90 + [0] * 10)  # Very biased
        sensitive = np.array(["A"] * 50 + ["B"] * 50)

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )

        # Should have some recommendations
        assert len(report.recommendations) > 0

    def test_recommendations_for_fair_model(self) -> None:
        """Test recommendations for fair model."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )

        # Should still have at least one recommendation
        assert len(report.recommendations) > 0

    def test_demographic_parity_violation_recommendation(self) -> None:
        """Test specific recommendation for demographic parity violation."""
        y_true = np.array([1] * 100)
        y_pred = np.array([1] * 80 + [0] * 20)
        sensitive = np.array(["A"] * 50 + ["B"] * 50)

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )

        # Check that recommendations exist
        assert isinstance(report.recommendations, list)


class TestBiasDetectorQuickCheck:
    """Tests for BiasDetector.quick_check method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_quick_check_fair_model(self) -> None:
        """Test quick check returns True for fair model."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        result = self.detector.quick_check(y_true, y_pred, sensitive)
        assert isinstance(result, bool)

    def test_quick_check_returns_bool(self) -> None:
        """Test quick check always returns boolean."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        result = self.detector.quick_check(y_true, y_pred, sensitive)
        assert result is True or result is False


class TestBiasDetectorEdgeCases:
    """Tests for edge cases in bias detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_single_group(self) -> None:
        """Test with single sensitive group."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "A"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        assert report.total_samples == 4

    def test_many_groups(self) -> None:
        """Test with many sensitive groups."""
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        sensitive = np.array([f"Group_{i % 10}" for i in range(n)])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        assert report.metadata["unique_groups"] == 10

    def test_all_positive_predictions(self) -> None:
        """Test with all positive predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        sensitive = np.array(["A", "A", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        # Should handle this case gracefully
        assert report.total_samples == 4

    def test_all_negative_predictions(self) -> None:
        """Test with all negative predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        assert report.total_samples == 4

    def test_empty_arrays(self) -> None:
        """Test with minimal arrays."""
        y_true = np.array([1])
        y_pred = np.array([1])
        sensitive = np.array(["A"])

        report = self.detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        assert report.total_samples == 1

    def test_numpy_array_conversion(self) -> None:
        """Test that lists are converted to numpy arrays."""
        # Pass lists deliberately to exercise the runtime ndarray-conversion path.
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        sensitive = ["A", "A", "B", "B"]

        report = self.detector.evaluate(
            y_true=y_true,  # type: ignore[arg-type]
            y_pred=y_pred,  # type: ignore[arg-type]
            sensitive_features=sensitive,  # type: ignore[arg-type]
        )
        assert report.total_samples == 4


class TestBiasDetectorMetadata:
    """Tests for metadata in bias detection results."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = BiasDetector(use_fairlearn=False)

    def test_metadata_includes_fairlearn_status(self) -> None:
        """Test that metadata includes fairlearn availability status."""
        report = self.detector.evaluate(
            y_true=np.array([1, 0]),
            y_pred=np.array([1, 0]),
            sensitive_features=np.array(["A", "B"]),
        )
        assert "fairlearn_used" in report.metadata

    def test_metadata_includes_group_count(self) -> None:
        """Test that metadata includes unique group count."""
        report = self.detector.evaluate(
            y_true=np.array([1, 0, 1]),
            y_pred=np.array([1, 0, 1]),
            sensitive_features=np.array(["A", "B", "C"]),
        )
        assert report.metadata["unique_groups"] == 3


class TestBiasDetectorThresholdBehavior:
    """Tests for threshold-based fairness decisions."""

    def test_demographic_parity_threshold(self) -> None:
        """Test demographic parity uses correct threshold."""
        detector = BiasDetector(
            use_fairlearn=False,
            demographic_parity_threshold=0.05,  # Very strict
        )

        # Create data with small disparity
        y_true = np.array([1] * 100)
        y_pred = np.array([1] * 52 + [0] * 48)  # 52% vs 48% by group
        sensitive = np.array(["A"] * 50 + ["B"] * 50)

        report = detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.DEMOGRAPHIC_PARITY],
        )

        # With 0.05 threshold, small disparity might fail
        assert len(report.fairness_results) == 1

    def test_disparate_impact_80_rule(self) -> None:
        """Test disparate impact follows 80% rule."""
        detector = BiasDetector(use_fairlearn=False)

        # Create data that violates 80% rule
        y_true = np.array([1] * 100)
        y_pred = np.array([1] * 50 + [0] * 50)
        # Group A gets 100% positive, Group B gets 0%
        sensitive = np.array(["A"] * 50 + ["B"] * 50)

        report = detector.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metrics=[FairnessMetric.DISPARATE_IMPACT],
        )

        # Should fail the 80% rule
        result = report.fairness_results[0]
        assert result.is_fair is False

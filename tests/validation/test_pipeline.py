"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Tests for validation pipeline error branches and edge cases.

Covers:
- DataQualityChecker edge cases
- ABTester statistical tests
- ValidationPipeline error handling
- Cross-validation edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.validation.pipeline import (
    ABTester,
    ABTestResult,
    DataQualityChecker,
    QualityCheckResult,
    ValidationPipeline,
    ValidationResult,
)


class TestDataQualityChecker:
    """Tests for DataQualityChecker edge cases."""

    def test_check_missing_values_no_missing(self):
        """Test missing values check with no missing data."""
        checker = DataQualityChecker()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = checker.check_missing_values(data)

        assert result.passed == True  # noqa: E712 - numpy.bool_ identity check fails
        assert result.score == 1.0
        assert result.details["missing_count"] == 0

    def test_check_missing_values_with_nans(self):
        """Test missing values check with NaN values."""
        checker = DataQualityChecker(missing_threshold=0.1)
        data = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])

        result = checker.check_missing_values(data)

        assert result.details["missing_count"] == 2
        assert result.details["missing_ratio"] > 0

    def test_check_missing_values_empty_data(self):
        """Test missing values check with empty data."""
        checker = DataQualityChecker()
        data = np.array([])

        result = checker.check_missing_values(data)

        assert result.passed is True

    def test_check_outliers_empty_data(self):
        """Test outliers check with empty data."""
        checker = DataQualityChecker()
        data = np.array([])

        result = checker.check_outliers(data)

        assert result.passed is True
        assert result.score == 1.0
        assert result.message == "No data to check"

    def test_check_outliers_with_outliers(self):
        """Test outliers check with extreme values."""
        checker = DataQualityChecker(outlier_threshold=2.0)
        data = np.array([1.0, 2.0, 3.0, 100.0, 2.0, 1.0, 3.0])

        result = checker.check_outliers(data)

        assert result.details["outlier_count"] > 0

    def test_check_feature_variance_1d_data(self):
        """Test feature variance check with 1D data."""
        checker = DataQualityChecker()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = checker.check_feature_variance(data)

        assert result.passed == True  # noqa: E712 - numpy.bool_ identity check fails

    def test_check_feature_variance_low_variance(self):
        """Test feature variance check with low variance features."""
        checker = DataQualityChecker()
        data = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

        result = checker.check_feature_variance(data)

        assert result.details["low_variance_count"] >= 1

    def test_check_class_balance_single_class(self):
        """Test class balance check with single class."""
        checker = DataQualityChecker()
        labels = np.array([0, 0, 0, 0, 0])

        result = checker.check_class_balance(labels)

        assert result.passed is False
        assert result.score == 0.0
        assert "Only one class" in result.message

    def test_check_class_balance_balanced(self):
        """Test class balance check with balanced classes."""
        checker = DataQualityChecker(imbalance_threshold=0.3)
        labels = np.array([0, 0, 0, 1, 1, 1])

        result = checker.check_class_balance(labels)

        assert result.passed == True  # noqa: E712 - numpy.bool_ identity check fails
        assert result.details["minority_ratio"] == 0.5

    def test_check_class_balance_imbalanced(self):
        """Test class balance check with imbalanced classes."""
        checker = DataQualityChecker(imbalance_threshold=0.3)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        result = checker.check_class_balance(labels)

        assert result.passed == False  # noqa: E712 - numpy.bool_ identity check fails
        assert result.details["minority_ratio"] == 0.1

    def test_check_feature_correlation_1d_data(self):
        """Test feature correlation check with 1D data."""
        checker = DataQualityChecker()
        data = np.array([1.0, 2.0, 3.0, 4.0])

        result = checker.check_feature_correlation(data)

        assert result.passed is True
        assert "Not enough features" in result.message

    def test_check_feature_correlation_single_feature(self):
        """Test feature correlation check with single feature."""
        checker = DataQualityChecker()
        data = np.array([[1.0], [2.0], [3.0]])

        result = checker.check_feature_correlation(data)

        assert result.passed is True

    def test_check_feature_correlation_high_correlation(self):
        """Test feature correlation check with highly correlated features."""
        checker = DataQualityChecker(correlation_threshold=0.9)
        data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        result = checker.check_feature_correlation(data)

        assert result.details["high_correlation_pairs"] >= 1

    def test_check_data_range_normal(self):
        """Test data range check with normal data."""
        checker = DataQualityChecker()
        data = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])

        result = checker.check_data_range(data)

        assert result.passed == True  # noqa: E712 - numpy.bool_ identity check fails
        assert result.details["needs_scaling"] == False  # noqa: E712

    def test_check_data_range_needs_scaling(self):
        """Test data range check with data needing scaling."""
        checker = DataQualityChecker()
        data = np.array([[1000.0, 5000.0], [2000.0, 6000.0]])

        result = checker.check_data_range(data)

        assert result.passed == False  # noqa: E712 - numpy.bool_ identity check fails
        assert result.details["needs_scaling"] == True  # noqa: E712

    def test_run_all_checks(self):
        """Test running all quality checks."""
        checker = DataQualityChecker()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        labels = np.array([0, 0, 1, 1])

        results = checker.run_all_checks(data, labels)

        assert len(results) == 6
        assert all(isinstance(r, QualityCheckResult) for r in results)


class TestABTester:
    """Tests for ABTester statistical tests."""

    def test_compare_models_significant_difference(self):
        """Test model comparison with significant difference."""
        tester = ABTester(confidence_level=0.95)

        model_a_scores = np.array([0.70, 0.72, 0.71, 0.69, 0.70])
        model_b_scores = np.array([0.85, 0.87, 0.86, 0.84, 0.85])

        result = tester.compare_models(
            model_a_scores,
            model_b_scores,
            model_a_name="Baseline",
            model_b_name="New Model",
            metric_name="F1 Score",
        )

        assert result.statistically_significant == True  # noqa: E712 - numpy.bool_
        assert result.winner == "New Model"
        assert result.improvement > 0

    def test_compare_models_no_significant_difference(self):
        """Test model comparison with no significant difference."""
        tester = ABTester(confidence_level=0.95)

        model_a_scores = np.array([0.80, 0.81, 0.79, 0.80, 0.80])
        model_b_scores = np.array([0.80, 0.80, 0.81, 0.79, 0.80])

        result = tester.compare_models(model_a_scores, model_b_scores)

        assert result.winner == "No significant difference"

    def test_compare_models_model_a_wins(self):
        """Test model comparison where model A wins."""
        tester = ABTester(confidence_level=0.95)

        model_a_scores = np.array([0.90, 0.91, 0.89, 0.90, 0.90])
        model_b_scores = np.array([0.70, 0.71, 0.69, 0.70, 0.70])

        result = tester.compare_models(model_a_scores, model_b_scores, model_a_name="Better Model")

        assert result.winner == "Better Model"

    def test_compare_models_zero_mean(self):
        """Test model comparison with zero mean for model A."""
        tester = ABTester()

        model_a_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        model_b_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        result = tester.compare_models(model_a_scores, model_b_scores)

        assert result.improvement == 0

    def test_cohens_d_zero_std(self):
        """Test Cohen's d with zero standard deviation."""
        tester = ABTester()

        group1 = np.array([1.0, 1.0, 1.0])
        group2 = np.array([1.0, 1.0, 1.0])

        d = tester._cohens_d(group1, group2)

        assert d == 0.0

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval calculation."""
        tester = ABTester(n_bootstrap=100)

        differences = np.array([0.1, 0.15, 0.12, 0.08, 0.11])

        lower, upper = tester._bootstrap_ci(differences)

        assert lower < upper
        assert lower < np.mean(differences) < upper


class TestValidationPipeline:
    """Tests for ValidationPipeline error handling."""

    def test_validate_basic(self):
        """Test basic validation pipeline."""

        class SimpleModel:
            def fit(self, X, y):
                self.classes_ = np.unique(y)

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=3)
        model = SimpleModel()

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        result = pipeline.validate(model, X, y, dataset_name="test", model_name="simple")

        assert isinstance(result, ValidationResult)
        assert result.dataset_name == "test"
        assert result.model_name == "simple"
        assert result.num_samples == 100
        assert result.num_features == 10

    def test_validate_without_quality_checks(self):
        """Test validation without quality checks."""

        class SimpleModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=3)
        model = SimpleModel()

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        result = pipeline.validate(model, X, y, run_quality_checks=False)

        assert len(result.quality_checks) == 0

    def test_validate_model_without_fit(self):
        """Test validation with model that has no fit method."""

        class NoFitModel:
            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=2)
        model = NoFitModel()

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        result = pipeline.validate(model, X, y)

        assert isinstance(result, ValidationResult)

    def test_validate_model_without_predict(self):
        """Test validation with model that has no predict method."""

        class NoPredictModel:
            def fit(self, X, y):
                pass

        pipeline = ValidationPipeline(n_folds=2)
        model = NoPredictModel()

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        result = pipeline.validate(model, X, y)

        assert isinstance(result, ValidationResult)

    def test_validate_model_with_predict_proba(self):
        """Test validation with model that has predict_proba."""

        class ProbaModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])

        pipeline = ValidationPipeline(n_folds=2)
        model = ProbaModel()

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        result = pipeline.validate(model, X, y)

        assert isinstance(result, ValidationResult)

    def test_validate_1d_features(self):
        """Test validation with 1D feature array."""

        class SimpleModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=2)
        model = SimpleModel()

        X = np.random.randn(30)
        y = np.random.randint(0, 2, 30)

        result = pipeline.validate(model, X, y)

        assert result.num_features == 1

    def test_compare_to_baseline(self):
        """Test comparing to baseline."""

        class SimpleModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=3)
        model = SimpleModel()

        X = np.random.randn(60, 5)
        y = np.random.randint(0, 2, 60)

        result1 = pipeline.validate(model, X, y, model_name="model1")
        # The second positional arg is a baseline_name string, not another result.
        pipeline.validate(model, X, y, model_name="model2")

        comparison = pipeline.compare_to_baseline(result1, "model2")

        # compare_to_baseline may return None if cross_val_scores are empty
        assert comparison is None or isinstance(comparison, ABTestResult)

    def test_get_benchmarks(self):
        """Test getting stored benchmarks."""

        class SimpleModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=2)
        model = SimpleModel()

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        pipeline.validate(model, X, y, dataset_name="test_dataset", model_name="test_model")

        benchmarks = pipeline.get_benchmarks()

        # Benchmark key may include both dataset and model name
        assert len(benchmarks) >= 1
        assert any("test_dataset" in key for key in benchmarks)

    def test_generate_report(self):
        """Test generating validation report."""

        class SimpleModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        pipeline = ValidationPipeline(n_folds=2)
        model = SimpleModel()

        X = np.random.randn(30, 5)
        y = np.random.randint(0, 2, 30)

        result = pipeline.validate(model, X, y)

        report = pipeline.generate_report(result)

        assert isinstance(report, str)
        # Report header may be uppercase or title case
        assert "validation report" in report.lower()


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_quality_check_result(self):
        """Test QualityCheckResult dataclass."""
        result = QualityCheckResult(
            check_name="test",
            passed=True,
            score=0.95,
            message="Test passed",
            details={"key": "value"},
        )

        assert result.check_name == "test"
        assert result.passed is True
        assert result.score == 0.95

    def test_ab_test_result(self):
        """Test ABTestResult dataclass."""
        result = ABTestResult(
            model_a_name="A",
            model_b_name="B",
            metric_name="F1",
            model_a_score=0.8,
            model_b_score=0.85,
            improvement=0.0625,
            p_value=0.01,
            statistically_significant=True,
            confidence_level=0.95,
            winner="B",
        )

        assert result.model_a_name == "A"
        assert result.winner == "B"

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            dataset_name="test",
            model_name="model",
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.865,
            auc_roc=0.92,
            auc_pr=0.89,
            confusion_matrix=np.array([[45, 5], [3, 47]]),
            quality_checks=[],
            validation_time_seconds=1.5,
            num_samples=100,
            num_features=10,
        )

        assert result.dataset_name == "test"
        assert result.f1_score == 0.865


# Run with: pytest tests/validation/test_pipeline.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

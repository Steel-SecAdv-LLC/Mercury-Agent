# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Validation Pipeline and Data Loaders."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from omni_mercury_engine.validation.data_loaders import (
    DatasetMetadata,
    MIMICLoader,
    NSLKDDLoader,
    USGSEarthquakeLoader,
)
from omni_mercury_engine.validation.pipeline import (
    ABTester,
    ABTestResult,
    DataQualityChecker,
    QualityCheckResult,
    ValidationPipeline,
    ValidationResult,
)


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test default values of metadata dataclass."""
        metadata = DatasetMetadata(
            name="test",
            source="synthetic",
            num_samples=1000,
            num_features=10,
            num_anomalies=100,
            anomaly_ratio=0.1,
        )
        assert metadata.name == "test"
        assert metadata.num_samples == 1000
        assert metadata.anomaly_ratio == 0.1


class TestNSLKDDLoader:
    """Tests for NSL-KDD dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create NSLKDDLoader instance."""
        return NSLKDDLoader()

    def test_initialization(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader is not None
        assert hasattr(loader, "load")

    def test_load_synthetic_data(self, loader: Any) -> None:
        """Test loading synthetic data."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=1000)

        assert data is not None
        assert labels is not None
        assert metadata is not None
        assert data.shape[0] == 1000
        assert len(labels) == 1000

    def test_synthetic_data_shape(self, loader: Any) -> None:
        """Test shape of synthetic data."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=500)

        assert data.shape[0] == 500
        assert len(labels) == 500
        assert data.shape[1] == len(loader.FEATURE_NAMES)

    def test_feature_names(self, loader: Any) -> None:
        """Test that feature names are provided."""
        assert len(loader.FEATURE_NAMES) > 0
        assert "duration" in loader.FEATURE_NAMES

    def test_label_distribution(self, loader: Any) -> None:
        """Test label distribution in synthetic data."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=1000)

        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2

    def test_metadata_populated(self, loader: Any) -> None:
        """Test that metadata is properly populated."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=1000)

        assert metadata.name == "NSL-KDD"
        assert metadata.num_samples == 1000
        assert metadata.anomaly_ratio > 0

    def test_train_test_split(self, loader: Any) -> None:
        """Test train/test split functionality."""
        loader.load(use_synthetic=True, n_samples=1000)
        X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.2)

        assert len(X_train) == 800
        assert len(X_test) == 200
        assert len(y_train) == 800
        assert len(y_test) == 200


class TestUSGSEarthquakeLoader:
    """Tests for USGS Earthquake dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create USGSEarthquakeLoader instance."""
        return USGSEarthquakeLoader()

    def test_initialization(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader is not None
        assert hasattr(loader, "load")

    def test_load_synthetic_data(self, loader: Any) -> None:
        """Test loading synthetic earthquake data."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=500)

        assert data is not None
        assert labels is not None
        assert metadata is not None

    def test_synthetic_earthquake_features(self, loader: Any) -> None:
        """Test synthetic earthquake feature generation."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=500)

        assert data.shape[0] == 500
        assert data.shape[1] == len(loader.FEATURE_NAMES)

    def test_magnitude_in_features(self, loader: Any) -> None:
        """Test that magnitude is first feature."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=100)

        magnitudes = data[:, 0]
        assert np.all(magnitudes >= 0)
        assert np.all(magnitudes <= 10)

    def test_metadata_populated(self, loader: Any) -> None:
        """Test that metadata is properly populated."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=500)

        assert metadata.name == "USGS Earthquake"
        assert metadata.num_samples == 500


class TestMIMICLoader:
    """Tests for MIMIC-III dataset loader."""

    @pytest.fixture
    def loader(self):
        """Create MIMICLoader instance."""
        return MIMICLoader()

    def test_initialization(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader is not None
        assert hasattr(loader, "load")

    def test_load_synthetic_data(self, loader: Any) -> None:
        """Test loading synthetic medical data."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=200)

        assert data is not None
        assert labels is not None
        assert metadata is not None

    def test_synthetic_medical_features(self, loader: Any) -> None:
        """Test synthetic medical feature generation."""
        data, labels, metadata = loader.load(use_synthetic=True, n_samples=200)

        assert data.shape[0] == 200

    def test_irb_status_check(self, loader: Any) -> None:
        """Test IRB status check method."""
        status = loader.check_irb_status()
        assert status is not None
        assert "using_synthetic" in status
        assert "can_access_real_data" in status


class TestQualityCheckResult:
    """Tests for QualityCheckResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = QualityCheckResult(
            check_name="test_check",
            passed=True,
            score=0.95,
            message="Test passed",
        )
        assert result.check_name == "test_check"
        assert result.passed is True
        assert result.score == 0.95


class TestDataQualityChecker:
    """Tests for DataQualityChecker."""

    @pytest.fixture
    def checker(self):
        """Create DataQualityChecker instance."""
        return DataQualityChecker()

    @pytest.fixture
    def clean_data(self):
        """Create clean dataset."""
        return np.random.randn(100, 10)

    @pytest.fixture
    def clean_labels(self):
        """Create clean labels."""
        return np.random.randint(0, 2, 100)

    @pytest.fixture
    def dirty_data(self):
        """Create dataset with quality issues."""
        data = np.random.randn(100, 10)
        data[0:10, 0] = np.nan
        data[50:55, 1] = 1000.0
        return data

    def test_initialization(self, checker: Any) -> None:
        """Test checker initialization."""
        assert checker is not None
        assert hasattr(checker, "run_all_checks")

    def test_run_all_checks(self, checker: Any, clean_data: Any, clean_labels: Any) -> None:
        """Test running all quality checks."""
        results = checker.run_all_checks(clean_data, clean_labels)

        assert len(results) == 6
        assert all(isinstance(r, QualityCheckResult) for r in results)

    def test_check_missing_values_clean(self, checker: Any, clean_data: Any) -> None:
        """Test missing value check on clean data."""
        result = checker.check_missing_values(clean_data)

        assert result.passed == True  # noqa: E712 - numpy.bool_ identity check fails
        assert result.score == 1.0

    def test_check_missing_values_dirty(self, checker: Any, dirty_data: Any) -> None:
        """Test missing value check on dirty data."""
        result = checker.check_missing_values(dirty_data)

        assert result.details["missing_count"] > 0

    def test_check_outliers(self, checker: Any, clean_data: Any) -> None:
        """Test outlier detection."""
        result = checker.check_outliers(clean_data)

        assert isinstance(result, QualityCheckResult)
        assert result.check_name == "outliers"

    def test_check_feature_variance(self, checker: Any, clean_data: Any) -> None:
        """Test feature variance check."""
        result = checker.check_feature_variance(clean_data)

        assert isinstance(result, QualityCheckResult)
        assert result.check_name == "feature_variance"

    def test_check_class_balance(self, checker: Any, clean_labels: Any) -> None:
        """Test class balance check."""
        result = checker.check_class_balance(clean_labels)

        assert isinstance(result, QualityCheckResult)
        assert result.check_name == "class_balance"

    def test_check_feature_correlation(self, checker: Any, clean_data: Any) -> None:
        """Test feature correlation check."""
        result = checker.check_feature_correlation(clean_data)

        assert isinstance(result, QualityCheckResult)
        assert result.check_name == "feature_correlation"

    def test_check_data_range(self, checker: Any, clean_data: Any) -> None:
        """Test data range check."""
        result = checker.check_data_range(clean_data)

        assert isinstance(result, QualityCheckResult)
        assert result.check_name == "data_range"


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = ABTestResult(
            model_a_name="Model A",
            model_b_name="Model B",
            metric_name="F1 Score",
            model_a_score=0.85,
            model_b_score=0.88,
            improvement=0.035,
            p_value=0.03,
            statistically_significant=True,
            confidence_level=0.95,
            winner="Model B",
        )
        assert result.model_a_name == "Model A"
        assert result.winner == "Model B"


class TestABTester:
    """Tests for ABTester."""

    @pytest.fixture
    def tester(self):
        """Create ABTester instance."""
        return ABTester()

    def test_initialization(self, tester: Any) -> None:
        """Test tester initialization."""
        assert tester is not None
        assert hasattr(tester, "compare_models")

    def test_compare_models(self, tester: Any) -> None:
        """Test model comparison."""
        model_a_scores = np.array([0.80, 0.82, 0.81, 0.83, 0.79])
        model_b_scores = np.array([0.85, 0.87, 0.86, 0.88, 0.84])

        result = tester.compare_models(model_a_scores, model_b_scores)

        assert isinstance(result, ABTestResult)
        assert result.model_b_score > result.model_a_score

    def test_compare_equal_models(self, tester: Any) -> None:
        """Test comparison of equal models."""
        scores = np.array([0.85, 0.85, 0.85, 0.85, 0.85])

        result = tester.compare_models(scores, scores)

        assert isinstance(result, ABTestResult)
        assert result.improvement == 0.0

    def test_statistical_significance(self, tester: Any) -> None:
        """Test statistical significance calculation."""
        model_a_scores = np.array([0.80, 0.82, 0.81, 0.83, 0.79])
        model_b_scores = np.array([0.85, 0.87, 0.86, 0.88, 0.84])

        result = tester.compare_models(model_a_scores, model_b_scores)

        assert "p_value" in result.__dict__
        assert "statistically_significant" in result.__dict__


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values of result dataclass."""
        result = ValidationResult(
            dataset_name="test",
            model_name="test_model",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_roc=0.90,
            auc_pr=0.88,
            confusion_matrix=np.array([[80, 20], [10, 90]]),
            quality_checks=[],
            validation_time_seconds=1.5,
            num_samples=200,
            num_features=10,
        )
        assert result.dataset_name == "test"
        assert result.f1_score == 0.85


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create ValidationPipeline instance."""
        return ValidationPipeline()

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for validation."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def sample_model(self):
        """Create mock model for validation."""
        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))
        return model

    def test_initialization(self, pipeline: Any) -> None:
        """Test pipeline initialization."""
        assert pipeline is not None
        assert hasattr(pipeline, "validate")

    def test_validate(self, pipeline: Any, sample_data: Any, sample_model: Any) -> None:
        """Test validation."""
        X, y = sample_data
        result = pipeline.validate(sample_model, X, y)

        assert isinstance(result, ValidationResult)
        assert result.num_samples == 100
        assert result.num_features == 10

    def test_validate_with_quality_checks(
        self, pipeline: Any, sample_data: Any, sample_model: Any
    ) -> None:
        """Test validation with quality checks."""
        X, y = sample_data
        result = pipeline.validate(sample_model, X, y, run_quality_checks=True)

        assert len(result.quality_checks) > 0

    def test_validate_without_quality_checks(
        self, pipeline: Any, sample_data: Any, sample_model: Any
    ) -> None:
        """Test validation without quality checks."""
        X, y = sample_data
        result = pipeline.validate(sample_model, X, y, run_quality_checks=False)

        assert len(result.quality_checks) == 0

    def test_cross_validation_scores(
        self, pipeline: Any, sample_data: Any, sample_model: Any
    ) -> None:
        """Test cross-validation scores are computed."""
        X, y = sample_data
        result = pipeline.validate(sample_model, X, y)

        assert len(result.cross_val_scores) == pipeline.n_folds


class TestValidationPipelineIntegration:
    """Integration tests for validation pipeline."""

    def test_full_pipeline_flow(self) -> None:
        """Test full validation pipeline flow."""
        pipeline = ValidationPipeline()
        nsl_loader = NSLKDDLoader()

        data, labels, metadata = nsl_loader.load(use_synthetic=True, n_samples=500)

        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

        result = pipeline.validate(model, data, labels)

        assert isinstance(result, ValidationResult)

    def test_multi_dataset_validation(self) -> None:
        """Test validation across multiple datasets."""
        pipeline = ValidationPipeline()

        loaders = [
            NSLKDDLoader(),
            USGSEarthquakeLoader(),
            MIMICLoader(),
        ]

        results = []
        for loader in loaders:
            data, labels, metadata = loader.load(use_synthetic=True, n_samples=100)

            model = Mock()
            model.fit = Mock(return_value=model)
            model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
            model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

            result = pipeline.validate(model, data, labels)
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_quality_check_integration(self) -> None:
        """Test data quality check integration."""
        pipeline = ValidationPipeline()
        checker = DataQualityChecker()

        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)

        quality_results = checker.run_all_checks(data, labels)

        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

        validation_result = pipeline.validate(model, data, labels)

        assert len(quality_results) > 0
        assert validation_result is not None


class TestValidationEdgeCases:
    """Edge case tests for validation pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create ValidationPipeline instance."""
        return ValidationPipeline()

    def test_single_sample(self, pipeline: Any) -> None:
        """Test validation with minimal samples."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

        result = pipeline.validate(model, X, y)
        assert isinstance(result, ValidationResult)

    def test_high_dimensional_data(self, pipeline: Any) -> None:
        """Test validation with high-dimensional data."""
        X = np.random.randn(100, 100)
        y = np.random.randint(0, 2, 100)

        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

        result = pipeline.validate(model, X, y)
        assert isinstance(result, ValidationResult)
        assert result.num_features == 100

    def test_imbalanced_labels(self, pipeline: Any) -> None:
        """Test validation with imbalanced labels."""
        X = np.random.randn(100, 10)
        y = np.array([0] * 95 + [1] * 5)

        model = Mock()
        model.fit = Mock(return_value=model)
        model.predict = Mock(side_effect=lambda X: np.random.randint(0, 2, len(X)))
        model.predict_proba = Mock(side_effect=lambda X: np.random.rand(len(X), 2))

        result = pipeline.validate(model, X, y)
        assert isinstance(result, ValidationResult)

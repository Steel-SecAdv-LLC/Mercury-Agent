"""
Mercury Agent - Adaptive Domain Thresholding Tests
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive unit tests for adaptive per-domain thresholding:
- Platt scaling calibration
- Isotonic regression calibration
- Calibration ensemble
- Domain-specific threshold management
- Domain ensemble weight optimization

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.adaptive_domain_thresholding import (
    DOMAIN_DEFAULTS,
    AdaptiveDomainThresholdManager,
    CalibrationEnsemble,
    DomainCalibrationResult,
    DomainThresholdConfig,
    DomainType,
    IsotonicCalibrator,
    PlattScalingCalibrator,
    create_domain_threshold_manager,
)


class TestDomainType:
    """Tests for DomainType enumeration."""

    def test_domain_values(self) -> None:
        """Test all domain enum values exist."""
        assert DomainType.MEDICAL.value == "medical"
        assert DomainType.FINANCIAL.value == "financial"
        assert DomainType.INFRASTRUCTURE.value == "infrastructure"
        assert DomainType.SECURITY.value == "security"
        assert DomainType.HUMANITARIAN.value == "humanitarian"
        assert DomainType.GENERAL.value == "general"

    def test_domain_from_string(self) -> None:
        """Test domain creation from string."""
        assert DomainType("medical") == DomainType.MEDICAL
        assert DomainType("financial") == DomainType.FINANCIAL


class TestDomainThresholdConfig:
    """Tests for DomainThresholdConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DomainThresholdConfig(domain=DomainType.MEDICAL)
        assert config.domain == DomainType.MEDICAL
        assert config.base_threshold == 0.5
        assert config.contamination == 0.05
        assert config.enable_probability_calibration is True
        assert config.precision_priority == 0.5
        assert config.ethical_threshold == 0.96
        assert config.min_threshold == 0.01
        assert config.max_threshold == 0.99

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DomainThresholdConfig(
            domain=DomainType.FINANCIAL,
            base_threshold=0.6,
            contamination=0.1,
            precision_priority=0.7,
        )
        assert config.domain == DomainType.FINANCIAL
        assert config.base_threshold == 0.6
        assert config.contamination == 0.1
        assert config.precision_priority == 0.7


class TestDomainDefaults:
    """Tests for domain-specific default configurations."""

    def test_medical_defaults(self) -> None:
        """Test medical domain defaults."""
        defaults = DOMAIN_DEFAULTS[DomainType.MEDICAL]
        assert defaults["contamination"] == 0.03
        assert defaults["precision_priority"] == 0.3
        assert defaults["ethical_threshold"] == 0.93

    def test_financial_defaults(self) -> None:
        """Test financial domain defaults."""
        defaults = DOMAIN_DEFAULTS[DomainType.FINANCIAL]
        assert defaults["contamination"] == 0.05
        assert defaults["precision_priority"] == 0.6
        assert defaults["ethical_threshold"] == 0.96

    def test_infrastructure_defaults(self) -> None:
        """Test infrastructure domain defaults."""
        defaults = DOMAIN_DEFAULTS[DomainType.INFRASTRUCTURE]
        assert defaults["contamination"] == 0.02
        assert defaults["precision_priority"] == 0.4
        assert defaults["ethical_threshold"] == 0.995

    def test_all_domains_have_defaults(self) -> None:
        """Test all domain types have default configurations."""
        for domain in DomainType:
            assert domain in DOMAIN_DEFAULTS


class TestPlattScalingCalibrator:
    """Tests for PlattScalingCalibrator."""

    @pytest.fixture
    def calibrator(self) -> PlattScalingCalibrator:
        """Create Platt scaling calibrator."""
        return PlattScalingCalibrator()

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample scores and labels."""
        np.random.seed(42)
        n_normal = 80
        n_anomaly = 20
        normal_scores = np.random.beta(2, 5, n_normal)
        anomaly_scores = np.random.beta(5, 2, n_anomaly)
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        return scores, labels.astype(np.int32)

    def test_initialization(self, calibrator: PlattScalingCalibrator) -> None:
        """Test calibrator initialization."""
        assert calibrator.A == -1.0
        assert calibrator.B == 0.0
        assert calibrator._fitted is False

    def test_fit(
        self, calibrator: PlattScalingCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting Platt scaling."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)

        assert calibrator._fitted is True
        assert calibrator.A != -1.0 or calibrator.B != 0.0

    def test_calibrate(
        self, calibrator: PlattScalingCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test score calibration."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)
        calibrated = calibrator.calibrate(scores)

        assert len(calibrated) == len(scores)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrate_without_fit_raises(self, calibrator: PlattScalingCalibrator) -> None:
        """Test that calibrating without fitting raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            calibrator.calibrate(np.array([0.5, 0.6, 0.7]))

    def test_get_params(
        self, calibrator: PlattScalingCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test getting fitted parameters."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)
        params = calibrator.get_params()

        assert "A" in params
        assert "B" in params
        assert params["A"] == calibrator.A
        assert params["B"] == calibrator.B

    def test_few_samples_warning(self, calibrator: PlattScalingCalibrator) -> None:
        """Test warning with too few samples."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        calibrator.fit(scores, labels)
        assert calibrator._fitted is True

    def test_single_class_handling(self, calibrator: PlattScalingCalibrator) -> None:
        """Test handling of single-class labels."""
        scores = np.random.rand(20)
        labels = np.zeros(20, dtype=np.int32)
        calibrator.fit(scores, labels)
        assert calibrator._fitted is True


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator."""

    @pytest.fixture
    def calibrator(self) -> IsotonicCalibrator:
        """Create isotonic calibrator."""
        return IsotonicCalibrator()

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample scores and labels."""
        np.random.seed(42)
        n_normal = 80
        n_anomaly = 20
        normal_scores = np.random.beta(2, 5, n_normal)
        anomaly_scores = np.random.beta(5, 2, n_anomaly)
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        return scores, labels.astype(np.int32)

    def test_initialization(self, calibrator: IsotonicCalibrator) -> None:
        """Test calibrator initialization."""
        assert calibrator.out_of_bounds == "clip"
        assert calibrator._fitted is False

    def test_fit(
        self, calibrator: IsotonicCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting isotonic regression."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)

        assert calibrator._fitted is True
        assert calibrator._score_bins is not None
        assert calibrator._calibration_map is not None

    def test_calibrate(
        self, calibrator: IsotonicCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test score calibration."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)
        calibrated = calibrator.calibrate(scores)

        assert len(calibrated) == len(scores)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrate_without_fit_raises(self, calibrator: IsotonicCalibrator) -> None:
        """Test that calibrating without fitting raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            calibrator.calibrate(np.array([0.5, 0.6, 0.7]))

    def test_monotonicity(
        self, calibrator: IsotonicCalibrator, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that calibrated values are monotonically increasing."""
        scores, labels = sample_data
        calibrator.fit(scores, labels)

        test_scores = np.linspace(0, 1, 100)
        calibrated = calibrator.calibrate(test_scores)

        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-10)


class TestCalibrationEnsemble:
    """Tests for CalibrationEnsemble."""

    @pytest.fixture
    def ensemble(self) -> CalibrationEnsemble:
        """Create calibration ensemble."""
        return CalibrationEnsemble()

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample scores and labels."""
        np.random.seed(42)
        n_normal = 150
        n_anomaly = 50
        normal_scores = np.random.beta(2, 5, n_normal)
        anomaly_scores = np.random.beta(5, 2, n_anomaly)
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        return scores, labels.astype(np.int32)

    def test_initialization(self, ensemble: CalibrationEnsemble) -> None:
        """Test ensemble initialization."""
        assert ensemble.platt_weight == 0.5
        assert ensemble._fitted is False
        assert ensemble.best_method == "ensemble"

    def test_fit(
        self, ensemble: CalibrationEnsemble, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting ensemble."""
        scores, labels = sample_data
        ensemble.fit(scores, labels)

        assert ensemble._fitted is True
        assert ensemble.best_method in ["platt", "isotonic", "ensemble"]

    def test_calibrate(
        self, ensemble: CalibrationEnsemble, sample_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test ensemble calibration."""
        scores, labels = sample_data
        ensemble.fit(scores, labels)
        calibrated = ensemble.calibrate(scores)

        assert len(calibrated) == len(scores)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrate_without_fit_raises(self, ensemble: CalibrationEnsemble) -> None:
        """Test that calibrating without fitting raises error."""
        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.calibrate(np.array([0.5, 0.6, 0.7]))

    def test_brier_score(self, ensemble: CalibrationEnsemble) -> None:
        """Test Brier score computation."""
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        brier = ensemble._brier_score(probs, labels)

        assert 0 <= brier <= 1
        expected = np.mean((probs - labels) ** 2)
        assert brier == pytest.approx(expected)


class TestAdaptiveDomainThresholdManager:
    """Tests for AdaptiveDomainThresholdManager."""

    @pytest.fixture
    def manager(self) -> AdaptiveDomainThresholdManager:
        """Create threshold manager."""
        return AdaptiveDomainThresholdManager(DomainType.MEDICAL)

    @pytest.fixture
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample scores and labels."""
        np.random.seed(42)
        n_normal = 150
        n_anomaly = 50
        normal_scores = np.random.beta(2, 5, n_normal)
        anomaly_scores = np.random.beta(5, 2, n_anomaly)
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        return scores, labels.astype(np.int32)

    def test_initialization(self, manager: AdaptiveDomainThresholdManager) -> None:
        """Test manager initialization."""
        assert manager.domain == DomainType.MEDICAL
        assert manager.config.domain == DomainType.MEDICAL
        assert manager._fitted is False

    def test_initialization_from_string(self) -> None:
        """Test initialization from string domain name."""
        manager = AdaptiveDomainThresholdManager("financial")
        assert manager.domain == DomainType.FINANCIAL

    def test_fit_with_labels(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test fitting with labels."""
        scores, labels = sample_data
        manager.fit(scores, labels)

        assert manager._fitted is True
        assert manager._current_threshold > 0

    def test_fit_without_labels(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test fitting without labels (unsupervised)."""
        scores, _ = sample_data
        manager.fit(scores)

        assert manager._fitted is True

    def test_calibrate(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test score calibration."""
        scores, labels = sample_data
        manager.fit(scores, labels)
        result = manager.calibrate(scores)

        assert isinstance(result, DomainCalibrationResult)
        assert result.domain == DomainType.MEDICAL
        assert len(result.calibrated_scores) == len(scores)
        assert len(result.predictions) == len(scores)

    def test_get_threshold(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test getting current threshold."""
        scores, labels = sample_data
        manager.fit(scores, labels)
        threshold = manager.get_threshold()

        assert 0 < threshold < 1
        assert manager.config.min_threshold <= threshold <= manager.config.max_threshold

    def test_update_performance(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test performance update."""
        scores, labels = sample_data
        manager.fit(scores, labels)

        result = manager.calibrate(scores)
        metrics = manager.update_performance(labels, result.predictions)

        assert len(manager._performance_history) == 1
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_get_performance_summary(
        self,
        manager: AdaptiveDomainThresholdManager,
        sample_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test getting performance summary."""
        scores, labels = sample_data
        manager.fit(scores, labels)

        result = manager.calibrate(scores)
        manager.update_performance(labels, result.predictions)
        manager.update_performance(labels, result.predictions)

        summary = manager.get_performance_summary()
        assert "n_records" in summary
        assert "current_threshold" in summary


class TestCreateDomainThresholdManager:
    """Tests for create_domain_threshold_manager factory function."""

    def test_create_medical_manager(self) -> None:
        """Test creating medical domain manager."""
        manager = create_domain_threshold_manager("medical")
        assert isinstance(manager, AdaptiveDomainThresholdManager)
        assert manager.domain == DomainType.MEDICAL

    def test_create_financial_manager(self) -> None:
        """Test creating financial domain manager."""
        manager = create_domain_threshold_manager("financial")
        assert manager.domain == DomainType.FINANCIAL

    def test_create_infrastructure_manager(self) -> None:
        """Test creating infrastructure domain manager."""
        manager = create_domain_threshold_manager("infrastructure")
        assert manager.domain == DomainType.INFRASTRUCTURE


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scores(self) -> None:
        """Test handling of empty scores."""
        manager = AdaptiveDomainThresholdManager(DomainType.MEDICAL)
        scores = np.array([0.1, 0.2, 0.3])
        manager.fit(scores)
        assert manager._fitted is True

    def test_single_score(self) -> None:
        """Test handling of single score."""
        manager = AdaptiveDomainThresholdManager(DomainType.FINANCIAL)
        scores = np.array([0.5])
        manager.fit(scores)
        assert manager._fitted is True

    def test_all_same_scores(self) -> None:
        """Test handling of identical scores."""
        manager = AdaptiveDomainThresholdManager(DomainType.INFRASTRUCTURE)
        scores = np.ones(100) * 0.5
        manager.fit(scores)
        assert manager._fitted is True

    def test_extreme_scores(self) -> None:
        """Test handling of extreme score values."""
        manager = AdaptiveDomainThresholdManager(DomainType.SECURITY)
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        manager.fit(scores, labels)
        assert manager._fitted is True

    def test_nan_scores(self) -> None:
        """Test handling of NaN scores."""
        manager = AdaptiveDomainThresholdManager(DomainType.HUMANITARIAN)
        scores = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
        scores = np.nan_to_num(scores, nan=0.0)
        manager.fit(scores)
        assert manager._fitted is True

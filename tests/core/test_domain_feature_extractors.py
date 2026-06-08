# Copyright (C) 2025 Steel Security Advisors LLC
"""Domain Feature Extractors Tests."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.domain_feature_extractors import (
    BaseDomainExtractor,
    Domain,
    DomainFeatureConfig,
    DomainFeatureExtractorFactory,
    DomainFeatureResult,
    FinancialFeatureExtractor,
    InfrastructureFeatureExtractor,
    MedicalFeatureExtractor,
    extract_financial_features,
    extract_infrastructure_features,
    extract_medical_features,
)


class TestDomainEnum:
    """Tests for Domain enumeration."""

    def test_domain_values(self) -> None:
        """Test all domain enum values exist."""
        assert Domain.MEDICAL.value == "medical"
        assert Domain.FINANCIAL.value == "financial"
        assert Domain.INFRASTRUCTURE.value == "infrastructure"
        assert Domain.SECURITY.value == "security"
        assert Domain.GENERAL.value == "general"

    def test_domain_from_string(self) -> None:
        """Test domain creation from string."""
        assert Domain("medical") == Domain.MEDICAL
        assert Domain("financial") == Domain.FINANCIAL
        assert Domain("infrastructure") == Domain.INFRASTRUCTURE


class TestDomainFeatureConfig:
    """Tests for DomainFeatureConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DomainFeatureConfig(domain=Domain.MEDICAL)
        assert config.domain == Domain.MEDICAL
        assert config.window_size == 60
        assert config.sampling_rate == 1.0
        assert config.enable_temporal is True
        assert config.enable_statistical is True
        assert config.enable_domain_specific is True
        assert config.contamination_estimate == 0.05

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DomainFeatureConfig(
            domain=Domain.FINANCIAL,
            window_size=120,
            sampling_rate=0.5,
            enable_temporal=False,
            contamination_estimate=0.1,
        )
        assert config.domain == Domain.FINANCIAL
        assert config.window_size == 120
        assert config.sampling_rate == 0.5
        assert config.enable_temporal is False
        assert config.contamination_estimate == 0.1


class TestDomainFeatureResult:
    """Tests for DomainFeatureResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result dataclass creation."""
        features = np.array([1.0, 2.0, 3.0])
        result = DomainFeatureResult(
            features=features,
            feature_names=["f1", "f2", "f3"],
            domain=Domain.MEDICAL,
        )
        assert np.array_equal(result.features, features)
        assert result.feature_names == ["f1", "f2", "f3"]
        assert result.domain == Domain.MEDICAL
        assert result.confidence == 1.0
        assert result.extraction_time_ms == 0.0


class TestMedicalFeatureExtractor:
    """Tests for MedicalFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> MedicalFeatureExtractor:
        """Create medical feature extractor."""
        return MedicalFeatureExtractor()

    @pytest.fixture
    def vital_signs_1d(self) -> np.ndarray:
        """Generate 1D vital sign data (heart rate)."""
        np.random.seed(42)
        return 70 + 10 * np.random.randn(100)

    @pytest.fixture
    def vital_signs_2d(self) -> np.ndarray:
        """Generate 2D vital sign data (multiple vitals)."""
        np.random.seed(42)
        n_samples = 100
        return np.column_stack(
            [
                70 + 10 * np.random.randn(n_samples),
                120 + 15 * np.random.randn(n_samples),
                80 + 10 * np.random.randn(n_samples),
                16 + 2 * np.random.randn(n_samples),
                97 + 1 * np.random.randn(n_samples),
            ]
        )

    def test_extractor_initialization(self, extractor: MedicalFeatureExtractor) -> None:
        """Test extractor initialization."""
        assert extractor.config.domain == Domain.MEDICAL
        assert extractor.config.window_size == 60
        assert extractor.sofa_weights is not None
        assert extractor.vital_ranges is not None

    def test_extract_1d_data(
        self, extractor: MedicalFeatureExtractor, vital_signs_1d: np.ndarray
    ) -> None:
        """Test feature extraction from 1D data."""
        result = extractor.extract(vital_signs_1d)

        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.MEDICAL
        assert len(result.features) > 0
        assert len(result.feature_names) == len(result.features)
        assert result.extraction_time_ms >= 0
        assert not np.any(np.isnan(result.features))
        assert not np.any(np.isinf(result.features))

    def test_extract_2d_data(
        self, extractor: MedicalFeatureExtractor, vital_signs_2d: np.ndarray
    ) -> None:
        """Test feature extraction from 2D data."""
        result = extractor.extract(vital_signs_2d)

        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.MEDICAL
        assert len(result.features) > 0
        assert result.metadata["n_vitals"] == 5
        assert result.metadata["n_samples"] == 100

    def test_statistical_features(self, extractor: MedicalFeatureExtractor) -> None:
        """Test statistical feature computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features, names = extractor._compute_statistical_features(data)

        assert "mean" in names
        assert "std" in names
        assert "skewness" in names
        assert "kurtosis" in names
        assert features[names.index("mean")] == pytest.approx(3.0)

    def test_temporal_features(self, extractor: MedicalFeatureExtractor) -> None:
        """Test temporal feature computation."""
        data = np.sin(np.linspace(0, 4 * np.pi, 100))
        features, names = extractor._compute_temporal_features(data)

        assert "trend_slope" in names
        assert "autocorr_lag1" in names
        assert "spectral_centroid" in names
        assert len(features) == len(names)

    def test_empty_data_handling(self, extractor: MedicalFeatureExtractor) -> None:
        """Test handling of empty data."""
        data = np.array([])
        features, names = extractor._compute_statistical_features(data)
        assert len(features) == 8
        assert all(f == 0.0 for f in features)

    def test_get_feature_names(
        self, extractor: MedicalFeatureExtractor, vital_signs_1d: np.ndarray
    ) -> None:
        """Test get_feature_names method."""
        extractor.extract(vital_signs_1d)
        names = extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_sofa_weights_customization(self) -> None:
        """Test SOFA weights customization."""
        custom_weights = {"respiratory": 0.30, "cardiovascular": 0.30}
        config = DomainFeatureConfig(
            domain=Domain.MEDICAL,
            medical_params={"sofa_weights": custom_weights},
        )
        extractor = MedicalFeatureExtractor(config)
        assert extractor.sofa_weights == custom_weights


class TestFinancialFeatureExtractor:
    """Tests for FinancialFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> FinancialFeatureExtractor:
        """Create financial feature extractor."""
        return FinancialFeatureExtractor()

    @pytest.fixture
    def transaction_data(self) -> np.ndarray:
        """Generate synthetic transaction data."""
        np.random.seed(42)
        n_samples = 200
        amounts = np.abs(np.random.lognormal(mean=5, sigma=1.5, size=n_samples))
        timestamps = np.cumsum(np.random.exponential(scale=60, size=n_samples))
        return np.column_stack([amounts, timestamps])

    def test_extractor_initialization(self, extractor: FinancialFeatureExtractor) -> None:
        """Test extractor initialization."""
        assert extractor.config.domain == Domain.FINANCIAL
        assert extractor.velocity_windows is not None
        assert len(extractor.velocity_windows) == 3

    def test_extract_transaction_data(
        self, extractor: FinancialFeatureExtractor, transaction_data: np.ndarray
    ) -> None:
        """Test feature extraction from transaction data."""
        result = extractor.extract(transaction_data)

        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.FINANCIAL
        assert len(result.features) > 0
        assert not np.any(np.isnan(result.features))

    def test_benford_features(self, extractor: FinancialFeatureExtractor) -> None:
        """Test Benford's Law feature computation."""
        amounts = np.array([123, 234, 345, 456, 567, 678, 789, 890, 901, 112, 223, 334])
        features, names = extractor._compute_benford_features(amounts)

        assert "benford_chi_square" in names
        assert "benford_mae" in names
        assert len(features) > 0

    def test_velocity_features(self, extractor: FinancialFeatureExtractor) -> None:
        """Test transaction velocity feature computation."""
        amounts = np.random.rand(200) * 1000
        features, names = extractor._compute_velocity_features(amounts)

        assert "velocity_10_sum" in names
        assert "velocity_50_sum" in names
        assert len(features) == len(names)

    def test_round_number_features(self, extractor: FinancialFeatureExtractor) -> None:
        """Test round number detection features."""
        amounts = np.array([100, 200, 300, 123.45, 567.89, 1000, 500])
        features, names = extractor._compute_round_number_features(amounts)

        assert "round_ends_in_0" in names
        round_ratio = features[names.index("round_ends_in_0")]
        assert 0.0 <= round_ratio <= 1.0

    def test_1d_data_handling(self, extractor: FinancialFeatureExtractor) -> None:
        """Test handling of 1D data (amounts only)."""
        amounts = np.abs(np.random.lognormal(mean=5, sigma=1.5, size=100))
        result = extractor.extract(amounts)

        assert isinstance(result, DomainFeatureResult)
        assert len(result.features) > 0


class TestInfrastructureFeatureExtractor:
    """Tests for InfrastructureFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> InfrastructureFeatureExtractor:
        """Create infrastructure feature extractor."""
        return InfrastructureFeatureExtractor()

    @pytest.fixture
    def scada_data(self) -> np.ndarray:
        """Generate synthetic SCADA sensor data."""
        np.random.seed(42)
        n_samples = 200
        n_sensors = 5
        base_signal = np.sin(np.linspace(0, 4 * np.pi, n_samples))
        data = np.column_stack(
            [base_signal + 0.1 * np.random.randn(n_samples) + i * 10 for i in range(n_sensors)]
        )
        return data

    def test_extractor_initialization(self, extractor: InfrastructureFeatureExtractor) -> None:
        """Test extractor initialization."""
        assert extractor.config.domain == Domain.INFRASTRUCTURE
        assert extractor.correlation_threshold is not None

    def test_extract_scada_data(
        self, extractor: InfrastructureFeatureExtractor, scada_data: np.ndarray
    ) -> None:
        """Test feature extraction from SCADA data."""
        result = extractor.extract(scada_data)

        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.INFRASTRUCTURE
        assert len(result.features) > 0
        assert not np.any(np.isnan(result.features))

    def test_correlation_matrix_features(
        self, extractor: InfrastructureFeatureExtractor, scada_data: np.ndarray
    ) -> None:
        """Test correlation matrix feature computation."""
        features, names = extractor._compute_correlation_matrix_features(scada_data)

        assert "corr_matrix_mean" in names
        assert "corr_matrix_std" in names
        assert len(features) == len(names)

    def test_setpoint_deviation_features(self, extractor: InfrastructureFeatureExtractor) -> None:
        """Test setpoint deviation feature computation."""
        data = np.column_stack(
            [
                100 + 5 * np.random.randn(100),
                50 + 2 * np.random.randn(100),
            ]
        )
        features, names = extractor._compute_setpoint_deviation_features(data)

        assert "setpoint_mean_deviation" in names
        assert len(features) > 0

    def test_alarm_features(self, extractor: InfrastructureFeatureExtractor) -> None:
        """Test alarm feature computation."""
        alarm_data = np.random.choice([0, 1], size=(100, 3), p=[0.9, 0.1]).astype(float)
        features, names = extractor._compute_alarm_features(alarm_data)

        assert "alarm_total_crossings" in names
        assert len(features) == len(names)

    def test_attack_indicator_features(
        self, extractor: InfrastructureFeatureExtractor, scada_data: np.ndarray
    ) -> None:
        """Test attack indicator feature computation."""
        features, names = extractor._compute_attack_indicator_features(scada_data)

        assert "attack_frozen_ratio" in names
        assert len(features) > 0


class TestDomainFeatureExtractorFactory:
    """Tests for DomainFeatureExtractorFactory."""

    def test_create_medical_extractor(self) -> None:
        """Test creating medical extractor via factory."""
        extractor = DomainFeatureExtractorFactory.create(Domain.MEDICAL)
        assert isinstance(extractor, MedicalFeatureExtractor)

    def test_create_financial_extractor(self) -> None:
        """Test creating financial extractor via factory."""
        extractor = DomainFeatureExtractorFactory.create(Domain.FINANCIAL)
        assert isinstance(extractor, FinancialFeatureExtractor)

    def test_create_infrastructure_extractor(self) -> None:
        """Test creating infrastructure extractor via factory."""
        extractor = DomainFeatureExtractorFactory.create(Domain.INFRASTRUCTURE)
        assert isinstance(extractor, InfrastructureFeatureExtractor)

    def test_create_from_string(self) -> None:
        """Test creating extractor from string domain name."""
        extractor = DomainFeatureExtractorFactory.create("medical")
        assert isinstance(extractor, MedicalFeatureExtractor)

    def test_create_with_config(self) -> None:
        """Test creating extractor with custom config."""
        config = DomainFeatureConfig(
            domain=Domain.MEDICAL,
            window_size=120,
        )
        extractor = DomainFeatureExtractorFactory.create(Domain.MEDICAL, config)
        assert extractor.config.window_size == 120

    def test_unsupported_domain_raises(self) -> None:
        """Test that unsupported domain raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            DomainFeatureExtractorFactory.create(Domain.GENERAL)

    def test_register_custom_extractor(self) -> None:
        """Test registering a custom extractor."""

        class CustomExtractor(BaseDomainExtractor):
            def extract(self, data: np.ndarray) -> DomainFeatureResult:
                return DomainFeatureResult(
                    features=np.array([1.0]),
                    feature_names=["custom"],
                    domain=Domain.SECURITY,
                )

            def get_feature_names(self) -> list[str]:
                return ["custom"]

        DomainFeatureExtractorFactory.register(Domain.SECURITY, CustomExtractor)
        extractor = DomainFeatureExtractorFactory.create(Domain.SECURITY)
        assert isinstance(extractor, CustomExtractor)


class TestConvenienceFunctions:
    """Tests for convenience extraction functions."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Generate sample data."""
        np.random.seed(42)
        return np.random.randn(100, 3)

    def test_extract_medical_features(self, sample_data: np.ndarray) -> None:
        """Test extract_medical_features convenience function."""
        result = extract_medical_features(sample_data)
        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.MEDICAL

    def test_extract_financial_features(self, sample_data: np.ndarray) -> None:
        """Test extract_financial_features convenience function."""
        result = extract_financial_features(np.abs(sample_data))
        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.FINANCIAL

    def test_extract_infrastructure_features(self, sample_data: np.ndarray) -> None:
        """Test extract_infrastructure_features convenience function."""
        result = extract_infrastructure_features(sample_data)
        assert isinstance(result, DomainFeatureResult)
        assert result.domain == Domain.INFRASTRUCTURE


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample(self) -> None:
        """Test extraction with single sample."""
        extractor = MedicalFeatureExtractor()
        data = np.array([[70, 120, 80]])
        result = extractor.extract(data)
        assert len(result.features) > 0

    def test_nan_handling(self) -> None:
        """Test NaN value handling."""
        extractor = MedicalFeatureExtractor()
        data = np.array([70, np.nan, 72, 71, np.nan, 73])
        result = extractor.extract(data)
        assert not np.any(np.isnan(result.features))

    def test_inf_handling(self) -> None:
        """Test infinity value handling."""
        extractor = FinancialFeatureExtractor()
        data = np.array([100, np.inf, 200, 300, -np.inf, 400])
        result = extractor.extract(np.abs(data))
        assert not np.any(np.isinf(result.features))

    def test_constant_data(self) -> None:
        """Test extraction with constant data."""
        extractor = InfrastructureFeatureExtractor()
        data = np.ones((100, 3)) * 50
        result = extractor.extract(data)
        assert len(result.features) > 0

    def test_large_data(self) -> None:
        """Test extraction with large dataset."""
        extractor = MedicalFeatureExtractor()
        data = np.random.randn(10000, 5)
        result = extractor.extract(data)
        assert len(result.features) > 0
        assert result.extraction_time_ms < 5000

"""
Mercury Agent - Tests for Disaster Dataset Loaders

Tests for FEMA disaster declarations and hazard mitigation loaders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.disaster import (
    FEMADisasterLoader,
    FEMAHazardMitigationLoader,
)


class TestFEMADisasterLoader:
    """Tests for FEMA disaster declarations loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=200,
            random_seed=42,
            preprocessing={
                "year_range": (2010, 2024),
                "declaration_types": ["DR", "EM"],
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> FEMADisasterLoader:
        """Create loader instance."""
        return FEMADisasterLoader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "fema_disaster"
        assert loader.REQUIRES_CREDENTIALS is False
        assert loader.year_range == (2010, 2024)

    def test_disaster_types(self, loader: Any) -> None:
        """Test disaster types are defined."""
        assert "DR" in loader.DISASTER_TYPES
        assert "EM" in loader.DISASTER_TYPES
        assert loader.DISASTER_TYPES["DR"] == "Major Disaster Declaration"

    def test_incident_types(self, loader: Any) -> None:
        """Test incident types list."""
        assert "Hurricane" in loader.INCIDENT_TYPES
        assert "Flood" in loader.INCIDENT_TYPES
        assert "Tornado" in loader.INCIDENT_TYPES
        assert "Earthquake" in loader.INCIDENT_TYPES
        assert len(loader.INCIDENT_TYPES) >= 10

    def test_feature_names(self, loader: Any) -> None:
        """Test feature names are correctly defined."""
        expected = [
            "disaster_number",
            "state_fips",
            "year",
            "month",
            "day",
            "incident_type_code",
            "declaration_type_code",
            "designated_area_code",
            "ia_program",
            "pa_program",
            "hm_program",
        ]
        assert expected == loader.FEATURE_NAMES

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        # Note: This will try real API first, fall back to synthetic
        result = loader.download()
        assert result is True

    def test_load_data(self, loader: Any) -> None:
        """Test loading disaster data."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert features.shape[1] == 11  # Number of features

    def test_preprocess(self, loader: Any) -> None:
        """Test preprocessing."""
        loader.download()
        features, _ = loader._load_raw()

        processed = loader.preprocess(features)

        assert processed.dtype == np.float32

    def test_year_range(self, loader: Any) -> None:
        """Test generated data is within year range."""
        loader.download()
        features, _ = loader._load_raw()

        years = features[:, 2]
        assert years.min() >= 2010
        assert years.max() <= 2024

    def test_state_fips_valid(self, loader: Any) -> None:
        """Test state FIPS codes are valid US states and territories."""
        loader.download()
        features, _ = loader._load_raw()

        state_fips = features[:, 1]
        assert state_fips.min() >= 1
        # Max FIPS includes US territories: PR=72, VI=78, GU=66, AS=60, MP=69
        assert state_fips.max() <= 78

    def test_program_flags_binary(self, loader: Any) -> None:
        """Test program flags are binary."""
        loader.download()
        features, _ = loader._load_raw()

        ia_program = features[:, 8]
        pa_program = features[:, 9]
        hm_program = features[:, 10]

        for prog in [ia_program, pa_program, hm_program]:
            unique_vals = np.unique(prog)
            assert all(v in [0, 1] for v in unique_vals)

    def test_major_disaster_labeling(self, loader: Any) -> None:
        """Test major disaster labeling logic."""
        loader.download()
        features, labels = loader._load_raw()

        # Check label logic: DR type + multiple programs
        for i in range(len(features)):
            is_dr = features[i, 6] == 0  # declaration_type_code == DR
            program_count = features[i, 8] + features[i, 9] + features[i, 10]
            expected_label = 1 if (is_dr and program_count >= 2) else 0
            assert labels[i] == expected_label


class TestFEMAHazardMitigationLoader:
    """Tests for FEMA hazard mitigation loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="fema_hazard_mitigation",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
            preprocessing={
                "year_range": (2015, 2024),
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> FEMAHazardMitigationLoader:
        """Create loader instance."""
        return FEMAHazardMitigationLoader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "fema_hazard_mitigation"
        assert loader.REQUIRES_CREDENTIALS is False

    def test_feature_names(self, loader: Any) -> None:
        """Test feature names."""
        expected = [
            "project_amount",
            "federal_share",
            "state_fips",
            "year",
            "project_type_code",
            "status_code",
            "program_type_code",
        ]
        assert expected == loader.FEATURE_NAMES

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        result = loader.download()
        assert result is True

    def test_load_data(self, loader: Any) -> None:
        """Test loading mitigation data."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 7

    def test_project_amounts_positive(self, loader: Any) -> None:
        """Test project amounts are positive."""
        loader.download()
        features, _ = loader._load_raw()

        amounts = features[:, 0]
        assert (amounts >= 0).all()

    def test_federal_share_less_than_total(self, loader: Any) -> None:
        """Test federal share is less than or equal to project amount."""
        loader.download()
        features, _ = loader._load_raw()

        project_amount = features[:, 0]
        federal_share = features[:, 1]
        assert (federal_share <= project_amount * 1.01).all()  # Allow small rounding

    def test_preprocess_log_transform(self, loader: Any) -> None:
        """Test preprocessing applies log transform to monetary values."""
        loader.download()
        features, _ = loader._load_raw()

        # Get raw monetary values
        raw_amounts = features[:, 0].copy()

        processed = loader.preprocess(features.copy())

        # Processed amounts should be log-transformed
        # (smaller range after log transform)
        assert processed[:, 0].std() < raw_amounts.std()


class TestFEMADatasetRegistry:
    """Test dataset registry for FEMA loaders."""

    def test_fema_disaster_registered(self) -> None:
        """Test FEMA disaster is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("fema_disaster")
        assert loader_class is FEMADisasterLoader

        # Test aliases
        assert DatasetRegistry.get("fema") is FEMADisasterLoader
        assert DatasetRegistry.get("disaster_declarations") is FEMADisasterLoader

    def test_hazard_mitigation_registered(self) -> None:
        """Test hazard mitigation is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("fema_hazard_mitigation")
        assert loader_class is FEMAHazardMitigationLoader

        # Test alias
        assert DatasetRegistry.get("hazard_mitigation") is FEMAHazardMitigationLoader


class TestFEMADisasterStatistics:
    """Integration tests for FEMA disaster statistics."""

    @pytest.fixture
    def loader(self, tmp_path: Any) -> FEMADisasterLoader:
        """Create FEMA loader with larger sample."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=500,
            random_seed=42,
            preprocessing={
                "year_range": (2000, 2024),
            },
        )
        loader = FEMADisasterLoader(config)
        loader.download()
        return loader

    def test_seasonal_patterns(self, loader: Any) -> None:
        """Test seasonal patterns in disaster data."""
        features, _ = loader._load_raw()
        months = features[:, 3]

        # Hurricane season (Aug-Oct) should have more events in Gulf states
        hurricane_states = [12, 22, 48, 37, 45]  # FL, LA, TX, NC, SC
        gulf_mask = np.isin(features[:, 1], hurricane_states)

        if gulf_mask.sum() > 0:
            gulf_months = months[gulf_mask]
            hurricane_season = np.isin(gulf_months, [8, 9, 10])
            # At least some hurricane season events
            assert hurricane_season.sum() > 0

    def test_incident_type_distribution(self, loader: Any) -> None:
        """Test incident types have variety."""
        features, _ = loader._load_raw()
        incident_codes = features[:, 5]

        unique_incidents = np.unique(incident_codes)
        # Should have at least a few different incident types
        assert len(unique_incidents) >= 3

    def test_get_statistics(self, loader: Any) -> None:
        """Test get_statistics method."""
        stats = loader.get_statistics()

        assert "n_samples" in stats
        assert "n_major_disasters" in stats
        assert "major_disaster_ratio" in stats
        assert "year_range" in stats
        assert "incident_type_distribution" in stats
        assert "declaration_type_distribution" in stats

        assert stats["n_samples"] == 500
        assert 0 <= stats["major_disaster_ratio"] <= 1


class TestFEMAAPIIntegration:
    """Tests for FEMA API integration (may require network)."""

    @pytest.fixture
    def loader(self, tmp_path: Any) -> FEMADisasterLoader:
        """Create FEMA loader."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
        )
        return FEMADisasterLoader(config)

    def test_rate_limiting(self, loader: Any) -> None:
        """Test rate limiting is implemented."""
        assert hasattr(loader, "_rate_limit")
        assert loader._request_delay >= 0.1  # At least 100ms delay

    def test_api_url_valid(self, loader: Any) -> None:
        """Test API URL is correctly configured."""
        from omni_mercury_engine.security.input_validation import TrustedEndpoints

        assert loader.API_URL == TrustedEndpoints.FEMA_DISASTER_DECLARATIONS
        assert "fema.gov" in loader.API_URL
        assert "v2" in loader.API_URL


class TestDisasterDataEdgeCases:
    """Edge case tests for disaster data."""

    def test_empty_year_range(self, tmp_path: Any) -> None:
        """Test handling of invalid year range."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=10,
            preprocessing={
                "year_range": (2025, 2025),  # Single year
            },
        )
        loader = FEMADisasterLoader(config)
        result = loader.download()
        assert result is True  # Should still generate synthetic

    def test_small_sample_size(self, tmp_path: Any) -> None:
        """Test with very small sample size."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=5,
        )
        loader = FEMADisasterLoader(config)
        loader.download()
        features, labels = loader._load_raw()

        assert len(features) == 5
        assert len(labels) == 5

    def test_large_sample_size(self, tmp_path: Any) -> None:
        """Test with large sample size."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=10000,
        )
        loader = FEMADisasterLoader(config)
        loader.download()
        features, labels = loader._load_raw()

        # API limits to 10000, synthetic should respect max_samples
        assert len(features) <= 10000

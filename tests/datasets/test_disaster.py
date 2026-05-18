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
        """Major disaster labels follow the minority-as-anomaly convention.

        As of v1.7.0 the FEMA Disaster loader auto-corrects label
        polarity so the *rarer* event class always carries label==1
        (the unsupervised-anomaly convention).  This test locks two
        invariants on the synthetic-fallback path used in CI:

        1.  Positive rate is at most 50 percent — anomaly class is
            the minority.
        2.  When `loader.labels_inverted` is False, label==1 matches
            the legacy "DR + multi-program" mask; when it is True,
            label==1 matches its complement.  This is the same
            invariant that `_select_anomaly_polarity` enforces and
            is what unblocks the previously-broken FEMA Disaster
            benchmark (see CHANGELOG `[Unreleased]`).
        """
        loader.download()
        features, labels = loader._load_raw()

        positive_rate = float(labels.mean())
        assert positive_rate <= 0.5, (
            f"FEMA Disaster positive rate is {positive_rate:.3f}; "
            "the anomaly class must be the minority "
            "(minority-as-anomaly invariant)."
        )

        candidate_major = (features[:, 6] == 0) & (
            (features[:, 8] + features[:, 9] + features[:, 10]) >= 2
        )
        expected_after_polarity = (~candidate_major) if loader.labels_inverted else candidate_major
        assert (labels == expected_after_polarity.astype(labels.dtype)).all()


class TestFEMAInvertedScoresCorrection:
    """Regression coverage for the v1.7.0 inverted-scores fix.

    The `CHANGELOG.md` reproducibility footnote previously flagged
    `fema_disaster` as a "known-broken loader producing inverted
    scores" — meaning the model's AUC routinely fell below 0.5
    because the loader handed it majority-as-positive labels.
    These tests pin the fix.
    """

    def test_polarity_flips_when_candidate_class_is_majority(self, tmp_path: Any) -> None:
        """`_select_anomaly_polarity` inverts a majority mask."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "d"),
            cache_dir=str(tmp_path / "c"),
        )
        loader = FEMADisasterLoader(config)
        mask = np.array([True, True, True, True, True, True, True, False, False, False])
        labels = loader._select_anomaly_polarity(mask)
        assert loader.labels_inverted is True
        assert labels.sum() == 3  # minority class wins
        assert (labels == (~mask).astype(np.int64)).all()

    def test_polarity_preserved_when_candidate_class_is_minority(self, tmp_path: Any) -> None:
        """`_select_anomaly_polarity` is a no-op when already minority."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "d"),
            cache_dir=str(tmp_path / "c"),
        )
        loader = FEMADisasterLoader(config)
        mask = np.array([True, True, False, False, False, False, False, False, False, False])
        labels = loader._select_anomaly_polarity(mask)
        assert loader.labels_inverted is False
        assert labels.sum() == 2
        assert (labels == mask.astype(np.int64)).all()

    def test_polarity_handles_empty_mask(self, tmp_path: Any) -> None:
        """No records → no flip, no crash."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "d"),
            cache_dir=str(tmp_path / "c"),
        )
        loader = FEMADisasterLoader(config)
        labels = loader._select_anomaly_polarity(np.array([], dtype=bool))
        assert loader.labels_inverted is False
        assert labels.size == 0

    def test_property_starts_false(self, tmp_path: Any) -> None:
        """`labels_inverted` defaults to False before any load."""
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "d"),
            cache_dir=str(tmp_path / "c"),
        )
        loader = FEMADisasterLoader(config)
        assert loader.labels_inverted is False

    def test_real_data_processing_invariant(self, tmp_path: Any) -> None:
        """Synthesised "real-data shape" records exercise the same path.

        Constructs an in-memory record set that mimics OpenFEMA's API
        response with a deliberately majority "DR + multi-program"
        slice, then runs the real-data processing pipeline.  Locks
        that the public path produces a minority-positive label set
        and reports the inversion via `labels_inverted`.
        """
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "d"),
            cache_dir=str(tmp_path / "c"),
        )
        loader = FEMADisasterLoader(config)

        # 8 records that would historically be label==1
        # (DR + IA + PA + HM), 2 minority records.
        majority_records = [
            {
                "declarationDate": "2020-01-01T00:00:00.000Z",
                "incidentType": "Hurricane",
                "declarationType": "DR",
                "fipsStateCode": "12",
                "designatedArea": "Statewide",
                "ihProgramDeclared": True,
                "paProgramDeclared": True,
                "hmProgramDeclared": True,
                "disasterNumber": 4000 + i,
            }
            for i in range(8)
        ]
        minority_records = [
            {
                "declarationDate": "2020-02-01T00:00:00.000Z",
                "incidentType": "Fire",
                "declarationType": "EM",
                "fipsStateCode": "6",
                "designatedArea": "Los Angeles (County)",
                "ihProgramDeclared": False,
                "paProgramDeclared": False,
                "hmProgramDeclared": False,
                "disasterNumber": 5000 + i,
            }
            for i in range(2)
        ]

        features, labels = loader._process_fema_data(majority_records + minority_records)
        assert features.shape == (10, 11)
        assert loader.labels_inverted is True
        assert labels.sum() == 2  # minority class wins
        # The 2 minority records carry the anomaly label.
        assert labels[-2:].tolist() == [1, 1]


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

"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for ADRepository dataset loaders.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.datasets import (
    ADREPOSITORY_DATASETS,
    ADRepositoryLoader,
    DatasetConfig,
    list_available_datasets,
    load_dataset,
)


class TestADRepositoryMetadata:
    """Test ADRepository dataset metadata."""

    def test_datasets_defined(self):
        """Verify all expected datasets are defined."""
        expected = {
            "fraud",
            "backdoor",
            "campaign",
            "thyroid",
            "donors",
            "census",
            "celeba",
            "smd",
            "swat",
            "dsads",
            "epilepsy",
        }
        assert expected.issubset(set(ADREPOSITORY_DATASETS.keys()))

    def test_dataset_info_complete(self):
        """Verify all datasets have required metadata fields."""
        required_fields = {"samples", "features", "anomaly_ratio", "domain", "description", "url"}

        for name, info in ADREPOSITORY_DATASETS.items():
            for field in required_fields:
                assert field in info, f"Dataset '{name}' missing field '{field}'"

    def test_list_available_datasets(self):
        """Test convenience function to list datasets."""
        datasets = list_available_datasets()
        assert isinstance(datasets, dict)
        assert len(datasets) >= 11
        assert "fraud" in datasets
        assert "thyroid" in datasets


class TestADRepositoryLoader:
    """Test ADRepository loader functionality."""

    def test_init_valid_dataset(self):
        """Test loader initialization with valid dataset."""
        config = DatasetConfig(name="thyroid", data_dir="./data/test")
        loader = ADRepositoryLoader(config, dataset_name="thyroid")

        assert loader.dataset_name == "thyroid"
        assert loader.dataset_info["samples"] == 7200
        assert loader.dataset_info["features"] == 21

    def test_init_invalid_dataset(self):
        """Test loader raises error for invalid dataset."""
        config = DatasetConfig(name="invalid", data_dir="./data/test")

        with pytest.raises(ValueError, match="Unknown dataset"):
            ADRepositoryLoader(config, dataset_name="nonexistent")

    def test_get_metadata(self):
        """Test metadata retrieval."""
        config = DatasetConfig(name="fraud", data_dir="./data/test")
        loader = ADRepositoryLoader(config, dataset_name="fraud")
        metadata = loader.get_metadata()

        assert metadata["name"] == "fraud"
        assert metadata["source"] == "ADRepository"
        assert metadata["samples"] == 284807
        assert metadata["features"] == 29
        assert metadata["domain"] == "finance"
        assert "Pang" in metadata["citation"]

    def test_synthetic_fallback(self):
        """Test synthetic fallback when download fails."""
        config = DatasetConfig(
            name="thyroid",
            data_dir="./data/test_synthetic",
            max_samples=1000,
        )
        loader = ADRepositoryLoader(config, dataset_name="thyroid")

        # Force synthetic fallback
        loader._create_synthetic_fallback()

        X, y = loader.load_data()

        assert X.shape[0] == 1000
        assert X.shape[1] == 21  # thyroid has 21 features
        assert len(y) == 1000
        assert y.sum() > 0  # Should have some anomalies
        assert not loader.is_real_data

    def test_load_with_max_samples(self):
        """Test loading with sample limit."""
        config = DatasetConfig(
            name="donors",
            data_dir="./data/test_limited",
            max_samples=500,
        )
        loader = ADRepositoryLoader(config, dataset_name="donors")
        loader._create_synthetic_fallback()

        X, y = loader.load_data()

        assert X.shape[0] <= 500
        assert len(y) <= 500

    def test_get_statistics(self):
        """Test statistics calculation."""
        config = DatasetConfig(
            name="campaign",
            data_dir="./data/test_stats",
            max_samples=1000,
        )
        loader = ADRepositoryLoader(config, dataset_name="campaign")
        loader._create_synthetic_fallback()
        loader.load_data()

        stats = loader.get_statistics()

        assert "n_samples" in stats
        assert "n_features" in stats
        assert "n_anomalies" in stats
        assert "anomaly_ratio" in stats
        assert "is_real_data" in stats
        assert stats["n_samples"] == 1000
        assert stats["n_features"] == 62


class TestLoadDatasetConvenience:
    """Test the load_dataset convenience function."""

    def test_load_thyroid_synthetic(self):
        """Test loading thyroid with synthetic fallback."""
        X, y, meta = load_dataset(
            "thyroid",
            data_dir="./data/test_convenience",
            max_samples=500,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, dict)
        assert X.shape[0] == len(y)
        assert meta["name"] == "thyroid"

    def test_load_invalid_dataset(self):
        """Test error on invalid dataset name."""
        with pytest.raises(ValueError):
            load_dataset("not_a_real_dataset")


class TestIntegrationWithEngine:
    """Integration tests with Mercury Agent engine."""

    def test_adrepository_with_detector(self):
        """Test using ADRepository data with anomaly detector."""
        from omni_mercury_engine import OmniMercuryEngine

        # Load synthetic data (faster for tests)
        config = DatasetConfig(
            name="backdoor",
            data_dir="./data/test_integration",
            max_samples=200,
        )
        loader = ADRepositoryLoader(config, dataset_name="backdoor")
        loader._create_synthetic_fallback()
        X, y = loader.load_data()

        # Run detection
        engine = OmniMercuryEngine()
        result = engine.detect(X)

        assert "detectors" in result
        assert "is_anomaly" in result

    def test_multiple_datasets_benchmark(self):
        """Test benchmarking across multiple ADRepository datasets."""

        datasets_to_test = ["thyroid", "backdoor", "campaign"]
        results = {}

        for name in datasets_to_test:
            config = DatasetConfig(
                name=name,
                data_dir=f"./data/test_benchmark_{name}",
                max_samples=500,
            )
            loader = ADRepositoryLoader(config, dataset_name=name)
            loader._create_synthetic_fallback()
            X, y = loader.load_data()

            # Simple detector test using Mercury-native scoring
            from omni_mercury_engine.ml.mercury_ml import StandardScaler

            X_scaled = StandardScaler().fit_transform(X)
            median = np.median(X_scaled, axis=0)
            mad = np.median(np.abs(X_scaled - median), axis=0) + 1e-10
            scores = np.max(np.abs(X_scaled - median) / mad, axis=1)

            if len(np.unique(y)) > 1:
                from omni_mercury_engine.ml.mercury_ml import roc_auc_score as native_roc_auc_score

                auc = native_roc_auc_score(y, scores)
                results[name] = auc

        # All datasets should produce valid AUC scores
        assert len(results) == 3
        for name, auc in results.items():
            assert 0 <= auc <= 1, f"Invalid AUC for {name}: {auc}"

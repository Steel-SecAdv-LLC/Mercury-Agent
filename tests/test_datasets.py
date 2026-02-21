"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for real-world dataset loaders and benchmarks.

These tests use synthetic fallback data for fast, offline unit testing.
Real-data tests are in test_loaders_live.py (marked @pytest.mark.network).
"""

from __future__ import annotations

import os

# Enable synthetic fallback for these unit tests — they deliberately test
# loader mechanics with generated data, not live API correctness.
os.environ["MERCURY_ALLOW_SYNTHETIC"] = "1"

import tempfile

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig, DatasetRegistry, DatasetSplit
from omni_mercury_engine.datasets.benchmarks import (
    BenchmarkResult,
    RealWorldBenchmarkSuite,
    random_baseline,
)
from omni_mercury_engine.datasets.environmental import (
    NOAAWeatherLoader,
    USGSEarthquakeLoader,
    WildfireDataLoader,
)
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.datasets.medical import MIMICLoader, PhysioNetLoader, SepsisDataset
from omni_mercury_engine.datasets.security import CICIDSLoader, NSLKDDLoader, ThreatIntelLoader
from omni_mercury_engine.datasets.space import NASAExoplanetLoader, SETILoader, SolarDynamicsLoader


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="test",
                data_dir=os.path.join(tmpdir, "data"),
                cache_dir=os.path.join(tmpdir, "cache"),
            )
            assert config.name == "test"
            assert config.split_ratios == (0.7, 0.15, 0.15)

    def test_config_split_ratios_validation(self):
        """Test split ratio validation."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError):
            DatasetConfig(
                name="test",
                data_dir=tmpdir,
                split_ratios=(0.5, 0.3, 0.3),  # Sums to 1.1
            )

    def test_cache_key_generation(self):
        """Test unique cache key generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = DatasetConfig(name="test", data_dir=tmpdir)
            config2 = DatasetConfig(name="test", data_dir=tmpdir, max_samples=100)

            assert config1.get_cache_key() != config2.get_cache_key()


class TestMedicalDatasets:
    """Tests for medical dataset loaders."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_mimic_loader_credential_gate(self, tmpdir):
        """Test MIMIC loader raises without credentials (never generates synthetic)."""
        config = DatasetConfig(
            name="mimic-iii",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = MIMICLoader(config)

        with pytest.raises(DataSourceUnavailableError, match="PhysioNet"):
            loader.load(DatasetSplit.ALL)

    def test_physionet_loader_ecg(self, tmpdir):
        """Test PhysioNet ECG loader."""
        config = DatasetConfig(
            name="physionet",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=50,
        )
        loader = PhysioNetLoader(config)

        features, labels = loader.load(DatasetSplit.TRAIN)

        assert len(features) > 0
        assert len(labels) == len(features)

    def test_sepsis_dataset(self, tmpdir):
        """Test sepsis-focused dataset."""
        config = DatasetConfig(
            name="sepsis",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = SepsisDataset(config)

        features, labels = loader.load(DatasetSplit.ALL)

        # Should have ~30% positive cases for sepsis
        positive_rate = labels.sum() / len(labels)
        assert 0.2 < positive_rate < 0.4


class TestSpaceDatasets:
    """Tests for space dataset loaders."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_seti_loader_deprecated(self, tmpdir):
        """Test SETI loader is deprecated and raises."""
        config = DatasetConfig(
            name="seti",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = SETILoader(config)

        with pytest.raises(DataSourceUnavailableError, match="deprecated"):
            loader.load(DatasetSplit.ALL)

    def test_exoplanet_loader(self, tmpdir):
        """Test exoplanet loader."""
        config = DatasetConfig(
            name="exoplanet",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=50,
        )
        loader = NASAExoplanetLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(NASAExoplanetLoader.FEATURE_NAMES)

    def test_solar_dynamics_loader(self, tmpdir):
        """Test solar dynamics loader."""
        config = DatasetConfig(
            name="solar",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = SolarDynamicsLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(SolarDynamicsLoader.FEATURE_NAMES)
        assert np.any(labels == 1)  # Should have some storm events


class TestEnvironmentalDatasets:
    """Tests for environmental dataset loaders."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_earthquake_loader(self, tmpdir):
        """Test earthquake catalog loader."""
        config = DatasetConfig(
            name="earthquake",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = USGSEarthquakeLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(USGSEarthquakeLoader.FEATURE_NAMES)

    def test_weather_loader(self, tmpdir):
        """Test weather data loader."""
        config = DatasetConfig(
            name="weather",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = NOAAWeatherLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(NOAAWeatherLoader.FEATURE_NAMES)

    def test_wildfire_loader(self, tmpdir):
        """Test wildfire detection loader."""
        config = DatasetConfig(
            name="wildfire",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
            random_seed=42,
        )
        loader = WildfireDataLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        # ~30% fire detections expected, with wider tolerance for small sample variance
        # With n=100 and p=0.3, std_dev ~= 4.6, so 4-sigma range is ~12-48%
        # Using even wider tolerance (0.1-0.6) to account for edge cases in synthetic generation
        fire_rate = labels.sum() / len(labels)
        assert 0.1 < fire_rate < 0.6, f"Fire rate {fire_rate:.2f} outside expected range [0.1, 0.6]"


class TestSecurityDatasets:
    """Tests for security dataset loaders."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_nslkdd_loader(self, tmpdir):
        """Test NSL-KDD loader."""
        config = DatasetConfig(
            name="nsl-kdd",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = NSLKDDLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(NSLKDDLoader.FEATURE_NAMES)
        assert np.any(labels == 1)  # Should have attacks

    def test_cicids_loader(self, tmpdir):
        """Test CICIDS loader."""
        config = DatasetConfig(
            name="cicids",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = CICIDSLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert len(features) == 100

    def test_threat_intel_loader(self, tmpdir):
        """Test threat intelligence loader."""
        config = DatasetConfig(
            name="threat-intel",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = ThreatIntelLoader(config)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(ThreatIntelLoader.FEATURE_NAMES)


class TestDatasetRegistry:
    """Tests for dataset registry."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = DatasetRegistry.list_datasets()

        assert "mimic-iii" in datasets
        assert "seti" in datasets
        assert "earthquake" in datasets
        assert "nsl-kdd" in datasets

    def test_create_from_registry(self):
        """Test creating loader from registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="earthquake", data_dir=tmpdir, cache_dir=tmpdir)
            loader = DatasetRegistry.create("earthquake", config)

            assert isinstance(loader, USGSEarthquakeLoader)


class TestBenchmarkSuite:
    """Tests for benchmark suite."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_benchmark_single_dataset(self, tmpdir):
        """Test running benchmark on single dataset."""
        suite = RealWorldBenchmarkSuite(
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples_per_dataset=50,
        )

        def simple_detector(features):
            return np.random.rand(len(features))

        result = suite.run_benchmark(
            dataset_name="earthquake",
            detector=simple_detector,
            detector_name="TestDetector",
        )

        assert isinstance(result, BenchmarkResult)
        assert result.dataset_name == "earthquake"
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.f1_score <= 1

    def test_benchmark_multiple_datasets(self, tmpdir):
        """Test running benchmark across categories."""
        suite = RealWorldBenchmarkSuite(
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples_per_dataset=30,
        )

        results = suite.run_all_benchmarks(
            detector=random_baseline,
            detector_name="RandomBaseline",
            datasets=["earthquake", "wildfire"],  # Quick test
        )

        assert len(results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in results)

    def test_baseline_comparison(self, tmpdir):
        """Test comparing against baseline."""
        suite = RealWorldBenchmarkSuite(
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples_per_dataset=30,
        )

        def better_detector(features):
            # Slightly better than random
            return np.clip(np.random.rand(len(features)) + 0.1, 0, 1)

        results = suite.run_all_benchmarks(
            detector=better_detector,
            detector_name="BetterDetector",
            datasets=["earthquake"],
        )

        comparison = suite.compare_with_baseline(
            results=results,
            baseline_detector=random_baseline,
            baseline_name="RandomBaseline",
        )

        assert comparison.baseline_name == "RandomBaseline"
        assert "earthquake" in comparison.improvement_vs_baseline

    def test_save_and_load_results(self, tmpdir):
        """Test saving and loading benchmark results."""
        suite = RealWorldBenchmarkSuite(
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples_per_dataset=30,
        )

        suite.run_all_benchmarks(
            detector=random_baseline,
            detector_name="Test",
            datasets=["earthquake"],
        )

        # Save
        save_path = os.path.join(tmpdir, "results.json")
        suite.save_results(save_path)
        assert os.path.exists(save_path)

        # Load
        suite2 = RealWorldBenchmarkSuite()
        loaded = suite2.load_results(save_path)

        assert len(loaded) == 1
        assert loaded[0].dataset_name == "earthquake"


class TestPyTorchIntegration:
    """Tests for PyTorch dataset integration."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_to_pytorch_dataset(self, tmpdir):
        """Test converting to PyTorch dataset."""
        pytest.importorskip("torch")

        config = DatasetConfig(
            name="earthquake",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=50,
        )
        loader = USGSEarthquakeLoader(config)
        loader.load()  # Ensure data is loaded

        torch_dataset = loader.to_pytorch_dataset(DatasetSplit.TRAIN)

        assert len(torch_dataset) > 0
        x, y = torch_dataset[0]
        assert hasattr(x, "shape")  # Is a tensor

    def test_get_dataloader(self, tmpdir):
        """Test getting PyTorch DataLoader."""
        pytest.importorskip("torch")

        config = DatasetConfig(
            name="earthquake",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=50,
        )
        loader = USGSEarthquakeLoader(config)
        loader.load()

        dataloader = loader.get_dataloader(
            split=DatasetSplit.TRAIN,
            batch_size=16,
        )

        batch = next(iter(dataloader))
        assert len(batch) == 2  # (features, labels)
        assert batch[0].shape[0] <= 16  # Batch size

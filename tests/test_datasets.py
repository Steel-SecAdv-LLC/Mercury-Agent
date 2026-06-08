# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for real-world dataset loaders and benchmarks."""

from __future__ import annotations

# ``MERCURY_ALLOW_SYNTHETIC=1`` is set at the suite level in
# ``tests/conftest.py`` (lifted from this module's import-time
# assignment so the contract holds under ``pytest-xdist -n 4``
# regardless of which worker collects which test file first).
import os
import tempfile
from typing import Any

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

    def test_config_creation(self) -> None:
        """Test basic config creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="test",
                data_dir=os.path.join(tmpdir, "data"),
                cache_dir=os.path.join(tmpdir, "cache"),
            )
            assert config.name == "test"
            assert config.split_ratios == (0.7, 0.15, 0.15)

    def test_config_split_ratios_validation(self) -> None:
        """Test split ratio validation."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError):
            DatasetConfig(
                name="test",
                data_dir=tmpdir,
                split_ratios=(0.5, 0.3, 0.3),  # Sums to 1.1
            )

    def test_cache_key_generation(self) -> None:
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

    def test_mimic_loader_credential_gate(self, tmpdir: Any) -> None:
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

    def test_physionet_loader_ecg(self, tmpdir: Any) -> None:
        """Test PhysioNet ECG loader.

        ``PhysioNetLoader.download()`` already takes the synthetic
        path automatically when no local data is present and
        ``MERCURY_ALLOW_SYNTHETIC=1`` is set (the conftest default for
        the unit-test suite) -- it makes no network call, so no
        monkey-patch is needed here.  Live coverage lives in
        ``tests/test_loaders_live.py @pytest.mark.network``.
        """
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

    def test_sepsis_dataset(self, tmpdir: Any) -> None:
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

    def test_seti_loader_deprecated(self, tmpdir: Any) -> None:
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

    def test_exoplanet_loader(self, tmpdir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test exoplanet loader synthetic-fallback path.

        Patches the upstream NASA TAP download to return False so the
        loader uses its synthetic generator, isolating the test from
        NASA Exoplanet Archive availability.  Live-API coverage is in
        ``tests/test_loaders_live.py @pytest.mark.network``.
        """
        config = DatasetConfig(
            name="exoplanet",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=50,
        )
        loader = NASAExoplanetLoader(config)
        monkeypatch.setattr(loader, "_download_from_nasa_tap", lambda: False)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(NASAExoplanetLoader.FEATURE_NAMES)

    def test_solar_dynamics_loader(self, tmpdir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test solar dynamics loader synthetic-fallback path.

        Patches the upstream NOAA SWPC download to return False so the
        loader exercises its synthetic solar-cycle generator,
        isolating the test from SWPC availability.  The ``labels == 1``
        assertion is satisfied by the synthetic ``kp_index >= 7``
        storm criterion (~22% of samples for the deterministic test
        seed).  Live-API coverage is in ``tests/test_loaders_live.py``.
        """
        config = DatasetConfig(
            name="solar",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = SolarDynamicsLoader(config)
        monkeypatch.setattr(loader, "_download_from_swpc", lambda: False)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(SolarDynamicsLoader.FEATURE_NAMES)
        assert np.any(labels == 1)  # Should have some storm events


class TestEnvironmentalDatasets:
    """Tests for environmental dataset loaders."""

    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_earthquake_loader(self, tmpdir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test earthquake catalog loader synthetic-fallback path.

        Patches the upstream USGS fdsnws download to return False so the
        loader exercises its synthetic Gutenberg-Richter generator.
        Live-API coverage is in
        ``tests/test_loaders_live.py @pytest.mark.network``.
        """
        config = DatasetConfig(
            name="earthquake",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = USGSEarthquakeLoader(config)
        monkeypatch.setattr(loader, "_download_from_usgs", lambda: False)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(USGSEarthquakeLoader.FEATURE_NAMES)

    def test_weather_loader(self, tmpdir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test weather loader synthetic-fallback path.

        The file's contract (see module docstring) is offline unit testing
        against the synthetic generator.  We force the synthetic path by
        patching the upstream Open-Meteo download to return False — the
        exact signal the loader already produces on a 503, timeout, or
        TLS-handshake stall.  Without this patch, the test makes eight
        sequential HTTP calls (one per LOCATIONS entry) with per-attempt
        timeouts, and an Open-Meteo outage trips pytest-timeout before
        the synthetic fallback is reached.  Live-API coverage lives in
        ``tests/test_loaders_live.py`` under ``@pytest.mark.network``.
        """
        config = DatasetConfig(
            name="weather",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = NOAAWeatherLoader(config)
        monkeypatch.setattr(loader, "_download_from_open_meteo", lambda: False)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(NOAAWeatherLoader.FEATURE_NAMES)
        assert loader.is_real_data is False

    def test_wildfire_loader(self, tmpdir: Any) -> None:
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

    def test_nslkdd_loader(self, tmpdir: Any) -> None:
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

    def test_cicids_loader(self, tmpdir: Any) -> None:
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

    def test_threat_intel_loader(self, tmpdir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test threat intelligence loader synthetic-fallback path.

        Patches the upstream MITRE ATT&CK download to return False so the
        loader exercises its synthetic threat-intel generator.  Live-API
        coverage is in ``tests/test_loaders_live.py``.
        """
        config = DatasetConfig(
            name="threat-intel",
            data_dir=tmpdir,
            cache_dir=tmpdir,
            max_samples=100,
        )
        loader = ThreatIntelLoader(config)
        monkeypatch.setattr(loader, "_download_from_mitre", lambda: False)

        features, labels = loader.load(DatasetSplit.ALL)

        assert features.shape[1] == len(ThreatIntelLoader.FEATURE_NAMES)


class TestDatasetRegistry:
    """Tests for dataset registry."""

    def test_list_datasets(self) -> None:
        """Test listing available datasets."""
        datasets = DatasetRegistry.list_datasets()

        assert "mimic-iii" in datasets
        assert "seti" in datasets
        assert "earthquake" in datasets
        assert "nsl-kdd" in datasets

    def test_create_from_registry(self) -> None:
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

    def test_benchmark_single_dataset(self, tmpdir: Any) -> None:
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

    def test_benchmark_multiple_datasets(self, tmpdir: Any) -> None:
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

    def test_baseline_comparison(self, tmpdir: Any) -> None:
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

    def test_save_and_load_results(self, tmpdir: Any) -> None:
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

    def test_to_pytorch_dataset(self, tmpdir: Any) -> None:
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

    def test_get_dataloader(self, tmpdir: Any) -> None:
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

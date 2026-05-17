"""
Mercury Agent - Tests for Climate and Ocean Dataset Loaders

Tests for Simons CMAP, World Ocean Database, and Copernicus Sea Level loaders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.climate import (
    CopernicusERA5Loader,
    CopernicusSeaLevelLoader,
    SimonsCMAPLoader,
    WorldOceanDatabaseLoader,
)


class TestSimonsCMAPLoader:
    """Tests for Simons CMAP ocean data loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="simons_cmap",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
            preprocessing={
                "variable_set": "physical",
                "region": {
                    "lat_min": -30,
                    "lat_max": 30,
                    "lon_min": -60,
                    "lon_max": 60,
                },
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> SimonsCMAPLoader:
        """Create loader instance."""
        return SimonsCMAPLoader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "simons_cmap"
        assert loader.REQUIRES_CREDENTIALS is True
        assert loader.variable_set == "physical"
        assert len(loader.FEATURE_NAMES) == 8

    def test_feature_names(self, loader: Any) -> None:
        """Test feature names are correctly defined."""
        expected = [
            "latitude",
            "longitude",
            "depth",
            "temperature",
            "salinity",
            "chlorophyll",
            "nitrate",
            "oxygen",
        ]
        assert expected == loader.FEATURE_NAMES

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        result = loader.download()
        assert result is True

        # Synthetic data should be marked as non-real
        assert loader.is_real_data is False

    def test_load_data(self, loader: Any) -> None:
        """Test loading data after download."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert features.shape[1] == 8  # Number of features
        assert labels.dtype == np.int64

    def test_preprocess(self, loader: Any) -> None:
        """Test preprocessing normalizes data."""
        loader.download()
        features, _ = loader._load_raw()

        processed = loader.preprocess(features)

        assert processed.dtype == np.float32
        # Check normalization (mean should be near 0, std near 1)
        assert np.abs(processed.mean()) < 0.5
        assert 0.5 < processed.std() < 1.5

    def test_variable_sets(self, loader: Any) -> None:
        """Test variable set options are defined."""
        assert "satellite" in loader.VARIABLE_SETS
        assert "biogeochemistry" in loader.VARIABLE_SETS
        assert "physical" in loader.VARIABLE_SETS


class TestWorldOceanDatabaseLoader:
    """Tests for World Ocean Database loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="world_ocean_database",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
            preprocessing={
                "instrument_type": "PFL",
                "year_range": (2020, 2024),
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> WorldOceanDatabaseLoader:
        """Create loader instance."""
        return WorldOceanDatabaseLoader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "world_ocean_database"
        assert loader.REQUIRES_CREDENTIALS is False
        assert loader.instrument_type == "PFL"

    def test_instrument_types(self, loader: Any) -> None:
        """Test instrument types are defined."""
        assert "OSD" in loader.INSTRUMENT_TYPES
        assert "CTD" in loader.INSTRUMENT_TYPES
        assert "PFL" in loader.INSTRUMENT_TYPES

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        result = loader.download()
        assert result is True
        assert loader.is_real_data is False

    def test_load_data(self, loader: Any) -> None:
        """Test loading WOD data."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 8  # lat, lon, depth, temp, sal, year, month, day
        assert labels.sum() >= 0  # Some anomalies

    def test_temperature_salinity_profiles(self, loader: Any) -> None:
        """Test that generated profiles have realistic ranges."""
        loader.download()
        features, _ = loader._load_raw()

        # Check temperature range (should be -2 to 35 C roughly)
        temps = features[:, 3]
        assert temps.min() >= -5
        assert temps.max() <= 40

        # Check salinity range (should be 30-40 PSU roughly)
        sal = features[:, 4]
        assert sal.min() >= 25
        assert sal.max() <= 45


class TestCopernicusSeaLevelLoader:
    """Tests for Copernicus sea level data loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="copernicus_sea_level",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
            preprocessing={
                "year_range": (2020, 2023),
                "region": {
                    "lat_min": -60,
                    "lat_max": 60,
                    "lon_min": -180,
                    "lon_max": 180,
                },
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> CopernicusSeaLevelLoader:
        """Create loader instance."""
        return CopernicusSeaLevelLoader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "copernicus_sea_level"
        assert loader.REQUIRES_CREDENTIALS is True
        assert loader.LICENSE == "CC BY 4.0"

    def test_feature_names(self, loader: Any) -> None:
        """Test sea level feature names."""
        expected_features = ["latitude", "longitude", "sla", "adt", "ugos", "vgos"]
        for feat in expected_features:
            assert feat in loader.FEATURE_NAMES

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        result = loader.download()
        assert result is True
        assert loader.is_real_data is False

    def test_load_data(self, loader: Any) -> None:
        """Test loading sea level data."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 8  # lat, lon, sla, adt, ugos, vgos, year, month

    def test_sea_level_anomaly_range(self, loader: Any) -> None:
        """Test SLA values are in realistic range."""
        loader.download()
        features, _ = loader._load_raw()

        # Sea level anomaly should typically be within -0.5 to 0.5 m
        sla = features[:, 2]
        assert sla.min() >= -1.0
        assert sla.max() <= 1.0

    def test_anomaly_labels(self, loader: Any) -> None:
        """Test anomaly labeling based on SLA threshold."""
        loader.download()
        features, labels = loader._load_raw()

        # Anomalies should be significant deviations
        sla = features[:, 2]
        expected_anomalies = np.abs(sla) > 0.15
        # Allow some variance due to velocity-based labels
        assert labels.sum() >= expected_anomalies.sum() * 0.5


class TestClimateDatasetRegistry:
    """Test dataset registry for climate loaders."""

    def test_simons_cmap_registered(self) -> None:
        """Test Simons CMAP is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("simons_cmap")
        assert loader_class is SimonsCMAPLoader

        # Test alias
        alias_class = DatasetRegistry.get("cmap")
        assert alias_class is SimonsCMAPLoader

    def test_wod_registered(self) -> None:
        """Test WOD is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("world_ocean_database")
        assert loader_class is WorldOceanDatabaseLoader

        # Test alias
        alias_class = DatasetRegistry.get("wod")
        assert alias_class is WorldOceanDatabaseLoader

    def test_copernicus_registered(self) -> None:
        """Test Copernicus sea level is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("copernicus_sea_level")
        assert loader_class is CopernicusSeaLevelLoader

        # Test alias
        alias_class = DatasetRegistry.get("sea_level")
        assert alias_class is CopernicusSeaLevelLoader


class TestOceanographicDataQuality:
    """Integration tests for oceanographic data quality."""

    @pytest.fixture
    def cmap_loader(self, tmp_path: Any) -> SimonsCMAPLoader:
        """Create CMAP loader."""
        config = DatasetConfig(
            name="simons_cmap",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=500,
            random_seed=42,
        )
        loader = SimonsCMAPLoader(config)
        loader.download()
        return loader

    def test_depth_distribution(self, cmap_loader: Any) -> None:
        """Test depth values follow expected distribution."""
        features, _ = cmap_loader._load_raw()
        depths = features[:, 2]

        # Most samples should be near surface
        shallow = (depths < 100).sum()
        deep = (depths > 500).sum()
        assert shallow > deep

    def test_temperature_depth_correlation(self, cmap_loader: Any) -> None:
        """Test temperature decreases with depth."""
        features, _ = cmap_loader._load_raw()
        depths = features[:, 2]
        temps = features[:, 3]

        # Calculate correlation
        shallow_temps = temps[depths < 50].mean()
        deep_temps = temps[depths > 200].mean()

        # Surface should be warmer than deep
        assert shallow_temps > deep_temps

    def test_oxygen_minimum_zone(self, cmap_loader: Any) -> None:
        """Test oxygen minimum zone pattern."""
        features, labels = cmap_loader._load_raw()
        oxygen = features[:, 7]

        # Low oxygen should trigger anomaly labels
        low_oxygen_mask = oxygen < 2.0
        if low_oxygen_mask.sum() > 0:
            # Most low-oxygen samples should be labeled as anomalies
            low_oxygen_labels = labels[low_oxygen_mask]
            assert low_oxygen_labels.mean() > 0.5


class TestCopernicusERA5Loader:
    """Tests for Copernicus ERA5 climate reanalysis loader."""

    @pytest.fixture
    def config(self, tmp_path: Any) -> DatasetConfig:
        """Create test configuration."""
        return DatasetConfig(
            name="copernicus_era5",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=100,
            random_seed=42,
            preprocessing={
                "variable_set": "surface",
                "year_range": (2020, 2023),
                "region": {
                    "lat_min": 30,
                    "lat_max": 50,
                    "lon_min": -130,
                    "lon_max": -70,
                },
                "hours": [0, 12],
            },
        )

    @pytest.fixture
    def loader(self, config: Any) -> CopernicusERA5Loader:
        """Create loader instance."""
        return CopernicusERA5Loader(config)

    def test_init(self, loader: Any) -> None:
        """Test loader initialization."""
        assert loader.DATASET_NAME == "copernicus_era5"
        assert loader.REQUIRES_CREDENTIALS is True
        assert loader.LICENSE == "CC BY 4.0"
        assert loader.variable_set == "surface"

    def test_feature_names(self, loader: Any) -> None:
        """Test ERA5 feature names."""
        expected_features = [
            "latitude",
            "longitude",
            "temperature_2m",
            "dewpoint_2m",
            "u_wind_10m",
            "v_wind_10m",
            "pressure",
            "precipitation",
        ]
        for feat in expected_features:
            assert feat in loader.FEATURE_NAMES

    def test_variable_sets(self, loader: Any) -> None:
        """Test variable set options."""
        assert "surface" in loader.VARIABLE_SETS
        assert "radiation" in loader.VARIABLE_SETS
        assert "soil" in loader.VARIABLE_SETS

    def test_synthetic_fallback(self, loader: Any) -> None:
        """Test synthetic data generation."""
        result = loader.download()
        assert result is True
        assert loader.is_real_data is False

    def test_load_data(self, loader: Any) -> None:
        """Test loading ERA5 data."""
        loader.download()
        features, labels = loader._load_raw()

        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert (
            features.shape[1] == 12
        )  # lat, lon, temp, dew, u, v, pres, precip, year, month, day, hour

    def test_temperature_range(self, loader: Any) -> None:
        """Test temperature values are in realistic range."""
        loader.download()
        features, _ = loader._load_raw()

        # 2m temperature in Celsius should be roughly -40 to 50
        temp = features[:, 2]
        assert temp.min() >= -50
        assert temp.max() <= 60

    def test_pressure_range(self, loader: Any) -> None:
        """Test pressure values are in realistic range."""
        loader.download()
        features, _ = loader._load_raw()

        # Surface pressure in hPa should be roughly 850-1050
        pressure = features[:, 6]
        assert pressure.min() >= 850
        assert pressure.max() <= 1100

    def test_wind_components(self, loader: Any) -> None:
        """Test wind components are reasonable."""
        loader.download()
        features, _ = loader._load_raw()

        u_wind = features[:, 4]
        v_wind = features[:, 5]

        # Wind speed should not exceed hurricane force (~70 m/s)
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        assert wind_speed.max() < 100

    def test_anomaly_detection(self, loader: Any) -> None:
        """Test anomaly labeling for climate extremes."""
        loader.download()
        features, labels = loader._load_raw()

        # Should detect some anomalies
        assert labels.sum() > 0
        # But not all should be anomalies
        assert labels.mean() < 0.5

    def test_preprocess(self, loader: Any) -> None:
        """Test preprocessing normalizes data."""
        loader.download()
        features, _ = loader._load_raw()

        processed = loader.preprocess(features)

        assert processed.dtype == np.float32
        # Check normalization
        assert np.abs(processed.mean()) < 0.5


class TestERA5DatasetRegistry:
    """Test dataset registry for ERA5 loader."""

    def test_era5_registered(self) -> None:
        """Test ERA5 is registered."""
        from omni_mercury_engine.datasets import DatasetRegistry

        loader_class = DatasetRegistry.get("copernicus_era5")
        assert loader_class is CopernicusERA5Loader

        # Test alias
        alias_class = DatasetRegistry.get("era5")
        assert alias_class is CopernicusERA5Loader

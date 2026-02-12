"""
Mercury Agent ♱ — Live Data Loader Tests

Tests each loader against real endpoints.
Mark with @pytest.mark.network for optional CI skipping.

Usage:
    pytest tests/test_loaders_live.py -m network
    pytest tests/test_loaders_live.py -m "not network"  # skip network tests
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

# Only run these tests when explicitly requested or in network-capable environments
pytestmark = [pytest.mark.network, pytest.mark.slow]


def _get_loader_config(name: str) -> "DatasetConfig":
    """Create a minimal DatasetConfig for testing."""
    from omni_mercury_engine.datasets.base import DatasetConfig
    return DatasetConfig(name=name, max_samples=500)


class TestFEMADisasterLive:
    """Test FEMA Disaster loader against live API."""

    def test_fema_download_and_load(self) -> None:
        from omni_mercury_engine.datasets.disaster import FEMADisasterLoader
        config = _get_loader_config("fema_disaster")
        loader = FEMADisasterLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0
        assert labels.shape[0] == features.shape[0]
        assert labels.sum() > 0  # Some anomalies present
        assert loader.is_real_data is True


class TestUSGSEarthquakeLive:
    """Test USGS Earthquake loader against live API."""

    def test_earthquake_download_and_load(self) -> None:
        from omni_mercury_engine.datasets.environmental import USGSEarthquakeLoader
        config = _get_loader_config("earthquake")
        loader = USGSEarthquakeLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0
        assert features.shape[1] >= 11  # 11 features expected


class TestNOAABuoyLive:
    """Test NOAA Buoy loader against live API."""

    def test_buoy_download_and_load(self) -> None:
        from omni_mercury_engine.datasets.ocean import NOAABuoyLoader
        config = _get_loader_config("noaa_buoy")
        loader = NOAABuoyLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0


class TestADBenchLive:
    """Test ADBench loader against GitHub."""

    def test_adbench_fraud(self) -> None:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader
        from omni_mercury_engine.datasets.base import DatasetConfig
        config = DatasetConfig(name="adbench", preprocessing={"dataset": "fraud"})
        loader = ADBenchLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0
        assert labels.sum() > 0
        sha = hashlib.sha256(features.tobytes()).hexdigest()
        assert len(sha) == 64

    def test_adbench_thyroid(self) -> None:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader
        from omni_mercury_engine.datasets.base import DatasetConfig
        config = DatasetConfig(name="adbench", preprocessing={"dataset": "thyroid"})
        loader = ADBenchLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0


class TestNASAExoplanetLive:
    """Test NASA Exoplanet Archive loader."""

    def test_exoplanet_download(self) -> None:
        from omni_mercury_engine.datasets.space import NASAExoplanetLoader
        config = _get_loader_config("nasa_exoplanet")
        loader = NASAExoplanetLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0


class TestBATADALLive:
    """Test BATADAL loader."""

    def test_batadal_download(self) -> None:
        from omni_mercury_engine.datasets.industrial import BATADALLoader
        config = _get_loader_config("batadal")
        loader = BATADALLoader(config)
        result = loader.download()
        assert result is True
        features, labels = loader.load()
        assert features.shape[0] > 0
        # Verify 43 SCADA sensor columns (or close to it)
        assert features.shape[1] >= 40


class TestCredentialGatedStubs:
    """Test that credential-gated loaders raise descriptive errors."""

    def test_swat_raises(self) -> None:
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.datasets.industrial import SWaTLoader
        config = _get_loader_config("swat")
        loader = SWaTLoader(config)
        with pytest.raises(DataSourceUnavailableError, match="iTrust"):
            loader.download()

    def test_wadi_raises(self) -> None:
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.datasets.industrial import WADILoader
        config = _get_loader_config("wadi")
        loader = WADILoader(config)
        with pytest.raises(DataSourceUnavailableError, match="iTrust"):
            loader.download()

    def test_seti_deprecated(self) -> None:
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.datasets.space import SETILoader
        config = _get_loader_config("seti")
        loader = SETILoader(config)
        with pytest.raises(DataSourceUnavailableError, match="deprecated"):
            loader.download()


class TestNoSyntheticDefault:
    """Verify that no loader silently returns synthetic data."""

    def test_mimic_raises_without_synthetic_flag(self) -> None:
        from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
        from omni_mercury_engine.datasets.medical import MIMICLoader
        config = _get_loader_config("mimic")
        loader = MIMICLoader(config)
        # Should raise because synthetic not allowed by default
        with pytest.raises(DataSourceUnavailableError):
            loader.download()

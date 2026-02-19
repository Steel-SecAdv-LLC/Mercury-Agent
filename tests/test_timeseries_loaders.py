"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for time-series dataset loaders.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.timeseries import (
    NABLoader,
    SMAPMSLLoader,
    SMDLoader,
)


class TestNABLoader:
    """Tests for NAB (Numenta Anomaly Benchmark) loader."""

    @pytest.fixture
    def nab_loader(self, tmp_path):
        """Create NABLoader instance."""
        config = DatasetConfig(
            name="NAB",
            data_dir=str(tmp_path),
        )
        return NABLoader(config)

    def test_initialization(self, nab_loader):
        """Should initialize with correct attributes."""
        assert nab_loader.config.name == "NAB"
        assert nab_loader.NAB_DATA_URL is not None
        assert nab_loader.NAB_LABELS_URL is not None

    def test_available_files(self, nab_loader):
        """Should have NAB data files."""
        assert hasattr(nab_loader, "NAB_FILES")
        assert len(nab_loader.NAB_FILES) > 0
        assert "realKnownCause" in nab_loader.NAB_FILES

    def test_dataset_name(self, nab_loader):
        """Should have correct dataset name."""
        assert nab_loader.DATASET_NAME == "nab"

    def test_download_method_exists(self, nab_loader):
        """NAB loader should have download method."""
        assert hasattr(nab_loader, "download")
        assert callable(nab_loader.download)

    def test_no_synthetic_fallback(self, nab_loader):
        """NAB loader should NOT have synthetic fallback."""
        synthetic_methods = [
            m for m in dir(nab_loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

    def test_real_url_references(self, nab_loader):
        """Should reference real GitHub URLs."""
        assert "numenta/NAB" in nab_loader.NAB_DATA_URL
        assert "numenta/NAB" in nab_loader.NAB_LABELS_URL


class TestSMDLoader:
    """Tests for SMD (Server Machine Dataset) loader."""

    @pytest.fixture
    def smd_loader(self, tmp_path):
        """Create SMDLoader instance."""
        config = DatasetConfig(
            name="SMD",
            data_dir=str(tmp_path),
        )
        return SMDLoader(config)

    def test_initialization(self, smd_loader):
        """Should initialize with correct attributes."""
        assert smd_loader.config.name == "SMD"
        assert smd_loader.SMD_BASE_URL is not None

    def test_dataset_name(self, smd_loader):
        """Should have correct dataset name."""
        assert smd_loader.DATASET_NAME == "smd"

    def test_machine_list(self, smd_loader):
        """Should have list of machine IDs."""
        assert hasattr(smd_loader, "MACHINES")
        assert len(smd_loader.MACHINES) > 0

    def test_download_method_exists(self, smd_loader):
        """SMD loader should have download method."""
        assert hasattr(smd_loader, "download")
        assert callable(smd_loader.download)

    def test_no_synthetic_fallback(self, smd_loader):
        """SMD loader should NOT have synthetic fallback."""
        synthetic_methods = [
            m for m in dir(smd_loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

    def test_real_url_references(self, smd_loader):
        """Should reference OmniAnomaly repository."""
        assert "OmniAnomaly" in smd_loader.SMD_BASE_URL


class TestSMAPMSLLoader:
    """Tests for SMAP/MSL (NASA telemetry) loader."""

    @pytest.fixture
    def smap_loader(self, tmp_path):
        """Create SMAP loader instance."""
        config = DatasetConfig(
            name="SMAP",
            data_dir=str(tmp_path),
            preprocessing={"dataset": "SMAP"},
        )
        return SMAPMSLLoader(config)

    @pytest.fixture
    def msl_loader(self, tmp_path):
        """Create MSL loader instance."""
        config = DatasetConfig(
            name="MSL",
            data_dir=str(tmp_path),
            preprocessing={"dataset": "MSL"},
        )
        return SMAPMSLLoader(config)

    def test_smap_initialization(self, smap_loader):
        """SMAP should initialize correctly."""
        assert smap_loader.config.name == "SMAP"
        assert smap_loader.dataset == "SMAP"

    def test_msl_initialization(self, msl_loader):
        """MSL should initialize correctly."""
        assert msl_loader.config.name == "MSL"
        assert msl_loader.dataset == "MSL"

    def test_dataset_name(self, smap_loader):
        """Should have correct dataset name."""
        assert smap_loader.DATASET_NAME == "smap_msl"

    def test_download_method_exists(self, smap_loader):
        """SMAP/MSL loader should have download method."""
        assert hasattr(smap_loader, "download")
        assert callable(smap_loader.download)

    def test_no_synthetic_fallback(self, smap_loader):
        """SMAP/MSL loader should NOT have synthetic fallback."""
        synthetic_methods = [
            m for m in dir(smap_loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

    def test_real_url_references(self, smap_loader):
        """Should reference telemanom repository."""
        assert "telemanom" in smap_loader.DATASET_URL


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""

    def test_all_loaders_have_download(self, tmp_path):
        """All loaders should implement download method."""
        config = DatasetConfig(name="test", data_dir=str(tmp_path))

        loaders = [
            NABLoader(config),
            SMDLoader(config),
            SMAPMSLLoader(
                DatasetConfig(
                    name="SMAP", data_dir=str(tmp_path), preprocessing={"dataset": "SMAP"}
                )
            ),
        ]

        for loader in loaders:
            assert hasattr(loader, "download")
            assert callable(loader.download)

    def test_all_loaders_have_citation(self, tmp_path):
        """All loaders should have citation info."""
        config = DatasetConfig(name="test", data_dir=str(tmp_path))

        loaders = [
            NABLoader(config),
            SMDLoader(config),
            SMAPMSLLoader(
                DatasetConfig(
                    name="SMAP", data_dir=str(tmp_path), preprocessing={"dataset": "SMAP"}
                )
            ),
        ]

        for loader in loaders:
            assert hasattr(loader, "CITATION")
            assert len(loader.CITATION) > 0

    def test_loaders_use_real_urls(self, tmp_path):
        """Loaders should reference real data sources, not synthetic."""
        nab = NABLoader(DatasetConfig(name="NAB", data_dir=str(tmp_path)))
        assert "numenta/NAB" in nab.NAB_DATA_URL

        smd = SMDLoader(DatasetConfig(name="SMD", data_dir=str(tmp_path)))
        assert "OmniAnomaly" in smd.SMD_BASE_URL


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = DatasetConfig(name="test")
        assert config.download is True
        assert config.data_dir == "./data"
        assert config.random_seed == 42

    def test_custom_values(self):
        """Should accept custom values."""
        config = DatasetConfig(
            name="custom",
            data_dir="/custom/path",
            download=False,
            random_seed=123,
        )
        assert config.download is False
        assert config.data_dir == "/custom/path"
        assert config.random_seed == 123

    def test_split_ratios(self):
        """Split ratios should sum to 1.0."""
        config = DatasetConfig(name="test")
        assert abs(sum(config.split_ratios) - 1.0) < 1e-6

    def test_preprocessing_dict(self):
        """Should accept preprocessing config."""
        config = DatasetConfig(name="test", preprocessing={"normalize": True, "window_size": 100})
        assert config.preprocessing["normalize"] is True
        assert config.preprocessing["window_size"] == 100


class TestRealDataRequirement:
    """Tests ensuring loaders require real data."""

    def test_nab_no_fallback(self, tmp_path):
        """NAB should not silently fall back to synthetic data."""
        config = DatasetConfig(name="NAB", data_dir=str(tmp_path))
        loader = NABLoader(config)

        # Check there's no _create_synthetic or _generate_fake method
        synthetic_methods = [
            m for m in dir(loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

    def test_smd_no_fallback(self, tmp_path):
        """SMD should not silently fall back to synthetic data."""
        config = DatasetConfig(name="SMD", data_dir=str(tmp_path))
        loader = SMDLoader(config)

        synthetic_methods = [
            m for m in dir(loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

    def test_smap_no_fallback(self, tmp_path):
        """SMAP should not silently fall back to synthetic data."""
        config = DatasetConfig(
            name="SMAP", data_dir=str(tmp_path), preprocessing={"dataset": "SMAP"}
        )
        loader = SMAPMSLLoader(config)

        synthetic_methods = [
            m for m in dir(loader) if "synthetic" in m.lower() or "fake" in m.lower()
        ]
        assert len(synthetic_methods) == 0

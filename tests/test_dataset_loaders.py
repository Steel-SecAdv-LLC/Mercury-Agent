"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Dataset Loaders

Comprehensive tests for industrial and UCR archive dataset loaders.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig, DatasetSplit


class TestDatasetImports:
    """Test that all dataset loaders can be imported."""

    def test_import_industrial_loaders(self):
        """Test importing industrial loaders."""
        from omni_mercury_engine.datasets.industrial import (
            BATADALLoader,
            SWaTLoader,
            WADILoader,
        )

        assert SWaTLoader is not None
        assert WADILoader is not None
        assert BATADALLoader is not None

    def test_import_ucr_loaders(self):
        """Test importing UCR archive loaders."""
        from omni_mercury_engine.datasets.ucr_archive import (
            CWRUBearingLoader,
            MBALoader,
            MSDSLoader,
            UCRLoader,
        )

        assert UCRLoader is not None
        assert MBALoader is not None
        assert CWRUBearingLoader is not None
        assert MSDSLoader is not None

    def test_import_from_package(self):
        """Test importing from main datasets package."""
        from omni_mercury_engine.datasets import (
            BATADALLoader,
            CWRUBearingLoader,
            MBALoader,
            MSDSLoader,
            SWaTLoader,
            UCRLoader,
            WADILoader,
        )

        # Verify all are accessible
        loaders = [
            SWaTLoader,
            WADILoader,
            BATADALLoader,
            UCRLoader,
            MBALoader,
            CWRUBearingLoader,
            MSDSLoader,
        ]
        assert all(loader is not None for loader in loaders)


class TestSWaTLoader:
    """Tests for SWaT (Secure Water Treatment) loader."""

    def test_swat_loader_init(self):
        """Test SWaT loader initialization."""
        from omni_mercury_engine.datasets.industrial import SWaTLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="swat",
                data_dir=str(tmpdir),
            )
            loader = SWaTLoader(config)

            assert loader.DATASET_NAME == "swat"
            assert loader.NUM_FEATURES == 51
            assert loader.ATTACK_COUNT == 36
            assert loader.REQUIRES_CREDENTIALS is True

    def test_swat_metadata(self):
        """Test SWaT metadata generation."""
        from omni_mercury_engine.datasets.industrial import SWaTLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="swat", data_dir=str(tmpdir))
            loader = SWaTLoader(config)

            metadata = loader.get_metadata()
            assert metadata.name == "SWaT"
            assert metadata.num_features == 51
            assert "Normal" in metadata.target_names
            assert "Attack" in metadata.target_names

    def test_swat_attack_count(self):
        """Test SWaT attack count."""
        from omni_mercury_engine.datasets.industrial import SWaTLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="swat", data_dir=str(tmpdir))
            loader = SWaTLoader(config)

            # SWaT has 36 documented attack scenarios
            assert loader.ATTACK_COUNT == 36


class TestWADILoader:
    """Tests for WADI (Water Distribution) loader."""

    def test_wadi_loader_init(self):
        """Test WADI loader initialization."""
        from omni_mercury_engine.datasets.industrial import WADILoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="wadi",
                data_dir=str(tmpdir),
            )
            loader = WADILoader(config)

            assert loader.DATASET_NAME == "wadi"
            assert loader.NUM_FEATURES == 123
            assert loader.ATTACK_COUNT == 15

    def test_wadi_metadata(self):
        """Test WADI metadata generation."""
        from omni_mercury_engine.datasets.industrial import WADILoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="wadi", data_dir=str(tmpdir))
            loader = WADILoader(config)

            metadata = loader.get_metadata()
            assert metadata.name == "WADI"
            assert metadata.num_features == 123


class TestBATADALLoader:
    """Tests for BATADAL loader."""

    def test_batadal_loader_init(self):
        """Test BATADAL loader initialization."""
        from omni_mercury_engine.datasets.industrial import BATADALLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="batadal",
                data_dir=str(tmpdir),
            )
            loader = BATADALLoader(config)

            assert loader.DATASET_NAME == "batadal"
            assert loader.NUM_FEATURES == 43
            assert loader.REQUIRES_CREDENTIALS is False

    def test_batadal_download_url(self):
        """Test BATADAL has valid download URL."""
        from omni_mercury_engine.datasets.industrial import BATADALLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="batadal", data_dir=str(tmpdir))
            loader = BATADALLoader(config)

            assert "batadal.net" in loader.DATASET_URL


class TestUCRLoader:
    """Tests for UCR Time Series Archive loader."""

    def test_ucr_loader_init(self):
        """Test UCR loader initialization."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="ucr", data_dir=str(tmpdir), preprocessing={"dataset_name": "ECG5000"}
            )
            loader = UCRLoader(config)

            assert loader.DATASET_NAME == "ucr"
            assert loader.dataset_name == "ECG5000"

    def test_ucr_default_dataset(self):
        """Test UCR default dataset is ECG5000."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="ucr", data_dir=str(tmpdir))
            loader = UCRLoader(config)

            assert loader.dataset_name == "ECG5000"

    def test_ucr_popular_datasets(self):
        """Test UCR popular datasets list."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="ucr", data_dir=str(tmpdir))
            loader = UCRLoader(config)

            datasets = loader.list_available_datasets()
            assert "ECG5000" in datasets
            assert "Wafer" in datasets
            assert "FordA" in datasets
            assert len(datasets) == 10  # POPULAR_DATASETS has 10 items

    def test_ucr_anomaly_label_conversion(self):
        """Test converting classification labels to anomaly labels."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="ucr", data_dir=str(tmpdir))
            loader = UCRLoader(config)

            # Simulate class labels (majority class 0, minority class 1)
            labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

            anomaly_labels = loader.convert_to_anomaly_labels(labels)

            # Class 1 should be anomaly (minority)
            assert anomaly_labels.sum() == 2
            assert (anomaly_labels == 1).sum() == 2

    def test_ucr_anomaly_specific_class(self):
        """Test specifying which class is anomaly."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="ucr", data_dir=str(tmpdir))
            loader = UCRLoader(config)

            labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])

            # Treat class 0 as anomaly
            anomaly_labels = loader.convert_to_anomaly_labels(labels, anomaly_class=0)

            assert anomaly_labels.sum() == 3  # Three samples of class 0


class TestMBALoader:
    """Tests for MBA (Machine Bearing Anomaly) / CWRU loader."""

    def test_mba_loader_init(self):
        """Test MBA loader initialization."""
        from omni_mercury_engine.datasets.ucr_archive import MBALoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="mba",
                data_dir=str(tmpdir),
            )
            loader = MBALoader(config)

            assert loader.DATASET_NAME == "mba"
            assert loader.SAMPLE_RATE == 12000  # 12kHz

    def test_mba_fault_types(self):
        """Test MBA fault types."""
        from omni_mercury_engine.datasets.ucr_archive import MBALoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="mba", data_dir=str(tmpdir))
            loader = MBALoader(config)

            assert "Normal" in loader.FAULT_TYPES
            assert "Inner_Race" in loader.FAULT_TYPES
            assert "Outer_Race" in loader.FAULT_TYPES
            assert "Ball" in loader.FAULT_TYPES

    def test_mba_metadata(self):
        """Test MBA metadata generation."""
        from omni_mercury_engine.datasets.ucr_archive import MBALoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="mba", data_dir=str(tmpdir))
            loader = MBALoader(config)

            metadata = loader.get_metadata()
            assert metadata.name == "MBA (CWRU Bearing)"
            assert metadata.num_features == 1024  # Window size
            assert "windowing" in metadata.preprocessing_applied


class TestCWRUBearingLoader:
    """Tests for CWRU Bearing loader (alias for MBA)."""

    def test_cwru_is_mba_alias(self):
        """Test that CWRUBearingLoader is an alias for MBALoader."""
        from omni_mercury_engine.datasets.ucr_archive import (
            CWRUBearingLoader,
            MBALoader,
        )

        assert issubclass(CWRUBearingLoader, MBALoader)
        assert CWRUBearingLoader.DATASET_NAME == "cwru_bearing"


class TestMSDSLoader:
    """Tests for MSDS (Multi-Source Data Stream) loader."""

    def test_msds_loader_init(self):
        """Test MSDS loader initialization."""
        from omni_mercury_engine.datasets.ucr_archive import MSDSLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="msds",
                data_dir=str(tmpdir),
                preprocessing={"n_sources": 4, "n_samples": 5000, "anomaly_ratio": 0.03},
            )
            loader = MSDSLoader(config)

            assert loader.DATASET_NAME == "msds"
            assert loader.n_sources == 4
            assert loader.n_samples == 5000
            assert loader.anomaly_ratio == 0.03

    def test_msds_default_params(self):
        """Test MSDS default parameters."""
        from omni_mercury_engine.datasets.ucr_archive import MSDSLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="msds", data_dir=str(tmpdir))
            loader = MSDSLoader(config)

            assert loader.n_sources == 3
            assert loader.n_samples == 10000
            assert loader.anomaly_ratio == 0.05

    def test_msds_synthetic_generation(self):
        """Test MSDS synthetic data generation."""
        from omni_mercury_engine.datasets.ucr_archive import MSDSLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="msds",
                data_dir=str(tmpdir),
                preprocessing={"n_sources": 2, "n_samples": 1000, "anomaly_ratio": 0.1},
            )
            loader = MSDSLoader(config)

            # Generate/load data
            features, labels = loader.load()

            # Check shapes
            assert features.shape[0] == 1000  # n_samples
            assert features.shape[1] == 20  # n_sources * 10 features
            assert labels.shape[0] == 1000

            # Check anomaly ratio (should be close to 10%)
            actual_ratio = labels.mean()
            assert 0.08 <= actual_ratio <= 0.12

    def test_msds_metadata(self):
        """Test MSDS metadata generation."""
        from omni_mercury_engine.datasets.ucr_archive import MSDSLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="msds", data_dir=str(tmpdir), preprocessing={"n_sources": 3}
            )
            loader = MSDSLoader(config)

            metadata = loader.get_metadata()
            assert metadata.name == "MSDS"
            assert metadata.num_features == 30  # 3 sources * 10 features
            assert "synthetic_generation" in metadata.preprocessing_applied


class TestBaselineResults:
    """Tests for baseline results in baselines.py."""

    def test_new_datasets_in_baselines(self):
        """Test that new datasets are in baseline results."""
        from omni_mercury_engine.evaluation.baselines import BASELINE_RESULTS

        # Check all new datasets are present
        assert "SWaT" in BASELINE_RESULTS
        assert "WADI" in BASELINE_RESULTS
        assert "UCR" in BASELINE_RESULTS
        assert "MBA" in BASELINE_RESULTS
        assert "MSDS" in BASELINE_RESULTS

    def test_tranad_results(self):
        """Test TranAD results for new datasets."""
        from omni_mercury_engine.evaluation.baselines import BASELINE_RESULTS

        # SWaT
        swat_tranad = BASELINE_RESULTS["SWaT"]["TranAD"]
        assert swat_tranad["f1"] == pytest.approx(0.8151, rel=0.01)

        # WADI
        wadi_tranad = BASELINE_RESULTS["WADI"]["TranAD"]
        assert wadi_tranad["f1"] == pytest.approx(0.4951, rel=0.01)

        # UCR
        ucr_tranad = BASELINE_RESULTS["UCR"]["TranAD"]
        assert ucr_tranad["f1"] == pytest.approx(0.9694, rel=0.01)

        # MBA
        mba_tranad = BASELINE_RESULTS["MBA"]["TranAD"]
        assert mba_tranad["f1"] > 0.98  # Very high for bearing data

        # MSDS
        msds_tranad = BASELINE_RESULTS["MSDS"]["TranAD"]
        assert msds_tranad["f1"] == pytest.approx(0.9262, rel=0.01)

    def test_baseline_citations(self):
        """Test baseline citations include new methods."""
        from omni_mercury_engine.evaluation.baselines import get_baseline_citations

        citations = get_baseline_citations()
        assert "GDN" in citations
        assert "USAD" in citations
        assert "MAAT" in citations

    def test_compare_to_baselines_new_datasets(self):
        """Test compare_to_baselines works with new datasets."""
        from omni_mercury_engine.evaluation.baselines import compare_to_baselines

        # Test with SWaT
        result = compare_to_baselines(
            dataset="SWaT", your_precision=0.85, your_recall=0.87, your_f1=0.86
        )
        assert result.dataset == "SWaT"
        assert result.your_f1 == 0.86
        assert result.best_baseline == "TranAD"

        # Test with MSDS
        result = compare_to_baselines(
            dataset="MSDS", your_precision=0.95, your_recall=0.96, your_f1=0.955
        )
        assert result.rank == 1  # Should beat TranAD (0.9262)

    def test_sota_for_new_datasets(self):
        """Test getting SOTA for new datasets."""
        from omni_mercury_engine.evaluation.baselines import get_sota_for_dataset

        # TranAD should be SOTA for all new datasets
        for dataset in ["SWaT", "WADI", "UCR", "MBA", "MSDS"]:
            sota_name, sota_metrics = get_sota_for_dataset(dataset)
            assert sota_name == "TranAD"
            assert "f1" in sota_metrics


class TestDatasetSplits:
    """Tests for dataset split functionality."""

    def test_msds_split_all(self):
        """Test MSDS ALL split."""
        from omni_mercury_engine.datasets.ucr_archive import MSDSLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(
                name="msds", data_dir=str(tmpdir), preprocessing={"n_samples": 500}
            )
            loader = MSDSLoader(config)

            features, labels = loader.load(split=DatasetSplit.ALL)
            assert features.shape[0] == 500


class TestDatasetRegistry:
    """Tests for dataset registry with new loaders."""

    def test_registry_contains_new_loaders(self):
        """Test that registry exports all new loaders."""
        from omni_mercury_engine import datasets

        # Check all new loaders are in __all__
        assert "SWaTLoader" in datasets.__all__
        assert "WADILoader" in datasets.__all__
        assert "BATADALLoader" in datasets.__all__
        assert "UCRLoader" in datasets.__all__
        assert "MBALoader" in datasets.__all__
        assert "CWRUBearingLoader" in datasets.__all__
        assert "MSDSLoader" in datasets.__all__


class TestIndustrialDatasetDomains:
    """Tests for industrial dataset domain-specific features."""

    def test_swat_feature_groups(self):
        """Test SWaT loader exposes feature groups."""
        from omni_mercury_engine.datasets.industrial import SWaTLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="swat", data_dir=str(tmpdir))
            loader = SWaTLoader(config)

            feature_groups = loader.FEATURE_GROUPS
            assert "P1" in feature_groups  # Process 1
            assert "P2" in feature_groups  # Process 2
            # Each process should have sensor/actuator lists

    def test_wadi_stages(self):
        """Test WADI loader exposes stage information."""
        from omni_mercury_engine.datasets.industrial import WADILoader

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(name="wadi", data_dir=str(tmpdir))
            loader = WADILoader(config)

            # WADI has multiple stages
            assert hasattr(loader, "STAGES") or loader.NUM_FEATURES == 123


class TestUCRDatasetVariants:
    """Tests for UCR dataset variants."""

    def test_ucr_different_datasets(self):
        """Test loading different UCR datasets."""
        from omni_mercury_engine.datasets.ucr_archive import UCRLoader

        datasets_to_test = ["ECG5000", "Wafer", "FordA"]

        for ds_name in datasets_to_test:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = DatasetConfig(
                    name="ucr", data_dir=str(tmpdir), preprocessing={"dataset_name": ds_name}
                )
                loader = UCRLoader(config)

                assert loader.dataset_name == ds_name
                metadata = loader.get_metadata()
                assert ds_name in metadata.name

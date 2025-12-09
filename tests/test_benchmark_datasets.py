"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""
Tests for Benchmark Dataset Loaders.

Tests MVTec AD, UCF-Crime, and Shanghai Tech Campus dataset loaders.
"""


import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestBaseImageDataset:
    """Tests for BaseImageDataset class."""

    def test_base_dataset_interface(self):
        """Test BaseImageDataset has required interface."""
        from omni_anomaly_engine.data.benchmarks.base_dataset import BaseImageDataset

        # BaseImageDataset is abstract, check interface exists
        assert hasattr(BaseImageDataset, "__getitem__")
        assert hasattr(BaseImageDataset, "__len__")


class TestBaseVideoDataset:
    """Tests for BaseVideoDataset class."""

    def test_base_video_interface(self):
        """Test BaseVideoDataset has required interface."""
        from omni_anomaly_engine.data.benchmarks.base_dataset import BaseVideoDataset

        assert hasattr(BaseVideoDataset, "__getitem__")
        assert hasattr(BaseVideoDataset, "__len__")


class TestMVTecADConfig:
    """Tests for MVTec AD configuration."""

    def test_default_config(self):
        """Test default MVTec AD config."""
        from omni_anomaly_engine.data.benchmarks.mvtec import MVTecADConfig

        config = MVTecADConfig()
        assert config.image_size == (224, 224)
        assert config.normalize is True

    def test_custom_config(self):
        """Test custom MVTec AD config."""
        from omni_anomaly_engine.data.benchmarks.mvtec import MVTecADConfig

        config = MVTecADConfig(
            root="./data/mvtec",
            category="bottle",
            image_size=(256, 256),
            normalize=False,
        )
        assert config.category == "bottle"
        assert config.image_size == (256, 256)


class TestMVTecADDataset:
    """Tests for MVTec AD dataset loader."""

    def test_mvtec_initialization(self, tmp_path):
        """Test MVTec AD dataset initialization."""
        from omni_anomaly_engine.data.benchmarks import MVTecADDataset
        from omni_anomaly_engine.data.benchmarks.mvtec import MVTecADConfig

        config = MVTecADConfig(root=str(tmp_path), category="bottle")
        dataset = MVTecADDataset(config=config)
        assert dataset is not None

    def test_mvtec_categories(self):
        """Test MVTec AD available categories."""
        from omni_anomaly_engine.data.benchmarks import MVTecADDataset

        categories = MVTecADDataset.get_categories()
        assert "bottle" in categories
        assert "cable" in categories
        assert "carpet" in categories
        assert len(categories) == 15

    def test_mvtec_config_from_dict(self, tmp_path):
        """Test MVTec AD config from dictionary."""
        from omni_anomaly_engine.data.benchmarks import MVTecADDataset

        config = {"root": str(tmp_path), "category": "cable"}
        dataset = MVTecADDataset(config=config)
        assert dataset.config.category == "cable"


class TestUCFCrimeConfig:
    """Tests for UCF-Crime configuration."""

    def test_default_config(self):
        """Test default UCF-Crime config."""
        from omni_anomaly_engine.data.benchmarks.ucf_crime import UCFCrimeConfig

        config = UCFCrimeConfig()
        assert config.frame_size == (224, 224)
        assert config.clip_length == 16

    def test_custom_config(self):
        """Test custom UCF-Crime config."""
        from omni_anomaly_engine.data.benchmarks.ucf_crime import UCFCrimeConfig

        config = UCFCrimeConfig(
            root="./data/ucf",
            split="test",
            clip_length=32,
            fps=15.0,
        )
        assert config.split == "test"
        assert config.clip_length == 32


class TestUCFCrimeDataset:
    """Tests for UCF-Crime dataset loader."""

    def test_ucf_initialization(self, tmp_path):
        """Test UCF-Crime dataset initialization."""
        from omni_anomaly_engine.data.benchmarks import UCFCrimeDataset
        from omni_anomaly_engine.data.benchmarks.ucf_crime import UCFCrimeConfig

        config = UCFCrimeConfig(root=str(tmp_path))
        dataset = UCFCrimeDataset(config=config)
        assert dataset is not None

    def test_ucf_anomaly_classes(self):
        """Test UCF-Crime anomaly classes."""
        from omni_anomaly_engine.data.benchmarks import UCFCrimeDataset

        classes = UCFCrimeDataset.get_anomaly_classes()
        assert "Abuse" in classes
        assert "Arrest" in classes
        assert "Assault" in classes
        assert len(classes) == 13


class TestShanghaiTechConfig:
    """Tests for Shanghai Tech Campus configuration."""

    def test_default_config(self):
        """Test default Shanghai Tech config."""
        from omni_anomaly_engine.data.benchmarks.shanghai_tech import ShanghaiTechConfig

        config = ShanghaiTechConfig()
        assert config.frame_size == (224, 224)

    def test_custom_config(self):
        """Test custom Shanghai Tech config."""
        from omni_anomaly_engine.data.benchmarks.shanghai_tech import ShanghaiTechConfig

        config = ShanghaiTechConfig(
            root="./data/shanghai",
            split="test",
            campus="campus1",
        )
        assert config.split == "test"
        assert config.campus == "campus1"


class TestShanghaiTechDataset:
    """Tests for Shanghai Tech Campus dataset loader."""

    def test_shanghai_initialization(self, tmp_path):
        """Test Shanghai Tech dataset initialization."""
        from omni_anomaly_engine.data.benchmarks import ShanghaiTechDataset
        from omni_anomaly_engine.data.benchmarks.shanghai_tech import ShanghaiTechConfig

        config = ShanghaiTechConfig(root=str(tmp_path))
        dataset = ShanghaiTechDataset(config=config)
        assert dataset is not None


class TestBenchmarkModuleImports:
    """Tests for benchmark module imports."""

    def test_module_imports(self):
        """Test all benchmark datasets can be imported."""
        from omni_anomaly_engine.data.benchmarks import (
            BaseImageDataset,
            BaseVideoDataset,
            MVTecADDataset,
            ShanghaiTechDataset,
            UCFCrimeDataset,
        )

        assert MVTecADDataset is not None
        assert UCFCrimeDataset is not None
        assert ShanghaiTechDataset is not None
        assert BaseImageDataset is not None
        assert BaseVideoDataset is not None

    def test_module_all_exports(self):
        """Test __all__ exports."""
        from omni_anomaly_engine.data import benchmarks

        assert "MVTecADDataset" in benchmarks.__all__
        assert "UCFCrimeDataset" in benchmarks.__all__
        assert "ShanghaiTechDataset" in benchmarks.__all__


class TestDatasetTransforms:
    """Tests for dataset transform utilities."""

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_default_transforms(self):
        """Test default image transforms."""
        from omni_anomaly_engine.data.benchmarks.base_dataset import get_default_transforms

        transform = get_default_transforms(image_size=(224, 224))
        assert transform is not None

        # Test transform on dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        from PIL import Image

        pil_image = Image.fromarray(dummy_image)
        transformed = transform(pil_image)

        assert transformed.shape == (3, 224, 224)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_no_normalize_transforms(self):
        """Test transforms without normalization."""
        from omni_anomaly_engine.data.benchmarks.base_dataset import get_default_transforms

        transform = get_default_transforms(image_size=(224, 224), normalize=False)
        assert transform is not None

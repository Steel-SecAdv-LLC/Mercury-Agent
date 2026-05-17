"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Tests for SOTA Visual Anomaly Detection modules.

Tests PatchCore, PaDiM, STFPM, Reverse Distillation, and CFlow detectors.
"""

import importlib.util
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    # Imported only for cast() type narrowing; the runtime classes pull in torch,
    # so keeping them under TYPE_CHECKING avoids importing torch at test-collection
    # time even when the suite is skipped for missing torch/torchvision.
    # PatchCoreConfig is omitted because test_patchcore_config constructs it at
    # runtime, so its import lives inside that function.
    from omni_mercury_engine.detectors.visual.cflow import CFlowConfig
    from omni_mercury_engine.detectors.visual.padim import PaDiMConfig
    from omni_mercury_engine.detectors.visual.reverse_distillation import (
        ReverseDistillationConfig,
    )

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


pytestmark = pytest.mark.visual


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestPatchCoreDetector:
    """Tests for PatchCore anomaly detector."""

    def test_patchcore_initialization(self) -> None:
        """Test PatchCore can be initialized with default config."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        detector = PatchCoreDetector()
        assert detector is not None
        assert detector.config is not None
        assert detector.config.backbone == "resnet18"

    def test_patchcore_config(self) -> None:
        """Test PatchCore with custom config."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector
        from omni_mercury_engine.detectors.visual.patchcore import PatchCoreConfig

        config = PatchCoreConfig(
            backbone="resnet18",
            coreset_ratio=0.1,
            k_nearest=3,
        )
        detector = PatchCoreDetector(config=config)
        # detector.config is typed as base VisualDetectorConfig but is actually
        # the PatchCoreConfig subclass at runtime.
        cfg = cast("PatchCoreConfig", detector.config)
        assert cfg.coreset_ratio == 0.1
        assert cfg.k_nearest == 3

    def test_patchcore_fit(self, sample_image_batch) -> None:
        """Test PatchCore fitting on normal images."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        detector = PatchCoreDetector()
        detector.fit(sample_image_batch)
        assert detector._is_fitted
        assert detector.memory_bank is not None

    @pytest.mark.slow
    def test_patchcore_detect(self, sample_image_batch, sample_image) -> None:
        """Test PatchCore anomaly detection."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        detector = PatchCoreDetector()
        detector.fit(sample_image_batch)

        result = detector.detect(sample_image)
        assert "scores" in result
        assert "anomaly_maps" in result
        assert "is_anomaly" in result


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestPaDiMDetector:
    """Tests for PaDiM anomaly detector."""

    def test_padim_initialization(self) -> None:
        """Test PaDiM can be initialized with default config."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        assert detector is not None
        assert cast("PaDiMConfig", detector.config).d_reduced == 100

    def test_padim_fit(self, sample_image_batch) -> None:
        """Test PaDiM fitting on normal images."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        detector.fit(sample_image_batch)
        assert detector._is_fitted
        assert detector.mean is not None
        assert detector.inv_covariance is not None

    @pytest.mark.slow
    def test_padim_detect(self, sample_image_batch, sample_image) -> None:
        """Test PaDiM anomaly detection."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        detector.fit(sample_image_batch)

        result = detector.detect(sample_image)
        assert "scores" in result
        assert "anomaly_maps" in result


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestSTFPMDetector:
    """Tests for STFPM teacher-student detector."""

    def test_stfpm_initialization(self) -> None:
        """Test STFPM can be initialized with default config."""
        from omni_mercury_engine.detectors.visual import STFPMDetector

        detector = STFPMDetector()
        assert detector is not None
        assert detector.config.backbone == "resnet18"

    def test_stfpm_config(self) -> None:
        """Test STFPM with custom config."""
        from omni_mercury_engine.detectors.visual import STFPMDetector
        from omni_mercury_engine.detectors.visual.stfpm import STFPMConfig

        config = STFPMConfig(
            backbone="resnet18",
            layers=["layer1", "layer2"],
            num_epochs=5,
        )
        detector = STFPMDetector(config=config)
        assert len(detector.config.layers) == 2


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestReverseDistillationDetector:
    """Tests for Reverse Distillation detector."""

    def test_reverse_distillation_initialization(self) -> None:
        """Test Reverse Distillation can be initialized."""
        from omni_mercury_engine.detectors.visual import ReverseDistillationDetector

        detector = ReverseDistillationDetector()
        assert detector is not None
        assert cast("ReverseDistillationConfig", detector.config).bottleneck_dim == 256

    def test_reverse_distillation_config(self) -> None:
        """Test Reverse Distillation with custom config."""
        from omni_mercury_engine.detectors.visual import ReverseDistillationDetector
        from omni_mercury_engine.detectors.visual.reverse_distillation import (
            ReverseDistillationConfig,
        )

        config = ReverseDistillationConfig(
            backbone="resnet18",
            bottleneck_dim=128,
            oce_gamma=0.5,
        )
        detector = ReverseDistillationDetector(config=config)
        cfg = cast("ReverseDistillationConfig", detector.config)
        assert cfg.bottleneck_dim == 128
        assert cfg.oce_gamma == 0.5


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestCFlowDetector:
    """Tests for CFlow normalizing flow detector."""

    def test_cflow_initialization(self) -> None:
        """Test CFlow can be initialized."""
        from omni_mercury_engine.detectors.visual import CFlowDetector

        detector = CFlowDetector()
        assert detector is not None
        assert cast("CFlowConfig", detector.config).n_flows == 8

    def test_cflow_config(self) -> None:
        """Test CFlow with custom config."""
        from omni_mercury_engine.detectors.visual import CFlowDetector
        from omni_mercury_engine.detectors.visual.cflow import CFlowConfig

        config = CFlowConfig(
            backbone="resnet18",
            n_flows=4,
            hidden_ratio=0.5,
        )
        detector = CFlowDetector(config=config)
        cfg = cast("CFlowConfig", detector.config)
        assert cfg.n_flows == 4
        assert cfg.hidden_ratio == 0.5


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestFeatureExtractor:
    """Tests for backbone feature extraction."""

    def test_feature_extractor_initialization(self) -> None:
        """Test FeatureExtractor can be initialized."""
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        extractor = FeatureExtractor(backbone_name="resnet18")
        assert extractor is not None

    def test_feature_extractor_forward(self, sample_image) -> None:
        """Test feature extraction forward pass."""
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        extractor = FeatureExtractor(
            backbone_name="resnet18",
            layers=["layer2", "layer3"],
        )

        features = extractor(sample_image)
        assert isinstance(features, dict)
        assert "layer2" in features or len(features) > 0


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestBaseVisualDetector:
    """Tests for base visual detector class."""

    def test_preprocessing(self, sample_image) -> None:
        """Test image preprocessing."""
        from omni_mercury_engine.detectors.visual import BaseVisualDetector

        detector = BaseVisualDetector()
        processed = detector.preprocess(sample_image)
        assert processed.shape[-2:] == (224, 224)

    def test_postprocessing(self, sample_image) -> None:
        """Test anomaly map postprocessing."""
        from omni_mercury_engine.detectors.visual import BaseVisualDetector

        detector = BaseVisualDetector()
        # Create dummy anomaly map
        anomaly_map = torch.randn(1, 56, 56)
        upsampled = detector.postprocess(anomaly_map, original_size=(224, 224))
        assert upsampled.shape[-2:] == (224, 224)

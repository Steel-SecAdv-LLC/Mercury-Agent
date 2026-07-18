# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for SOTA Visual Anomaly Detection modules.

Tests PatchCore, PaDiM, STFPM, Reverse Distillation, and CFlow detectors.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, cast

import numpy as np
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

OFF_MODALITY_MESSAGE = r"expects a batch of images with shape \[B, C, H, W\] or \[B, H, W, C\]"


def _synthetic_image_set() -> tuple[Any, Any]:
    """Build a structured normal batch and a [normal, anomalous-patch] test pair.

    Normal images share a smooth sinusoidal texture with small noise; the
    second test image carries a bright 50x50 square patch (rows 80:130,
    cols 90:140) that is far outside the normal appearance distribution.
    """
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 224), torch.linspace(0, 1, 224), indexing="ij")
    base = 0.4 + 0.2 * torch.sin(12 * torch.pi * xx) * torch.sin(12 * torch.pi * yy)
    normal = base.expand(6, 3, 224, 224).clone() + 0.02 * torch.randn(6, 3, 224, 224)
    test = base.expand(2, 3, 224, 224).clone() + 0.02 * torch.randn(2, 3, 224, 224)
    test[1, :, 80:130, 90:140] = 0.98
    return normal.clamp(0, 1), test.clamp(0, 1)


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestOffModalityRejection:
    """Non-image input must be rejected early with an actionable message.

    Regression: a generic ``[n, d]`` float array previously crashed deep in
    ``torch.nn.functional.interpolate`` ("Input and output must have the same
    number of spatial dimensions...") instead of naming the expected image
    modality.
    """

    @pytest.mark.parametrize(
        "detector_name",
        ["patchcore", "padim", "stfpm", "reverse_distillation", "cflow"],
    )
    def test_generic_2d_array_rejected(self, detector_name: str) -> None:
        """fit/extract_features on a (200, 8) array raise a clean ValueError."""
        from omni_mercury_engine.detectors import visual

        classes: dict[str, type[Any]] = {
            "patchcore": visual.PatchCoreDetector,
            "padim": visual.PaDiMDetector,
            "stfpm": visual.STFPMDetector,
            "reverse_distillation": visual.ReverseDistillationDetector,
            "cflow": visual.CFlowDetector,
        }
        cls = classes[detector_name]
        detector = cls({"backbone": "resnet18"})
        data = np.random.default_rng(0).random((200, 8)).astype(np.float32)

        with pytest.raises(ValueError, match=OFF_MODALITY_MESSAGE):
            detector.fit(data)
        with pytest.raises(ValueError, match=OFF_MODALITY_MESSAGE):
            detector.extract_features(data)


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

    def test_patchcore_fit(self, sample_image_batch: Any) -> None:
        """Test PatchCore fitting on normal images."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        detector = PatchCoreDetector()
        detector.fit(sample_image_batch)
        assert detector._is_fitted
        assert detector.memory_bank is not None

    @pytest.mark.slow
    def test_patchcore_detect(self, sample_image_batch: Any, sample_image: Any) -> None:
        """Test PatchCore anomaly detection."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        detector = PatchCoreDetector()
        detector.fit(sample_image_batch)

        result = detector.detect(sample_image)
        assert "scores" in result
        assert "anomaly_maps" in result
        assert "is_anomaly" in result

    @pytest.mark.slow
    def test_patchcore_scores_injected_patch_higher(self) -> None:
        """An injected anomaly patch scores clearly above a normal image."""
        from omni_mercury_engine.detectors.visual import PatchCoreDetector

        normal, test = _synthetic_image_set()
        detector = PatchCoreDetector({"backbone": "resnet18"})
        detector.fit(normal)

        result = detector.detect(test)
        scores = result["scores"]
        assert np.isfinite(scores).all()
        assert scores[1] > scores[0]
        assert result["anomaly_maps"].shape == (2, 224, 224)


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestPaDiMDetector:
    """Tests for PaDiM anomaly detector."""

    def test_padim_initialization(self) -> None:
        """Test PaDiM can be initialized with default config."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        assert detector is not None
        assert cast("PaDiMConfig", detector.config).d_reduced == 100

    def test_padim_fit(self, sample_image_batch: Any) -> None:
        """Test PaDiM fitting on normal images."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        detector.fit(sample_image_batch)
        assert detector._is_fitted
        assert detector.mean is not None
        assert detector.inv_covariance is not None

    @pytest.mark.slow
    def test_padim_detect(self, sample_image_batch: Any, sample_image: Any) -> None:
        """Test PaDiM anomaly detection."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        detector = PaDiMDetector()
        detector.fit(sample_image_batch)

        result = detector.detect(sample_image)
        assert "scores" in result
        assert "anomaly_maps" in result

    @pytest.mark.slow
    def test_padim_scores_injected_patch_higher(self) -> None:
        """An injected anomaly patch scores clearly above a normal image."""
        from omni_mercury_engine.detectors.visual import PaDiMDetector

        normal, test = _synthetic_image_set()
        detector = PaDiMDetector({"backbone": "resnet18"})
        detector.fit(normal)

        result = detector.detect(test)
        scores = result["scores"]
        assert np.isfinite(scores).all()
        assert scores[1] > scores[0]
        assert result["anomaly_maps"].shape == (2, 224, 224)


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

    @pytest.mark.slow
    def test_stfpm_fit_detect_on_images(self) -> None:
        """Short fit/detect cycle on synthetic images returns finite scores."""
        from omni_mercury_engine.detectors.visual import STFPMDetector

        normal, test = _synthetic_image_set()
        detector = STFPMDetector({"backbone": "resnet18", "num_epochs": 1})
        detector.fit(normal[:2])

        result = detector.detect(test)
        assert np.isfinite(result["scores"]).all()
        assert result["anomaly_maps"].shape == (2, 224, 224)


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

    @pytest.mark.slow
    def test_reverse_distillation_fit_detect_on_images(self) -> None:
        """Short fit/detect cycle on synthetic images returns finite scores."""
        from omni_mercury_engine.detectors.visual import ReverseDistillationDetector

        normal, test = _synthetic_image_set()
        detector = ReverseDistillationDetector({"backbone": "resnet18", "num_epochs": 1})
        detector.fit(normal[:2])

        result = detector.detect(test)
        assert np.isfinite(result["scores"]).all()
        assert result["anomaly_maps"].shape == (2, 224, 224)


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

    @pytest.mark.slow
    def test_cflow_detect_runs_without_grad(self) -> None:
        """detect() completes after fit (regression: the flow forward ran
        outside ``torch.no_grad`` and ``.numpy()`` failed on a grad tensor)."""
        from omni_mercury_engine.detectors.visual import CFlowDetector

        normal, test = _synthetic_image_set()
        detector = CFlowDetector({"backbone": "resnet18", "num_epochs": 1})
        detector.fit(normal[:2])

        result = detector.detect(test)
        assert np.isfinite(result["scores"]).all()
        assert result["anomaly_maps"].shape == (2, 224, 224)


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestFeatureExtractor:
    """Tests for backbone feature extraction."""

    def test_feature_extractor_initialization(self) -> None:
        """Test FeatureExtractor can be initialized."""
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        extractor = FeatureExtractor(backbone_name="resnet18")
        assert extractor is not None

    def test_feature_extractor_forward(self, sample_image: Any) -> None:
        """Test feature extraction forward pass."""
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        extractor = FeatureExtractor(
            backbone_name="resnet18",
            layers=["layer2", "layer3"],
        )

        features = extractor(sample_image)
        assert isinstance(features, dict)
        assert "layer2" in features or len(features) > 0


def _concrete_visual_detector() -> Any:
    """Minimal concrete subclass to exercise the base's concrete helpers.

    The base is a genuine ABC (``fit`` / ``detect`` / ``extract_features``
    are ``@abstractmethod``), so concrete contract methods are stubbed here.
    """
    from omni_mercury_engine.detectors.visual.base_visual import BaseVisualDetector

    class _Concrete(BaseVisualDetector):
        def fit(self, data: Any) -> BaseVisualDetector:
            return self

        def detect(self, data: Any) -> dict[str, Any]:
            return {}

        def extract_features(self, data: Any) -> torch.Tensor:
            return torch.zeros(1)

    return _Concrete()


@pytest.mark.skipif(not HAS_TORCH or not HAS_TORCHVISION, reason="torch/torchvision not installed")
class TestBaseVisualDetector:
    """Contract tests for the visual detector base (ROADMAP #4).

    The base is a genuine ABC: ``fit`` / ``detect`` / ``extract_features``
    are ``@abstractmethod`` (no ``NotImplementedError`` stub on the public
    path), so the base cannot be instantiated directly. The native SOTA
    detectors are the concrete implementations.
    """

    def test_base_is_abstract_not_instantiable(self) -> None:
        """Direct instantiation raises TypeError, not a runtime NotImplementedError."""
        from omni_mercury_engine.detectors.visual import BaseVisualDetector

        with pytest.raises(TypeError):
            BaseVisualDetector()  # type: ignore[abstract]

    def test_contract_methods_are_abstract(self) -> None:
        """The three contract methods are declared abstract."""
        from omni_mercury_engine.detectors.visual import BaseVisualDetector

        assert BaseVisualDetector.__abstractmethods__ == frozenset(
            {"fit", "detect", "extract_features"}
        )

    def test_preprocessing(self, sample_image: Any) -> None:
        """The concrete ``preprocess`` helper works on a concrete subclass."""
        detector = _concrete_visual_detector()
        processed = detector.preprocess(sample_image)
        assert processed.shape[-2:] == (224, 224)

    @pytest.mark.parametrize(
        "shape",
        [(200, 8), (16,), (4, 224, 224), (2, 5, 32, 32)],
    )
    def test_preprocess_rejects_non_image_input(self, shape: tuple[int, ...]) -> None:
        """Off-modality arrays are rejected with the expected-shape message."""
        detector = _concrete_visual_detector()
        with pytest.raises(ValueError, match=OFF_MODALITY_MESSAGE):
            detector.preprocess(np.zeros(shape, dtype=np.float32))

    def test_postprocessing(self, sample_image: Any) -> None:
        """The concrete ``postprocess`` helper works on a concrete subclass."""
        detector = _concrete_visual_detector()
        # Create dummy anomaly map
        anomaly_map = torch.randn(1, 56, 56)
        upsampled = detector.postprocess(anomaly_map, original_size=(224, 224))
        assert upsampled.shape[-2:] == (224, 224)

    def _spy_preresize_shape(self, monkeypatch: Any) -> dict[str, tuple[int, ...]]:
        """Capture the tensor shape reaching the resize step (post layout choice)."""
        captured: dict[str, tuple[int, ...]] = {}
        real_interpolate = torch.nn.functional.interpolate

        def _spy(inp: Any, *args: Any, **kwargs: Any) -> Any:
            captured["shape"] = tuple(inp.shape)
            return real_interpolate(inp, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "interpolate", _spy)
        return captured

    def test_channel_first_input_with_1_or_3_width_is_not_permuted(self, monkeypatch: Any) -> None:
        """A channel-first batch whose width is 1 or 3 keeps its channel axis.

        Regression: the layout heuristic keyed solely on the trailing axis, so a
        genuine channel-first tensor whose width happened to be 1 or 3 (e.g.
        ``[B, 3, H, 3]``) was wrongly permuted into an invalid layout. It must be
        read as ``[B, C, H, W]`` and reach resize with its (H, W) intact.
        """
        detector = _concrete_visual_detector()
        captured = self._spy_preresize_shape(monkeypatch)

        out = detector.preprocess(torch.rand(2, 3, 32, 3))

        assert captured["shape"] == (2, 3, 32, 3)  # not permuted to (2, 3, 3, 32)
        assert out.shape == (2, 3, 224, 224)

    def test_channel_last_input_is_permuted(self, monkeypatch: Any) -> None:
        """A channel-last ``[B, H, W, C]`` batch is transposed to channel-first."""
        detector = _concrete_visual_detector()
        captured = self._spy_preresize_shape(monkeypatch)

        out = detector.preprocess(torch.rand(2, 32, 16, 3))

        assert captured["shape"] == (2, 3, 32, 16)  # permuted from channel-last
        assert out.shape == (2, 3, 224, 224)

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Base classes for visual anomaly detection.

Provides unified interface for all visual anomaly detection algorithms,
ensuring consistent API across PatchCore, PaDiM, STFPM, and other methods.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.base import BaseDetector, DetectorMetrics
from omni_mercury_engine.core.conformal_prediction import (
    split_calibration_indices,
    split_conformal_threshold,
)
from omni_mercury_engine.core.exceptions import DataException, InputModalityError
from omni_mercury_engine.security.safe_torch import safe_torch_load

logger = logging.getLogger(__name__)


class BackboneType(Enum):
    """Supported backbone architectures for feature extraction."""

    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    WIDE_RESNET50_2 = "wide_resnet50_2"
    EFFICIENTNET_B4 = "efficientnet_b4"
    VIT_BASE = "vit_base_patch16_224"
    DINO_V2 = "dinov2_vitb14"


@dataclass
class VisualDetectorConfig:
    """Configuration for visual anomaly detectors.

    Attributes:
        backbone: Pre-trained backbone architecture for feature extraction
        image_size: Input image dimensions (height, width)
        layers: Feature extraction layers to use from backbone
        device: Computation device (cuda/cpu/mps)
        batch_size: Batch size for inference
        threshold: Anomaly score threshold for binary classification.
            ``None`` (the default) calibrates it during ``fit`` from
            held-out normal images by split conformal prediction, so the
            decision is on the detector's own score scale.  A float is used
            verbatim and ``fit`` then trains on every image.
        false_alarm_rate: Target probability that a normal image is flagged
            when the threshold is calibrated (split-conformal level alpha).
        calibration_fraction: Share of the normal images held out from
            training to calibrate the threshold (at least one image each
            for training and calibration).
        calibration_seed: Seed of the deterministic train/calibration split.
        normalize_features: Whether to L2-normalize extracted features
        pool_strategy: Feature pooling strategy ('avg', 'max', 'adaptive')
    """

    backbone: str = "resnet18"
    image_size: tuple[int, int] = (224, 224)
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    threshold: float | None = None
    normalize_features: bool = True
    pool_strategy: str = "adaptive"
    false_alarm_rate: float = 0.05
    calibration_fraction: float = 0.2
    calibration_seed: int = 0


class BaseVisualDetector(BaseDetector, nn.Module):
    """Abstract base class for visual anomaly detectors.

    .. note:: **Transparent abstract contract (ROADMAP deferred item #4).**
        ``fit`` / ``detect`` / ``extract_features`` are genuine
        ``@abstractmethod`` declarations, so :class:`BaseVisualDetector`
        cannot be instantiated directly (a ``TypeError`` is raised) — there
        is no ``NotImplementedError`` stub on the public path. The native
        SOTA detectors (PatchCore, PaDiM, STFPM, ReverseDistillation, CFlow)
        are the concrete implementations, consistent with the 2026-05
        strategic decision to keep native detectors.

    Extends BaseDetector with visual-specific functionality including
    image preprocessing, feature extraction, and anomaly map generation.

    All visual detectors must implement:
        - fit(): Learn normal feature distribution
        - detect(): Compute anomaly scores and maps
        - extract_features(): Extract backbone features

    Attributes:
        config: Detector configuration
        backbone: Pre-trained feature extractor
        device: Computation device
    """

    def __init__(self, config: VisualDetectorConfig | dict[str, Any] | None = None) -> None:
        """Initialize visual detector.

        Args:
            config: Detector configuration (VisualDetectorConfig or dict)
        """
        # Initialize nn.Module first (required for PyTorch)
        nn.Module.__init__(self)

        # Handle config conversion
        if config is None:
            self.visual_config = VisualDetectorConfig()
        elif isinstance(config, dict):
            self.visual_config = VisualDetectorConfig(**config)
        else:
            self.visual_config = config

        # Expose config reference for internal use
        self._visual_config_ref = self.visual_config

        # Initialize BaseDetector attributes
        # Note: We don't call BaseDetector.__init__() because we inherit from nn.Module
        # and need to avoid MRO conflicts. Instead, we manually set the required attributes.
        # ``inf`` until ``fit`` calibrates it: no decision before calibration.
        configured = self.visual_config.threshold
        self.threshold = math.inf if configured is None else float(configured)
        #: How ``threshold`` was set: ``{"method": "fixed", ...}`` for a
        #: configured value, ``{"method": "split_conformal", ...}`` with the
        #: calibration size, the requested and the achieved false-alarm
        #: bound when it was calibrated by ``fit``; ``None`` before ``fit``.
        self.calibration: dict[str, Any] | None = None
        self._is_fitted = False
        self._name = self.__class__.__name__
        # BaseDetector's contract (its ``metrics`` property) promises a
        # real DetectorMetrics, so honour it even though we skip
        # BaseDetector.__init__ to avoid the nn.Module MRO conflict.
        self._metrics = DetectorMetrics()

        self.device = torch.device(self.visual_config.device)
        self._backbone: nn.Module | None = None
        self._feature_dim: int = 0

        # Image normalization (ImageNet stats) - use _norm prefix to avoid conflicts with subclass attributes
        self.register_buffer("_norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def visual_detector_config(self) -> VisualDetectorConfig:
        """Get the visual detector configuration."""
        return self._visual_config_ref

    @property
    def config(self) -> VisualDetectorConfig:
        """Get the detector configuration (alias for backward compatibility)."""
        return self._visual_config_ref

    @config.setter
    def config(self, value: VisualDetectorConfig | dict[str, Any]) -> None:
        """Set the detector configuration.

        Args:
            value: VisualDetectorConfig instance or dict.
        """
        if isinstance(value, VisualDetectorConfig):
            self._visual_config_ref = value
            self.visual_config = value
        elif isinstance(value, dict):
            # Rebuild the detector's own config dataclass: a saved
            # ``PatchCoreConfig`` dict carries subclass fields
            # (``coreset_sampling_ratio`` ...) that the base
            # ``VisualDetectorConfig`` rejects, which made every concrete
            # visual detector impossible to ``load``.
            config_cls = type(getattr(self, "_visual_config_ref", None) or VisualDetectorConfig())
            self._visual_config_ref = config_cls(**value)
            self.visual_config = self._visual_config_ref

    @property
    def backbone(self) -> nn.Module:
        """Get the backbone feature extractor."""
        if self._backbone is None:
            raise RuntimeError("Backbone not initialized. Call _init_backbone() first.")
        return self._backbone

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension after extraction."""
        return self._feature_dim

    def _init_backbone(self) -> None:
        """Initialize the backbone network.

        Override in subclasses if needed.
        """
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        self._backbone = FeatureExtractor(
            backbone_name=self.visual_config.backbone,
            layers=self.visual_config.layers,
            device=self.device,
        )
        self._backbone.eval()
        self._backbone.to(self.device)

        # Compute feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, *self.visual_config.image_size).to(self.device)
            features = self._backbone(dummy)
            if isinstance(features, dict):
                self._feature_dim = sum(f.shape[1] for f in features.values())
            else:
                self._feature_dim = features.shape[1]

    def preprocess(self, images: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Preprocess images for feature extraction.

        Args:
            images: Input images [B, C, H, W] or [B, H, W, C], values in [0, 255] or [0, 1]

        Returns:
            Normalized tensor [B, 3, H, W] on device

        Raises:
            ValueError: If the input is not a 4-D image batch with 1 or 3
                channels (off-modality input, e.g. a generic 2-D feature
                array).
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        if images.dim() != 4 or (images.shape[1] not in (1, 3) and images.shape[-1] not in (1, 3)):
            raise InputModalityError(
                f"{self._name} expects a batch of images with shape "
                f"[B, C, H, W] or [B, H, W, C] where C is 1 or 3 channels; "
                f"got input of shape {tuple(images.shape)}"
            )

        # Disambiguate channel layout. Prefer channel-first ([B, C, H, W]) when
        # the leading axis is already a valid 1/3-channel count; only treat the
        # input as channel-last ([B, H, W, C]) — and permute — when the leading
        # axis is not a valid channel count but the trailing one is. This avoids
        # mis-permuting a genuine channel-first tensor whose width happens to be
        # 1 or 3 (e.g. [B, 3, H, 3]).
        if images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
            images = images.permute(0, 3, 1, 2)

        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0

        # Resize if needed
        if images.shape[-2:] != self.visual_config.image_size:
            images = torch.nn.functional.interpolate(
                images,
                size=self.visual_config.image_size,
                mode="bilinear",
                align_corners=False,
            )

        # Apply ImageNet normalization
        images = images.to(self.device)
        images = (images - self._norm_mean) / self._norm_std  # type: ignore[operator, unused-ignore]

        return images

    def postprocess_anomaly_map(
        self,
        anomaly_map: torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Postprocess anomaly map to original image size.

        Args:
            anomaly_map: Raw anomaly map from detector
            original_size: Target size (H, W) for resizing

        Returns:
            Smoothed and resized anomaly map
        """
        # Gaussian smoothing for cleaner maps
        if anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)

        kernel_size = 33
        sigma = 4.0
        kernel = self._get_gaussian_kernel(kernel_size, sigma).to(anomaly_map.device)
        anomaly_map = torch.nn.functional.conv2d(anomaly_map, kernel, padding=kernel_size // 2)

        # Resize to original size
        if original_size is not None:
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )

        return anomaly_map.squeeze(1)

    def postprocess(
        self,
        anomaly_map: torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Alias for postprocess_anomaly_map for API compatibility.

        Args:
            anomaly_map: Raw anomaly map from detector
            original_size: Target size (H, W) for resizing

        Returns:
            Smoothed and resized anomaly map
        """
        return self.postprocess_anomaly_map(anomaly_map, original_size)

    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for smoothing."""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    @abstractmethod
    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseVisualDetector:
        """Fit detector to normal (non-anomalous) images.

        Args:
            data: Normal images [N, C, H, W] or [N, H, W, C]

        Returns:
            Self for method chaining

        Abstract contract — every native visual detector implements this.
        """
        ...

    @abstractmethod
    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies in images.

        Args:
            data: Test images [N, C, H, W] or [N, H, W, C]

        Returns:
            Dict containing:
                - scores: Image-level anomaly scores [N]
                - anomaly_maps: Pixel-level anomaly maps [N, H, W]
                - is_anomaly: Binary anomaly flags [N]
                - features: Extracted features for fusion [N, D]

        Abstract contract — every native visual detector implements this.
        """
        ...

    @abstractmethod
    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, feature_dim] normalized for fusion

        Abstract contract — every native visual detector implements this.
        """
        ...

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for integration with PyTorch pipelines.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Detection results dict
        """
        return self.detect(x)

    def _split_for_calibration(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split normal images into training and calibration subsets.

        A fixed configured ``threshold`` needs no calibration: every image
        trains.  Otherwise a seeded ``calibration_fraction`` of the batch is
        held out so the threshold is calibrated on images the model never
        trained on -- scoring the training images themselves would put the
        threshold inside the in-sample error (a memory bank scores its own
        patches near zero), and normal images would then be flagged.

        Args:
            data: Raw image batch, before :meth:`preprocess`.

        Returns:
            ``(training_images, calibration_images)``; the second is ``None``
            when no calibration will run.

        Raises:
            DataException: If calibration is required and fewer than two
                images were given.
        """
        cfg = self.visual_config
        if cfg.threshold is not None:
            self.threshold = float(cfg.threshold)
            self.calibration = {"method": "fixed", "threshold": self.threshold}
            return data, None
        if data.dim() != 4:
            # Not an image batch: ``preprocess`` refuses it with the typed
            # modality error, which is the error the caller should see.
            return data, None
        n = int(data.shape[0])
        if n < 2:
            raise DataException(
                f"{self._name} calibrates its decision threshold on held-out normal "
                f"images and needs at least 2 to fit (got {n}). Pass more normal images, "
                "or configure an explicit `threshold`."
            )
        if not 0.0 < cfg.false_alarm_rate < 1.0:
            raise DataException(f"false_alarm_rate must lie in (0, 1); got {cfg.false_alarm_rate}")
        try:
            fit_idx, cal_idx = split_calibration_indices(
                n, cfg.calibration_fraction, cfg.calibration_seed
            )
        except ValueError as e:
            raise DataException(str(e)) from e
        index = torch.as_tensor(np.concatenate([fit_idx, cal_idx]), device=data.device)
        n_fit = len(fit_idx)
        return data[index[:n_fit]], data[index[n_fit:]]

    def _calibrate_threshold(self, calibration: torch.Tensor | None) -> None:
        """Set ``threshold`` by split conformal prediction on held-out normals.

        With ``n`` calibration scores and target level ``alpha``, the
        threshold is the ``k``-th smallest score, ``k = min(ceil((n + 1)(1 -
        alpha)), n)``.  For a new normal image exchangeable with the
        calibration images, ``P(score > threshold) <= (n + 1 - k)/(n + 1)``
        (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2021).  That
        achieved bound is recorded in :attr:`calibration`; when too few
        images exist to reach ``alpha`` it is larger than ``alpha`` and a
        warning says so -- the guarantee is reported, never overstated.

        Args:
            calibration: Held-out normal images, or ``None`` (no-op).

        Raises:
            DataException: If a calibration score is not finite.
        """
        if calibration is None:
            return
        # Raw scores only: no decision is made while the threshold is unset.
        self.threshold = math.inf
        scores = np.asarray(self.detect(calibration)["scores"], dtype=np.float64).reshape(-1)
        try:
            result = split_conformal_threshold(scores, self.visual_config.false_alarm_rate)
        except ValueError as e:
            raise DataException(f"{self._name}: cannot calibrate: {e}") from e
        self.threshold = result.threshold
        self.calibration = result.as_record()
        if result.false_alarm_bound > result.false_alarm_rate:
            logger.warning(
                "%s: %d calibration image(s) bound the false-alarm rate by %.3f, above the "
                "requested %.3f; fit on more normal images to tighten it",
                self._name,
                result.n_calibration,
                result.false_alarm_bound,
                result.false_alarm_rate,
            )

    def _fitted_state(self) -> dict[str, Any]:
        """Fitted state that ``state_dict`` does not carry (plain tensor attributes).

        Subclasses holding their fitted model outside registered modules and
        buffers (a memory bank, Gaussian statistics, a random projection)
        return it here so :meth:`save` persists it; without this a saved
        detector reloads with weights but no fitted model.
        """
        return {}

    def _build_architecture_for_load(self) -> None:
        """Create lazily-built submodules so ``load_state_dict`` has targets.

        Detectors that build modules on the first ``fit`` batch build them
        here from the configured ``image_size`` (``fit`` sees preprocessed,
        i.e. resized, images, so the shapes are identical).
        """

    def _restore_fitted_state(self, state: dict[str, Any]) -> None:
        """Reinstate :meth:`_fitted_state` after the weights are loaded."""

    def _load_probe(self) -> torch.Tensor:
        """A zero batch at the configured input size, for shape inference."""
        height, width = self.visual_config.image_size
        return torch.zeros(1, 3, int(height), int(width), device=self.device)

    def save(self, path: str) -> None:
        """Save detector state to file.

        The config dataclass is flattened to a primitive dict before
        ``torch.save`` so the checkpoint round-trips cleanly under
        ``weights_only=True`` on load -- PyTorch's safe-load mode
        refuses arbitrary classes, and a stored ``VisualDetectorConfig``
        instance would not be on the safe-globals list. The on-disk
        schema is now ``{"config": dict, "state_dict": ..., "is_fitted":
        bool}`` regardless of any future config field additions.
        """
        torch.save(
            {
                "config": asdict(self.visual_config),
                "state_dict": self.state_dict(),
                "is_fitted": self._is_fitted,
                "threshold": self.threshold,
                "calibration": self.calibration,
                "fitted_state": self._fitted_state(),
            },
            path,
        )

    def load(self, path: str) -> BaseVisualDetector:
        """Load detector state from file.

        Args:
            path: Path to saved detector state file

        Security Note:
            Loading is hard-pinned to ``weights_only=True`` so the
            checkpoint cannot execute arbitrary code via pickle. The
            previous ``allow_unsafe=True`` escape hatch has been
            removed; legacy checkpoints that cannot load with
            ``weights_only=True`` must be re-saved by the operator (or
            converted via ``tools/migrate_pkl.py`` if they originated
            as raw pickles).

            Round-trip contract: ``save()`` writes the config as a
            primitive dict and ``load()`` reconstructs the dataclass
            via keyword expansion. Checkpoints saved by an older
            code path that stored the full ``VisualDetectorConfig``
            object will fail under ``weights_only=True`` and must be
            re-saved.

        Raises:
            RuntimeError: If the checkpoint cannot load with
                ``weights_only=True``.
            TypeError: If the saved config dict references fields
                that the current ``VisualDetectorConfig`` no longer
                accepts (schema drift).
        """
        try:
            checkpoint = safe_torch_load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint at '{path}' cannot be loaded safely (weights_only=True). "
                "Re-save the checkpoint with the current code path; the unsafe "
                "fallback was removed because pickle has no role in the Mercury "
                f"Agent runtime. Original error: {e}"
            ) from e

        config_payload = checkpoint["config"]
        if isinstance(config_payload, (VisualDetectorConfig, dict)):
            # Route through the ``config`` property setter rather than
            # assigning ``self.visual_config`` directly. The setter
            # also maintains ``self._visual_config_ref`` (which backs
            # the ``config`` / ``visual_detector_config`` accessors).
            # A direct attribute assignment would leave those two
            # references out of sync after a load -- ``detector.config``
            # would still return the pre-load config while
            # ``detector.visual_config`` carried the freshly loaded
            # checkpoint config. The setter accepts both the dataclass
            # instance (legacy on-disk shape) and a dict (current
            # ``asdict``-serialised shape).
            self.config = config_payload
        else:
            raise RuntimeError(
                f"Checkpoint at '{path}' has 'config' of type "
                f"{type(config_payload).__name__}; expected dict (from "
                "dataclasses.asdict) or VisualDetectorConfig."
            )
        self._build_architecture_for_load()
        self.load_state_dict(checkpoint["state_dict"])
        self._restore_fitted_state(checkpoint.get("fitted_state") or {})
        self._is_fitted = checkpoint["is_fitted"]
        # A calibrated threshold lives on the detector, not in its config;
        # checkpoints written before calibration existed carry neither key
        # and keep the configured threshold.
        if "threshold" in checkpoint:
            self.threshold = checkpoint["threshold"]
            self.calibration = checkpoint.get("calibration")
        else:
            configured = self.visual_config.threshold
            self.threshold = math.inf if configured is None else float(configured)
        return self

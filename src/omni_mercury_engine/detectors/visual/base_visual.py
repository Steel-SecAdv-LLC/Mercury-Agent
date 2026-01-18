"""
Mercury Agent ♱
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
Base classes for visual anomaly detection.

Provides unified interface for all visual anomaly detection algorithms,
ensuring consistent API across PatchCore, PaDiM, STFPM, and other methods.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.base import BaseDetector


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
        threshold: Anomaly score threshold for binary classification
        normalize_features: Whether to L2-normalize extracted features
        pool_strategy: Feature pooling strategy ('avg', 'max', 'adaptive')
    """

    backbone: str = "resnet18"
    image_size: tuple[int, int] = (224, 224)
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    threshold: float = 0.5
    normalize_features: bool = True
    pool_strategy: str = "adaptive"


class BaseVisualDetector(BaseDetector, nn.Module):
    """Abstract base class for visual anomaly detectors.

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
        self.threshold = self.visual_config.threshold
        self._is_fitted = False
        self._name = self.__class__.__name__
        self._metrics = None  # Will be initialized on first access if needed

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
            self._visual_config_ref = VisualDetectorConfig(**value)
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
        """Initialize the backbone network. Override in subclasses if needed."""
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
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Handle channel-last format
        if images.dim() == 4 and images.shape[-1] in [1, 3]:
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
        images = (images - self._norm_mean) / self._norm_std

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

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseVisualDetector:
        """Fit detector to normal (non-anomalous) images.

        Args:
            data: Normal images [N, C, H, W] or [N, H, W, C]

        Returns:
            Self for method chaining

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement fit() for visual detectors.")

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

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement detect() for visual detectors.")

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, feature_dim] normalized for fusion

        Note:
            Subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement extract_features() for visual detectors."
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for integration with PyTorch pipelines.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Detection results dict
        """
        return self.detect(x)

    def save(self, path: str) -> None:
        """Save detector state to file."""
        torch.save(
            {
                "config": self.visual_config,
                "state_dict": self.state_dict(),
                "is_fitted": self._is_fitted,
            },
            path,
        )

    def load(self, path: str, *, allow_unsafe: bool = False) -> BaseVisualDetector:
        """Load detector state from file.

        Args:
            path: Path to saved detector state file
            allow_unsafe: If True, allows loading checkpoints that require full
                pickle deserialization. Only set this to True for legacy checkpoints
                from trusted sources. Default is False for maximum security.

        Security Note:
            By default, this uses weights_only=True to prevent arbitrary code
            execution from untrusted checkpoint files. Standard state_dict
            checkpoints load safely.

            If you encounter errors with legacy checkpoints that stored custom
            objects, you can pass allow_unsafe=True after verifying the checkpoint
            source is trusted (i.e., generated by this application).

            See: https://pytorch.org/docs/stable/generated/torch.load.html

        Raises:
            RuntimeError: If checkpoint requires unsafe loading but allow_unsafe=False
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Default: safe loading with weights_only=True
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            if allow_unsafe:
                logger.warning(
                    "Safe checkpoint loading failed. Falling back to unsafe mode "
                    "as explicitly requested. Only do this for trusted checkpoints. "
                    f"Original error: {e}"
                )
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )  # nosec B614 - intentional for trusted checkpoints with allow_unsafe=True
            else:
                raise RuntimeError(
                    f"Checkpoint at '{path}' cannot be loaded safely (weights_only=True). "
                    "This may indicate the checkpoint contains custom pickled objects. "
                    "If you trust this checkpoint source, re-run with allow_unsafe=True. "
                    f"Original error: {e}"
                ) from e

        self.visual_config = checkpoint["config"]
        self.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = checkpoint["is_fitted"]
        return self

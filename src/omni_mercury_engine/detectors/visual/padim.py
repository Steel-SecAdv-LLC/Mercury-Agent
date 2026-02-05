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
PaDiM: Patch Distribution Modeling Framework for Anomaly Detection

Implementation of PaDiM algorithm from ICPR 2020.
Fastest inference among SOTA methods with competitive accuracy.

Key Features:
    1. Each patch position modeled as multivariate Gaussian
    2. Mahalanobis distance for anomaly scoring
    3. No training required - fits distributions directly
    4. Random dimensionality reduction for efficiency

Reference:
    Defard et al. "PaDiM: a Patch Distribution Modeling Framework
    for Anomaly Detection and Localization"
    https://arxiv.org/abs/2011.08785
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from omni_mercury_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class PaDiMConfig(VisualDetectorConfig):
    """Configuration for PaDiM detector.

    Attributes:
        d_reduced: Reduced feature dimension (random projection)
        epsilon: Regularization for covariance matrix inversion
    """

    d_reduced: int = 100  # Reduced dimension for efficiency
    epsilon: float = 0.01  # Covariance regularization
    layers: list[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])


class PaDiMDetector(BaseVisualDetector):
    """PaDiM anomaly detector.

    Models patch-level feature distributions as multivariate Gaussians.
    Uses Mahalanobis distance for anomaly scoring.

    Fastest inference time among SOTA visual AD methods:
    - No memory bank lookup required
    - Direct distance computation

    Example:
        >>> detector = PaDiMDetector()
        >>> detector.fit(normal_images)
        >>> results = detector.detect(test_images)
    """

    def __init__(self, config: PaDiMConfig | dict[str, Any] | None = None) -> None:
        """Initialize PaDiM detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            config = PaDiMConfig()
        elif isinstance(config, dict):
            config = PaDiMConfig(**config)

        super().__init__(config)
        self.padim_config: PaDiMConfig = config

        # Initialize backbone
        self._init_backbone()

        # Distribution parameters (populated during fit)
        self.mean: torch.Tensor | None = None  # [H*W, d_reduced]
        self.cov_inv: torch.Tensor | None = None  # [H*W, d_reduced, d_reduced]

        # Random projection matrix
        self.projection: torch.Tensor | None = None

        # Spatial info
        self._spatial_shape: tuple[int, int] | None = None

    @property
    def inv_covariance(self) -> torch.Tensor | None:
        """Alias for cov_inv for test compatibility."""
        return self.cov_inv

    def _get_random_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """Generate random projection matrix for dimensionality reduction.

        Uses random Gaussian projection following Johnson-Lindenstrauss lemma.

        Args:
            input_dim: Input feature dimension
            output_dim: Target reduced dimension

        Returns:
            Projection matrix [input_dim, output_dim]
        """
        # Random Gaussian projection
        projection = torch.randn(input_dim, output_dim)
        # Orthogonalize for better preservation of distances
        projection, _ = torch.linalg.qr(projection)
        return projection

    def _aggregate_features(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Aggregate multi-layer features.

        Args:
            features: Dict of layer features {layer_name: [B, C, H, W]}

        Returns:
            Tuple of:
                - Aggregated features [B, H*W, total_channels]
                - Spatial shape (H, W)
        """
        feature_list = list(features.values())

        # Use smallest spatial size as reference
        min_h = min(f.shape[2] for f in feature_list)
        min_w = min(f.shape[3] for f in feature_list)

        resized = []
        for feat in feature_list:
            if feat.shape[2] != min_h or feat.shape[3] != min_w:
                feat = nn.functional.interpolate(
                    feat, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            resized.append(feat)

        # Concatenate: [B, sum(C_i), H, W]
        aggregated = torch.cat(resized, dim=1)

        # Reshape to [B, H*W, C]
        batch_size, channels, h, w = aggregated.shape
        patches = aggregated.permute(0, 2, 3, 1).reshape(batch_size, h * w, channels)

        return patches, (h, w)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> PaDiMDetector:
        """Fit detector by computing Gaussian parameters for each position.

        Args:
            data: Normal (non-anomalous) images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)
        n_samples = data.shape[0]
        logger.info(f"Fitting PaDiM on {n_samples} images")

        # Collect all features
        all_features = []
        batch_size = self.padim_config.batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            patches, spatial_shape = self._aggregate_features(features)
            all_features.append(patches)

            if self._spatial_shape is None:
                self._spatial_shape = spatial_shape

        # Stack all features: [N, H*W, C]
        all_features_tensor = torch.cat(all_features, dim=0)
        n_positions = all_features_tensor.shape[1]
        feature_dim = all_features_tensor.shape[2]

        logger.info(
            f"Feature shape: {n_samples} samples, {n_positions} positions, "
            f"{feature_dim} dimensions"
        )

        # Create random projection if needed
        if feature_dim > self.padim_config.d_reduced:
            self.projection = self._get_random_projection(
                feature_dim, self.padim_config.d_reduced
            ).to(self.device)

            # Apply projection
            all_features_tensor = torch.matmul(all_features_tensor, self.projection)
            logger.info(
                f"Applied random projection: {feature_dim} -> " f"{self.padim_config.d_reduced}"
            )

        d = all_features_tensor.shape[2]

        # Compute mean for each position: [H*W, d]
        self.mean = all_features_tensor.mean(dim=0)

        # Compute covariance for each position: [H*W, d, d]
        # Centered features
        centered = all_features_tensor - self.mean.unsqueeze(0)

        # Covariance: E[XX^T]
        # For each position, compute covariance across samples
        cov = torch.zeros(n_positions, d, d, device=self.device)
        for pos in range(n_positions):
            pos_features = centered[:, pos, :]  # [N, d]
            cov[pos] = torch.matmul(pos_features.T, pos_features) / (n_samples - 1)

        # Regularize and invert covariance
        identity = torch.eye(d, device=self.device).unsqueeze(0)
        cov_reg = cov + self.padim_config.epsilon * identity

        # Batch matrix inversion
        self.cov_inv = torch.linalg.inv(cov_reg)

        logger.info(f"Fitted Gaussian distributions for {n_positions} positions")
        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using Mahalanobis distance.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Dict containing:
                - scores: Image-level anomaly scores [N]
                - anomaly_maps: Pixel-level anomaly maps [N, H, W]
                - is_anomaly: Binary anomaly flags [N]
                - features: Extracted features for fusion [N, D]
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        original_size = data.shape[-2:]
        data = self.preprocess(data)

        all_scores = []
        all_maps = []
        all_features = []

        batch_size = self.padim_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            patches, spatial_shape = self._aggregate_features(features)

            # Apply projection if used
            if self.projection is not None:
                patches = torch.matmul(patches, self.projection)

            # Compute Mahalanobis distance
            distances = self._compute_mahalanobis(patches)

            # Reshape to spatial map
            h, w = spatial_shape
            score_maps = distances.view(-1, h, w)

            # Image-level scores
            image_scores = score_maps.view(score_maps.shape[0], -1).max(dim=1)[0]

            # Upsample maps
            score_maps_up = nn.functional.interpolate(
                score_maps.unsqueeze(1),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            # Gaussian smoothing
            score_maps_np = score_maps_up.cpu().numpy()
            for j in range(len(score_maps_np)):
                score_maps_np[j] = gaussian_filter(score_maps_np[j], sigma=4)

            all_scores.append(image_scores)
            all_maps.append(torch.from_numpy(score_maps_np).to(self.device))
            all_features.append(patches.mean(dim=1))

        scores = torch.cat(all_scores, dim=0)
        anomaly_maps = torch.cat(all_maps, dim=0)
        features = torch.cat(all_features, dim=0)

        is_anomaly = scores > self.threshold

        return {
            "scores": scores.cpu().numpy(),
            "anomaly_maps": anomaly_maps.cpu().numpy(),
            "is_anomaly": is_anomaly.cpu().numpy(),
            "features": features.cpu(),
        }

    def _compute_mahalanobis(self, patches: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for each patch position.

        Mahalanobis distance: sqrt((x - mu)^T @ Sigma^-1 @ (x - mu))

        Args:
            patches: Patch features [B, H*W, d]

        Returns:
            Distance scores [B, H*W]
        """
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("Detector must be fitted before computing Mahalanobis distance")
        # Center features: [B, H*W, d]
        centered = patches - self.mean.unsqueeze(0)

        # Mahalanobis distance for each position
        # (x - mu)^T @ Sigma^-1 @ (x - mu)
        # Einstein summation for batch operation
        # centered: [B, P, d], cov_inv: [P, d, d]
        # Result: [B, P]

        # Step 1: (x - mu) @ Sigma^-1 -> [B, P, d]
        intermediate = torch.einsum("bpd,pde->bpe", centered, self.cov_inv)

        # Step 2: element-wise multiply and sum -> [B, P]
        distances_sq = torch.einsum("bpd,bpd->bp", intermediate, centered)

        # Take sqrt (Mahalanobis distance)
        distances = torch.sqrt(torch.clamp(distances_sq, min=0))

        return distances

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, 128] normalized for fusion
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)

        all_features = []
        batch_size = self.padim_config.batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            patches, _ = self._aggregate_features(features)

            if self.projection is not None:
                patches = torch.matmul(patches, self.projection)

            # Global average
            global_feat = patches.mean(dim=1)
            all_features.append(global_feat)

        features = torch.cat(all_features, dim=0)

        # Project to 128D
        if features.shape[1] != 128:
            if not hasattr(self, "_fusion_projection"):
                self._fusion_projection = nn.Linear(features.shape[1], 128).to(features.device)
            features = self._fusion_projection(features)

        features = nn.functional.normalize(features, p=2, dim=1)
        return features

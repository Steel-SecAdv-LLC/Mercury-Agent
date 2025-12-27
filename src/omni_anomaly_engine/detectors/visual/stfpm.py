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
STFPM: Student-Teacher Feature Pyramid Matching for Anomaly Detection

Implementation of STFPM from arXiv 2021.
Optimal for real-time applications due to low inference latency.

Key Features:
    1. Knowledge distillation from teacher to student network
    2. Multi-scale feature pyramid matching
    3. Anomaly = feature discrepancy between teacher and student
    4. No memory bank required

Reference:
    Wang et al. "Student-Teacher Feature Pyramid Matching for
    Anomaly Detection"
    https://arxiv.org/abs/2103.04257
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from omni_anomaly_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class STFPMConfig(VisualDetectorConfig):
    """Configuration for STFPM detector.

    Attributes:
        learning_rate: Learning rate for student training
        num_epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
    """

    learning_rate: float = 0.4
    num_epochs: int = 100
    weight_decay: float = 1e-5
    layers: list[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])


class StudentNetwork(nn.Module):
    """Lightweight student network that learns to mimic teacher.

    Uses same architecture as teacher but trained from scratch
    on normal data only.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: list[str] | None = None,
    ):
        """Initialize student network.

        Args:
            backbone_name: Backbone architecture (must match teacher)
            layers: Layers for feature extraction
        """
        super().__init__()

        self.layers = layers or ["layer1", "layer2", "layer3"]

        # Create student backbone (not pretrained)
        try:
            from torchvision import models

            backbone_map = {
                "resnet18": models.resnet18,
                "resnet50": models.resnet50,
                "wide_resnet50_2": models.wide_resnet50_2,
            }

            if backbone_name in backbone_map:
                self.backbone = backbone_map[backbone_name](weights=None)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")

        except ImportError:
            raise ImportError("torchvision required for STFPM")

        # Feature extraction hooks
        self._features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register hooks to capture intermediate features."""
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name, None)
            if layer is not None:
                layer.register_forward_hook(self._hook_fn(layer_name))

    def _hook_fn(self, layer_name: str) -> Any:
        """Create hook function for layer."""

        def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self._features[layer_name] = output

        return hook

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning multi-scale features."""
        self._features.clear()
        _ = self.backbone(x)
        return {k: v.clone() for k, v in self._features.items()}


class STFPMDetector(BaseVisualDetector):
    """STFPM anomaly detector using teacher-student distillation.

    Trains a student network to match teacher features on normal data.
    Anomalies cause feature discrepancy between teacher and student.

    Advantages:
    - Fastest inference among knowledge distillation methods
    - No memory bank required
    - Lightweight model

    Example:
        >>> detector = STFPMDetector()
        >>> detector.fit(normal_images)  # Trains student
        >>> results = detector.detect(test_images)
    """

    def __init__(self, config: STFPMConfig | dict[str, Any] | None = None) -> None:
        """Initialize STFPM detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            config = STFPMConfig()
        elif isinstance(config, dict):
            config = STFPMConfig(**config)

        super().__init__(config)
        self.stfpm_config: STFPMConfig = config
        # Override _config to use the specific STFPM config
        self._config = config

        # Initialize teacher (pretrained, frozen)
        self._init_backbone()
        self.teacher = self.backbone

        # Initialize student (trainable)
        self.student = StudentNetwork(
            backbone_name=config.backbone,
            layers=config.layers,
        ).to(self.device)

    def _compute_layer_loss(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature matching loss for a single layer.

        Uses normalized L2 distance.

        Args:
            teacher_feat: Teacher features [B, C, H, W]
            student_feat: Student features [B, C, H, W]

        Returns:
            Loss tensor
        """
        # L2 normalize along channel dimension
        teacher_norm = nn.functional.normalize(teacher_feat, p=2, dim=1)
        student_norm = nn.functional.normalize(student_feat, p=2, dim=1)

        # Mean squared error
        loss = torch.mean((teacher_norm - student_norm) ** 2)
        return loss

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> STFPMDetector:
        """Train student network on normal data.

        Args:
            data: Normal images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, np.ndarray[Any, Any]):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)
        n_samples = data.shape[0]
        logger.info(f"Training STFPM student on {n_samples} images")

        # Setup optimizer
        optimizer = optim.SGD(
            self.student.parameters(),
            lr=self.stfpm_config.learning_rate,
            momentum=0.9,
            weight_decay=self.stfpm_config.weight_decay,
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.stfpm_config.num_epochs // 2, gamma=0.1
        )

        # Create data loader
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.stfpm_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        self.student.train()
        self.teacher.eval()

        for epoch in range(self.stfpm_config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)

                # Get teacher features (no grad)
                with torch.no_grad():
                    teacher_features = self.teacher(batch)

                # Get student features
                student_features = self.student(batch)

                # Compute loss for each layer
                total_loss = torch.tensor(0.0, device=self.device)
                for layer in self.stfpm_config.layers:
                    if layer in teacher_features and layer in student_features:
                        layer_loss = self._compute_layer_loss(
                            teacher_features[layer],
                            student_features[layer],
                        )
                        total_loss = total_loss + layer_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    f"Epoch {epoch + 1}/{self.stfpm_config.num_epochs}, Loss: {avg_loss:.6f}"
                )

        self.student.eval()
        self._is_fitted = True
        logger.info("STFPM training complete")
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using teacher-student discrepancy.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Detection results dict
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if isinstance(data, np.ndarray[Any, Any]):
            data = torch.from_numpy(data).float()

        original_size = data.shape[-2:]
        data = self.preprocess(data)

        all_scores = []
        all_maps = []
        all_features = []

        self.student.eval()
        self.teacher.eval()

        batch_size = self.stfpm_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                teacher_features = self.teacher(batch)
                student_features = self.student(batch)

            # Compute anomaly maps from feature discrepancy
            anomaly_maps = self._compute_anomaly_maps(
                teacher_features, student_features, original_size
            )

            # Image-level scores
            image_scores = anomaly_maps.view(len(batch), -1).max(dim=1)[0]

            # Extract features for fusion (concatenate discrepancies)
            feat_list = []
            for layer in self.stfpm_config.layers:
                if layer in teacher_features:
                    t_feat = teacher_features[layer]
                    s_feat = student_features[layer]
                    diff = (t_feat - s_feat).pow(2).mean(dim=[2, 3])
                    feat_list.append(diff)
            features = torch.cat(feat_list, dim=1)

            all_scores.append(image_scores)
            all_maps.append(anomaly_maps)
            all_features.append(features)

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

    def _compute_anomaly_maps(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute anomaly maps from feature discrepancy.

        Args:
            teacher_features: Teacher layer features
            student_features: Student layer features
            original_size: Target output size

        Returns:
            Anomaly maps [B, H, W]
        """
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.zeros(batch_size, *original_size, device=self.device)

        for layer in self.stfpm_config.layers:
            if layer not in teacher_features or layer not in student_features:
                continue

            t_feat = teacher_features[layer]
            s_feat = student_features[layer]

            # Normalize features
            t_norm = nn.functional.normalize(t_feat, p=2, dim=1)
            s_norm = nn.functional.normalize(s_feat, p=2, dim=1)

            # Compute squared difference
            diff = (t_norm - s_norm).pow(2).mean(dim=1)  # [B, H, W]

            # Upsample to original size
            diff_up = nn.functional.interpolate(
                diff.unsqueeze(1),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            anomaly_map = anomaly_map + diff_up

        # Average across layers
        anomaly_map = anomaly_map / len(self.stfpm_config.layers)

        # Gaussian smoothing
        anomaly_map_np = anomaly_map.cpu().numpy()
        for i in range(batch_size):
            anomaly_map_np[i] = gaussian_filter(anomaly_map_np[i], sigma=4)

        return torch.from_numpy(anomaly_map_np).to(self.device)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, 128] normalized for fusion
        """
        if isinstance(data, np.ndarray[Any, Any]):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)

        all_features = []

        self.student.eval()
        self.teacher.eval()

        batch_size = self.stfpm_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                teacher_features = self.teacher(batch)
                student_features = self.student(batch)

            # Concatenate layer-wise discrepancies
            feat_list = []
            for layer in self.stfpm_config.layers:
                if layer in teacher_features:
                    t_feat = teacher_features[layer]
                    s_feat = student_features[layer]
                    # Global average of squared difference
                    diff = (t_feat - s_feat).pow(2).mean(dim=[2, 3])
                    feat_list.append(diff)

            features = torch.cat(feat_list, dim=1)
            all_features.append(features)

        features = torch.cat(all_features, dim=0)

        # Project to 128D
        if features.shape[1] != 128:
            if not hasattr(self, "_fusion_projection"):
                self._fusion_projection = nn.Linear(features.shape[1], 128).to(features.device)
            features = self._fusion_projection(features)

        features = nn.functional.normalize(features, p=2, dim=1)
        return features

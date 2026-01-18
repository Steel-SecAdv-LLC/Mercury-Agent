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
Dual-Student Knowledge Distillation for Anomaly Detection

Implementation based on recent SOTA papers (2024):
- "Dual-Student Knowledge Distillation for Visual Anomaly Detection"
- "Generalist Multi-Class AD via Distillation to Two Heterogeneous Students"

Key Innovation:
Two students with inverted architectures learning from one teacher.
- Student 1 (Encoder-Decoder): Specialized for patch-level defects
- Student 2 (Encoder-Encoder): Optimized for semantic anomalies

The dual structure enhances normal consistency while introducing
diversity for anomaly representation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn, optim

logger = logging.getLogger(__name__)


@dataclass
class DualStudentConfig:
    """Configuration for Dual-Student distillation.

    Attributes:
        backbone: Backbone architecture for teacher/students
        hidden_dim: Hidden dimension for students
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        temperature: Distillation temperature
        alpha: Weight for distillation loss vs reconstruction
    """

    backbone: str = "resnet18"
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    temperature: float = 4.0
    alpha: float = 0.5
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EncoderDecoderStudent(nn.Module):
    """Encoder-Decoder student for patch-level anomaly detection.

    Learns to reconstruct teacher features, specialized for
    detecting local/patch-level defects.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        """Initialize encoder-decoder student.

        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Reconstructed features [B, C, H, W]
        """
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded: torch.Tensor = self.decoder(bottleneck)
        return decoded


class EncoderEncoderStudent(nn.Module):
    """Encoder-Encoder student for semantic anomaly detection.

    Uses skip connections to preserve semantic information,
    specialized for detecting semantic/contextual anomalies.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
    ):
        """Initialize encoder-encoder student.

        Args:
            in_channels: Input feature channels
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # First encoder block
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Second encoder block with skip
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Attention for feature recombination
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: Input features [B, C, H, W]

        Returns:
            Processed features [B, C, H, W]
        """
        # First encoding
        enc1 = self.enc1(x)

        # Second encoding with residual
        enc2 = self.enc2(enc1) + enc1

        # Attention modulation
        attn = self.attention(enc2)
        attn = attn.view(-1, enc2.shape[1], 1, 1)
        modulated = enc2 * attn

        # Output
        output: torch.Tensor = self.output(modulated)
        return output


class DualStudentDistillation(nn.Module):
    """Dual-Student Knowledge Distillation for anomaly detection.

    Trains two student networks with different architectures
    to learn from a pre-trained teacher. Anomalies are detected
    based on discrepancy between teacher and student outputs.

    Example:
        >>> distiller = DualStudentDistillation()
        >>> distiller.fit(normal_images)
        >>> anomaly_scores = distiller.detect(test_images)
    """

    def __init__(self, config: DualStudentConfig | dict[str, Any] | None = None) -> None:
        """Initialize dual-student distillation.

        Args:
            config: Distillation configuration
        """
        super().__init__()

        if config is None:
            self.config = DualStudentConfig()
        elif isinstance(config, dict):
            self.config = DualStudentConfig(**config)
        else:
            self.config = config

        self.device = torch.device(self.config.device)

        # Teacher (pre-trained, frozen)
        self.teacher: nn.Module | None = None

        # Students
        self.student1: EncoderDecoderStudent | None = None
        self.student2: EncoderEncoderStudent | None = None

        self._is_fitted = False
        self._feature_dim: int = 0

    def _initialize_networks(self, sample_input: torch.Tensor) -> None:
        """Initialize networks based on input dimensions.

        Args:
            sample_input: Sample input for dimension inference
        """
        # Initialize teacher
        from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor

        self.teacher = FeatureExtractor(
            backbone_name=self.config.backbone,
            layers=["layer2", "layer3"],
            device=self.device,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Get feature dimensions
        with torch.no_grad():
            features = self.teacher(sample_input)
            total_channels = sum(f.shape[1] for f in features.values())

        self._feature_dim = total_channels

        # Initialize students
        self.student1 = EncoderDecoderStudent(total_channels, self.config.hidden_dim).to(
            self.device
        )

        self.student2 = EncoderEncoderStudent(total_channels, self.config.hidden_dim).to(
            self.device
        )

        logger.info(f"Initialized dual-student networks (feature_dim={total_channels})")

    def _aggregate_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale teacher features.

        Args:
            features: Dict of layer features

        Returns:
            Aggregated features [B, C_total, H, W]
        """
        feature_list = list(features.values())

        # Find minimum spatial size
        min_h = min(f.shape[2] for f in feature_list)
        min_w = min(f.shape[3] for f in feature_list)

        resized = []
        for feat in feature_list:
            if feat.shape[2] != min_h or feat.shape[3] != min_w:
                feat = nn.functional.interpolate(
                    feat, size=(min_h, min_w), mode="bilinear", align_corners=False
                )
            resized.append(feat)

        return torch.cat(resized, dim=1)

    def _distillation_loss(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Uses cosine similarity for better gradient flow.

        Args:
            teacher_feat: Teacher features [B, C, H, W]
            student_feat: Student features [B, C, H, W]

        Returns:
            Loss tensor
        """
        # Normalize along channel dimension
        t_norm = nn.functional.normalize(teacher_feat, p=2, dim=1)
        s_norm = nn.functional.normalize(student_feat, p=2, dim=1)

        # Cosine similarity loss (1 - cosine similarity)
        similarity = (t_norm * s_norm).sum(dim=1)
        loss = (1 - similarity).mean()

        return loss

    def fit(self, data: torch.Tensor) -> DualStudentDistillation:
        """Train students on normal data.

        Args:
            data: Normal images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)

        # Initialize networks
        if self.student1 is None:
            self._initialize_networks(data[:1])

        logger.info(f"Training dual-student on {len(data)} images")

        # Assert networks are initialized
        assert self.student1 is not None
        assert self.student2 is not None
        assert self.teacher is not None

        # Setup optimizer
        params = list(self.student1.parameters()) + list(self.student2.parameters())
        optimizer = optim.Adam(params, lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)

        # Data loader
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        self.student1.train()
        self.student2.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)

                # Get teacher features
                with torch.no_grad():
                    teacher_features = self.teacher(batch)
                    teacher_agg = self._aggregate_features(teacher_features)

                # Get student outputs
                student1_out = self.student1(teacher_agg)
                student2_out = self.student2(teacher_agg)

                # Distillation losses
                loss1 = self._distillation_loss(teacher_agg, student1_out)
                loss2 = self._distillation_loss(teacher_agg, student2_out)

                # Combined loss
                total_loss = loss1 + loss2

                optimizer.zero_grad()
                total_loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.6f}")

        self.student1.eval()
        self.student2.eval()
        self._is_fitted = True

        logger.info("Dual-student training complete")
        return self

    def detect(self, data: torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using dual-student discrepancy.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Detection results dict
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before detect()")

        data = data.to(self.device)

        # Assert networks are initialized
        assert self.student1 is not None
        assert self.student2 is not None
        assert self.teacher is not None

        self.student1.eval()
        self.student2.eval()

        with torch.no_grad():
            # Teacher features
            teacher_features = self.teacher(data)
            teacher_agg = self._aggregate_features(teacher_features)

            # Student outputs
            student1_out = self.student1(teacher_agg)
            student2_out = self.student2(teacher_agg)

            # Anomaly scores = max discrepancy from teacher
            diff1 = (teacher_agg - student1_out).pow(2).mean(dim=1)  # [B, H, W]
            diff2 = (teacher_agg - student2_out).pow(2).mean(dim=1)  # [B, H, W]

            # Take maximum discrepancy
            anomaly_maps = torch.max(diff1, diff2)

            # Image-level scores
            scores = anomaly_maps.view(len(data), -1).max(dim=1)[0]

        return {
            "scores": scores.cpu().numpy(),
            "anomaly_maps": anomaly_maps.cpu().numpy(),
            "is_anomaly": (scores > scores.mean()).cpu().numpy(),
            "student1_maps": diff1.cpu().numpy(),
            "student2_maps": diff2.cpu().numpy(),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass for PyTorch integration."""
        return self.detect(x)

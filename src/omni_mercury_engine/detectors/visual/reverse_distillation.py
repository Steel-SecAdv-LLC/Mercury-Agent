# Copyright (C) 2025 Steel Security Advisors LLC
"""Reverse Distillation for Anomaly Detection.

Implementation of Reverse Distillation from CVPR 2022.
Solves the student mimicry problem in knowledge distillation.

Key Innovations:
    1. Reverse information flow: Student → Bottleneck → Decoder → Compare with Teacher
    2. One-class embedding bottleneck
    3. Multi-scale feature reconstruction
    4. Prevents student from learning to replicate anomalies

Reference:
    Deng & Li. "Anomaly Detection via Reverse Distillation from
    One-Class Embedding"
    https://arxiv.org/abs/2201.10703
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor
from omni_mercury_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ReverseDistillationConfig(VisualDetectorConfig):
    """Configuration for Reverse Distillation detector.

    Attributes:
        bottleneck_dim: Dimension of the one-class embedding bottleneck
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
        oce_gamma: Gamma parameter for one-class embedding loss
        layers: Feature extraction layers to use
    """

    bottleneck_dim: int = 256
    learning_rate: float = 0.005
    num_epochs: int = 200
    weight_decay: float = 1e-5
    oce_gamma: float = 0.1  # One-class embedding gamma parameter
    layers: list[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3"])


class OCEBottleneck(nn.Module):
    """One-Class Embedding Bottleneck.

    Compresses features into a compact representation that captures only normal patterns.
    """

    def __init__(self, in_channels: int, bottleneck_dim: int = 256) -> None:
        """Initialize bottleneck.

        Args:
            in_channels: Input channel dimension
            bottleneck_dim: Bottleneck dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, 1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 1),
            nn.BatchNorm2d(bottleneck_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck."""
        result: torch.Tensor = self.encoder(x)
        return result


class MultiScaleDecoder(nn.Module):
    """Multi-scale feature decoder for reconstruction.

    Reconstructs features at multiple scales to match teacher.
    """

    def __init__(
        self,
        bottleneck_dim: int,
        output_channels: list[int],
        spatial_sizes: list[tuple[int, int]],
    ):
        """Initialize decoder.

        Args:
            bottleneck_dim: Input bottleneck dimension
            output_channels: Output channels for each scale
            spatial_sizes: Spatial sizes for each scale
        """
        super().__init__()

        self.decoders = nn.ModuleList()
        for out_ch in output_channels:
            decoder = nn.Sequential(
                nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_dim, out_ch, 1),
            )
            self.decoders.append(decoder)

        self.spatial_sizes = spatial_sizes

    def forward(self, bottleneck: torch.Tensor) -> list[torch.Tensor]:
        """Decode bottleneck to multi-scale features.

        Args:
            bottleneck: Bottleneck features [B, C, H, W]

        Returns:
            List of decoded features at each scale
        """
        outputs = []
        for decoder, size in zip(self.decoders, self.spatial_sizes, strict=False):
            # Upsample bottleneck to target size
            x = nn.functional.interpolate(
                bottleneck, size=size, mode="bilinear", align_corners=False
            )
            outputs.append(decoder(x))
        return outputs


class ReverseDistillationDetector(BaseVisualDetector):
    """Reverse Distillation anomaly detector.

    Uses reverse information flow to prevent student mimicry:
    Teacher → Student Encoder → Bottleneck → Decoder → Compare with Teacher

    Example:
        >>> detector = ReverseDistillationDetector()
        >>> detector.fit(normal_images)
        >>> results = detector.detect(test_images)
    """

    def __init__(self, config: ReverseDistillationConfig | dict[str, Any] | None = None) -> None:
        """Initialize detector."""
        if config is None:
            config = ReverseDistillationConfig()
        elif isinstance(config, dict):
            config = ReverseDistillationConfig(**config)

        super().__init__(config)
        self.rd_config: ReverseDistillationConfig = config

        # Initialize teacher (pretrained, frozen)
        self._init_backbone()
        self.teacher = self.backbone

        # Student encoder (same architecture, trainable)
        self.student_encoder = FeatureExtractor(
            backbone_name=config.backbone,
            layers=config.layers,
            device=self.device,
            pretrained=False,  # Train from scratch
        )

        # Unfreeze student
        for param in self.student_encoder.parameters():
            param.requires_grad = True

        # These will be initialized after first forward pass
        self.bottleneck: OCEBottleneck | None = None
        self.decoder: MultiScaleDecoder | None = None

        self._layer_channels: dict[str, int] = {}
        self._layer_sizes: dict[str, tuple[int, int]] = {}

    def _initialize_components(self, sample_input: torch.Tensor) -> None:
        """Initialize bottleneck and decoder based on feature dimensions.

        Args:
            sample_input: Sample input for dimension inference
        """
        with torch.no_grad():
            features = self.teacher(sample_input)

        # Get layer info
        for layer, feat in features.items():
            self._layer_channels[layer] = feat.shape[1]
            self._layer_sizes[layer] = (feat.shape[2], feat.shape[3])

        # Total channels for bottleneck input
        total_channels = sum(self._layer_channels.values())

        # Initialize bottleneck
        self.bottleneck = OCEBottleneck(total_channels, self.rd_config.bottleneck_dim).to(
            self.device
        )

        # Initialize decoder
        output_channels = [self._layer_channels[layer] for layer in self.rd_config.layers]
        spatial_sizes = [self._layer_sizes[layer] for layer in self.rd_config.layers]

        self.decoder = MultiScaleDecoder(
            self.rd_config.bottleneck_dim,
            output_channels,
            spatial_sizes,
        ).to(self.device)

        logger.info(
            f"Initialized RD components: bottleneck={self.rd_config.bottleneck_dim}, "
            f"layers={list(self._layer_channels.keys())}"
        )

    def _aggregate_student_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale student features for bottleneck.

        Args:
            features: Dict of layer features

        Returns:
            Aggregated features [B, sum(C_i), H_min, W_min]
        """
        # Find minimum spatial size
        min_h = min(f.shape[2] for f in features.values())
        min_w = min(f.shape[3] for f in features.values())

        resized = []
        for layer in self.rd_config.layers:
            if layer in features:
                feat = features[layer]
                if feat.shape[2] != min_h or feat.shape[3] != min_w:
                    feat = nn.functional.interpolate(
                        feat, size=(min_h, min_w), mode="bilinear", align_corners=False
                    )
                resized.append(feat)

        return torch.cat(resized, dim=1)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> ReverseDistillationDetector:
        """Train student encoder, bottleneck, and decoder.

        Args:
            data: Normal images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)
        logger.info(f"Training Reverse Distillation on {data.shape[0]} images")

        # Initialize components
        if self.bottleneck is None:
            self._initialize_components(data[:1])

        # Assert components are initialized
        assert self.bottleneck is not None
        assert self.decoder is not None

        # Setup optimizer for student, bottleneck, decoder
        params = (
            list(self.student_encoder.parameters())
            + list(self.bottleneck.parameters())
            + list(self.decoder.parameters())
        )
        optimizer = optim.Adam(
            params,
            lr=self.rd_config.learning_rate,
            weight_decay=self.rd_config.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.rd_config.num_epochs)

        # Data loader
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.rd_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        self.student_encoder.train()
        self.bottleneck.train()
        self.decoder.train()
        self.teacher.eval()

        for epoch in range(self.rd_config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)

                # Teacher features (frozen)
                with torch.no_grad():
                    teacher_features = self.teacher(batch)

                # Student features
                student_features = self.student_encoder(batch)

                # Aggregate and pass through bottleneck
                aggregated = self._aggregate_student_features(student_features)
                bottleneck_out = self.bottleneck(aggregated)

                # Decode to multi-scale
                decoded_features = self.decoder(bottleneck_out)

                # Compute reconstruction loss
                total_loss = torch.tensor(0.0, device=self.device)
                for i, layer in enumerate(self.rd_config.layers):
                    if layer in teacher_features:
                        t_feat = teacher_features[layer]
                        d_feat = decoded_features[i]

                        # Cosine similarity loss
                        t_norm = nn.functional.normalize(t_feat, p=2, dim=1)
                        d_norm = nn.functional.normalize(d_feat, p=2, dim=1)
                        loss = 1 - (t_norm * d_norm).sum(dim=1).mean()
                        total_loss = total_loss + loss

                optimizer.zero_grad()
                total_loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"Epoch {epoch + 1}/{self.rd_config.num_epochs}, Loss: {avg_loss:.6f}")

        self.student_encoder.eval()
        self.bottleneck.eval()
        self.decoder.eval()
        self._is_fitted = True

        logger.info("Reverse Distillation training complete")
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using reconstruction error.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Detection results dict
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        original_size: tuple[int, int] = (data.shape[-2], data.shape[-1])
        data = self.preprocess(data)

        all_scores = []
        all_maps = []
        all_features = []

        # Assert components are initialized
        assert self.bottleneck is not None
        assert self.decoder is not None

        self.student_encoder.eval()
        self.bottleneck.eval()
        self.decoder.eval()
        self.teacher.eval()

        batch_size = self.rd_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                teacher_features = self.teacher(batch)
                student_features = self.student_encoder(batch)

                aggregated = self._aggregate_student_features(student_features)
                bottleneck_out = self.bottleneck(aggregated)
                decoded_features = self.decoder(bottleneck_out)

            # Compute anomaly maps
            anomaly_map = self._compute_anomaly_map(
                teacher_features, decoded_features, original_size
            )

            # Image-level scores
            image_scores = anomaly_map.view(len(batch), -1).max(dim=1)[0]

            # Features for fusion (bottleneck representation)
            features = nn.functional.adaptive_avg_pool2d(bottleneck_out, 1).flatten(1)

            all_scores.append(image_scores)
            all_maps.append(anomaly_map)
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

    def _compute_anomaly_map(
        self,
        teacher_features: dict[str, torch.Tensor],
        decoded_features: list[torch.Tensor],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        """Compute anomaly map from reconstruction error.

        Args:
            teacher_features: Teacher layer features
            decoded_features: Decoded layer features
            original_size: Target output size

        Returns:
            Anomaly map [B, H, W]
        """
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.zeros(batch_size, *original_size, device=self.device)

        for i, layer in enumerate(self.rd_config.layers):
            if layer not in teacher_features:
                continue

            t_feat = teacher_features[layer]
            d_feat = decoded_features[i]

            # Cosine distance
            t_norm = nn.functional.normalize(t_feat, p=2, dim=1)
            d_norm = nn.functional.normalize(d_feat, p=2, dim=1)

            # 1 - cosine similarity
            distance = 1 - (t_norm * d_norm).sum(dim=1)

            # Upsample
            distance_up = nn.functional.interpolate(
                distance.unsqueeze(1),
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            anomaly_map = anomaly_map + distance_up

        anomaly_map = anomaly_map / len(self.rd_config.layers)

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
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)

        # Assert bottleneck is initialized
        assert self.bottleneck is not None

        all_features = []

        batch_size = self.rd_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                student_features = self.student_encoder(batch)
                aggregated = self._aggregate_student_features(student_features)
                bottleneck_out = self.bottleneck(aggregated)

            # Global average of bottleneck
            features = nn.functional.adaptive_avg_pool2d(bottleneck_out, 1).flatten(1)
            all_features.append(features)

        features = torch.cat(all_features, dim=0)

        # Project to 128D
        if features.shape[1] != 128:
            if not hasattr(self, "_fusion_projection"):
                self._fusion_projection = nn.Linear(features.shape[1], 128).to(features.device)
            features = self._fusion_projection(features)

        features = nn.functional.normalize(features, p=2, dim=1)
        return features

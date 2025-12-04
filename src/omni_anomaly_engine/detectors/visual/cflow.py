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

"""
CFlow-AD: Real-Time Unsupervised Anomaly Detection with Localization via
Conditional Normalizing Flows

Implementation of CFlow from WACV 2022.
Uses normalizing flows for precise anomaly localization.

Key Features:
    1. Conditional normalizing flows for density estimation
    2. Position-encoding aware anomaly detection
    3. Precise pixel-level localization
    4. Faster training than memory-bank methods

Reference:
    Gudovskiy et al. "CFLOW-AD: Real-Time Unsupervised Anomaly Detection
    with Localization via Conditional Normalizing Flows"
    https://arxiv.org/abs/2107.12571
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from omni_anomaly_engine.detectors.visual.backbone import FeatureExtractor
from omni_anomaly_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class CFlowConfig(VisualDetectorConfig):
    """Configuration for CFlow detector.

    Attributes:
        num_flows: Number of flow layers
        hidden_dim: Hidden dimension in flow networks
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
    """

    num_flows: int = 8
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    clamp_value: float = 3.0
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial conditioning."""

    def __init__(self, channels: int, max_size: int = 64):
        """Initialize positional encoding.

        Args:
            channels: Number of encoding channels
            max_size: Maximum spatial size
        """
        super().__init__()

        self.channels = channels

        # Create position encodings
        pe = torch.zeros(max_size, max_size, channels)

        y_pos = torch.arange(max_size).unsqueeze(1).float()
        x_pos = torch.arange(max_size).unsqueeze(0).float()

        div_term = torch.exp(
            torch.arange(0, channels, 2).float()
            * (-math.log(10000.0) / channels)
        )

        # Alternate between sin and cos for x and y
        for i in range(channels // 4):
            pe[:, :, 4 * i] = torch.sin(y_pos * div_term[i])
            pe[:, :, 4 * i + 1] = torch.cos(y_pos * div_term[i])
            pe[:, :, 4 * i + 2] = torch.sin(x_pos.T * div_term[i])
            pe[:, :, 4 * i + 3] = torch.cos(x_pos.T * div_term[i])

        self.register_buffer("pe", pe)

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Get positional encoding for given size.

        Args:
            h: Height
            w: Width

        Returns:
            Positional encoding [1, channels, h, w]
        """
        pe = self.pe[:h, :w, :].permute(2, 0, 1).unsqueeze(0)
        return pe


class AffineCoupling(nn.Module):
    """Affine coupling layer for normalizing flow.

    Implements the affine transformation:
    z1 = x1
    z2 = x2 * exp(s(x1)) + t(x1)
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        hidden_dim: int = 256,
        clamp: float = 3.0,
    ):
        """Initialize affine coupling.

        Args:
            in_channels: Input channel dimension
            cond_channels: Conditioning channel dimension
            hidden_dim: Hidden layer dimension
            clamp: Clamping value for stability
        """
        super().__init__()

        self.clamp = clamp
        self.split_dim = in_channels // 2

        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Conv2d(self.split_dim + cond_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.split_dim, 1),
        )

        self.translate_net = nn.Sequential(
            nn.Conv2d(self.split_dim + cond_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.split_dim, 1),
        )

        # Initialize last layer to zero for stable training
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.translate_net[-1].weight)
        nn.init.zeros_(self.translate_net[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward/inverse affine coupling.

        Args:
            x: Input tensor [B, C, H, W]
            cond: Conditioning tensor [B, C_cond, H, W]
            reverse: If True, apply inverse transformation

        Returns:
            Tuple of (output, log_det_jacobian)
        """
        x1, x2 = x.chunk(2, dim=1)

        # Compute scale and translation
        cond_input = torch.cat([x1, cond], dim=1)
        log_scale = self.scale_net(cond_input)
        log_scale = self.clamp * torch.tanh(log_scale / self.clamp)
        translate = self.translate_net(cond_input)

        if reverse:
            z1 = x1
            z2 = (x2 - translate) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=[1, 2, 3])
        else:
            z1 = x1
            z2 = x2 * torch.exp(log_scale) + translate
            log_det = log_scale.sum(dim=[1, 2, 3])

        z = torch.cat([z1, z2], dim=1)
        return z, log_det


class ConditionalNormalizingFlow(nn.Module):
    """Conditional normalizing flow for anomaly detection.

    Learns the density of normal features conditioned on position.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        num_flows: int = 8,
        hidden_dim: int = 256,
        clamp: float = 3.0,
    ):
        """Initialize normalizing flow.

        Args:
            in_channels: Input feature channels
            cond_channels: Conditioning channels (position encoding)
            num_flows: Number of flow layers
            hidden_dim: Hidden dimension in coupling layers
            clamp: Clamping value for stability
        """
        super().__init__()

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                AffineCoupling(in_channels, cond_channels, hidden_dim, clamp)
            )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all flows.

        Args:
            x: Input features [B, C, H, W]
            cond: Position conditioning [B, C_cond, H, W]

        Returns:
            Tuple of (latent z, total log determinant)
        """
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)

        for flow in self.flows:
            z, log_det = flow(z, cond)
            total_log_det = total_log_det + log_det

            # Channel-wise permutation
            z = z.flip(dims=[1])

        return z, total_log_det

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse pass (sampling direction).

        Args:
            z: Latent tensor [B, C, H, W]
            cond: Position conditioning [B, C_cond, H, W]

        Returns:
            Tuple of (reconstructed x, total log determinant)
        """
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)

        for flow in reversed(self.flows):
            x = x.flip(dims=[1])
            x, log_det = flow(x, cond, reverse=True)
            total_log_det = total_log_det + log_det

        return x, total_log_det

    def log_prob(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of x.

        Args:
            x: Input features [B, C, H, W]
            cond: Position conditioning [B, C_cond, H, W]

        Returns:
            Log probability [B]
        """
        z, log_det = self.forward(x, cond)

        # Standard normal prior
        log_prior = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=[1, 2, 3])

        return log_prior + log_det


class CFlowDetector(BaseVisualDetector):
    """CFlow anomaly detector using conditional normalizing flows.

    Models feature density conditioned on spatial position.
    Anomaly score = negative log likelihood.

    Example:
        >>> detector = CFlowDetector()
        >>> detector.fit(normal_images)
        >>> results = detector.detect(test_images)
    """

    def __init__(self, config: CFlowConfig | dict[str, Any] | None = None):
        """Initialize CFlow detector."""
        if config is None:
            config = CFlowConfig()
        elif isinstance(config, dict):
            config = CFlowConfig(**config)

        super().__init__(config)
        self.cflow_config: CFlowConfig = config

        # Initialize backbone
        self._init_backbone()

        # Flow models (one per layer)
        self.flows: nn.ModuleDict = nn.ModuleDict()
        self.position_encodings: nn.ModuleDict = nn.ModuleDict()

        self._initialized = False

    def _initialize_flows(self, sample_input: torch.Tensor) -> None:
        """Initialize flow models based on feature dimensions."""
        with torch.no_grad():
            features = self.backbone(sample_input)

        for layer in self.cflow_config.layers:
            if layer not in features:
                continue

            feat = features[layer]
            in_channels = feat.shape[1]
            h, w = feat.shape[2], feat.shape[3]

            # Ensure even channels for splitting
            if in_channels % 2 != 0:
                in_channels += 1

            # Position encoding (same dimension as features)
            cond_channels = in_channels
            self.position_encodings[layer] = PositionalEncoding2D(
                cond_channels, max(h, w)
            ).to(self.device)

            # Normalizing flow
            self.flows[layer] = ConditionalNormalizingFlow(
                in_channels=in_channels,
                cond_channels=cond_channels,
                num_flows=self.cflow_config.num_flows,
                hidden_dim=self.cflow_config.hidden_dim,
                clamp=self.cflow_config.clamp_value,
            ).to(self.device)

            logger.info(
                f"Initialized flow for {layer}: "
                f"in_channels={in_channels}, spatial=({h}, {w})"
            )

        self._initialized = True

    def fit(self, data: np.ndarray | torch.Tensor) -> "CFlowDetector":
        """Train normalizing flows on normal data.

        Args:
            data: Normal images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)
        logger.info(f"Training CFlow on {data.shape[0]} images")

        # Initialize flows
        if not self._initialized:
            self._initialize_flows(data[:1])

        # Optimizer for all flows
        params = []
        for flow in self.flows.values():
            params.extend(flow.parameters())

        optimizer = optim.Adam(
            params,
            lr=self.cflow_config.learning_rate,
            weight_decay=1e-5,
        )

        # Data loader
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cflow_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Training loop
        for flow in self.flows.values():
            flow.train()

        for epoch in range(self.cflow_config.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)

                with torch.no_grad():
                    features = self.backbone(batch)

                total_loss = torch.tensor(0.0, device=self.device)

                for layer in self.cflow_config.layers:
                    if layer not in features or layer not in self.flows:
                        continue

                    feat = features[layer]
                    b, c, h, w = feat.shape

                    # Pad if odd channels
                    if c % 2 != 0:
                        feat = torch.cat([feat, feat[:, :1]], dim=1)

                    # Get position encoding
                    pos_enc = self.position_encodings[layer](h, w)
                    pos_enc = pos_enc.expand(b, -1, -1, -1)

                    # Negative log likelihood
                    log_prob = self.flows[layer].log_prob(feat, pos_enc)
                    loss = -log_prob.mean()
                    total_loss = total_loss + loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    f"Epoch {epoch + 1}/{self.cflow_config.num_epochs}, "
                    f"NLL: {avg_loss:.4f}"
                )

        for flow in self.flows.values():
            flow.eval()

        self._is_fitted = True
        logger.info("CFlow training complete")
        return self

    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies using negative log likelihood.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Detection results dict
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

        for flow in self.flows.values():
            flow.eval()

        batch_size = self.cflow_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            # Compute anomaly maps from all layers
            batch_maps = []
            for layer in self.cflow_config.layers:
                if layer not in features or layer not in self.flows:
                    continue

                feat = features[layer]
                b, c, h, w = feat.shape

                if c % 2 != 0:
                    feat = torch.cat([feat, feat[:, :1]], dim=1)

                pos_enc = self.position_encodings[layer](h, w).expand(b, -1, -1, -1)

                # Get per-pixel log probability
                z, _ = self.flows[layer].forward(feat, pos_enc)

                # Anomaly score = sum of squared latent (negative log prior)
                pixel_scores = 0.5 * (z ** 2).sum(dim=1)  # [B, H, W]

                # Upsample to original size
                pixel_scores_up = nn.functional.interpolate(
                    pixel_scores.unsqueeze(1),
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                batch_maps.append(pixel_scores_up)

            # Average across layers
            anomaly_map = torch.stack(batch_maps).mean(dim=0)

            # Gaussian smoothing
            anomaly_map_np = anomaly_map.cpu().numpy()
            for j in range(len(batch)):
                anomaly_map_np[j] = gaussian_filter(anomaly_map_np[j], sigma=4)

            anomaly_map = torch.from_numpy(anomaly_map_np).to(self.device)

            # Image-level scores
            image_scores = anomaly_map.view(len(batch), -1).max(dim=1)[0]

            # Features for fusion (concatenate mean latents)
            feat_list = []
            for layer in self.cflow_config.layers:
                if layer in features:
                    feat_list.append(features[layer].mean(dim=[2, 3]))
            fusion_features = torch.cat(feat_list, dim=1)

            all_scores.append(image_scores)
            all_maps.append(anomaly_map)
            all_features.append(fusion_features)

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

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
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

        batch_size = self.cflow_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            feat_list = []
            for layer in self.cflow_config.layers:
                if layer in features:
                    feat_list.append(features[layer].mean(dim=[2, 3]))

            fusion_features = torch.cat(feat_list, dim=1)
            all_features.append(fusion_features)

        features = torch.cat(all_features, dim=0)

        # Project to 128D
        if features.shape[1] != 128:
            if not hasattr(self, "_fusion_projection"):
                self._fusion_projection = nn.Linear(
                    features.shape[1], 128
                ).to(features.device)
            features = self._fusion_projection(features)

        features = nn.functional.normalize(features, p=2, dim=1)
        return features

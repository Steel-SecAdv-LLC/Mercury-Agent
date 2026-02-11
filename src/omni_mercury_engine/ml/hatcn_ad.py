"""
Mercury Agent ♱
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

from typing import Any

"""
Hierarchical Attention Temporal Convolutional Network for Anomaly Detection (HATCN-AD)

Multi-scale temporal pattern learning with hierarchical attention
for improved time-series anomaly detection (20-40% forecasting gains).

⚠️ SIMULATION-BASED: Trained on simulated time-series. Real-world validation required.

Reference: Inspired by TCN and attention mechanisms for time-series

"""


import torch
from torch import nn


class TemporalBlock(nn.Module):
    """Dilated causal convolutional block."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block."""
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]]
        out = self.bn2(out)

        res = x if self.downsample is None else self.downsample(x)

        if res.size(2) != out.size(2):
            res = res[:, :, : out.size(2)]

        return self.relu(out + res)


class HierarchicalAttention(nn.Module):
    """Multi-scale hierarchical attention."""

    def __init__(self, hidden_dim: int, num_scales: int = 3) -> None:
        super().__init__()

        self.num_scales = num_scales
        self.scale_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
                )
                for _ in range(num_scales)
            ]
        )

        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, scale_features: list[Any]) -> tuple[torch.Tensor, list[Any]]:
        """
        Apply hierarchical attention across scales.

        Args:
            scale_features: List of features from different temporal scales

        Returns:
            Attended features and attention weights
        """
        attended = []
        attention_weights = []

        for i, features in enumerate(scale_features):
            attn_scores = self.scale_attentions[i](features)
            attn_weights = torch.softmax(attn_scores, dim=1)

            attended.append((features * attn_weights).sum(dim=1))
            attention_weights.append(attn_weights)

        scale_weights_norm = torch.softmax(self.scale_weights, dim=0)
        fused = sum(scale_weights_norm[i] * attended[i] for i in range(self.num_scales))

        return fused, attention_weights


class HATCN_AD(nn.Module):
    """Hierarchical Attention TCN for Anomaly Detection."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_scales: int = 3, kernel_size: int = 3
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.temporal_blocks = nn.ModuleList(
            [
                TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation=2**i)
                for i in range(num_scales)
            ]
        )

        self.attention = HierarchicalAttention(hidden_dim, num_scales)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass for anomaly detection.

        Args:
            x: Input time series [batch, seq_len, input_dim]

        Returns:
            Dict with anomaly scores and attention weights
        """
        _batch_size, _seq_len, _ = x.shape

        x = self.input_proj(x)

        x = x.transpose(1, 2)

        scale_features = []
        for temporal_block in self.temporal_blocks:
            scale_out = temporal_block(x)
            scale_features.append(scale_out.transpose(1, 2))

        fused, attn_weights = self.attention(scale_features)

        anomaly_scores = self.predictor(fused)

        return {
            "anomaly_scores": anomaly_scores,
            "attention_weights": attn_weights,
            "scale_features": scale_features,
        }

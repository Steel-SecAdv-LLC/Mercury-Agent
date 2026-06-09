# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Attention mechanisms for detector fusion and cross-modal integration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadDetectorAttention(nn.Module):
    """Multi-head attention over different detector outputs."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        """Initialize the instance."""
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:.

            query, key, value: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(attn_output + query)
        return output, attn_weights


class TemporalAttention(nn.Module):
    """Attention mechanism for time series anomalies."""

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        """Initialize the instance."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:.

            x: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, head_dim)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, head_dim)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        output = self.out_proj(attn_output)

        return output, attn_weights


class SpatialAttention(nn.Module):
    """Spatial attention for geographic anomalies."""

    def __init__(self, channels: int) -> None:
        """Initialize the instance."""
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:.

            x: [batch_size, channels, height, width]

        Returns:
            attended: [batch_size, channels, height, width]
        """
        attention_map = torch.sigmoid(self.conv(x))
        return x * attention_map


class CrossModalAttention(nn.Module):
    """Cross-modal attention between different modalities."""

    def __init__(self, dim1: int, dim2: int, hidden_dim: int) -> None:
        """Initialize the instance."""
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:.

            x1: [batch_size, seq_len1, dim1]
            x2: [batch_size, seq_len2, dim2]

        Returns:
            fused: [batch_size, seq_len1, hidden_dim]
            attention_weights: Attention weights
        """
        proj1 = self.proj1(x1)
        proj2 = self.proj2(x2)

        fused, attn_weights = self.attention(proj1, proj2, proj2)

        return fused, attn_weights

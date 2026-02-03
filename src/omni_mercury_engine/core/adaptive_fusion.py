"""
Mercury Agent - Adaptive Fusion Architecture
Copyright (C) 2025 Steel Security Advisory LLC

This module provides enhanced fusion capabilities including:
- Adaptive attention head count based on input complexity
- Temperature-scaled attention with learnable parameters
- Uncertainty quantification with confidence intervals
- Sparse attention patterns for efficiency
- Attention visualization tools

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification result with confidence intervals."""

    mean: torch.Tensor
    std: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
            "lower_bound": self.lower_bound.detach().cpu().numpy().tolist(),
            "upper_bound": self.upper_bound.detach().cpu().numpy().tolist(),
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "confidence_level": self.confidence_level,
        }


@dataclass
class AttentionVisualization:
    """Attention weights visualization data."""

    attention_weights: torch.Tensor
    detector_names: list[str]
    head_contributions: torch.Tensor
    temperature: float
    sparsity_ratio: float

    def get_top_contributors(self, k: int = 5) -> list[tuple[str, float]]:
        """Get top k contributing detectors."""
        mean_weights = self.attention_weights.mean(dim=(0, 1))
        if mean_weights.dim() > 1:
            mean_weights = mean_weights.mean(dim=0)

        values, indices = torch.topk(mean_weights, min(k, len(self.detector_names)))
        return [(self.detector_names[idx.item()], val.item()) for idx, val in zip(indices, values)]


class TemperatureScaledAttention(nn.Module):
    """
    Attention mechanism with learnable temperature scaling.

    Temperature controls the sharpness of attention distribution:
    - Low temperature (< 1): Sharper, more focused attention
    - High temperature (> 1): Softer, more distributed attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
    ):
        """
        Initialize temperature-scaled attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            initial_temperature: Initial temperature value
            learnable_temperature: Whether temperature is learnable
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Explicit validation instead of assert (remains active in optimized code)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer("temperature", torch.tensor(initial_temperature))

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        logger.info(
            f"TemperatureScaledAttention initialized: embed_dim={embed_dim}, "
            f"num_heads={num_heads}, initial_temp={initial_temperature}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply temperature-scaled attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            return_attention: Whether to return attention weights

        Returns:
            output: Attended output [batch_size, seq_len, embed_dim]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores with temperature scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.scale * self.temperature.clamp(min=0.1)
        )

        # Validate attention scores for NaN/Inf before softmax
        if torch.any(~torch.isfinite(scores)):
            logger.warning("Non-finite attention scores detected, replacing with zeros")
            scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()  # type: ignore[no-any-return]


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for efficiency with many detectors.

    Implements top-k attention where only the k most relevant
    detectors contribute to each output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k: int | None = None,
        sparsity_ratio: float = 0.5,
    ):
        """
        Initialize sparse attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_k: Number of top elements to keep (overrides sparsity_ratio)
            sparsity_ratio: Ratio of elements to keep (0.0-1.0)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k = top_k
        self.sparsity_ratio = sparsity_ratio

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        logger.info(
            f"SparseAttention initialized: embed_dim={embed_dim}, "
            f"top_k={top_k}, sparsity_ratio={sparsity_ratio}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply sparse top-k attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            return_attention: Whether to return attention weights

        Returns:
            output: Attended output [batch_size, seq_len, embed_dim]
            attention_weights: Optional sparse attention weights
        """
        batch_size, seq_len, _ = query.shape

        # Determine k
        if self.top_k is not None:
            k = min(self.top_k, seq_len)
        else:
            k = max(1, int(seq_len * self.sparsity_ratio))

        # Project Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_proj = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / self.scale

        # Apply top-k sparsity
        if k < seq_len:
            top_scores, top_indices = torch.topk(scores, k, dim=-1)

            # Create sparse attention mask
            sparse_scores = torch.full_like(scores, float("-inf"))
            sparse_scores.scatter_(-1, top_indices, top_scores)
            scores = sparse_scores

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_sparsity_ratio(self) -> float:
        """Get effective sparsity ratio."""
        return self.sparsity_ratio


class AdaptiveHeadAttention(nn.Module):
    """
    Attention with adaptive head count based on input complexity.

    Dynamically adjusts the number of active attention heads based on
    the complexity of the input (e.g., number of active detectors).
    """

    def __init__(
        self,
        embed_dim: int,
        min_heads: int = 1,
        max_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize adaptive head attention.

        Args:
            embed_dim: Embedding dimension
            min_heads: Minimum number of attention heads
            max_heads: Maximum number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.min_heads = min_heads
        self.max_heads = max_heads

        # Explicit validation instead of assert (remains active in optimized code)
        if embed_dim % max_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by max_heads ({max_heads})"
            )

        self.head_dim = embed_dim // max_heads

        # Projections for all possible heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Head selection network
        self.head_selector = nn.Sequential(
            nn.Linear(embed_dim, max_heads),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # Track active heads for monitoring
        self._active_heads = max_heads

        logger.info(
            f"AdaptiveHeadAttention initialized: embed_dim={embed_dim}, "
            f"min_heads={min_heads}, max_heads={max_heads}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply adaptive head attention.

        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            return_attention: Whether to return attention weights

        Returns:
            output: Attended output [batch_size, seq_len, embed_dim]
            attention_weights: Optional attention weights
        """
        batch_size, seq_len, _ = query.shape

        # Compute head importance based on input
        input_summary = query.mean(dim=1)  # [batch_size, embed_dim]
        head_importance = self.head_selector(input_summary)  # [batch_size, max_heads]

        # Determine number of active heads (based on importance threshold)
        active_mask = head_importance > 0.5
        num_active = active_mask.sum(dim=-1).float().mean().item()
        self._active_heads = max(self.min_heads, min(self.max_heads, int(num_active) + 1))

        # Project Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.max_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.max_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.max_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply head importance weighting
        head_weights = head_importance.unsqueeze(-1).unsqueeze(-1)  # [batch, heads, 1, 1]
        scores = scores * head_weights

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Weight output by head importance
        output = output * head_weights

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output, None

    def get_active_heads(self) -> int:
        """Get number of currently active heads."""
        return self._active_heads


class UncertaintyQuantifier(nn.Module):
    """
    Uncertainty quantification module for fusion outputs.

    Estimates epistemic (model) and aleatoric (data) uncertainty
    using Monte Carlo dropout and ensemble disagreement.
    """

    def __init__(
        self,
        embed_dim: int,
        n_mc_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            embed_dim: Embedding dimension
            n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
            dropout_rate: Dropout rate for MC dropout
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_mc_samples = n_mc_samples

        # MC Dropout layers
        self.mc_dropout = nn.Dropout(dropout_rate)

        # Aleatoric uncertainty head
        self.aleatoric_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus(),  # Ensure positive variance
        )

        logger.info(
            f"UncertaintyQuantifier initialized: embed_dim={embed_dim}, "
            f"n_mc_samples={n_mc_samples}"
        )

    def forward(
        self,
        x: torch.Tensor,
        confidence_level: float = 0.95,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for input tensor.

        Args:
            x: Input tensor [batch_size, embed_dim]
            confidence_level: Confidence level for intervals (default: 0.95)

        Returns:
            UncertaintyEstimate with mean, std, and confidence intervals
        """
        # Monte Carlo sampling for epistemic uncertainty
        self.train()  # Enable dropout
        mc_samples = []
        for _ in range(self.n_mc_samples):
            sample = self.mc_dropout(x)
            mc_samples.append(sample)

        mc_samples = torch.stack(mc_samples, dim=0)  # [n_samples, batch, embed_dim]

        # Epistemic uncertainty (model uncertainty from MC dropout)
        mean = mc_samples.mean(dim=0)
        epistemic_var = mc_samples.var(dim=0).mean().item()

        # Aleatoric uncertainty (data uncertainty)
        aleatoric_var = self.aleatoric_head(x).squeeze(-1)  # [batch_size]
        aleatoric_uncertainty = aleatoric_var.mean().item()

        # Total uncertainty
        total_std = torch.sqrt(mc_samples.var(dim=0) + aleatoric_var.unsqueeze(-1))

        # Compute confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        lower_bound = mean - z_score * total_std
        upper_bound = mean + z_score * total_std

        return UncertaintyEstimate(
            mean=mean,
            std=total_std,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            epistemic_uncertainty=epistemic_var,
            aleatoric_uncertainty=aleatoric_uncertainty,
            confidence_level=confidence_level,
        )


class AdaptiveFusionLayer(nn.Module):
    """
    Enhanced fusion layer combining all adaptive mechanisms.

    Integrates:
    - Adaptive head count based on input complexity
    - Temperature-scaled attention
    - Sparse attention for efficiency
    - Uncertainty quantification
    """

    def __init__(
        self,
        embed_dim: int,
        min_heads: int = 1,
        max_heads: int = 8,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
        enable_sparse: bool = True,
        sparsity_ratio: float = 0.5,
        enable_uncertainty: bool = True,
        n_mc_samples: int = 10,
    ):
        """
        Initialize adaptive fusion layer.

        Args:
            embed_dim: Embedding dimension
            min_heads: Minimum attention heads
            max_heads: Maximum attention heads
            dropout: Dropout probability
            initial_temperature: Initial attention temperature
            learnable_temperature: Whether temperature is learnable
            enable_sparse: Whether to enable sparse attention
            sparsity_ratio: Ratio of elements to keep in sparse attention
            enable_uncertainty: Whether to enable uncertainty quantification
            n_mc_samples: Number of MC samples for uncertainty
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.enable_sparse = enable_sparse
        self.enable_uncertainty = enable_uncertainty

        # Adaptive head attention
        self.adaptive_attention = AdaptiveHeadAttention(
            embed_dim=embed_dim,
            min_heads=min_heads,
            max_heads=max_heads,
            dropout=dropout,
        )

        # Temperature-scaled attention
        self.temp_attention = TemperatureScaledAttention(
            embed_dim=embed_dim,
            num_heads=max_heads,
            dropout=dropout,
            initial_temperature=initial_temperature,
            learnable_temperature=learnable_temperature,
        )

        # Sparse attention (optional)
        if enable_sparse:
            self.sparse_attention = SparseAttention(
                embed_dim=embed_dim,
                num_heads=max_heads,
                dropout=dropout,
                sparsity_ratio=sparsity_ratio,
            )

        # Uncertainty quantifier (optional)
        if enable_uncertainty:
            self.uncertainty_quantifier = UncertaintyQuantifier(
                embed_dim=embed_dim,
                n_mc_samples=n_mc_samples,
                dropout_rate=dropout,
            )

        # Fusion gate to combine different attention outputs
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        logger.info(
            f"AdaptiveFusionLayer initialized: embed_dim={embed_dim}, "
            f"heads={min_heads}-{max_heads}, sparse={enable_sparse}, "
            f"uncertainty={enable_uncertainty}"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_uncertainty: bool = False,
    ) -> dict[str, Any]:
        """
        Apply adaptive fusion.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary containing:
            - output: Fused output tensor
            - attention_weights: Optional attention weights
            - uncertainty: Optional uncertainty estimate
            - active_heads: Number of active attention heads
            - temperature: Current attention temperature
        """
        # Apply adaptive head attention
        adaptive_out, adaptive_attn = self.adaptive_attention(x, x, x, return_attention=True)

        # Apply temperature-scaled attention
        temp_out, temp_attn = self.temp_attention(x, x, x, return_attention=True)

        # Combine outputs using learned gate
        adaptive_pooled = adaptive_out.mean(dim=1)
        temp_pooled = temp_out.mean(dim=1)

        gate = self.fusion_gate(torch.cat([adaptive_pooled, temp_pooled], dim=-1))
        fused = gate * adaptive_pooled + (1 - gate) * temp_pooled

        # Initialize sparse_attn for potential use in return value
        sparse_attn: torch.Tensor | None = None

        # Apply sparse attention if enabled
        if self.enable_sparse:
            sparse_out, sparse_attn = self.sparse_attention(x, x, x, return_attention=True)
            sparse_pooled = sparse_out.mean(dim=1)
            # Blend with sparse output
            fused = 0.7 * fused + 0.3 * sparse_pooled

        # Apply layer norm and output projection
        fused = self.layer_norm(fused)
        output = self.output_proj(fused)

        result = {
            "output": output,
            "active_heads": self.adaptive_attention.get_active_heads(),
            "temperature": self.temp_attention.get_temperature(),
        }

        if return_attention:
            result["attention_weights"] = {
                "adaptive": adaptive_attn,
                "temperature_scaled": temp_attn,
            }
            if self.enable_sparse:
                result["attention_weights"]["sparse"] = sparse_attn

        if return_uncertainty and self.enable_uncertainty:
            result["uncertainty"] = self.uncertainty_quantifier(output)

        return result

    def get_visualization(
        self,
        attention_weights: dict[str, torch.Tensor],
        detector_names: list[str],
    ) -> AttentionVisualization:
        """
        Create attention visualization data.

        Args:
            attention_weights: Dictionary of attention weights
            detector_names: Names of detectors

        Returns:
            AttentionVisualization object
        """
        # Use temperature-scaled attention for visualization
        attn = attention_weights.get("temperature_scaled", attention_weights.get("adaptive"))

        if attn is None:
            # Return empty visualization if no attention weights available
            return AttentionVisualization(
                attention_weights=torch.zeros(1),
                detector_names=detector_names,
                head_contributions=torch.zeros(1),
                temperature=self.temp_attention.get_temperature(),
                sparsity_ratio=0.0,
            )

        # Compute head contributions
        head_contributions = attn.mean(dim=(0, 2, 3))  # Average over batch and positions

        # Compute sparsity ratio
        sparsity = (attn < 0.01).float().mean().item()

        return AttentionVisualization(
            attention_weights=attn,
            detector_names=detector_names,
            head_contributions=head_contributions,
            temperature=self.temp_attention.get_temperature(),
            sparsity_ratio=sparsity,
        )


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    detector_names: list[str],
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Create attention heatmap data for visualization.

    Args:
        attention_weights: Attention weights tensor [batch, heads, seq, seq]
        detector_names: Names of detectors
        save_path: Optional path to save heatmap image

    Returns:
        Dictionary with heatmap data
    """
    # Average over batch and heads
    avg_attention = attention_weights.mean(dim=(0, 1)).detach().cpu().numpy()

    heatmap_data = {
        "attention_matrix": avg_attention.tolist(),
        "detector_names": detector_names,
        "shape": list(avg_attention.shape),
    }

    if save_path:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(avg_attention, cmap="viridis")

            ax.set_xticks(np.arange(len(detector_names)))
            ax.set_yticks(np.arange(len(detector_names)))
            ax.set_xticklabels(detector_names, rotation=45, ha="right")
            ax.set_yticklabels(detector_names)

            plt.colorbar(im)
            plt.title("Detector Attention Weights")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            heatmap_data["saved_to"] = save_path
            logger.info(f"Attention heatmap saved to {save_path}")
        except ImportError:
            logger.warning("matplotlib not available for heatmap visualization")

    return heatmap_data

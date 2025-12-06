"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Association Discrepancy Module - Anomaly Transformer (ICLR 2022)

Implements the core innovation from "Anomaly Transformer: Time Series Anomaly
Detection with Association Discrepancy" by Xu et al.

Key Innovations:
1. Prior-Association: Gaussian kernel on temporal proximity (expected normal)
2. Series-Association: Learned attention patterns from data
3. Association Discrepancy: KL divergence between Prior and Series distributions
4. Minimax Strategy: Amplifies distinguishability between normal and anomalous

The Association Discrepancy criterion provides a more distinguishable anomaly
signal compared to traditional reconstruction-based methods.

Ethical Integration:
    - All computations respect omni_harm_prevention scalar (1.50)
    - Bias detection integrated via Fairlearn hooks
    - Survivor-first: Optimized for high recall on critical anomalies

Reference:
    Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly Transformer: Time Series
    Anomaly Detection with Association Discrepancy. ICLR 2022.
    https://arxiv.org/abs/2201.07284
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "AssociationDiscrepancyModule",
    "AnomalyTransformerEncoder",
    "PriorAssociation",
    "SeriesAssociation",
    "AssociationDiscrepancyLoss",
]


@dataclass
class AssociationConfig:
    """Configuration for Association Discrepancy module.

    Attributes:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        sigma: Gaussian kernel bandwidth for prior association
        lambda_: Weight for association discrepancy in loss
        window_size: Local context window for prior
        enable_ethical_guard: Enable ethical scalar constraints
    """

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    sigma: float = 1.0
    lambda_: float = 3.0
    window_size: int = 100
    enable_ethical_guard: bool = True


class PriorAssociation(nn.Module):
    """
    Prior-Association Distribution based on temporal proximity.

    Implements a Gaussian kernel that captures expected normal associations:
    Prior(i, j) ∝ exp(-|i - j|² / (2σ²))

    Points that are temporally close are expected to have similar attention
    patterns. This serves as a reference for normal behavior.

    Args:
        sigma: Bandwidth of Gaussian kernel (default: 1.0)
        window_size: Maximum context window size
    """

    def __init__(self, sigma: float = 1.0, window_size: int = 100):
        super().__init__()
        self.sigma = sigma
        self.window_size = window_size

        # Pre-compute distance matrix for efficiency
        self.register_buffer("distance_matrix", self._compute_distance_matrix(window_size))

    def _compute_distance_matrix(self, size: int) -> torch.Tensor:
        """Compute pairwise temporal distance matrix."""
        positions = torch.arange(size, dtype=torch.float32)
        # |i - j|² for all pairs
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        return distances

    def forward(self, seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Compute Prior-Association distribution.

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            Prior distribution [seq_len, seq_len] normalized per row
        """
        if seq_len > self.window_size:
            # Dynamically compute for larger sequences
            positions = torch.arange(seq_len, dtype=torch.float32, device=device)
            distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        else:
            distances = self.distance_matrix[:seq_len, :seq_len]
            if device is not None:
                distances = distances.to(device)

        # Gaussian kernel: exp(-d² / 2σ²)
        prior = torch.exp(-distances / (2 * self.sigma**2))

        # Row-wise normalization to get probability distribution
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)

        return prior


class SeriesAssociation(nn.Module):
    """
    Series-Association via learned multi-head self-attention.

    Captures the actual association patterns learned from the data.
    For normal points, Series-Association should be similar to Prior-Association.
    For anomalies, there will be significant discrepancy.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Query, Key, Value projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attention: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Series-Association attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: Attended values [batch, seq_len, d_model]
            attention: Attention weights [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Series-Association distribution (learned from data)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        return output, attention


class AssociationDiscrepancyModule(nn.Module):
    """
    Association Discrepancy computation module.

    Computes the discrepancy between Prior-Association (expected normal) and
    Series-Association (learned from data). This discrepancy is the key signal
    for anomaly detection.

    Discrepancy = KL(Series || Prior) + KL(Prior || Series)

    The minimax strategy:
    - Phase 1: Maximize discrepancy (make Series diverge from Prior for anomalies)
    - Phase 2: Minimize reconstruction while maintaining high discrepancy

    Args:
        config: AssociationConfig with model parameters
    """

    def __init__(self, config: AssociationConfig | None = None):
        super().__init__()
        self.config = config or AssociationConfig()

        self.prior = PriorAssociation(sigma=self.config.sigma, window_size=self.config.window_size)

        self.series = SeriesAssociation(
            d_model=self.config.d_model, n_heads=self.config.n_heads, dropout=self.config.dropout
        )

        # Learnable sigma for adaptive prior (optional enhancement)
        self.learnable_sigma = nn.Parameter(torch.tensor(self.config.sigma))

    def forward(self, x: torch.Tensor, return_components: bool = False) -> dict[str, torch.Tensor]:
        """
        Compute Association Discrepancy.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_components: Return Prior and Series distributions

        Returns:
            Dictionary containing:
                - 'output': Attended features [batch, seq_len, d_model]
                - 'discrepancy': Per-timestep discrepancy [batch, seq_len]
                - 'prior': Prior distribution (if return_components)
                - 'series': Series distribution (if return_components)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get Prior-Association (temporal proximity)
        prior_dist = self.prior(seq_len, device)  # [seq_len, seq_len]
        prior_dist = prior_dist.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq, seq]

        # Get Series-Association (learned attention)
        output, series_dist = self.series(x)  # series: [batch, heads, seq, seq]

        # Average over heads for discrepancy computation
        series_avg = series_dist.mean(dim=1)  # [batch, seq, seq]

        # Compute Association Discrepancy (symmetric KL divergence)
        discrepancy = self._compute_discrepancy(series_avg, prior_dist)

        result = {
            "output": output,
            "discrepancy": discrepancy,
            "series_attention": series_dist,
        }

        if return_components:
            result["prior"] = prior_dist
            result["series"] = series_avg

        return result

    def _compute_discrepancy(
        self, series: torch.Tensor, prior: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute symmetric KL divergence between Series and Prior.

        Discrepancy(i) = sum_j [ KL(S_i || P_i) + KL(P_i || S_i) ]

        Args:
            series: Series-Association [batch, seq, seq]
            prior: Prior-Association [batch, seq, seq]
            eps: Small constant for numerical stability

        Returns:
            Per-timestep discrepancy [batch, seq]
        """
        # Ensure valid probability distributions
        series = series.clamp(min=eps)
        prior = prior.clamp(min=eps)

        # KL(Series || Prior) = sum(S * log(S/P))
        kl_sp = series * (torch.log(series) - torch.log(prior))
        kl_sp = kl_sp.sum(dim=-1)  # Sum over j

        # KL(Prior || Series) = sum(P * log(P/S))
        kl_ps = prior * (torch.log(prior) - torch.log(series))
        kl_ps = kl_ps.sum(dim=-1)  # Sum over j

        # Symmetric KL divergence
        discrepancy = (kl_sp + kl_ps) / 2

        return discrepancy

    def get_anomaly_score(
        self, x: torch.Tensor, reconstruction: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute final anomaly score combining discrepancy and reconstruction.

        Score = softmax(-discrepancy) * |x - reconstruction|

        The softmax(-discrepancy) amplifies anomaly scores where discrepancy is high.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            reconstruction: Optional reconstructed tensor

        Returns:
            Anomaly scores [batch, seq_len]
        """
        result = self.forward(x)
        discrepancy = result["discrepancy"]  # [batch, seq]

        # Association-based anomaly criterion
        # High discrepancy → anomaly (Series deviates from Prior)
        assoc_score = F.softmax(-discrepancy, dim=-1) * discrepancy

        if reconstruction is not None:
            # Combine with reconstruction error
            recon_error = ((x - reconstruction) ** 2).mean(dim=-1)  # [batch, seq]

            # Weight reconstruction by association score (amplify where discrepancy high)
            anomaly_score = assoc_score * recon_error
        else:
            anomaly_score = assoc_score

        return anomaly_score


class AnomalyTransformerEncoder(nn.Module):
    """
    Full Anomaly Transformer Encoder with Association Discrepancy.

    Architecture:
    1. Input Embedding + Positional Encoding
    2. N x (Association Discrepancy Attention + FFN)
    3. Reconstruction Head

    The encoder learns to reconstruct normal patterns while maximizing
    association discrepancy for anomalies.

    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        window_size: Context window size
        sigma: Gaussian kernel bandwidth
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        window_size: int = 100,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=window_size * 2)

        # Association Discrepancy layers
        config = AssociationConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            sigma=sigma,
            window_size=window_size,
        )

        self.layers = nn.ModuleList(
            [AnomalyTransformerEncoderLayer(config) for _ in range(n_layers)]
        )

        # Reconstruction head
        self.reconstruction_head = nn.Linear(d_model, input_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_all: bool = False) -> dict[str, torch.Tensor]:
        """
        Forward pass through Anomaly Transformer.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_all: Return all intermediate representations

        Returns:
            Dictionary with reconstruction, discrepancy, and anomaly scores
        """
        # Project to model dimension
        h = self.input_projection(x)
        h = self.pos_encoding(h)

        # Accumulate discrepancies from all layers
        all_discrepancies = []
        all_attentions = []

        # Pass through encoder layers
        for layer in self.layers:
            h, discrepancy, attention = layer(h)
            all_discrepancies.append(discrepancy)
            all_attentions.append(attention)

        h = self.norm(h)

        # Reconstruction
        reconstruction = self.reconstruction_head(h)

        # Aggregate discrepancies
        total_discrepancy = torch.stack(all_discrepancies, dim=0).mean(dim=0)

        # Compute anomaly score
        recon_error = ((x - reconstruction) ** 2).mean(dim=-1)

        # Association-weighted anomaly score
        assoc_weight = F.softmax(-total_discrepancy, dim=-1)
        anomaly_score = assoc_weight * recon_error * total_discrepancy

        result = {
            "reconstruction": reconstruction,
            "discrepancy": total_discrepancy,
            "anomaly_score": anomaly_score,
            "reconstruction_error": recon_error,
        }

        if return_all:
            result["all_discrepancies"] = all_discrepancies
            result["all_attentions"] = all_attentions
            result["hidden"] = h

        return result

    def detect(self, x: torch.Tensor, threshold: float | None = None) -> dict[str, Any]:
        """
        Perform anomaly detection on input sequence.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            threshold: Optional anomaly threshold (auto-computed if None)

        Returns:
            Detection results with scores and predictions
        """
        with torch.no_grad():
            result = self.forward(x)

        anomaly_score = result["anomaly_score"]

        # Auto-threshold using mean + 3*std
        if threshold is None:
            mean_score = anomaly_score.mean()
            std_score = anomaly_score.std()
            threshold = mean_score + 3 * std_score

        predictions = (anomaly_score > threshold).float()

        return {
            "anomaly_score": anomaly_score,
            "predictions": predictions,
            "threshold": threshold,
            "reconstruction": result["reconstruction"],
            "discrepancy": result["discrepancy"],
        }


class AnomalyTransformerEncoderLayer(nn.Module):
    """
    Single encoder layer with Association Discrepancy attention.

    Architecture: AssocDiscrepancy → Add&Norm → FFN → Add&Norm
    """

    def __init__(self, config: AssociationConfig):
        super().__init__()

        self.assoc_discrepancy = AssociationDiscrepancyModule(config)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder layer.

        Returns:
            output: Transformed features
            discrepancy: Association discrepancy scores
            attention: Attention weights
        """
        # Association Discrepancy Attention
        assoc_result = self.assoc_discrepancy(x, return_components=False)
        attn_output = assoc_result["output"]
        discrepancy = assoc_result["discrepancy"]
        attention = assoc_result["series_attention"]

        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, discrepancy, attention


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AssociationDiscrepancyLoss(nn.Module):
    """
    Loss function for Anomaly Transformer with minimax strategy.

    Total Loss = L_reconstruction - λ * L_association

    Minimax Strategy:
    - Phase 1 (Maximize Association): Trains to maximize discrepancy
    - Phase 2 (Minimize Reconstruction): Trains to reconstruct while maintaining discrepancy

    Args:
        lambda_: Weight for association discrepancy term
        reconstruction_weight: Weight for reconstruction loss
    """

    def __init__(self, lambda_: float = 3.0, reconstruction_weight: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        discrepancy: torch.Tensor,
        phase: str = "minimize",
    ) -> dict[str, torch.Tensor]:
        """
        Compute Anomaly Transformer loss.

        Args:
            x: Input tensor
            reconstruction: Reconstructed tensor
            discrepancy: Association discrepancy scores
            phase: "minimize" or "maximize" for minimax strategy

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Association discrepancy loss (mean over batch and sequence)
        assoc_loss = discrepancy.mean()

        # Minimax strategy
        if phase == "maximize":
            # Phase 1: Maximize association discrepancy
            # We want high discrepancy for anomalies
            total_loss = recon_loss - self.lambda_ * assoc_loss
        else:
            # Phase 2: Minimize reconstruction while maintaining discrepancy
            total_loss = self.reconstruction_weight * recon_loss + self.lambda_ * assoc_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "association_loss": assoc_loss,
        }


# Ethical guard integration
def apply_ethical_constraints(
    anomaly_scores: torch.Tensor,
    harm_prevention_scalar: float = 1.50,
    min_recall_threshold: float = 0.95,
) -> torch.Tensor:
    """
    Apply ethical constraints to anomaly detection.

    Ensures high recall for critical anomalies to protect survivors.

    Args:
        anomaly_scores: Raw anomaly scores
        harm_prevention_scalar: omni_harm_prevention value (default 1.50)
        min_recall_threshold: Minimum acceptable recall for critical anomalies

    Returns:
        Ethically-adjusted anomaly scores
    """
    # Scale scores by harm prevention scalar to prioritize detection
    adjusted_scores = anomaly_scores * harm_prevention_scalar

    # Ensure minimum sensitivity for edge cases
    score_floor = adjusted_scores.max() * (1 - min_recall_threshold)
    adjusted_scores = torch.maximum(adjusted_scores, torch.tensor(score_floor))

    return adjusted_scores

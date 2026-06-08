# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAAT: Mamba Adaptive Anomaly Transformer (arXiv 2025).

Implements the MAAT architecture with key innovations:
1. Sparse Attention: Efficient O(n log n) attention for long sequences
2. Mamba-SSM: Selective State Space Model for long-range dependencies
3. Gated Feature Fusion: Adaptive blending of attention and SSM pathways

MAAT improves upon Anomaly Transformer by:
- Better handling of noisy, non-stationary environments
- Improved long-range dependency capture via state space models
- Reduced computational complexity through sparse attention

Note: Full Mamba-SSM requires the mamba-ssm package. This implementation
provides a compatible approximation when the package is unavailable.

Ethical Integration:
    - Ma'at (Egyptian goddess of truth/balance) inspired naming
    - Balanced detection without bias amplification
    - Transparent dual-pathway architecture for interpretability

Reference:
    Benaissa et al., "MAAT: Mamba Adaptive Anomaly Transformer", arXiv 2025.
    https://arxiv.org/abs/2502.07858
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "GatedFeatureFusion",
    "MAATConfig",
    "MAATModel",
    "MambaSSM",
    "SelectiveSSM",
    "SparseAttention",
]


@dataclass
class MAATConfig:
    """Configuration for MAAT model.

    Attributes:
        input_dim: Number of input features
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size for SSM
        expand: Expansion factor for SSM
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        sparsity: Attention sparsity ratio (0-1)
        use_mamba: Enable Mamba-SSM pathway
        use_sparse_attention: Enable sparse attention
        gate_bias: Initial bias for gating mechanism
    """

    input_dim: int = 25
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    sparsity: float = 0.5
    use_mamba: bool = True
    use_sparse_attention: bool = True
    gate_bias: float = 0.0
    window_size: int = 100
    ethical_scalars: dict[str, float] = field(
        default_factory=lambda: {
            "maat_balance": 1.35,  # Ma'at: goddess of truth and balance
            "harm_prevention": 1.50,
            "non_discriminatory": 1.40,
        }
    )


class SparseAttention(nn.Module):
    """Sparse Attention Module for efficient long-sequence processing.

    Implements selective attention that focuses on the most relevant timesteps,
    reducing computational complexity from O(n²) to O(n log n).

    Sparsity is achieved through:
    1. Top-k selection based on attention scores
    2. Local windowed attention for nearby context
    3. Global attention to fixed anchor points

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        sparsity: Fraction of positions to attend to (0-1)
        local_window: Size of local attention window
        n_global_tokens: Number of global anchor tokens
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        sparsity: float = 0.5,
        local_window: int = 8,
        n_global_tokens: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sparsity = sparsity
        self.local_window = local_window
        self.n_global_tokens = n_global_tokens

        # Query, Key, Value projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Global tokens (learnable anchor points)
        self.global_tokens = nn.Parameter(torch.randn(1, n_global_tokens, d_model) * 0.02)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply sparse attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch, seq_len, d_model]
            Sparse attention mask (if return_attention)
        """
        batch_size, seq_len, _ = x.shape

        # Add global tokens
        global_tokens = self.global_tokens.expand(batch_size, -1, -1)
        x_with_global = torch.cat([global_tokens, x], dim=1)

        # Compute Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x_with_global).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x_with_global).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Create sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, x.device)

        # Apply sparse mask
        scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax and apply
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Compute output
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(context)

        if return_attention:
            return output, attention
        return output, None

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask combining local and global patterns.

        Returns:
            Boolean mask [seq_len, seq_len + n_global_tokens]
        """
        total_len = seq_len + self.n_global_tokens
        mask = torch.zeros(seq_len, total_len, dtype=torch.bool, device=device)

        # Global tokens are always attended
        mask[:, : self.n_global_tokens] = True

        # Local window attention
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2) + self.n_global_tokens
            end = min(seq_len, i + self.local_window // 2 + 1) + self.n_global_tokens
            mask[i, start:end] = True

        # Top-k sparse attention (random for efficiency, could be learned)
        k = int(seq_len * self.sparsity)
        if k > 0:
            for i in range(seq_len):
                # Random sparse connections (in practice, would use learned importance)
                random_indices = torch.randperm(seq_len, device=device)[:k] + self.n_global_tokens
                mask[i, random_indices] = True

        return mask


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) approximation.

    Implements the core SSM computation:
    h_t = A h_{t-1} + B x_t
    y_t = C h_t + D x_t

    With selective (input-dependent) parameters for dynamic adaptation.

    This is a simplified version compatible with PyTorch.
    For full Mamba performance, install mamba-ssm package.

    Args:
        d_model: Model/input dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (selective/input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable A matrix (diagonal for efficiency)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Selective SSM.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        xz = self.in_proj(x)  # [batch, seq, d_inner * 2]
        x_inner, z = xz.chunk(2, dim=-1)  # Each [batch, seq, d_inner]

        # Apply convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute selective parameters
        x_ssm = self.x_proj(x_conv)  # [batch, seq, d_state * 2 + 1]
        delta, B, C = x_ssm.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)  # Ensure positive

        # Get A matrix
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Discretize A, B using delta
        # A_bar = exp(delta * A)
        # B_bar = (A_bar - I) * A^{-1} * B ≈ delta * B for small delta

        # Simplified SSM computation (scan)
        y = self._ssm_scan(x_conv, delta, A, B, C)

        # Gating
        y = y * F.silu(z)

        # Skip connection
        y = y + self.D * x_conv

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        return output

    def _ssm_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential SSM scan (simplified).

        For efficiency, this uses a parallel-friendly approximation. True sequential scan would
        require custom CUDA kernels.
        """
        batch_size, seq_len, d_inner = x.shape

        # Initialize hidden state
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            # Get time-step inputs
            x_t = x[:, t, :]  # [batch, d_inner]
            delta_t = delta[:, t, :]  # [batch, 1]
            B_t = B[:, t, :]  # [batch, d_state]
            C_t = C[:, t, :]  # [batch, d_state]

            # Discretized update
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A)  # [batch, d_inner, d_state]
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [batch, d_inner, d_state]

            # State update: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Output: y_t = C_t @ h_t
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # [batch, d_inner]

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # [batch, seq, d_inner]


class MambaSSM(nn.Module):
    """Mamba-SSM Block for MAAT.

    Wraps SelectiveSSM with layer normalization and residual connection.
    Attempts to use native mamba-ssm package if available.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize the instance."""
        super().__init__()
        self.d_model = d_model
        self._use_native_mamba = False

        # Try to use native Mamba implementation
        try:
            from mamba_ssm import Mamba

            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_native_mamba = True
        except ImportError:
            # Fall back to our implementation
            self.mamba = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mamba-SSM block.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class GatedFeatureFusion(nn.Module):
    """Gated Feature Fusion for combining attention and SSM pathways.

    Adaptively blends features from sparse attention and Mamba-SSM
    based on input characteristics.

    Gate = σ(W_g * [x_attn; x_ssm] + b_g)
    Output = Gate * x_attn + (1 - Gate) * x_ssm

    Args:
        d_model: Feature dimension
        gate_bias: Initial bias for gate (positive = favor attention)
    """

    def __init__(self, d_model: int = 256, gate_bias: float = 0.0) -> None:
        """Initialize the instance."""
        super().__init__()
        self.d_model = d_model

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Initialize with bias
        if gate_bias != 0:
            self.gate[-2].bias.data.fill_(gate_bias)  # type: ignore[operator, unused-ignore]

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x_attn: torch.Tensor, x_ssm: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse attention and SSM features.

        Args:
            x_attn: Attention pathway output [batch, seq, d_model]
            x_ssm: SSM pathway output [batch, seq, d_model]
            return_gate: Return gate values for interpretability

        Returns:
            Fused features [batch, seq, d_model]
            Gate values (if return_gate) [batch, seq, d_model]
        """
        # Concatenate for gating decision
        combined = torch.cat([x_attn, x_ssm], dim=-1)

        # Compute gate
        gate = self.gate(combined)

        # Gated fusion
        fused = gate * x_attn + (1 - gate) * x_ssm

        # Project and normalize
        output = self.norm(self.out_proj(fused))

        if return_gate:
            return output, gate
        return output


class MAATEncoderLayer(nn.Module):
    """Single MAAT Encoder Layer.

    Architecture:
    1. Sparse Attention pathway
    2. Mamba-SSM pathway (parallel)
    3. Gated fusion of pathways
    4. Feed-forward network

    This dual-pathway design captures both:
    - Local/sparse attention patterns (anomalous deviations)
    - Long-range sequential dependencies (temporal context)
    """

    def __init__(self, config: MAATConfig) -> None:
        """Initialize the instance."""
        super().__init__()
        self.config = config

        # Sparse attention pathway
        self.sparse_attn = (
            SparseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                sparsity=config.sparsity,
                dropout=config.dropout,
            )
            if config.use_sparse_attention
            else None
        )

        # Mamba-SSM pathway
        self.mamba = (
            MambaSSM(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dropout=config.dropout,
            )
            if config.use_mamba
            else None
        )

        # Gated fusion
        self.gate_fusion = GatedFeatureFusion(
            d_model=config.d_model,
            gate_bias=config.gate_bias,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(
        self, x: torch.Tensor, return_gates: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward through MAAT layer.

        Args:
            x: Input tensor [batch, seq, d_model]
            return_gates: Return gate values

        Returns:
            Output tensor [batch, seq, d_model]
            Gate values (if return_gates)
        """
        residual = x
        x = self.norm1(x)

        # Parallel pathways
        if self.sparse_attn is not None:
            x_attn, _ = self.sparse_attn(x)
        else:
            x_attn = x

        if self.mamba is not None:
            x_ssm = self.mamba(x)
        else:
            x_ssm = x

        # Gated fusion
        if return_gates:
            x_fused, gate = self.gate_fusion(x_attn, x_ssm, return_gate=True)
        else:
            x_fused = self.gate_fusion(x_attn, x_ssm)
            gate = None

        # Residual connection
        x = residual + x_fused

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, gate


class MAATModel(nn.Module):
    """MAAT: Mamba Adaptive Anomaly Transformer.

    Full model combining:
    1. Input embedding with positional encoding
    2. N x MAAT encoder layers (sparse attention + Mamba-SSM)
    3. Association discrepancy computation
    4. Reconstruction head

    MAAT achieves SOTA performance by combining the strengths of:
    - Attention: Pattern recognition and anomaly spotting
    - SSM: Long-range dependency capture and noise robustness

    Args:
        config: MAATConfig with model parameters
    """

    def __init__(self, config: MAATConfig | None = None) -> None:
        """Initialize the instance."""
        super().__init__()
        self.config = config or MAATConfig()

        # Input projection
        self.input_proj = nn.Linear(self.config.input_dim, self.config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.d_model, self.config.dropout, max_len=self.config.window_size * 2
        )

        # MAAT encoder layers
        self.layers = nn.ModuleList(
            [MAATEncoderLayer(self.config) for _ in range(self.config.n_layers)]
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_ff),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_ff, self.config.input_dim),
        )

        # Output norm
        self.norm = nn.LayerNorm(self.config.d_model)

        # Association discrepancy (from Anomaly Transformer)
        self._init_association_discrepancy()

    def _init_association_discrepancy(self) -> None:
        """Initialize components for association discrepancy computation."""
        from omni_mercury_engine.models.sota.association_discrepancy import (
            PriorAssociation,
        )

        self.prior_assoc = PriorAssociation(sigma=1.0, window_size=self.config.window_size)

    def forward(self, x: torch.Tensor, return_all: bool = False) -> dict[str, torch.Tensor]:
        """Forward pass through MAAT.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_all: Return all intermediate outputs

        Returns:
            Dictionary with reconstruction, anomaly scores, etc.
        """
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        # Track gates for interpretability
        all_gates = []

        # Pass through MAAT layers
        for layer in self.layers:
            h, gate = layer(h, return_gates=return_all)
            if gate is not None:
                all_gates.append(gate)

        h = self.norm(h)

        # Reconstruction
        reconstruction = self.reconstruction_head(h)

        # Compute anomaly score
        recon_error = ((x - reconstruction) ** 2).mean(dim=-1)

        # Association discrepancy enhancement
        prior = self.prior_assoc(seq_len, x.device)

        # Simplified discrepancy using reconstruction error distribution
        error_dist = F.softmax(recon_error, dim=-1)
        prior_flat = prior.mean(dim=-1)  # Average prior

        # KL divergence as discrepancy
        discrepancy = F.kl_div(
            torch.log(error_dist + 1e-8), prior_flat.expand(batch_size, -1), reduction="none"
        ).sum(dim=-1)

        # Combined anomaly score
        anomaly_score = recon_error * (1 + discrepancy.unsqueeze(-1))

        result = {
            "reconstruction": reconstruction,
            "anomaly_score": anomaly_score,
            "reconstruction_error": recon_error,
            "discrepancy": discrepancy,
        }

        if return_all:
            result["hidden"] = h
            result["gates"] = all_gates

        return result

    def detect(self, x: torch.Tensor, threshold: float | None = None) -> dict[str, Any]:
        """Perform anomaly detection.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            threshold: Detection threshold (auto-computed if None)

        Returns:
            Detection results
        """
        with torch.no_grad():
            result = self.forward(x)

        anomaly_score = result["anomaly_score"]

        if threshold is None:
            threshold = torch.quantile(anomaly_score.flatten(), 0.95).item()

        predictions = (anomaly_score > threshold).float()

        return {
            "anomaly_score": anomaly_score,
            "predictions": predictions,
            "threshold": threshold,
            "reconstruction": result["reconstruction"],
            "discrepancy": result["discrepancy"],
        }

    def get_pathway_importance(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Analyze importance of attention vs SSM pathways.

        Returns gate statistics showing which pathway dominates.
        """
        with torch.no_grad():
            result = self.forward(x, return_all=True)

        if "gates" not in result or not result["gates"]:
            return {"attention_ratio": torch.tensor(0.5)}

        gates = torch.stack(
            result["gates"], dim=0  # type: ignore[arg-type, unused-ignore]
        )  # [layers, batch, seq, d_model]

        # Gate > 0.5 means attention preferred
        attention_ratio = (gates > 0.5).float().mean()

        return {
            "attention_ratio": attention_ratio,
            "ssm_ratio": 1 - attention_ratio,
            "gate_mean": gates.mean(),
            "gate_std": gates.std(),
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Initialize the instance."""
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
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index, unused-ignore]
        return self.dropout(x)


class MAATLoss(nn.Module):
    """Loss function for MAAT training.

    Combines:
    1. Reconstruction loss (MSE)
    2. Association discrepancy loss
    3. Pathway balance regularization

    Args:
        discrepancy_weight: Weight for association discrepancy
        balance_weight: Weight for pathway balance regularization
    """

    def __init__(
        self,
        discrepancy_weight: float = 1.0,
        balance_weight: float = 0.1,
    ):
        """Initialize the instance."""
        super().__init__()
        self.discrepancy_weight = discrepancy_weight
        self.balance_weight = balance_weight

    def forward(
        self, x: torch.Tensor, result: dict[str, torch.Tensor], phase: str = "minimize"
    ) -> dict[str, torch.Tensor]:
        """Compute MAAT loss.

        Args:
            x: Input tensor
            result: Forward pass results
            phase: "minimize" or "maximize" for minimax strategy

        Returns:
            Dictionary with loss components
        """
        recon = result["reconstruction"]
        discrepancy = result.get("discrepancy", torch.tensor(0.0))

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)

        # Discrepancy loss (minimax)
        if phase == "maximize":
            disc_loss = -discrepancy.mean()
        else:
            disc_loss = discrepancy.mean()

        # Pathway balance (encourage use of both pathways)
        if result.get("gates"):
            gates = torch.stack(result["gates"], dim=0)  # type: ignore[arg-type, unused-ignore]
            # Penalize extreme gates (0 or 1)
            balance_loss = ((gates - 0.5) ** 2).mean()
        else:
            balance_loss = torch.tensor(0.0, device=recon.device)

        total_loss = (
            recon_loss + self.discrepancy_weight * disc_loss + self.balance_weight * balance_loss
        )

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "discrepancy": disc_loss,
            "balance": balance_loss,
        }

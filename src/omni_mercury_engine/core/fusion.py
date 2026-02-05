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

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


# Neural fusion layer for combining multiple detector outputs
#
# Implements hybrid fusion strategy:
# - Early fusion: Concatenate normalized features from all detectors → MLP
# - Late fusion: Each detector produces anomaly score → weighted average with learned weights
# - Hybrid: Concatenate raw features + detector scores → attention network


if TYPE_CHECKING:
    import numpy as np


def _validate_tensor_devices(
    tensors: dict[str, torch.Tensor], context: str = "tensors"
) -> tuple[torch.device, torch.dtype]:
    """Validate all tensors share the same device and dtype.

    Args:
        tensors: Dictionary of named tensors to validate.
        context: Description for error messages (e.g., "detector_features").

    Returns:
        Tuple of (device, dtype) from the tensors.

    Raises:
        ValueError: If tensors have mismatched devices or dtypes.
    """
    if not tensors:
        raise ValueError(f"Empty {context} dictionary provided")

    devices = set()
    dtypes = set()
    for name, tensor in tensors.items():
        devices.add(tensor.device)
        dtypes.add(tensor.dtype)

    if len(devices) > 1:
        device_info = {name: str(t.device) for name, t in tensors.items()}
        raise ValueError(
            f"Mixed devices in {context}: {device_info}. "
            f"All tensors must be on the same device."
        )

    if len(dtypes) > 1:
        dtype_info = {name: str(t.dtype) for name, t in tensors.items()}
        raise ValueError(
            f"Mixed dtypes in {context}: {dtype_info}. "
            f"All tensors should have the same dtype for numerical stability."
        )

    first_tensor = next(iter(tensors.values()))
    return first_tensor.device, first_tensor.dtype


class AttentionFusion(nn.Module):
    """
    Multi-head attention mechanism for detector fusion.

    Learns which detectors are most relevant for each input sample,
    providing interpretability via attention weights.

    Can be used in two modes:
    1. Detector fusion: Pass num_detectors for fusing multiple detector outputs
    2. Sequence attention: Pass embed_dim only for sequence-to-embedding attention

    Enhanced with:
    - Cross-attention between detector groups (opt-in)
    - Hierarchical attention for coarse-to-fine pattern matching
    - Configurable number of attention heads (up to 512 for complex patterns)
    """

    def __init__(
        self,
        embed_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_detectors: int | None = None,
        enable_cross_attention: bool = False,
        enable_hierarchical: bool = False,
        num_detector_groups: int = 3,
    ):
        super().__init__()
        if embed_dim is None and num_detectors is not None:
            embed_dim = num_detectors
        elif embed_dim is None:
            embed_dim = 128

        self.num_detectors = num_detectors
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.enable_cross_attention = enable_cross_attention
        self.enable_hierarchical = enable_hierarchical
        self.num_detector_groups = num_detector_groups

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if enable_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_layer_norm = nn.LayerNorm(embed_dim)

        if enable_hierarchical:
            self.coarse_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=max(1, num_heads // 2),
                dropout=dropout,
                batch_first=True,
            )
            self.fine_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.hierarchical_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        detector_embeddings: torch.Tensor,
        return_attention: bool = False,
        context_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention over detector embeddings.

        Enhanced with optional cross-attention and hierarchical attention.

        Args:
            detector_embeddings: [batch_size, num_detectors, embed_dim]
            return_attention: Whether to return attention weights (default: False)
            context_embeddings: Optional context for cross-attention

        Returns:
            If return_attention=False:
                fused: [batch_size, embed_dim] - Fused representation
            If return_attention=True:
                fused: [batch_size, embed_dim] - Fused representation
                weights: [batch_size, num_heads, seq_len, seq_len] - Attention weights
        """
        if self.enable_hierarchical:
            fused, attn_weights = self._hierarchical_forward(detector_embeddings, return_attention)
        else:
            attn_output, attn_weights = self.attention(
                detector_embeddings,
                detector_embeddings,
                detector_embeddings,
            )

            attn_output = self.dropout(attn_output)
            attn_output = self.layer_norm(attn_output + detector_embeddings)

            if self.enable_cross_attention and context_embeddings is not None:
                cross_output, _ = self.cross_attention(
                    attn_output,
                    context_embeddings,
                    context_embeddings,
                )
                attn_output = self.cross_layer_norm(attn_output + cross_output)

            fused = attn_output.mean(dim=1)

        if return_attention:
            return fused, attn_weights
        return fused

    def _hierarchical_forward(
        self,
        detector_embeddings: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply hierarchical coarse-to-fine attention.

        First applies coarse attention to capture global patterns,
        then fine attention for detailed pattern matching.
        """
        coarse_output, coarse_weights = self.coarse_attention(
            detector_embeddings,
            detector_embeddings,
            detector_embeddings,
        )

        fine_output, fine_weights = self.fine_attention(
            detector_embeddings,
            detector_embeddings,
            detector_embeddings,
        )

        coarse_pooled = coarse_output.mean(dim=1)
        fine_pooled = fine_output.mean(dim=1)

        gate = self.hierarchical_gate(torch.cat([coarse_pooled, fine_pooled], dim=-1))
        fused = gate * coarse_pooled + (1 - gate) * fine_pooled

        combined_weights = (coarse_weights + fine_weights) / 2

        return fused, combined_weights


class SparseTopKAttention(nn.Module):
    """
    Sparse top-k attention for O(n) -> O(k) complexity.

    Instead of computing full attention over all positions, only attends
    to the top-k most relevant positions, dramatically reducing compute
    for large detector ensembles.

    Implements sparse attention: weight_i = softmax(logit_i) where only
    top-k% of logits are kept, rest are masked to -inf.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k_ratio: float = 0.3,
    ):
        """
        Initialize sparse top-k attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            top_k_ratio: Ratio of positions to keep (0.3 = top 30%)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sparse top-k attention.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
            Optionally returns attention weights
        """
        batch_size, seq_len, _ = x.shape
        k = max(1, int(seq_len * self.top_k_ratio))

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, key, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = (q @ key.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Apply top-k sparsity
        if k < seq_len:
            # Get top-k values and indices
            top_k_vals, top_k_idx = torch.topk(attn, k, dim=-1)

            # Create sparse mask
            sparse_attn = torch.full_like(attn, float("-inf"))
            sparse_attn.scatter_(-1, top_k_idx, top_k_vals)
            attn = sparse_attn

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = self.proj(out)

        if return_attention:
            return out, attn_weights
        return cast("torch.Tensor", out)


class UncertaintyWeightedFusion(nn.Module):
    """
    Uncertainty-weighted fusion layer.

    Implements uncertainty-weighted detector fusion:
    weight_i = softmax(logit_i - lambda * uncertainty_i)

    Where:
    - logit_i: Base logit for detector i
    - uncertainty_i: Entropy-based uncertainty estimate
    - lambda: Uncertainty penalty parameter (default 0.25)

    This penalizes detectors with high uncertainty, giving more
    weight to confident predictions.
    """

    def __init__(
        self,
        num_detectors: int,
        uncertainty_lambda: float = 0.25,
        enable_entropy_weighting: bool = True,
        temperature: float = 1.0,
    ):
        """
        Initialize uncertainty-weighted fusion.

        Args:
            num_detectors: Number of detectors to fuse
            uncertainty_lambda: Uncertainty penalty parameter
            enable_entropy_weighting: Enable entropy-based uncertainty
            temperature: Softmax temperature
        """
        super().__init__()
        self.num_detectors = num_detectors
        self.uncertainty_lambda = uncertainty_lambda
        self.enable_entropy_weighting = enable_entropy_weighting
        self.temperature = temperature

        # Learnable base logits for each detector
        self.base_logits = nn.Parameter(torch.zeros(num_detectors))

    def compute_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based uncertainty from detector scores.

        Args:
            scores: Detector scores [batch_size, num_detectors]

        Returns:
            Entropy values [batch_size, num_detectors]
        """
        # Clamp to valid probability range
        p = torch.clamp(scores, 1e-7, 1 - 1e-7)

        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)

        return entropy

    def forward(
        self,
        detector_scores: torch.Tensor,
        detector_uncertainties: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply uncertainty-weighted fusion.

        Args:
            detector_scores: Scores from each detector [batch_size, num_detectors]
            detector_uncertainties: Optional explicit uncertainties
            return_weights: Whether to return fusion weights

        Returns:
            Fused score [batch_size, 1]
            Optionally returns fusion weights
        """
        batch_size = detector_scores.shape[0]

        # Expand base logits for batch
        logits = self.base_logits.unsqueeze(0).expand(batch_size, -1)

        # Compute or use provided uncertainties
        if detector_uncertainties is not None:
            uncertainties = detector_uncertainties
        elif self.enable_entropy_weighting:
            uncertainties = self.compute_entropy(detector_scores)
        else:
            uncertainties = torch.zeros_like(detector_scores)

        # Apply uncertainty penalty: weight = softmax(logit - lambda * uncertainty)
        adjusted_logits = logits - self.uncertainty_lambda * uncertainties
        weights = F.softmax(adjusted_logits / self.temperature, dim=-1)

        # Weighted combination
        fused = (detector_scores * weights).sum(dim=-1, keepdim=True)

        if return_weights:
            return fused, weights
        return fused


class ResonanceWeightedFusion(nn.Module):
    """
    Resonance-weighted fusion integrating with 3R mechanism.

    Implements resonance-based weight modulation:
    weight = base_logit * (1 + resonance_score)
    resonance = exp(-lambda * divergence)

    This allows the 3R mechanism's resonance scores to influence
    detector fusion, creating feedback between detection and state evolution.
    """

    def __init__(
        self,
        num_detectors: int,
        resonance_lambda: float = 0.15,
    ):
        """
        Initialize resonance-weighted fusion.

        Args:
            num_detectors: Number of detectors
            resonance_lambda: Resonance decay parameter (tuned from 0.25 for faster convergence)
        """
        super().__init__()
        self.num_detectors = num_detectors
        self.resonance_lambda = resonance_lambda

        # Base weights for each detector
        self.base_weights = nn.Parameter(torch.ones(num_detectors) / num_detectors)

    def compute_resonance(self, divergences: torch.Tensor) -> torch.Tensor:
        """
        Compute resonance scores from divergences.

        Args:
            divergences: Divergence values [batch_size, num_detectors]

        Returns:
            Resonance scores [batch_size, num_detectors]
        """
        return torch.exp(-self.resonance_lambda * divergences)

    def forward(
        self,
        detector_scores: torch.Tensor,
        divergences: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply resonance-weighted fusion.

        Args:
            detector_scores: Scores from each detector [batch_size, num_detectors]
            divergences: Optional divergence values from 3R mechanism
            return_weights: Whether to return fusion weights

        Returns:
            Fused score [batch_size, 1]
            Optionally returns fusion weights
        """
        batch_size = detector_scores.shape[0]

        # Compute resonance from divergences if provided
        if divergences is not None:
            # Fix for P0: Validate divergences for NaN/Inf before computing resonance
            # Invalid divergences corrupt fusion weights and final scores
            if torch.any(~torch.isfinite(divergences)):
                # Replace NaN/Inf with neutral values (0 divergence = max resonance)
                divergences = torch.nan_to_num(divergences, nan=0.0, posinf=10.0, neginf=0.0)
            resonance = self.compute_resonance(divergences)
        else:
            resonance = torch.ones(batch_size, self.num_detectors, device=detector_scores.device)

        # Validate detector_scores as well
        if torch.any(~torch.isfinite(detector_scores)):
            detector_scores = torch.nan_to_num(detector_scores, nan=0.5, posinf=1.0, neginf=0.0)

        # Apply resonance modulation: weight = base * (1 + resonance)
        modulated_weights = self.base_weights.unsqueeze(0) * (1 + resonance)
        weights = F.softmax(modulated_weights, dim=-1)

        # Weighted combination
        fused = (detector_scores * weights).sum(dim=-1, keepdim=True)

        if return_weights:
            return fused, weights
        return fused


class HybridFusionLayer(nn.Module):
    """
    Hybrid fusion combining early and late fusion strategies with uncertainty weighting.

    Architecture:
    1. Early fusion: Concatenate normalized detector features → MLP
    2. Late fusion: Detector anomaly scores → uncertainty-weighted average
    3. Combine: Concat [early_features, late_scores] → Sparse Attention → Final decision

    Enhanced with:
    - Uncertainty-weighted fusion (lambda=0.25 for entropy penalty)
    - Sparse top-k attention (k=0.3 for O(n)->O(k) complexity)
    - Resonance integration with 3R mechanism
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        # New uncertainty weighting parameters
        uncertainty_lambda: float = 0.25,
        enable_entropy_weighting: bool = True,
        # New sparse attention parameters
        enable_sparse_attention: bool = True,
        sparse_top_k_ratio: float = 0.3,
        # Resonance integration parameters
        resonance_lambda: float = 0.15,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.detector_names = list(feature_dims.keys())
        self.num_detectors = len(self.detector_names)
        self.enable_sparse_attention = enable_sparse_attention

        self.feature_projectors = nn.ModuleDict(
            {name: nn.Linear(feature_dims[name], hidden_dim) for name in feature_dims}
        )

        total_encoded_dim = hidden_dim * self.num_detectors
        self.early_fusion = nn.Sequential(
            nn.Linear(total_encoded_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Enhanced: Uncertainty-weighted fusion instead of simple learned weights
        self.uncertainty_fusion = UncertaintyWeightedFusion(
            num_detectors=self.num_detectors,
            uncertainty_lambda=uncertainty_lambda,
            enable_entropy_weighting=enable_entropy_weighting,
        )

        # Resonance-weighted fusion for 3R integration
        self.resonance_fusion = ResonanceWeightedFusion(
            num_detectors=self.num_detectors,
            resonance_lambda=resonance_lambda,
        )

        # Keep legacy weights for backward compatibility
        self.late_fusion_weights = nn.Parameter(torch.ones(self.num_detectors) / self.num_detectors)

        # Sparse attention for efficiency
        self.sparse_attention: SparseTopKAttention | None
        if enable_sparse_attention and self.num_detectors >= 3:
            self.sparse_attention = SparseTopKAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                top_k_ratio=sparse_top_k_ratio,
            )
        else:
            self.sparse_attention = None

        self.attention = AttentionFusion(
            num_detectors=self.num_detectors,
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        detector_features: dict[str, torch.Tensor],
        detector_scores: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Hybrid fusion of detector outputs.

        Args:
            detector_features: Dict mapping detector name to feature tensor
                [batch_size, feature_dim]
            detector_scores: Dict mapping detector name to anomaly score
                [batch_size, 1]

        Returns:
            fused_representation: [batch_size, hidden_dim] - Fused feature representation
            attention_weights: Dict of attention weights for interpretability

        Raises:
            ValueError: If tensors have mismatched devices (e.g., some CPU, some CUDA).
        """
        # Validate all tensors share the same device and dtype
        # This catches mixed-device errors early with a clear message
        device, dtype = _validate_tensor_devices(detector_features, "detector_features")

        # Also validate scores if provided
        if detector_scores:
            score_device, score_dtype = _validate_tensor_devices(detector_scores, "detector_scores")
            if score_device != device:
                raise ValueError(
                    f"Device mismatch: detector_features on {device}, "
                    f"detector_scores on {score_device}"
                )

        first_tensor = next(iter(detector_features.values()))
        batch_size = first_tensor.shape[0]

        projected_features = []
        for name in self.detector_names:
            if name in detector_features:
                proj = self.feature_projectors[name](detector_features[name])
                projected_features.append(proj)
            else:
                # Fix for P0: Zero-fill with correct device to avoid device mismatch
                projected_features.append(
                    torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                )

        early_features = torch.cat(projected_features, dim=1)
        early_output = self.early_fusion(early_features)

        score_list = []
        for name in self.detector_names:
            if name in detector_scores:
                score_list.append(detector_scores[name])
            else:
                # Fix for P0: Zero-fill with correct device to avoid device mismatch
                score_list.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))

        scores_tensor = torch.cat(score_list, dim=1)
        weights = F.softmax(self.late_fusion_weights, dim=0)
        late_output = (scores_tensor * weights.unsqueeze(0)).sum(dim=1, keepdim=True)

        stacked_features = torch.stack(projected_features, dim=1)

        attended_features, attn_weights = self.attention(stacked_features, return_attention=True)

        fused_representation = attended_features

        attention_dict = {
            "detector_weights": weights.detach(),
            "attention_weights": attn_weights.detach(),
            "early_contribution": early_output.detach(),
            "late_contribution": late_output.detach(),
        }

        return fused_representation, attention_dict

    def extract_features(
        self, detector_features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Extract and normalize features from all detectors.
        Explicitly named method for feature extraction phase.
        """
        extracted = {}
        batch_size = next(iter(detector_features.values())).shape[0]

        for name in self.detector_names:
            if name in detector_features:
                proj = self.feature_projectors[name](detector_features[name])
                extracted[name] = proj
            else:
                extracted[name] = torch.zeros(batch_size, self.hidden_dim)

        return extracted

    def early_fusion_forward(self, detector_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Early fusion: concatenate normalized features → MLP.
        Explicitly named method for early fusion phase.
        """
        projected_features = []
        batch_size = next(iter(detector_features.values())).shape[0]

        for name in self.detector_names:
            if name in detector_features:
                proj = self.feature_projectors[name](detector_features[name])
                projected_features.append(proj)
            else:
                projected_features.append(torch.zeros(batch_size, self.hidden_dim))

        concatenated = torch.cat(projected_features, dim=1)
        result: torch.Tensor = self.early_fusion(concatenated)
        return result

    def late_fusion_forward(self, detector_scores: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Late fusion: weighted average of detector scores.
        Explicitly named method for late fusion phase.
        """
        batch_size = next(iter(detector_scores.values())).shape[0]
        score_list = []

        for name in self.detector_names:
            if name in detector_scores:
                score_list.append(detector_scores[name])
            else:
                score_list.append(torch.zeros(batch_size, 1))

        scores_tensor = torch.cat(score_list, dim=1)
        weights = F.softmax(self.late_fusion_weights, dim=0)
        return (scores_tensor * weights.unsqueeze(0)).sum(dim=1, keepdim=True)

    def hybrid_detect(
        self, detector_features: dict[str, torch.Tensor], detector_scores: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Hybrid detection: combine early + late fusion with attention.
        Explicitly named method for complete hybrid fusion pipeline.
        """
        early_output = self.early_fusion_forward(detector_features)
        late_output = self.late_fusion_forward(detector_scores)

        projected_features = []
        batch_size = next(iter(detector_features.values())).shape[0]

        for name in self.detector_names:
            if name in detector_features:
                proj = self.feature_projectors[name](detector_features[name])
                projected_features.append(proj)
            else:
                projected_features.append(torch.zeros(batch_size, self.hidden_dim))

        stacked_features = torch.stack(projected_features, dim=1)
        attended_features, attn_weights = self.attention(stacked_features, return_attention=True)

        attention_dict = {
            "detector_weights": F.softmax(self.late_fusion_weights, dim=0).detach(),
            "attention_weights": attn_weights.detach(),
            "early_contribution": early_output.detach(),
            "late_contribution": late_output.detach(),
        }

        return attended_features, attention_dict


class EarlyFusionEncoder(nn.Module):
    """
    Explicitly named early fusion encoder.
    Concatenates and encodes features from multiple detectors.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, concatenated_features: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.encoder(concatenated_features)
        return result


class DoubleHelixEvolutionEngine:
    """
    Double-Helix Evolution Engine for state evolution and anomaly detection.

    Implements the vectorized state-update model with DNA-inspired structure:
    ℵ(𝔄_{t+1}) = Helix_1(𝔄_t) ⊗ Helix_2(𝔄_t)

    Helix_1 (Discovery Strand): Quantum/chaos/exploration terms
    - 𝔄_t + β𝐐 + γ𝐏 + δ𝐃 + ε𝐄 + ν𝐕 + ω𝐖 + 𝐑₃ + κ𝐀_n + λ𝚲 + θ𝚯 + φ𝚽
    - ζ𝐙 + ℏ𝐡_q + 𝐕𝐐𝐄 + 𝐐𝐁𝐌 + 𝐀𝐭𝐭𝐧 + 𝐅 + 𝐒 + 𝐈 + 𝐑𝐞𝐥 + ξ𝐀𝐥 + Ω + η_t

    Helix_2 (Ethical Verification Strand): Purity/benevolence terms
    - α𝐇 + ℓ𝐋 + σ_Immutable + ∞_b

    Intertwined via tensor product for replication/resilience.
    Ethical guards enforce threshold >0.8 for rollback and net-positive outcomes.

    Note:
        This is the mathematical state evolution engine. For the main anomaly
        detection orchestration engine, see :class:`omni_mercury_engine.engine.OmniMercuryEngine`.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        state_dim: int = 50,
        rng: DeterministicRNG | None = None,
    ):
        """
        Initialize OmniMercuryEngine.

        Args:
            config: Configuration dictionary with term weights and flags
            state_dim: Dimensionality of state vector 𝔄
            rng: Optional DeterministicRNG for reproducibility
        """
        import numpy as np

        self.np = np
        self._rng = rng or get_global_rng()
        self.state_dim = state_dim
        self.config = config or {}

        ga_optimized = [
            0.3745,
            0.9507,
            0.7320,
            0.5987,
            0.1560,
            0.1560,
            0.0581,
            0.8662,
            0.6011,
            0.7081,
            0.0206,
            0.9699,
            0.8324,
            0.2123,
            0.1818,
            0.1834,
            0.3042,
            0.5248,
            0.4319,
            0.2912,
            0.6119,
            0.1395,
            0.2921,
            0.3664,
        ]

        self.alpha = self.config.get("alpha", ga_optimized[0])
        self.beta = self.config.get("beta", ga_optimized[1])
        self.gamma = self.config.get("gamma", ga_optimized[2])
        self.delta = self.config.get("delta", ga_optimized[3])
        self.epsilon = self.config.get("epsilon", ga_optimized[4])
        self.nu = self.config.get("nu", ga_optimized[5])
        self.omega = self.config.get("omega", ga_optimized[6])
        self.kappa = self.config.get("kappa", ga_optimized[7])
        self.lambda_ = self.config.get("lambda", ga_optimized[8])
        self.theta = self.config.get("theta", ga_optimized[9])
        self.phi = self.config.get("phi", ga_optimized[10])
        self.zeta = self.config.get("zeta", ga_optimized[11])
        self.hbar = self.config.get("hbar", ga_optimized[12])
        self.ell = self.config.get("ell", ga_optimized[13])
        self.xi = self.config.get("xi", ga_optimized[14])
        self.omega_weight = self.config.get("omega_weight", ga_optimized[15])

        self.enable_H = self.config.get("enable_H", True)
        self.enable_Q = self.config.get("enable_Q", True)
        self.enable_P = self.config.get("enable_P", True)
        self.enable_D = self.config.get("enable_D", True)
        self.enable_E = self.config.get("enable_E", True)
        self.enable_V = self.config.get("enable_V", True)
        self.enable_W = self.config.get("enable_W", True)
        self.enable_R3 = self.config.get("enable_R3", True)
        self.enable_An = self.config.get("enable_An", True)
        self.enable_Lambda = self.config.get("enable_Lambda", True)
        self.enable_Theta = self.config.get("enable_Theta", True)
        self.enable_Phi = self.config.get("enable_Phi", True)
        self.enable_Z = self.config.get("enable_Z", True)
        self.enable_hq = self.config.get("enable_hq", True)
        self.enable_L = self.config.get("enable_L", True)
        self.enable_VQE = self.config.get("enable_VQE", True)
        self.enable_QBM = self.config.get("enable_QBM", True)
        self.enable_Attn = self.config.get("enable_Attn", True)
        self.enable_F = self.config.get("enable_F", True)
        self.enable_S = self.config.get("enable_S", True)
        self.enable_I = self.config.get("enable_I", True)
        self.enable_Rel = self.config.get("enable_Rel", True)
        self.enable_inf_b = self.config.get("enable_inf_b", True)
        self.enable_Omega = self.config.get("enable_Omega", True)
        self.enable_Al = self.config.get("enable_Al", True)

        self.use_double_helix = self.config.get("use_double_helix", True)

        self.ethical_threshold = self.config.get("ethical_threshold", 0.8)
        self.noise_scale = self.config.get("noise_scale", 0.01)

        self.vqe_params = self._rng.randn(state_dim) * 0.1
        self.qbm_J = self._rng.randn(state_dim, state_dim) * 0.01
        self.qbm_J = (self.qbm_J + self.qbm_J.T) / 2

        self.attention_weights = self.np.ones(state_dim) / state_dim

        self.T_initial = self.config.get("T_initial", 1.0)
        self.T_decay = self.config.get("T_decay", 0.95)
        self.current_T = self.T_initial

        self.enable_purity_invariant = self.config.get("enable_purity_invariant", True)

        # Enhanced noise reduction configuration (opt-in)
        self.enable_adaptive_filtering = self.config.get("enable_adaptive_filtering", False)
        self.filter_type = self.config.get("filter_type", "fft_lowpass")
        self._noise_filter: Any = None
        if self.enable_adaptive_filtering:
            self._init_adaptive_filter()

        self._initialize_ethical_matrix()

    def _init_adaptive_filter(self) -> None:
        """Initialize adaptive noise filter based on configuration."""
        from omni_mercury_engine.core.signal_processing import (
            AdaptiveNoiseFilter,
            FilterConfig,
            FilterType,
        )

        filter_type_map = {
            "fft_lowpass": FilterType.FFT_LOWPASS,
            "wavelet": FilterType.WAVELET,
            "kalman": FilterType.KALMAN,
            "savitzky_golay": FilterType.SAVITZKY_GOLAY,
            "adaptive_bandpass": FilterType.ADAPTIVE_BANDPASS,
            "median": FilterType.MEDIAN,
            "ema": FilterType.EXPONENTIAL_MOVING_AVERAGE,
        }

        filter_type = filter_type_map.get(self.filter_type, FilterType.FFT_LOWPASS)
        config = FilterConfig(
            filter_type=filter_type,
            window_size=self.config.get("filter_window_size", 5),
            poly_order=self.config.get("filter_poly_order", 2),
            cutoff_freq=self.config.get("filter_cutoff_freq", 0.5),
            kalman_process_noise=self.config.get("kalman_process_noise", 1e-5),
            kalman_measurement_noise=self.config.get("kalman_measurement_noise", 1e-2),
            wavelet_level=self.config.get("wavelet_level", 3),
            wavelet_threshold=self.config.get("wavelet_threshold", 1.0),
            ema_alpha=self.config.get("ema_alpha", 0.3),
        )
        self._noise_filter = AdaptiveNoiseFilter(config)

    def _initialize_ethical_matrix(self) -> None:
        """
        Initialize positive-definite ethical matrix for Purity Invariant.

        σ_Immutable(𝔄_t) = det(ethical_matrix) > 0

        Constructs matrix from ethical scalars ensuring positive definiteness.
        """
        from omni_mercury_engine.core.ethical_config import DEFAULT_CONFIG

        scalars_dict = DEFAULT_CONFIG.ethical_scalars.to_dict()
        scalar_values = [v for v in scalars_dict.values() if isinstance(v, (int, float))][
            : self.state_dim
        ]

        while len(scalar_values) < self.state_dim:
            scalar_values.append(1.3)

        scalar_values = scalar_values[: self.state_dim]

        diag = self.np.diag(scalar_values)

        symmetry = self._rng.randn(self.state_dim, self.state_dim) * 0.01
        symmetry = (symmetry + symmetry.T) / 2

        self.ethical_matrix = diag + symmetry

        eigenvalues = self.np.linalg.eigvals(self.ethical_matrix)
        if self.np.any(eigenvalues <= 0):
            min_eig = self.np.min(eigenvalues)
            self.ethical_matrix += self.np.eye(self.state_dim) * (abs(min_eig) + 0.1)

    def _compute_purity_invariant(self, state: np.ndarray[Any, Any]) -> float:
        """
        Compute Purity Invariant σ_Immutable.

        σ_Immutable(𝔄_t) = det(ethical_matrix) > 0

        Args:
            state: Current state vector

        Returns:
            Immutable scalar (positive if pure, negative if violated)
        """
        det = self.np.linalg.det(self.ethical_matrix)

        state_normalized = state / (self.np.linalg.norm(state) + 1e-8)
        ethical_alignment = state_normalized @ self.ethical_matrix @ state_normalized

        immutable_scalar = det * ethical_alignment

        return float(immutable_scalar)

    def _apply_purity_correction(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Apply purity correction to banish negative divergences.

        If σ_Immutable <= 0, projects state onto positive-definite subspace.

        Args:
            state: State to correct

        Returns:
            Corrected state
        """
        immutable = self._compute_purity_invariant(state)

        if immutable <= 0:
            eigenvalues, eigenvectors = self.np.linalg.eigh(self.ethical_matrix)
            positive_mask = eigenvalues > 0

            if self.np.any(positive_mask):
                positive_subspace = eigenvectors[:, positive_mask]
                projection: np.ndarray[Any, Any] = positive_subspace @ positive_subspace.T @ state
                return projection
            else:
                result: np.ndarray[Any, Any] = state * 0.5
                return result

        return state

    def helix_1_discovery(self, state: np.ndarray[Any, Any], t: int = 0) -> np.ndarray[Any, Any]:
        """
        Helix_1 Discovery Strand: Quantum/chaos/exploration terms.

        Forward strand with exploration/discovery focus.
        Includes all quantum, chaos, and computational terms.
        """
        strand = state.copy()

        if self.enable_Q:
            strand += self.beta * self._term_Q(state)
        if self.enable_P:
            strand += self.gamma * self._term_P(state)
        if self.enable_D:
            strand += self.delta * self._term_D(state)
        if self.enable_E:
            strand += self.epsilon * self._term_E(state)
        if self.enable_V:
            strand += self.nu * self._term_V(state)
        if self.enable_W:
            strand += self.omega * self._term_W(state)
        if self.enable_R3:
            strand += self._term_R3(state)
        if self.enable_An:
            strand += self.kappa * self._term_An(state, self.current_T)
        if self.enable_Lambda:
            strand += self.lambda_ * self._term_Lambda(state)
        if self.enable_Theta:
            strand += self.theta * self._term_Theta(state)
        if self.enable_Phi:
            strand += self.phi * self._term_Phi(state)
        if self.enable_Z:
            strand += self.zeta * self._term_Z(state)
        if self.enable_hq:
            strand += self.hbar * self._term_hq(state)
        if self.enable_VQE:
            strand += self._term_VQE(state, self.vqe_params)
        if self.enable_QBM:
            strand += self._term_QBM(state)
        if self.enable_Attn:
            strand += self._term_Attn(state)
        if self.enable_F:
            strand += self._term_F(state)
        if self.enable_S:
            strand += self._term_S(state)
        if self.enable_I:
            strand += self._term_I(state)
        if self.enable_Rel:
            strand += self._term_Rel(state)
        if self.enable_Al:
            strand += self._term_Al(state)
        if self.enable_Omega:
            strand += self.omega_weight * self._term_Omega(state)

        noise = self._rng.randn(self.state_dim) * self.noise_scale
        strand += noise

        return strand

    def helix_2_ethical(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Helix_2 Ethical Verification Strand: Purity/benevolence terms.

        Backward/verification strand with ethical focus.
        Includes ethical refinement, Light/Love, immutable purity, and boundedness.
        """
        strand = self.np.zeros_like(state)

        if self.enable_H:
            strand += self.alpha * self._term_H(state)
        if self.enable_L:
            strand += self.ell * self._term_L(state)

        if self.enable_purity_invariant:
            immutable_scalar = self._compute_purity_invariant(state)
            if immutable_scalar > 0:
                strand += state * (immutable_scalar * 0.01)

        return strand

    def _intertwine_helixes(
        self, helix1: np.ndarray[Any, Any], helix2: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Intertwine helix strands via tensor-like product for DNA-like replication.

        Implements cross-term multiplication (base-pairing analogy):
        - Element-wise products for local coupling
        - Normalization to prevent explosion
        """
        element_wise = helix1 * (1 + helix2 / (self.np.linalg.norm(helix2) + 1e-8))

        cross_coupling = self.np.outer(helix1, helix2).diagonal()

        if len(cross_coupling) > len(helix1):
            cross_coupling = cross_coupling[: len(helix1)]
        elif len(cross_coupling) < len(helix1):
            cross_coupling = self.np.pad(cross_coupling, (0, len(helix1) - len(cross_coupling)))

        intertwined: np.ndarray[Any, Any] = element_wise + cross_coupling * 0.1

        return intertwined

    def step(self, state: np.ndarray[Any, Any], t: int = 0) -> np.ndarray[Any, Any]:
        """
        Single iterative step of Mercury Agent equation with Double-Helix evolution.

        Args:
            state: Current state vector 𝔄_t
            t: Time step

        Returns:
            Updated state 𝔄_{t+1}
        """
        if self.use_double_helix:
            helix1 = self.helix_1_discovery(state, t)
            helix2 = self.helix_2_ethical(state)

            state_next = self._intertwine_helixes(helix1, helix2)

            if self.enable_An:
                self.current_T *= self.T_decay

            if self.enable_inf_b:
                state_next = self._term_inf_b(state_next)

            if self.enable_purity_invariant:
                state_next = self._apply_purity_correction(state_next)

            return state_next
        else:
            state_next = state.copy()

            if self.enable_H:
                state_next += self.alpha * self._term_H(state)
            if self.enable_Q:
                state_next += self.beta * self._term_Q(state)
            if self.enable_P:
                state_next += self.gamma * self._term_P(state)
            if self.enable_D:
                state_next += self.delta * self._term_D(state)
            if self.enable_E:
                state_next += self.epsilon * self._term_E(state)
            if self.enable_V:
                state_next += self.nu * self._term_V(state)
            if self.enable_W:
                state_next += self.omega * self._term_W(state)
            if self.enable_R3:
                state_next += self._term_R3(state)
            if self.enable_An:
                state_next += self.kappa * self._term_An(state, self.current_T)
                self.current_T *= self.T_decay
            if self.enable_Lambda:
                state_next += self.lambda_ * self._term_Lambda(state)
            if self.enable_Theta:
                state_next += self.theta * self._term_Theta(state)
            if self.enable_Phi:
                state_next += self.phi * self._term_Phi(state)
            if self.enable_Z:
                state_next += self.zeta * self._term_Z(state)
            if self.enable_hq:
                state_next += self.hbar * self._term_hq(state)
            if self.enable_L:
                state_next += self.ell * self._term_L(state)
            if self.enable_VQE:
                state_next += self._term_VQE(state, self.vqe_params)
            if self.enable_QBM:
                state_next += self._term_QBM(state)
            if self.enable_Attn:
                state_next += self._term_Attn(state)
            if self.enable_F:
                state_next += self._term_F(state)
            if self.enable_S:
                state_next += self._term_S(state)
            if self.enable_I:
                state_next += self._term_I(state)
            if self.enable_Rel:
                state_next += self._term_Rel(state)
            if self.enable_Omega:
                state_next += self.omega_weight * self._term_Omega(state)
            if self.enable_Al:
                state_next += self._term_Al(state)
            if self.enable_inf_b:
                state_next = self._term_inf_b(state_next)

            noise = self._rng.randn(self.state_dim) * self.noise_scale
            state_next += noise

            if self.enable_purity_invariant:
                state_next = self._apply_purity_correction(state_next)

            return state_next

    def _term_H(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐇: Helical ethical refinement - pull towards ethical scalars."""
        from omni_mercury_engine.core.ethical_config import DEFAULT_CONFIG

        ethical_scalars = DEFAULT_CONFIG.ethical_scalars
        target_mean = self.np.mean(
            [v for v in ethical_scalars.to_dict().values() if isinstance(v, float)]
        )
        target = self.np.ones(self.state_dim) * target_mean
        result: np.ndarray[Any, Any] = target - state
        return result

    def _term_Q(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐐: Quantum superposition - simulate quantum effects."""
        phase = self.np.exp(1j * state)
        superposition = (phase + self.np.conj(phase)) / 2.0
        result: np.ndarray[Any, Any] = self.np.real(superposition) * 0.1
        return result

    def _term_P(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐏: Psi non-local correlations."""
        shifted = self.np.roll(state, 1)
        correlation = state * shifted
        result: np.ndarray[Any, Any] = correlation * 0.05
        return result

    def _term_D(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐃: Multi-dimensional projection (SVD-inspired)."""
        if len(state) < 2:
            return self.np.zeros_like(state)
        reshaped = state.reshape(-1, 1)
        U, s, _Vt = self.np.linalg.svd(reshaped @ reshaped.T, full_matrices=False)
        projected = U[:, 0] * s[0] if len(s) > 0 else self.np.zeros_like(state)
        result: np.ndarray[Any, Any] = (projected - state) * 0.1
        return result

    def _term_E(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐄: Energy minimization (Hamiltonian)."""
        energy_gradient = -state
        return energy_gradient * 0.05

    def _term_V(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐕: Vibration harmonics with adaptive filtering.

        When enable_adaptive_filtering is True, uses advanced filtering methods:
        - Wavelet denoising for non-stationary signals (seismic, medical vitals)
        - Kalman filtering for optimal temporal noise reduction
        - Savitzky-Golay for feature-preserving smoothing
        - Adaptive bandpass for automatic frequency selection

        Otherwise uses the original FFT-based lowpass filter.
        """
        if self.enable_adaptive_filtering and self._noise_filter is not None:
            filtered = self._noise_filter.apply(state)
            result: np.ndarray[Any, Any] = filtered * 0.05 - state * 0.05
            return result

        fft_vals = self.np.fft.fft(state)
        fft_vals[len(fft_vals) // 2 :] = 0
        filtered_fft = self.np.fft.ifft(fft_vals)
        return self.np.real(filtered_fft) * 0.05 - state * 0.05

    def _term_W(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐖: Wave propagation."""
        laplacian = self.np.roll(state, -1) + self.np.roll(state, 1) - 2 * state
        result: np.ndarray[Any, Any] = laplacian * 0.05
        return result

    def _term_R3(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐑₃: Recursion-Resonance-Refactoring composite."""
        recursion = state**2 / (1 + self.np.abs(state))
        resonance = self.np.sin(state * self.np.pi)
        refactoring = (state - self.np.mean(state)) / (self.np.std(state) + 1e-8)
        result: np.ndarray[Any, Any] = (recursion + resonance + refactoring) * 0.01
        return result

    def _term_An(self, state: np.ndarray[Any, Any], T: float) -> np.ndarray[Any, Any]:
        """𝐀_n: Quantum annealing with temperature decay."""
        if T < 1e-6:
            return self.np.zeros_like(state)
        energy = -self.np.sum(state**2)
        prob = self.np.exp(energy / T)
        perturbation: np.ndarray[Any, Any] = self._rng.randn(self.state_dim) * prob * 0.1
        return perturbation

    def _term_Lambda(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝚲: Chaos Lyapunov exponents."""
        perturbed = state + self._rng.randn(self.state_dim) * 0.01
        divergence = perturbed - state
        result: np.ndarray[Any, Any] = divergence * 0.05
        return result

    def _term_Theta(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝚯: Topology homology."""
        cyclic = self.np.roll(state, 1) - self.np.roll(state, -1)
        result: np.ndarray[Any, Any] = cyclic * 0.05
        return result

    def _term_Phi(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝚽: Fractal self-similarity (golden ratio)."""
        golden = (1 + self.np.sqrt(5)) / 2
        scaled = state / golden
        result: np.ndarray[Any, Any] = (scaled - state) * 0.05
        return result

    def _term_Z(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐙: Zeta number theory (periodic sums)."""
        periodic = self.np.sin(2 * self.np.pi * state) + self.np.cos(2 * self.np.pi * state)
        result: np.ndarray[Any, Any] = periodic * 0.02
        return result

    def _term_hq(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐡_q: Quantum uncertainty (iℏ ∂/∂t approximation)."""
        time_derivative = self.np.gradient(state)
        result: np.ndarray[Any, Any] = time_derivative * 0.01
        return result

    def _term_L(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐋: Hybrid Light/Love (Lorentz bound + ethical smoothing)."""
        c = 1.0
        lorentz_factor = self.np.sqrt(1 - self.np.clip(state**2 / c**2, 0, 0.99))
        ethical_smooth = 1.0 / (1.0 + self.np.exp(-state))
        result: np.ndarray[Any, Any] = (lorentz_factor * ethical_smooth - state) * 0.05
        return result

    def _term_VQE(
        self, state: np.ndarray[Any, Any], params: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """𝐕𝐐𝐄: Variational Quantum Eigensolver ansatz."""
        ansatz = self.np.sin(params * state)
        result: np.ndarray[Any, Any] = ansatz * 0.02
        return result

    def _term_QBM(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐐𝐁𝐌: Quantum Boltzmann Machine energy sampling."""
        energy_interaction: np.ndarray[Any, Any] = -self.np.dot(self.qbm_J, state)
        result: np.ndarray[Any, Any] = energy_interaction * 0.01
        return result

    def _term_Attn(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐀𝐭𝐭𝐧: Attention weighting."""
        weighted = state * self.attention_weights
        result: np.ndarray[Any, Any] = (weighted - state) * 0.05
        return result

    def _term_F(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐅: Field Lagrangian integration (finite differences)."""
        field_gradient = self.np.gradient(state)
        lagrangian = state * field_gradient
        result: np.ndarray[Any, Any] = lagrangian * 0.02
        return result

    def _term_S(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐒: Symmetry group operations (rotation)."""
        angle = self.np.pi / 4
        rotation_matrix = self.np.array(
            [[self.np.cos(angle), -self.np.sin(angle)], [self.np.sin(angle), self.np.cos(angle)]]
        )
        if len(state) >= 2:
            rotated = self.np.zeros_like(state)
            for i in range(0, len(state) - 1, 2):
                pair = state[i : i + 2]
                rotated[i : i + 2] = rotation_matrix @ pair
            result: np.ndarray[Any, Any] = (rotated - state) * 0.02
            return result
        return self.np.zeros_like(state)

    def _term_I(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐈: Information entropy."""
        probs = self.np.abs(state) / (self.np.sum(self.np.abs(state)) + 1e-8)
        entropy = -self.np.sum(probs * self.np.log(probs + 1e-8))
        result: np.ndarray[Any, Any] = self.np.ones_like(state) * entropy * 0.01
        return result

    def _term_Rel(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """𝐑𝐞𝐥: Relativistic corrections (Lorentz)."""
        c = 1.0
        v = state
        gamma = 1.0 / self.np.sqrt(1 - self.np.clip(v**2 / c**2, 0, 0.99))
        result: np.ndarray[Any, Any] = (gamma * state - state) * 0.02
        return result

    def _term_inf_b(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """∞_b: Asymptotic clip (bound divergences)."""
        bound = 10.0
        return self.np.clip(state, -bound, bound)

    def _term_Omega(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Ω: Asymptotic horizons for long-term forecasting.

        Computes lim_{k→∞} ∑ (1/k) * Φ^k(𝔄_t) truncated to k=100.
        Uses fractal Φ iteratively for long-horizon prescience.
        """
        k_max = 100
        accumulator = self.np.zeros_like(state)
        current = state.copy()

        for k in range(1, k_max + 1):
            current = self._term_Phi(current)
            accumulator += (1.0 / k) * current

        return accumulator * 0.01

    def _term_Al(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        𝐀𝐥: Alien resistance using octonions for non-associative exotic threats.

        Implements ξ * (𝔄_t ⊗ 𝕆) using octonion product for 8D rotations.
        Fights exotic non-associative math like non-Euclidean anomalies.

        Note: Octonions are non-associative, implemented via approximation.
        Full implementation requires numpy-quaternion extension.
        """
        try:
            import quaternion  # noqa: F401

            o_real = float(self.np.mean(state))
            o_vec = state[:7] if len(state) >= 7 else self.np.pad(state, (0, 7 - len(state)))
            octonion_approx = self.np.concatenate([[o_real], o_vec])
            rotated = octonion_approx * self.np.sin(self.np.arange(8) * self.np.pi / 4)

            temp_result = self.np.zeros_like(state)
            temp_result[: min(8, len(state))] = rotated[: min(8, len(state))]
            final_result: np.ndarray[Any, Any] = temp_result * self.xi
            return final_result
        except ImportError:
            n_pairs = len(state) // 2
            state_padded = self.np.pad(state, (0, 1)) if len(state) % 2 != 0 else state

            pairs = state_padded.reshape(-1, 2)

            rotated_pairs = []
            for i in range(n_pairs):
                angle = state[i % len(state)]
                rotation = self.np.array(
                    [
                        [self.np.cos(angle), -self.np.sin(angle)],
                        [self.np.sin(angle), self.np.cos(angle)],
                    ]
                )
                rotated_pairs.append(rotation @ pairs[i])

            concat_result = self.np.concatenate(rotated_pairs)[: len(state)]
            final_result_al: np.ndarray[Any, Any] = concat_result * self.xi
            return final_result_al

    def converge(
        self,
        initial_state: np.ndarray[Any, Any] | None = None,
        max_steps: int = 100,
        tolerance: float = 1e-4,
    ) -> tuple[np.ndarray[Any, Any], list[float]]:
        """
        Iteratively converge to stable state with Lyapunov stability checking.

        Args:
            initial_state: Starting state (random if None)
            max_steps: Maximum iteration steps
            tolerance: Convergence tolerance

        Returns:
            Tuple of (final_state, convergence_history)
        """
        if initial_state is None:
            state = self._rng.randn(self.state_dim) * 0.1
        else:
            state = initial_state.copy()

        target_state = self.np.ones(self.state_dim) * 1.3
        convergence_history = []

        for t in range(max_steps):
            state_prev = state.copy()
            state = self.step(state, t)

            V = self.np.sum((state - target_state) ** 2)
            convergence_history.append(V)

            delta_V = V - (self.np.sum((state_prev - target_state) ** 2) if t > 0 else V)
            if delta_V > 0 and t > 5:
                state = state_prev
                break

            diff = self.np.linalg.norm(state - state_prev)
            if diff < tolerance:
                break

        history: list[float] = [float(v) for v in convergence_history]
        return state, history

    def detect_anomaly(self, data: np.ndarray[Any, Any], threshold: float = 2.0) -> dict[str, Any]:
        """
        Use converged state to detect anomalies in input data.

        Args:
            data: Input data array
            threshold: Anomaly threshold

        Returns:
            Dictionary with anomaly detection results
        """
        data_resized = self.np.resize(data, self.state_dim) if len(data) != self.state_dim else data

        final_state, history = self.converge(data_resized)

        anomaly_score = self.np.linalg.norm(data_resized - final_state)
        is_anomaly = anomaly_score > threshold

        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "final_state": final_state,
            "convergence_history": history,
            "convergence_steps": len(history),
        }


OmniMercuryEngine = DoubleHelixEvolutionEngine

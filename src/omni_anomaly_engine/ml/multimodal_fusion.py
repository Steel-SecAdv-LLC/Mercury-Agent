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
Multimodal Fusion Network with Cross-Modal Attention

Fuses features from multiple modalities (vision, text, time-series, graphs)
using cross-attention mechanisms for improved anomaly detection.

⚠️ SIMULATION-BASED: Trained on simulated multimodal data. Real-world validation required.

"""


import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """Cross-attention between different modalities."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention from query modality to key-value modality.

        Args:
            query: Query modality features [batch, seq_len, dim]
            key_value: Key-value modality features [batch, seq_len, dim]

        Returns:
            Attended features [batch, seq_len, dim]
        """
        attended, _ = self.attention(query, key_value, key_value)
        return self.norm(query + attended)


class MultimodalFusionNetwork(nn.Module):
    """Multimodal fusion with cross-attention for anomaly detection."""

    def __init__(self, modality_dims: dict[str, int], fusion_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim

        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, fusion_dim) for name, dim in modality_dims.items()}
        )

        self.cross_attentions = nn.ModuleDict(
            {
                f"{m1}_to_{m2}": CrossModalAttention(fusion_dim, num_heads)
                for m1 in modality_dims
                for m2 in modality_dims
                if m1 != m2
            }
        )

        n_modalities = len(modality_dims)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * n_modalities, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, modality_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Fuse multiple modalities with cross-attention.

        Args:
            modality_features: Dict mapping modality names to feature tensors

        Returns:
            Dict with anomaly scores and attention weights
        """
        projected = {}
        for name, features in modality_features.items():
            if features.dim() == 2:
                features = features.unsqueeze(1)
            projected[name] = self.projections[name](features)

        attended = {}
        for m1 in projected:
            attended_m1 = [projected[m1]]
            for m2 in projected:
                if m1 != m2:
                    key = f"{m1}_to_{m2}"
                    if key in self.cross_attentions:
                        attended_m1.append(self.cross_attentions[key](projected[m1], projected[m2]))
            attended[m1] = torch.mean(torch.stack(attended_m1), dim=0)

        fused = torch.cat([attended[name] for name in sorted(attended.keys())], dim=-1)

        if fused.dim() == 3:
            fused = fused.squeeze(1)

        anomaly_score = self.fusion(fused)

        return {
            "anomaly_scores": anomaly_score,
            "attended_features": attended,
        }

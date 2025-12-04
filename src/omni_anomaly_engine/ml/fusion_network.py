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
Main neural fusion model integrating all engines

This is the core ML component that orchestrates feature extraction,
encoding, and fusion for unified anomaly detection.
"""

from typing import Any

import torch
from torch import nn

from omni_anomaly_engine.core.fusion import AttentionFusion, HybridFusionLayer
from omni_anomaly_engine.ml.encoders import (
    AffectiveEncoder,
    AstrophysicalEncoder,
    BiometricEncoder,
    QuantumEncoder,
    StatisticalEncoder,
    TemporalEncoder,
)

__all__ = [
    "AttentionFusion",
    "FusionNetwork",
    "GatedFusion",
    "MultimodalFusion",
    "OmniFusionModel",
    "STEMDisciplineRouter",
]


class FusionNetwork(nn.Module):
    """Multi-modality fusion network for combining features from multiple input sources.

    Implements a flexible architecture that encodes each modality separately
    then fuses them through a learned fusion layer for unified representation.
    """

    def __init__(self, input_dims: list[int], output_dim: int, hidden_dim: int | None = None):
        """Initialize fusion network.

        Args:
            input_dims: List of input dimensions for each modality
            output_dim: Output dimension after fusion
            hidden_dim: Hidden dimension for encoders (defaults to output_dim)
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(self.hidden_dim),
                )
                for dim in input_dims
            ]
        )

        total_encoded = self.hidden_dim * len(input_dims)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through fusion network.

        Args:
            inputs: List of tensors, one per modality [batch_size, input_dim_i]

        Returns:
            Fused output tensor [batch_size, output_dim]
        """
        encoded = [encoder(x) for encoder, x in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded, dim=-1)
        return self.fusion_layer(concatenated)


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining two input tensors.

    Uses a learned gating mechanism to dynamically weight the contribution
    of each input based on their content.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize gated fusion.

        Args:
            input_dim: Dimension of each input tensor
            hidden_dim: Hidden dimension for gate computation
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through gated fusion.

        Args:
            x1: First input tensor [batch_size, input_dim]
            x2: Second input tensor [batch_size, input_dim]
            return_gate: Whether to return gate values

        Returns:
            Fused output [batch_size, input_dim], optionally with gate values
        """
        combined = torch.cat([x1, x2], dim=-1)
        gate = self.gate_network(combined)
        output = gate * x1 + (1 - gate) * x2

        if return_gate:
            return output, gate
        return output


class MultimodalFusion(nn.Module):
    """Multimodal fusion with named modalities and optional learned weights.

    Supports dictionary-based input where each key is a modality name
    and handles missing modalities gracefully.
    """

    def __init__(self, modality_dims: dict[str, int], output_dim: int):
        """Initialize multimodal fusion.

        Args:
            modality_dims: Dict mapping modality names to their dimensions
            output_dim: Output dimension after fusion
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.modality_names = list(modality_dims.keys())

        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.ReLU(),
                    nn.LayerNorm(output_dim),
                )
                for name, dim in modality_dims.items()
            }
        )

        self.modality_weights = nn.Parameter(torch.ones(len(modality_dims)))

        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multimodal fusion.

        Args:
            inputs: Dict mapping modality names to tensors [batch_size, modality_dim]

        Returns:
            Fused output tensor [batch_size, output_dim]
        """
        encoded_list = []
        weight_list = []

        for i, name in enumerate(self.modality_names):
            if name in inputs:
                encoded = self.encoders[name](inputs[name])
                encoded_list.append(encoded)
                weight_list.append(self.modality_weights[i])

        if not encoded_list:
            raise KeyError("No valid modalities provided in inputs")

        weights = torch.softmax(torch.stack(weight_list), dim=0)
        weighted_sum = sum(w * e for w, e in zip(weights, encoded_list))

        return self.fusion_layer(weighted_sum)


class OmniFusionModel(nn.Module):
    """
    Unified fusion model integrating all 13 engines through neural network.

    Architecture:
    1. Feature extraction from each detector/model
    2. Neural encoding to fixed-size embeddings
    3. Hybrid fusion (early + late + attention)
    4. Multi-task heads (anomaly detection, classification, regression)
    """

    def __init__(
        self,
        feature_dims: dict[str, int] | None = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()

        if feature_dims is None:
            feature_dims = {
                "statistical": 10,
                "temporal": 32,
                "spatial": 32,
                "dimensional": 50,
                "directive": 20,
                "quantum": 16,
                "astrophysical": 24,
                "biometric": 128,
                "affective": 64,
                "neural": 48,
                "consciousness": 32,
                "security": 40,
                "resilience": 16,
            }

        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim

        self.encoders = nn.ModuleDict(
            {
                "statistical": StatisticalEncoder(
                    feature_dims.get("statistical", 10), 64, hidden_dim
                ),
                "temporal": TemporalEncoder(
                    feature_dims.get("temporal", 64),
                    hidden_dim // 2,
                    hidden_dim,
                    num_layers=2,
                ),
                "biometric": BiometricEncoder(input_channels=3, output_dim=hidden_dim),
                "quantum": QuantumEncoder(feature_dims.get("quantum", 8), hidden_dim),
                "astrophysical": AstrophysicalEncoder(
                    feature_dims.get("astrophysical", 24), hidden_dim
                ),
                "affective": AffectiveEncoder(
                    feature_dims.get("affective", 64),
                    hidden_dim // 2,
                    hidden_dim,
                    num_layers=2,
                ),
            }
        )

        self.generic_encoders = nn.ModuleDict()
        for name, dim in feature_dims.items():
            if name not in self.encoders:
                self.generic_encoders[name] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )

        self.fusion_layer = HybridFusionLayer(
            feature_dims=dict.fromkeys(feature_dims.keys(), hidden_dim),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        detector_features: dict[str, torch.Tensor],
        detector_scores: dict[str, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through fusion network.

        Args:
            detector_features: Dict of raw features from each detector
            detector_scores: Dict of individual detector anomaly scores
            return_attention: Whether to return attention weights

        Returns:
            Dict containing:
                - anomaly_probs: [batch_size, 1]
                - class_logits: [batch_size, num_classes]
                - regression_output: [batch_size, 1]
                - attention_weights: Dict (if return_attention=True)
        """
        encoded_features = {}

        for name, features in detector_features.items():
            if name in self.encoders:
                encoded_features[name] = self.encoders[name](features)
            elif name in self.generic_encoders:
                encoded_features[name] = self.generic_encoders[name](features)
            elif features.dim() == 2 and features.shape[1] == self.hidden_dim:
                encoded_features[name] = features
            elif features.dim() == 2:
                proj = nn.Linear(features.shape[1], self.hidden_dim).to(features.device)
                encoded_features[name] = proj(features)

        if detector_scores is None:
            detector_scores = {
                name: torch.zeros(features.shape[0], 1)
                for name, features in encoded_features.items()
            }

        fused_repr, attention_weights = self.fusion_layer(encoded_features, detector_scores)

        anomaly_probs = torch.sigmoid(self.anomaly_head(fused_repr))
        class_logits = self.classification_head(fused_repr)
        regression_output = self.regression_head(fused_repr)

        output = {
            "anomaly_probs": anomaly_probs,
            "class_logits": class_logits,
            "regression_output": regression_output,
        }

        if return_attention:
            output["attention_weights"] = attention_weights

        return output

    def get_detector_importance(
        self, detector_features: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        Get importance scores for each detector based on attention weights.

        Args:
            detector_features: Dict of features from each detector

        Returns:
            Dict mapping detector name to importance score (0-1)
        """
        with torch.no_grad():
            output = self.forward(
                detector_features,
                return_attention=True,
            )

            weights = output["attention_weights"]["detector_weights"]

            importance = {
                name: float(weights[i]) for i, name in enumerate(self.feature_dims.keys())
            }

        return importance


class STEMDisciplineRouter:
    """Routes data to appropriate engines based on STEM discipline.

    Maps specific scientific disciplines to specialized detection engines
    for optimized multi-engine fusion. Improves detection accuracy through
    domain-aware weight selection.

    Example:
        router = STEMDisciplineRouter()
        weights = router.route(data, discipline='biology')

        fusion_model = OmniFusionModel(...)
        output = fusion_model(detector_features, detector_scores=weights)
    """

    def __init__(self):
        """Initialize STEM discipline routing mappings."""
        self.discipline_weights = {
            "biology": {
                "biometric": 0.90,
                "neural": 0.70,
                "affective": 0.50,
                "statistical": 0.60,
                "temporal": 0.40,
            },
            "biotechnology": {
                "biometric": 0.85,
                "neural": 0.65,
                "statistical": 0.70,
                "quantum": 0.40,
            },
            "genetics": {
                "biometric": 0.95,
                "statistical": 0.80,
                "dimensional": 0.50,
            },
            "neuroscience": {
                "neural": 0.95,
                "biometric": 0.80,
                "affective": 0.75,
                "consciousness": 0.70,
            },
            "physics": {
                "quantum": 0.90,
                "astrophysical": 0.85,
                "dimensional": 0.70,
                "statistical": 0.65,
                "temporal": 0.60,
            },
            "quantum_mechanics": {
                "quantum": 0.95,
                "dimensional": 0.75,
                "statistical": 0.70,
            },
            "astrophysics": {
                "astrophysical": 0.95,
                "quantum": 0.70,
                "dimensional": 0.65,
                "statistical": 0.70,
            },
            "particle_physics": {
                "quantum": 0.90,
                "astrophysical": 0.70,
                "dimensional": 0.80,
            },
            "chemistry": {
                "biometric": 0.60,
                "statistical": 0.75,
                "dimensional": 0.55,
                "quantum": 0.50,
            },
            "biochemistry": {
                "biometric": 0.80,
                "statistical": 0.70,
                "neural": 0.50,
            },
            "materials_science": {
                "dimensional": 0.80,
                "quantum": 0.60,
                "statistical": 0.70,
            },
            "earth_sciences": {
                "spatial": 0.85,
                "temporal": 0.75,
                "statistical": 0.70,
                "dimensional": 0.60,
            },
            "geology": {
                "spatial": 0.90,
                "temporal": 0.70,
                "statistical": 0.75,
            },
            "oceanography": {
                "spatial": 0.85,
                "temporal": 0.80,
                "statistical": 0.75,
                "dimensional": 0.60,
            },
            "meteorology": {
                "temporal": 0.90,
                "spatial": 0.85,
                "statistical": 0.80,
            },
            "computer_science": {
                "neural": 0.85,
                "security": 0.90,
                "statistical": 0.80,
                "temporal": 0.65,
            },
            "artificial_intelligence": {
                "neural": 0.95,
                "consciousness": 0.75,
                "statistical": 0.80,
                "affective": 0.60,
            },
            "cybersecurity": {
                "security": 1.00,
                "neural": 0.70,
                "statistical": 0.80,
                "quantum": 0.50,
            },
            "electrical_engineering": {
                "temporal": 0.80,
                "statistical": 0.75,
                "dimensional": 0.60,
            },
            "aerospace_engineering": {
                "astrophysical": 0.85,
                "spatial": 0.80,
                "temporal": 0.70,
                "dimensional": 0.65,
            },
            "biomedical_engineering": {
                "biometric": 0.90,
                "neural": 0.75,
                "affective": 0.60,
                "statistical": 0.70,
            },
            "civil_engineering": {
                "spatial": 0.85,
                "statistical": 0.75,
                "resilience": 0.70,
            },
            "mechanical_engineering": {
                "temporal": 0.75,
                "statistical": 0.80,
                "dimensional": 0.65,
            },
            "statistics": {
                "statistical": 1.00,
                "temporal": 0.70,
                "dimensional": 0.60,
            },
            "data_science": {
                "statistical": 0.95,
                "neural": 0.80,
                "temporal": 0.70,
            },
            "applied_mathematics": {
                "statistical": 0.90,
                "dimensional": 0.80,
                "quantum": 0.60,
            },
        }

    def route(
        self, data: torch.Tensor, discipline: str, data_type: str | None = None
    ) -> dict[str, float]:
        """Route data to appropriate engines based on STEM discipline.

        Args:
            data: Input data tensor
            discipline: STEM discipline (e.g., 'biology', 'physics', 'cybersecurity')
            data_type: Optional data type hint ('timeseries', 'image', 'tabular')

        Returns:
            Dict mapping engine names to weight scores (0.0-1.0)
        """
        if discipline not in self.discipline_weights:
            return self._default_weights()

        weights = self.discipline_weights[discipline].copy()

        if data_type:
            weights = self._adjust_for_data_type(weights, data_type)

        return weights

    def _adjust_for_data_type(self, weights: dict[str, float], data_type: str) -> dict[str, float]:
        """Adjust weights based on data type characteristics."""
        adjusted = weights.copy()

        if data_type == "timeseries":
            adjusted["temporal"] = adjusted.get("temporal", 0.5) * 1.3
        elif data_type == "image":
            adjusted["spatial"] = adjusted.get("spatial", 0.5) * 1.3
            adjusted["biometric"] = adjusted.get("biometric", 0.5) * 1.2
        elif data_type == "tabular":
            adjusted["statistical"] = adjusted.get("statistical", 0.5) * 1.2

        max_weight = max(adjusted.values())
        if max_weight > 1.0:
            adjusted = {k: v / max_weight for k, v in adjusted.items()}

        return adjusted

    def _default_weights(self) -> dict[str, float]:
        """Return default weights when discipline is unknown."""
        return {
            "statistical": 0.75,
            "temporal": 0.60,
            "spatial": 0.60,
            "dimensional": 0.55,
            "neural": 0.65,
            "quantum": 0.50,
            "biometric": 0.50,
            "affective": 0.45,
            "astrophysical": 0.45,
            "security": 0.60,
            "consciousness": 0.40,
            "resilience": 0.50,
        }

    def explain_routing(self, discipline: str) -> dict[str, Any]:
        """Explain why engines were prioritized for a discipline.

        Args:
            discipline: STEM discipline

        Returns:
            Dict with explanation and top engines
        """
        if discipline not in self.discipline_weights:
            return {
                "discipline": discipline,
                "status": "unknown",
                "explanation": "Discipline not in routing table, using default weights",
                "top_engines": list(self._default_weights().keys())[:3],
            }

        weights = self.discipline_weights[discipline]
        sorted_engines = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        explanations = {
            "biology": (
                "Biometric patterns dominate biological data, "
                "with neural and affective engines for behavioral aspects"
            ),
            "physics": (
                "Quantum and astrophysical engines excel at physics simulations, "
                "dimensional for complex spaces"
            ),
            "cybersecurity": (
                "Security engine designed specifically for threat detection, "
                "with neural for pattern recognition"
            ),
            "neuroscience": (
                "Neural and consciousness engines specialized for brain data, "
                "biometric for physiological signals"
            ),
            "chemistry": (
                "Dimensional analysis for molecular structures, "
                "statistical for reaction kinetics"
            ),
        }

        return {
            "discipline": discipline,
            "status": "routed",
            "explanation": explanations.get(
                discipline, f"Optimized for {discipline} domain patterns"
            ),
            "top_engines": [name for name, _ in sorted_engines[:3]],
            "weights": dict(sorted_engines),
        }

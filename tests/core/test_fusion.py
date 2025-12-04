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
Test fusion mechanisms
"""

import torch

from omni_anomaly_engine.core.fusion import HybridFusionLayer
from omni_anomaly_engine.ml.fusion_network import OmniFusionModel


def test_hybrid_fusion_initialization():
    """Test hybrid fusion can be initialized"""
    fusion = HybridFusionLayer(feature_dims={"detector1": 64, "detector2": 64}, hidden_dim=128)
    assert fusion is not None


def test_feature_level_fusion():
    """Test feature-level fusion"""
    fusion = HybridFusionLayer(feature_dims={"statistical": 128, "temporal": 128}, hidden_dim=128)

    detector_features = {
        "statistical": torch.randn(32, 128),
        "temporal": torch.randn(32, 128),
    }

    detector_scores = {
        "statistical": torch.randn(32, 1),
        "temporal": torch.randn(32, 1),
    }

    fused, attn = fusion(detector_features, detector_scores)
    assert fused.shape[0] == 32
    assert fused.shape[1] == 128


def test_fusion_network():
    """Test neural fusion network"""
    network = OmniFusionModel(feature_dims={"detector1": 64, "detector2": 64}, hidden_dim=128)

    detector_features = {
        "detector1": torch.randn(32, 64),
        "detector2": torch.randn(32, 64),
    }

    output = network(detector_features)

    assert "anomaly_probs" in output
    assert output["anomaly_probs"].shape == (32, 1)

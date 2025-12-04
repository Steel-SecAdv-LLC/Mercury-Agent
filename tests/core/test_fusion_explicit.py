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

"""Tests for explicit hybrid fusion methods"""

import torch

from omni_anomaly_engine.core.fusion import EarlyFusionEncoder, HybridFusionLayer


def test_extract_features():
    """Test explicit extract_features method"""
    feature_dims = {"det1": 128, "det2": 128, "det3": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_features = {
        "det1": torch.randn(4, 128),
        "det2": torch.randn(4, 128),
        "det3": torch.randn(4, 128),
    }

    extracted = fusion.extract_features(detector_features)
    assert len(extracted) == 3
    assert all(v.shape == (4, 64) for v in extracted.values())


def test_early_fusion_forward():
    """Test explicit early fusion method"""
    feature_dims = {"det1": 128, "det2": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_features = {
        "det1": torch.randn(3, 128),
        "det2": torch.randn(3, 128),
    }

    output = fusion.early_fusion_forward(detector_features)
    assert output.shape == (3, 64)


def test_late_fusion_forward():
    """Test explicit late fusion method"""
    feature_dims = {"det1": 128, "det2": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_scores = {
        "det1": torch.randn(3, 1),
        "det2": torch.randn(3, 1),
    }

    output = fusion.late_fusion_forward(detector_scores)
    assert output.shape == (3, 1)


def test_hybrid_detect():
    """Test explicit hybrid_detect method"""
    feature_dims = {"det1": 128, "det2": 128, "det3": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_features = {
        "det1": torch.randn(2, 128),
        "det2": torch.randn(2, 128),
        "det3": torch.randn(2, 128),
    }

    detector_scores = {
        "det1": torch.randn(2, 1),
        "det2": torch.randn(2, 1),
        "det3": torch.randn(2, 1),
    }

    fused, attention_dict = fusion.hybrid_detect(detector_features, detector_scores)
    assert fused.shape == (2, 64)
    assert "detector_weights" in attention_dict
    assert "attention_weights" in attention_dict
    assert "early_contribution" in attention_dict
    assert "late_contribution" in attention_dict


def test_early_fusion_encoder():
    """Test EarlyFusionEncoder class"""
    encoder = EarlyFusionEncoder(input_dim=256, hidden_dim=128)

    concatenated = torch.randn(5, 256)
    output = encoder(concatenated)

    assert output.shape == (5, 128)

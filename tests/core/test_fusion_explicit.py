# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for explicit hybrid fusion methods."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.fusion import EarlyFusionEncoder, HybridFusionLayer


def test_extract_features() -> None:
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


def test_early_fusion_forward() -> None:
    """Test explicit early fusion method"""
    feature_dims = {"det1": 128, "det2": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_features = {
        "det1": torch.randn(3, 128),
        "det2": torch.randn(3, 128),
    }

    output = fusion.early_fusion_forward(detector_features)
    assert output.shape == (3, 64)


def test_late_fusion_forward() -> None:
    """Test explicit late fusion method"""
    feature_dims = {"det1": 128, "det2": 128}
    fusion = HybridFusionLayer(feature_dims, hidden_dim=64)

    detector_scores = {
        "det1": torch.randn(3, 1),
        "det2": torch.randn(3, 1),
    }

    output = fusion.late_fusion_forward(detector_scores)
    assert output.shape == (3, 1)


def test_hybrid_detect() -> None:
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


def test_early_fusion_encoder() -> None:
    """Test EarlyFusionEncoder class"""
    encoder = EarlyFusionEncoder(input_dim=256, hidden_dim=128)

    concatenated = torch.randn(5, 256)
    output = encoder(concatenated)

    assert output.shape == (5, 128)

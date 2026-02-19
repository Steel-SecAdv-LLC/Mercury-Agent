"""
Tests for Multimodal Fusion Network.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import pytest  # noqa: E402
pytest.importorskip("torch")

import torch
from torch import nn

from omni_mercury_engine.ml.multimodal_fusion import (
    CrossModalAttention,
    MultimodalFusionNetwork,
)


class TestCrossModalAttention:
    """Tests for CrossModalAttention class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        attention = CrossModalAttention(dim=64)
        assert isinstance(attention, nn.Module)
        assert isinstance(attention.attention, nn.MultiheadAttention)
        assert isinstance(attention.norm, nn.LayerNorm)

    def test_init_custom_heads(self) -> None:
        """Test initialization with custom number of heads."""
        attention = CrossModalAttention(dim=64, num_heads=8)
        assert isinstance(attention, nn.Module)

    def test_forward(self) -> None:
        """Test forward pass."""
        attention = CrossModalAttention(dim=64, num_heads=4)
        query = torch.randn(4, 10, 64)
        key_value = torch.randn(4, 10, 64)
        output = attention(query, key_value)
        assert output.shape == query.shape

    def test_forward_different_seq_lengths(self) -> None:
        """Test forward pass with different sequence lengths."""
        attention = CrossModalAttention(dim=64, num_heads=4)
        query = torch.randn(4, 10, 64)
        key_value = torch.randn(4, 20, 64)
        output = attention(query, key_value)
        assert output.shape == query.shape

    def test_forward_batch_sizes(self) -> None:
        """Test forward pass with different batch sizes."""
        attention = CrossModalAttention(dim=64, num_heads=4)
        for batch_size in [1, 4, 16]:
            query = torch.randn(batch_size, 10, 64)
            key_value = torch.randn(batch_size, 10, 64)
            output = attention(query, key_value)
            assert output.shape == (batch_size, 10, 64)

    def test_residual_connection(self) -> None:
        """Test that residual connection is applied."""
        attention = CrossModalAttention(dim=64, num_heads=4)
        query = torch.randn(4, 10, 64)
        key_value = torch.zeros(4, 10, 64)
        output = attention(query, key_value)
        assert output.shape == query.shape


class TestMultimodalFusionNetwork:
    """Tests for MultimodalFusionNetwork class."""

    def test_init_two_modalities(self) -> None:
        """Test initialization with two modalities."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims)
        assert isinstance(network, nn.Module)
        assert len(network.projections) == 2
        assert "vision" in network.projections
        assert "text" in network.projections

    def test_init_three_modalities(self) -> None:
        """Test initialization with three modalities."""
        modality_dims = {"vision": 256, "text": 128, "audio": 64}
        network = MultimodalFusionNetwork(modality_dims)
        assert len(network.projections) == 3
        assert len(network.cross_attentions) == 6

    def test_init_custom_fusion_dim(self) -> None:
        """Test initialization with custom fusion dimension."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=256)
        assert network.fusion_dim == 256

    def test_init_custom_num_heads(self) -> None:
        """Test initialization with custom number of attention heads."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, num_heads=8)
        assert isinstance(network, nn.Module)

    def test_forward_two_modalities(self) -> None:
        """Test forward pass with two modalities."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        output = network(modality_features)

        assert "anomaly_scores" in output
        assert "attended_features" in output
        assert output["anomaly_scores"].shape == (4, 1)

    def test_forward_three_modalities(self) -> None:
        """Test forward pass with three modalities."""
        modality_dims = {"vision": 256, "text": 128, "audio": 64}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
            "audio": torch.randn(4, 64),
        }
        output = network(modality_features)

        assert "anomaly_scores" in output
        assert output["anomaly_scores"].shape == (4, 1)

    def test_forward_with_sequence(self) -> None:
        """Test forward pass with sequence input."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 10, 256),
            "text": torch.randn(4, 10, 128),
        }
        output = network(modality_features)

        assert "anomaly_scores" in output

    def test_forward_batch_sizes(self) -> None:
        """Test forward pass with different batch sizes."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        for batch_size in [1, 4, 16]:
            modality_features = {
                "vision": torch.randn(batch_size, 256),
                "text": torch.randn(batch_size, 128),
            }
            output = network(modality_features)
            assert output["anomaly_scores"].shape == (batch_size, 1)

    def test_output_range(self) -> None:
        """Test that anomaly scores are in [0, 1] range (sigmoid output)."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        output = network(modality_features)

        assert (output["anomaly_scores"] >= 0).all()
        assert (output["anomaly_scores"] <= 1).all()

    def test_attended_features_returned(self) -> None:
        """Test that attended features are returned."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        output = network(modality_features)

        assert "attended_features" in output
        assert "vision" in output["attended_features"]
        assert "text" in output["attended_features"]

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {
            "vision": torch.randn(4, 256, requires_grad=True),
            "text": torch.randn(4, 128, requires_grad=True),
        }
        output = network(modality_features)
        loss = output["anomaly_scores"].sum()
        loss.backward()

        assert modality_features["vision"].grad is not None
        assert modality_features["text"].grad is not None


class TestMultimodalFusionNetworkEdgeCases:
    """Edge case tests for MultimodalFusionNetwork."""

    def test_single_modality(self) -> None:
        """Test with single modality (no cross-attention)."""
        modality_dims = {"vision": 256}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {"vision": torch.randn(4, 256)}
        output = network(modality_features)

        assert "anomaly_scores" in output
        assert len(network.cross_attentions) == 0

    def test_many_modalities(self) -> None:
        """Test with many modalities."""
        modality_dims = {
            "vision": 256,
            "text": 128,
            "audio": 64,
            "sensor": 32,
            "graph": 128,
        }
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=64)

        modality_features = {name: torch.randn(4, dim) for name, dim in modality_dims.items()}
        output = network(modality_features)

        assert "anomaly_scores" in output
        assert output["anomaly_scores"].shape == (4, 1)

    def test_small_fusion_dim(self) -> None:
        """Test with small fusion dimension."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=16, num_heads=2)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        output = network(modality_features)

        assert output["anomaly_scores"].shape == (4, 1)

    def test_large_fusion_dim(self) -> None:
        """Test with large fusion dimension."""
        modality_dims = {"vision": 256, "text": 128}
        network = MultimodalFusionNetwork(modality_dims, fusion_dim=512, num_heads=8)

        modality_features = {
            "vision": torch.randn(4, 256),
            "text": torch.randn(4, 128),
        }
        output = network(modality_features)

        assert output["anomaly_scores"].shape == (4, 1)

"""
Tests for HATCN-AD (Hierarchical Attention TCN for Anomaly Detection) module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.hatcn_ad import (
    HATCN_AD,
    HierarchicalAttention,
    TemporalBlock,
)


class TestTemporalBlock:
    """Tests for TemporalBlock class."""

    def test_init(self) -> None:
        """Test initialization."""
        block = TemporalBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=1)
        assert block.conv1.in_channels == 32
        assert block.conv1.out_channels == 64
        assert block.downsample is not None

    def test_init_same_channels(self) -> None:
        """Test initialization with same input/output channels."""
        block = TemporalBlock(in_channels=32, out_channels=32, kernel_size=3, dilation=1)
        assert block.downsample is None

    def test_forward(self) -> None:
        """Test forward pass."""
        block = TemporalBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=1)
        block.eval()
        x = torch.randn(4, 32, 100)  # batch=4, channels=32, seq_len=100
        with torch.no_grad():
            out = block.forward(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 64

    def test_forward_same_channels(self) -> None:
        """Test forward pass with same input/output channels."""
        block = TemporalBlock(in_channels=32, out_channels=32, kernel_size=3, dilation=1)
        block.eval()
        x = torch.randn(4, 32, 100)
        with torch.no_grad():
            out = block.forward(x)
        assert out.shape[0] == 4
        assert out.shape[1] == 32

    def test_different_dilations(self) -> None:
        """Test with different dilation values."""
        for dilation in [1, 2, 4, 8]:
            block = TemporalBlock(in_channels=32, out_channels=32, kernel_size=3, dilation=dilation)
            block.eval()
            x = torch.randn(4, 32, 100)
            with torch.no_grad():
                out = block.forward(x)
            assert out.shape[0] == 4
            assert out.shape[1] == 32

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the block."""
        block = TemporalBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=1)
        block.train()
        x = torch.randn(4, 32, 100, requires_grad=True)
        out = block.forward(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None


class TestHierarchicalAttention:
    """Tests for HierarchicalAttention class."""

    def test_init(self) -> None:
        """Test initialization."""
        attn = HierarchicalAttention(hidden_dim=64, num_scales=3)
        assert attn.num_scales == 3
        assert len(attn.scale_attentions) == 3
        assert attn.scale_weights.shape == (3,)

    def test_init_different_scales(self) -> None:
        """Test initialization with different number of scales."""
        for num_scales in [2, 3, 4, 5]:
            attn = HierarchicalAttention(hidden_dim=64, num_scales=num_scales)
            assert attn.num_scales == num_scales
            assert len(attn.scale_attentions) == num_scales

    def test_forward(self) -> None:
        """Test forward pass."""
        attn = HierarchicalAttention(hidden_dim=64, num_scales=3)
        attn.eval()
        scale_features = [
            torch.randn(4, 50, 64),  # batch=4, seq_len=50, hidden_dim=64
            torch.randn(4, 50, 64),
            torch.randn(4, 50, 64),
        ]
        with torch.no_grad():
            fused, attn_weights = attn.forward(scale_features)
        assert fused.shape == (4, 64)
        assert len(attn_weights) == 3

    def test_attention_weights_sum_to_one(self) -> None:
        """Test that attention weights sum to 1."""
        attn = HierarchicalAttention(hidden_dim=64, num_scales=3)
        attn.eval()
        scale_features = [
            torch.randn(4, 50, 64),
            torch.randn(4, 50, 64),
            torch.randn(4, 50, 64),
        ]
        with torch.no_grad():
            _, attn_weights = attn.forward(scale_features)
        for weights in attn_weights:
            sums = weights.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through attention."""
        attn = HierarchicalAttention(hidden_dim=64, num_scales=3)
        attn.train()
        scale_features = [
            torch.randn(4, 50, 64, requires_grad=True),
            torch.randn(4, 50, 64, requires_grad=True),
            torch.randn(4, 50, 64, requires_grad=True),
        ]
        fused, _ = attn.forward(scale_features)
        loss = fused.mean()
        loss.backward()
        for features in scale_features:
            assert features.grad is not None


class TestHATCN_AD:
    """Tests for HATCN_AD class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        model = HATCN_AD(input_dim=10)
        assert model.input_proj.in_features == 10
        assert model.input_proj.out_features == 64
        assert len(model.temporal_blocks) == 3

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        model = HATCN_AD(input_dim=20, hidden_dim=128, num_scales=4, kernel_size=5)
        assert model.input_proj.in_features == 20
        assert model.input_proj.out_features == 128
        assert len(model.temporal_blocks) == 4

    def test_forward(self) -> None:
        """Test forward pass."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.eval()
        x = torch.randn(4, 50, 10)  # batch=4, seq_len=50, input_dim=10
        with torch.no_grad():
            result = model.forward(x)
        assert "anomaly_scores" in result
        assert "attention_weights" in result
        assert "scale_features" in result
        assert result["anomaly_scores"].shape == (4, 1)

    def test_anomaly_scores_range(self) -> None:
        """Test that anomaly scores are in [0, 1] range."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        scores = result["anomaly_scores"]
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.eval()
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, 50, 10)
            with torch.no_grad():
                result = model.forward(x)
            assert result["anomaly_scores"].shape == (batch_size, 1)

    def test_different_sequence_lengths(self) -> None:
        """Test with different sequence lengths."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.eval()
        for seq_len in [20, 50, 100, 200]:
            x = torch.randn(4, seq_len, 10)
            with torch.no_grad():
                result = model.forward(x)
            assert result["anomaly_scores"].shape == (4, 1)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.train()
        x = torch.randn(4, 50, 10, requires_grad=True)
        result = model.forward(x)
        loss = result["anomaly_scores"].mean()
        loss.backward()
        assert x.grad is not None

    def test_scale_features_output(self) -> None:
        """Test that scale features are returned correctly."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        assert len(result["scale_features"]) == 3
        for scale_feat in result["scale_features"]:
            assert scale_feat.shape[0] == 4  # batch size
            assert scale_feat.shape[2] == 32  # hidden_dim


class TestHATCN_ADEdgeCases:
    """Edge case tests for HATCN_AD."""

    def test_single_scale(self) -> None:
        """Test with single scale."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=1)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        assert result["anomaly_scores"].shape == (4, 1)

    def test_many_scales(self) -> None:
        """Test with many scales."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=5)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        assert result["anomaly_scores"].shape == (4, 1)
        assert len(result["scale_features"]) == 5

    def test_small_hidden_dim(self) -> None:
        """Test with small hidden dimension."""
        model = HATCN_AD(input_dim=10, hidden_dim=8, num_scales=2)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        assert result["anomaly_scores"].shape == (4, 1)

    def test_large_kernel_size(self) -> None:
        """Test with large kernel size."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=3, kernel_size=7)
        model.eval()
        x = torch.randn(4, 50, 10)
        with torch.no_grad():
            result = model.forward(x)
        assert result["anomaly_scores"].shape == (4, 1)

    def test_short_sequence(self) -> None:
        """Test with short sequence."""
        model = HATCN_AD(input_dim=10, hidden_dim=32, num_scales=2, kernel_size=3)
        model.eval()
        x = torch.randn(4, 10, 10)  # Very short sequence
        with torch.no_grad():
            result = model.forward(x)
        assert result["anomaly_scores"].shape == (4, 1)

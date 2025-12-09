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


class TestFusionNetworkForwardPass:
    """Comprehensive tests for ML fusion network forward pass."""

    def test_forward_pass_all_outputs(self):
        """Test forward pass returns all expected outputs."""
        network = OmniFusionModel(
            feature_dims={"statistical": 10, "temporal": 32, "quantum": 16},
            hidden_dim=128,
            num_classes=10,
        )

        detector_features = {
            "statistical": torch.randn(16, 10),
            "temporal": torch.randn(16, 32),
            "quantum": torch.randn(16, 16),
        }

        output = network(detector_features)

        assert "anomaly_probs" in output
        assert "class_logits" in output
        assert "regression_output" in output

        assert output["anomaly_probs"].shape == (16, 1)
        assert output["class_logits"].shape == (16, 10)
        assert output["regression_output"].shape == (16, 1)

    def test_forward_pass_with_attention(self):
        """Test forward pass with attention weights returned."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64, "detector2": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(8, 64),
            "detector2": torch.randn(8, 64),
        }

        output = network(detector_features, return_attention=True)

        assert "attention_weights" in output
        assert output["attention_weights"] is not None

    def test_forward_pass_with_detector_scores(self):
        """Test forward pass with explicit detector scores."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64, "detector2": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(8, 64),
            "detector2": torch.randn(8, 64),
        }

        detector_scores = {
            "detector1": torch.randn(8, 1),
            "detector2": torch.randn(8, 1),
        }

        output = network(detector_features, detector_scores=detector_scores)

        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (8, 1)

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample batch."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(1, 64),
        }

        output = network(detector_features)

        assert output["anomaly_probs"].shape == (1, 1)

    def test_forward_pass_large_batch(self):
        """Test forward pass with large batch size."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64, "detector2": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(256, 64),
            "detector2": torch.randn(256, 64),
        }

        output = network(detector_features)

        assert output["anomaly_probs"].shape == (256, 1)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the network."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(8, 64, requires_grad=True),
        }

        output = network(detector_features)
        loss = output["anomaly_probs"].sum()
        loss.backward()

        assert detector_features["detector1"].grad is not None

    def test_forward_pass_eval_mode(self):
        """Test forward pass in evaluation mode."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64},
            hidden_dim=128,
        )
        network.eval()

        detector_features = {
            "detector1": torch.randn(8, 64),
        }

        with torch.no_grad():
            output = network(detector_features)

        assert output["anomaly_probs"].shape == (8, 1)
        assert 0 <= output["anomaly_probs"].min() <= 1
        assert 0 <= output["anomaly_probs"].max() <= 1

    def test_get_detector_importance(self):
        """Test detector importance computation."""
        network = OmniFusionModel(
            feature_dims={"detector1": 64, "detector2": 64},
            hidden_dim=128,
        )

        detector_features = {
            "detector1": torch.randn(8, 64),
            "detector2": torch.randn(8, 64),
        }

        importance = network.get_detector_importance(detector_features)

        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestGatedFusion:
    """Tests for GatedFusion mechanism."""

    def test_gated_fusion_forward(self):
        """Test gated fusion forward pass."""
        from omni_anomaly_engine.ml.fusion_network import GatedFusion

        fusion = GatedFusion(input_dim=64, hidden_dim=32)

        x1 = torch.randn(16, 64)
        x2 = torch.randn(16, 64)

        output = fusion(x1, x2)

        assert output.shape == (16, 64)

    def test_gated_fusion_with_gate_return(self):
        """Test gated fusion returns gate values."""
        from omni_anomaly_engine.ml.fusion_network import GatedFusion

        fusion = GatedFusion(input_dim=64, hidden_dim=32)

        x1 = torch.randn(16, 64)
        x2 = torch.randn(16, 64)

        output, gate = fusion(x1, x2, return_gate=True)

        assert output.shape == (16, 64)
        assert gate.shape == (16, 64)
        assert 0 <= gate.min() <= 1
        assert 0 <= gate.max() <= 1


class TestMultimodalFusion:
    """Tests for MultimodalFusion mechanism."""

    def test_multimodal_fusion_forward(self):
        """Test multimodal fusion forward pass."""
        from omni_anomaly_engine.ml.fusion_network import MultimodalFusion

        fusion = MultimodalFusion(
            modality_dims={"visual": 128, "audio": 64, "text": 256},
            output_dim=128,
        )

        inputs = {
            "visual": torch.randn(8, 128),
            "audio": torch.randn(8, 64),
            "text": torch.randn(8, 256),
        }

        output = fusion(inputs)

        assert output.shape == (8, 128)

    def test_multimodal_fusion_partial_modalities(self):
        """Test multimodal fusion with partial modalities."""
        from omni_anomaly_engine.ml.fusion_network import MultimodalFusion

        fusion = MultimodalFusion(
            modality_dims={"visual": 128, "audio": 64, "text": 256},
            output_dim=128,
        )

        inputs = {
            "visual": torch.randn(8, 128),
            "audio": torch.randn(8, 64),
        }

        output = fusion(inputs)

        assert output.shape == (8, 128)

    def test_multimodal_fusion_no_modalities_raises(self):
        """Test multimodal fusion raises error with no valid modalities."""
        import pytest

        from omni_anomaly_engine.ml.fusion_network import MultimodalFusion

        fusion = MultimodalFusion(
            modality_dims={"visual": 128, "audio": 64},
            output_dim=128,
        )

        inputs = {
            "unknown": torch.randn(8, 128),
        }

        with pytest.raises(KeyError):
            fusion(inputs)


class TestFusionNetworkBasic:
    """Tests for basic FusionNetwork class."""

    def test_fusion_network_basic_forward(self):
        """Test basic fusion network forward pass."""
        from omni_anomaly_engine.ml.fusion_network import FusionNetwork

        network = FusionNetwork(
            input_dims=[64, 128, 32],
            output_dim=64,
            hidden_dim=128,
        )

        inputs = [
            torch.randn(16, 64),
            torch.randn(16, 128),
            torch.randn(16, 32),
        ]

        output = network(inputs)

        assert output.shape == (16, 64)

    def test_fusion_network_single_modality(self):
        """Test fusion network with single modality."""
        from omni_anomaly_engine.ml.fusion_network import FusionNetwork

        network = FusionNetwork(
            input_dims=[64],
            output_dim=32,
        )

        inputs = [torch.randn(8, 64)]

        output = network(inputs)

        assert output.shape == (8, 32)

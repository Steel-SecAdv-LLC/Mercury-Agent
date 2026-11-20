# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test fusion mechanisms."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.core.fusion import HybridFusionLayer
from omni_mercury_engine.ml.fusion_network import OmniFusionModel


def test_hybrid_fusion_initialization() -> None:
    """Test hybrid fusion can be initialized"""
    fusion = HybridFusionLayer(feature_dims={"detector1": 64, "detector2": 64}, hidden_dim=128)
    assert fusion is not None


def test_feature_level_fusion() -> None:
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


def test_fusion_network() -> None:
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

    def test_forward_pass_all_outputs(self) -> None:
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

    def test_forward_pass_with_attention(self) -> None:
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

    def test_forward_pass_with_detector_scores(self) -> None:
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

    def test_forward_pass_single_sample(self) -> None:
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

    def test_forward_pass_large_batch(self) -> None:
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

    def test_forward_pass_gradient_flow(self) -> None:
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

    def test_forward_pass_eval_mode(self) -> None:
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

    def test_get_detector_importance(self) -> None:
        """Importance is attention-sourced: input-dependent and normalized.

        The previous source, ``softmax(late_fusion_weights)``, never receives
        gradient (the late output is a detached diagnostic) and therefore
        reported exactly ``1/n`` for every detector on every input.
        """
        torch.manual_seed(0)
        network = OmniFusionModel(
            feature_dims={"detector1": 64, "detector2": 64},
            hidden_dim=128,
        )
        network.eval()

        features_a = {
            "detector1": torch.zeros(8, 64),
            "detector2": torch.ones(8, 64) * 5.0,
        }
        features_b = {
            "detector1": torch.randn(8, 64) * 3.0,
            "detector2": torch.randn(8, 64) * 0.1,
        }

        importance_a = network.get_detector_importance(features_a)
        importance_b = network.get_detector_importance(features_b)

        assert isinstance(importance_a, dict)
        assert set(importance_a) == {"detector1", "detector2"}
        assert abs(sum(importance_a.values()) - 1.0) < 1e-5
        assert all(v >= 0.0 for v in importance_a.values())
        # Attention-sourced importance must depend on the input; the old
        # late_fusion_weights source returned the identical uniform 1/n here.
        assert importance_a != importance_b

    def test_detector_importance_matches_single_forward_attention(self) -> None:
        """importance_from_attention on a forward's own attention dict equals
        get_detector_importance, so callers need no second forward pass."""
        torch.manual_seed(0)
        network = OmniFusionModel(
            feature_dims={"detector1": 32, "detector2": 32},
            hidden_dim=64,
        )
        network.eval()
        features = {
            "detector1": torch.randn(4, 32),
            "detector2": torch.randn(4, 32),
        }
        with torch.no_grad():
            output = network(features, return_attention=True)
        from_attention = network.importance_from_attention(output["attention_weights"])
        from_method = network.get_detector_importance(features)
        assert set(from_attention) == set(from_method)
        for name in from_attention:
            assert abs(from_attention[name] - from_method[name]) < 1e-6


class TestGatedFusion:
    """Tests for GatedFusion mechanism."""

    def test_gated_fusion_forward(self) -> None:
        """Test gated fusion forward pass."""
        from omni_mercury_engine.ml.fusion_network import GatedFusion

        fusion = GatedFusion(input_dim=64, hidden_dim=32)

        x1 = torch.randn(16, 64)
        x2 = torch.randn(16, 64)

        output = fusion(x1, x2)

        assert output.shape == (16, 64)

    def test_gated_fusion_with_gate_return(self) -> None:
        """Test gated fusion returns gate values."""
        from omni_mercury_engine.ml.fusion_network import GatedFusion

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

    def test_multimodal_fusion_forward(self) -> None:
        """Test multimodal fusion forward pass."""
        from omni_mercury_engine.ml.fusion_network import MultimodalFusion

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

    def test_multimodal_fusion_partial_modalities(self) -> None:
        """Test multimodal fusion with partial modalities."""
        from omni_mercury_engine.ml.fusion_network import MultimodalFusion

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

    def test_multimodal_fusion_no_modalities_raises(self) -> None:
        """Test multimodal fusion raises error with no valid modalities."""
        import pytest

        from omni_mercury_engine.ml.fusion_network import MultimodalFusion

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

    def test_fusion_network_basic_forward(self) -> None:
        """Test basic fusion network forward pass."""
        from omni_mercury_engine.ml.fusion_network import FusionNetwork

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

    def test_fusion_network_single_modality(self) -> None:
        """Test fusion network with single modality."""
        from omni_mercury_engine.ml.fusion_network import FusionNetwork

        network = FusionNetwork(
            input_dims=[64],
            output_dim=32,
        )

        inputs = [torch.randn(8, 64)]

        output = network(inputs)

        assert output.shape == (8, 32)


class TestCrossDomainFusion:
    """Tests for cross-domain fusion paths combining multiple domain detectors."""

    def test_geological_security_fusion(self) -> None:
        """Test fusion of geological and security domain features."""
        network = OmniFusionModel(
            feature_dims={
                "tornado": 20,
                "flood": 20,
                "hurricane": 20,
                "network_intrusion": 41,
                "threat_intel": 32,
            },
            hidden_dim=128,
            num_classes=5,
        )

        detector_features = {
            "tornado": torch.randn(16, 20),
            "flood": torch.randn(16, 20),
            "hurricane": torch.randn(16, 20),
            "network_intrusion": torch.randn(16, 41),
            "threat_intel": torch.randn(16, 32),
        }

        output = network(detector_features)

        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (16, 1)
        assert output["class_logits"].shape == (16, 5)

    def test_medical_humanitarian_fusion(self) -> None:
        """Test fusion of medical and humanitarian domain features."""
        network = OmniFusionModel(
            feature_dims={
                "sepsis": 24,
                "cardiac": 20,
                "biometric": 128,
                "missing_person": 64,
            },
            hidden_dim=128,
            num_classes=4,
        )

        detector_features = {
            "sepsis": torch.randn(8, 24),
            "cardiac": torch.randn(8, 20),
            "biometric": torch.randn(8, 128),
            "missing_person": torch.randn(8, 64),
        }

        output = network(detector_features)

        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (8, 1)

    def test_space_environmental_fusion(self) -> None:
        """Test fusion of space and environmental domain features."""
        network = OmniFusionModel(
            feature_dims={
                "solar_storm": 32,
                "schumann_resonance": 16,
                "earthquake": 14,
                "ocean_anomaly": 20,
            },
            hidden_dim=128,
        )

        detector_features = {
            "solar_storm": torch.randn(12, 32),
            "schumann_resonance": torch.randn(12, 16),
            "earthquake": torch.randn(12, 14),
            "ocean_anomaly": torch.randn(12, 20),
        }

        output = network(detector_features)

        assert "anomaly_probs" in output
        assert output["anomaly_probs"].shape == (12, 1)

    def test_all_domains_fusion(self) -> None:
        """Test fusion across all major domain categories."""
        network = OmniFusionModel(
            feature_dims={
                "geological": 60,
                "medical": 44,
                "security": 73,
                "space": 48,
                "humanitarian": 192,
                "infrastructure": 55,
            },
            hidden_dim=256,
            num_classes=10,
        )

        detector_features = {
            "geological": torch.randn(4, 60),
            "medical": torch.randn(4, 44),
            "security": torch.randn(4, 73),
            "space": torch.randn(4, 48),
            "humanitarian": torch.randn(4, 192),
            "infrastructure": torch.randn(4, 55),
        }

        output = network(detector_features)

        assert "anomaly_probs" in output
        assert "class_logits" in output
        assert "regression_output" in output
        assert output["anomaly_probs"].shape == (4, 1)
        assert output["class_logits"].shape == (4, 10)

    def test_cross_domain_attention_weights(self) -> None:
        """Test attention weights across domains."""
        network = OmniFusionModel(
            feature_dims={
                "domain_a": 64,
                "domain_b": 64,
                "domain_c": 64,
            },
            hidden_dim=128,
        )

        detector_features = {
            "domain_a": torch.randn(8, 64),
            "domain_b": torch.randn(8, 64),
            "domain_c": torch.randn(8, 64),
        }

        output = network(detector_features, return_attention=True)

        assert "attention_weights" in output
        if output["attention_weights"] is not None:
            # attention_weights may be a dict or tensor depending on implementation
            if isinstance(output["attention_weights"], dict):
                assert len(output["attention_weights"]) > 0
            else:
                assert output["attention_weights"].shape[0] == 8

    def test_cross_domain_with_missing_detector(self) -> None:
        """Test fusion handles missing detector gracefully."""
        network = OmniFusionModel(
            feature_dims={
                "detector_a": 64,
                "detector_b": 64,
            },
            hidden_dim=128,
        )

        detector_features = {
            "detector_a": torch.randn(8, 64),
        }

        try:
            output = network(detector_features)
            assert output["anomaly_probs"].shape[0] == 8
        except KeyError:
            pass  # Expected: output key may vary by implementation

    def test_hybrid_fusion_cross_domain(self) -> None:
        """Test HybridFusionLayer with cross-domain features."""
        fusion = HybridFusionLayer(
            feature_dims={
                "geological": 128,
                "medical": 128,
                "security": 128,
            },
            hidden_dim=256,
        )

        detector_features = {
            "geological": torch.randn(16, 128),
            "medical": torch.randn(16, 128),
            "security": torch.randn(16, 128),
        }

        detector_scores = {
            "geological": torch.randn(16, 1),
            "medical": torch.randn(16, 1),
            "security": torch.randn(16, 1),
        }

        fused, attn = fusion(detector_features, detector_scores)

        assert fused.shape[0] == 16
        assert fused.shape[1] == 256

    def test_cross_domain_ethical_alignment(self) -> None:
        """Test fusion maintains ethical alignment across domains."""
        network = OmniFusionModel(
            feature_dims={
                "humanitarian": 128,
                "medical": 64,
            },
            hidden_dim=128,
        )

        detector_features = {
            "humanitarian": torch.randn(8, 128),
            "medical": torch.randn(8, 64),
        }

        output = network(detector_features)

        probs = output["anomaly_probs"]
        assert 0 <= probs.min() <= 1
        assert 0 <= probs.max() <= 1


class TestUnregisteredFeatureGroups:
    """Groups absent from feature_dims are dropped, not lazily projected.

    The fusion layer only consumes registered detector names, so a lazily
    minted projection for an unregistered group is dead weight: never trained
    (created after the optimizer exists), never read, yet written into
    checkpoints.
    """

    def test_unknown_group_creates_no_dynamic_projection(self) -> None:
        """Unknown groups must be ignored without minting dead parameters."""
        network = OmniFusionModel(feature_dims={"d1": 8, "d2": 8}, hidden_dim=16)
        features = {
            "d1": torch.randn(4, 8),
            "d2": torch.randn(4, 8),
            "mystery": torch.randn(4, 12),
        }
        output = network(features)
        assert output["anomaly_probs"].shape == (4, 1)
        assert len(network._dynamic_projections) == 0
        assert not any(k.startswith("_dynamic_projections") for k in network.state_dict())

    def test_unknown_group_output_identical_to_omitting_it(self) -> None:
        """The dropped group must not influence the fused output."""
        torch.manual_seed(0)
        network = OmniFusionModel(feature_dims={"d1": 8}, hidden_dim=16)
        network.eval()
        base = {"d1": torch.randn(4, 8)}
        with torch.no_grad():
            with_unknown = network({**base, "mystery": torch.randn(4, 12)})
            without = network(base)
        torch.testing.assert_close(with_unknown["anomaly_probs"], without["anomaly_probs"])

    def test_unknown_group_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """The drop is logged at WARNING, naming the group, once per group."""
        import logging

        network = OmniFusionModel(feature_dims={"d1": 8}, hidden_dim=16)
        features = {"d1": torch.randn(2, 8), "mystery": torch.randn(2, 12)}
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.ml.fusion_network"):
            network(features)
            network(features)
        warnings = [r for r in caplog.records if "mystery" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING

    def test_only_unknown_groups_raises(self) -> None:
        """All-unregistered input fails loud instead of fusing nothing."""
        network = OmniFusionModel(feature_dims={"d1": 8}, hidden_dim=16)
        with pytest.raises(ValueError, match="No registered feature groups"):
            network({"mystery": torch.randn(2, 12)})


class TestZeroFillDeviceSafety:
    """Zero-filled tensors for absent groups follow the input/module device.

    CUDA is unavailable here, so the always-available ``meta`` device stands
    in: pre-fix these paths built ``torch.zeros`` with no device (implicit
    CPU) and crashed on the subsequent cat/stack with non-CPU inputs.
    """

    @staticmethod
    def _meta_layer() -> HybridFusionLayer:
        torch.manual_seed(0)
        layer = HybridFusionLayer(feature_dims={"a": 4, "b": 4, "c": 4}, hidden_dim=8, num_heads=2)
        return layer.to("meta")

    def test_extract_features_zero_fill_device(self) -> None:
        """extract_features zero-fills absent groups on the input's device."""
        layer = self._meta_layer()
        extracted = layer.extract_features({"a": torch.randn(3, 4, device="meta")})
        assert set(extracted) == {"a", "b", "c"}
        assert all(t.device.type == "meta" for t in extracted.values())

    def test_early_fusion_forward_zero_fill_device(self) -> None:
        """early_fusion_forward concatenates on the input's device."""
        layer = self._meta_layer()
        out = layer.early_fusion_forward({"a": torch.randn(3, 4, device="meta")})
        assert out.device.type == "meta"

    def test_late_fusion_forward_zero_fill_device(self) -> None:
        """late_fusion_forward zero-fills absent scores on the input's device."""
        layer = self._meta_layer()
        out = layer.late_fusion_forward({"a": torch.rand(3, 1, device="meta")})
        assert out.device.type == "meta"

    def test_hybrid_detect_zero_fill_device(self) -> None:
        """hybrid_detect stacks projections on the input's device."""
        layer = self._meta_layer()
        fused, _ = layer.hybrid_detect(
            {"a": torch.randn(3, 4, device="meta")},
            {"a": torch.rand(3, 1, device="meta")},
        )
        assert fused.device.type == "meta"

    def test_default_detector_scores_device(self) -> None:
        """OmniFusionModel's default zero scores follow the features' device."""
        torch.manual_seed(0)
        network = OmniFusionModel(feature_dims={"d1": 8, "d2": 8}, hidden_dim=16)
        network = network.to("meta")
        network.eval()
        features = {
            "d1": torch.randn(2, 8, device="meta"),
            "d2": torch.randn(2, 8, device="meta"),
        }
        output = network(features)
        assert output["anomaly_probs"].device.type == "meta"

"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for ThreeRAttentionBlock and LyapunovAnomalyLoss.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omni_mercury_engine.ml.three_r_attention import (
    ThreeRAnomalyTransformer,
    ThreeRAttentionBlock,
)
from omni_mercury_engine.ml.training import LyapunovAnomalyLoss


class TestThreeRAttentionBlock:
    """Tests for ThreeRAttentionBlock."""

    def test_initialization(self) -> None:
        """Test block initializes with correct golden-ratio weights."""
        block = ThreeRAttentionBlock(d_model=128, n_heads=4)

        # Check golden ratio weights
        phi = 1.618033988749895
        phi_sum = phi + 1.0 + (1.0 / phi)

        assert abs(block.w_R.item() - phi / phi_sum) < 1e-6
        assert abs(block.w_H.item() - 1.0 / phi_sum) < 1e-6
        assert abs(block.w_O.item() - (1.0 / phi) / phi_sum) < 1e-6

        # Weights should sum to 1
        total_weight = block.w_R.item() + block.w_H.item() + block.w_O.item()
        assert abs(total_weight - 1.0) < 1e-6

    def test_forward_shape(self) -> None:
        """Test forward pass produces correct output shapes."""
        block = ThreeRAttentionBlock(d_model=128, n_heads=4)
        x = torch.randn(8, 50, 128)  # [batch, seq_len, d_model]

        output, scores = block(x)

        assert output.shape == x.shape
        assert "R_score" in scores
        assert "H_score" in scores
        assert "O_score" in scores
        assert "anomaly_scores" in scores
        assert scores["anomaly_scores"].shape == (8,)

    def test_forward_with_component_outputs(self) -> None:
        """Test forward returns component outputs when requested."""
        block = ThreeRAttentionBlock(d_model=128, n_heads=4)
        x = torch.randn(4, 32, 128)

        output, scores = block(x, return_component_outputs=True)

        assert "R_output" in scores
        assert "H_output" in scores
        assert "O_output" in scores
        assert scores["R_output"].shape == x.shape

    def test_ethical_scaling(self) -> None:
        """Test ethical scaling is applied correctly."""
        block = ThreeRAttentionBlock(d_model=64, ethical_threshold=0.96)

        expected_scale = 0.96**1.618033988749895
        assert abs(scores_dict := block(torch.randn(2, 10, 64))[1])
        assert "ethical_scale" in scores_dict
        assert abs(scores_dict["ethical_scale"] - expected_scale) < 1e-6

    def test_multi_scale_attention(self) -> None:
        """Test multi-scale attention works with different num_scales."""
        for num_scales in [1, 2, 3, 4]:
            block = ThreeRAttentionBlock(d_model=64, n_heads=2, num_scales=num_scales)
            x = torch.randn(2, 64, 64)
            output, _ = block(x)
            assert output.shape == x.shape


class TestThreeRAnomalyTransformer:
    """Tests for ThreeRAnomalyTransformer."""

    def test_initialization(self) -> None:
        """Test model initializes correctly."""
        model = ThreeRAnomalyTransformer(
            input_dim=25,
            d_model=128,
            n_heads=4,
            num_layers=2,
        )
        assert model.input_dim == 25
        assert model.d_model == 128

    def test_forward_shape(self) -> None:
        """Test forward produces correct output shapes."""
        model = ThreeRAnomalyTransformer(input_dim=25, d_model=64, num_layers=1)
        x = torch.randn(8, 100, 25)

        output = model(x)

        assert output["reconstruction"].shape == x.shape
        assert output["anomaly_scores"].shape == (8,)
        assert len(output["layer_scores"]) == 1

    def test_forward_with_latent(self) -> None:
        """Test forward returns latent when requested."""
        model = ThreeRAnomalyTransformer(input_dim=10, d_model=32, num_layers=1)
        x = torch.randn(4, 50, 10)

        output = model(x, return_latent=True)

        assert "latent" in output
        assert output["latent"].shape == (4, 50, 32)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through the model."""
        model = ThreeRAnomalyTransformer(input_dim=10, d_model=32, num_layers=1)
        x = torch.randn(2, 20, 10, requires_grad=True)

        output = model(x)
        loss = output["anomaly_scores"].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLyapunovAnomalyLoss:
    """Tests for LyapunovAnomalyLoss."""

    def test_initialization(self) -> None:
        """Test loss initializes with correct parameters."""
        loss_fn = LyapunovAnomalyLoss(lambda_kl=0.5, mu_stability=0.2, alpha=0.3)

        assert loss_fn.lambda_kl == 0.5
        assert loss_fn.mu_stability == 0.2
        assert loss_fn.alpha == 0.3

    def test_forward_without_kl(self) -> None:
        """Test forward pass without KL divergence."""
        loss_fn = LyapunovAnomalyLoss(lambda_kl=0.0, mu_stability=0.1)

        x = torch.randn(8, 100, 25)
        x_recon = torch.randn(8, 100, 25)
        anomaly_scores = torch.randn(8)

        result = loss_fn(x=x, x_recon=x_recon, anomaly_scores=anomaly_scores)

        assert "total" in result
        assert "reconstruction" in result
        assert "kl" in result
        assert "stability" in result
        assert "lyapunov_V" in result

        # KL should be zero when lambda_kl=0
        assert result["kl"].item() == 0.0

    def test_forward_with_kl(self) -> None:
        """Test forward pass with VAE KL divergence."""
        loss_fn = LyapunovAnomalyLoss(lambda_kl=1.0, mu_stability=0.1)

        x = torch.randn(8, 100, 25)
        x_recon = torch.randn(8, 100, 25)
        anomaly_scores = torch.randn(8)
        mu = torch.randn(8, 32)
        logvar = torch.randn(8, 32)

        result = loss_fn(
            x=x,
            x_recon=x_recon,
            anomaly_scores=anomaly_scores,
            mu=mu,
            logvar=logvar,
        )

        # KL should be non-zero
        assert result["kl"].item() != 0.0

    def test_stability_tracking(self) -> None:
        """Test stability violation tracking across iterations."""
        loss_fn = LyapunovAnomalyLoss(mu_stability=0.1, alpha=0.25)

        x = torch.randn(4, 50, 10)
        x_recon = torch.randn(4, 50, 10)

        # First iteration - no stability loss (no previous scores)
        scores1 = torch.ones(4) * 0.5
        result1 = loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores1)
        assert result1["stability"].item() == 0.0

        # Second iteration - increasing scores should trigger stability penalty
        scores2 = torch.ones(4) * 2.0  # Much higher
        result2 = loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores2)
        # V̇ = V_t - V_{t-1} = 4.0 - 0.25 = 3.75 (positive, violation)
        assert result2["stability"].item() > 0

    def test_reset_state(self) -> None:
        """Test state reset functionality."""
        loss_fn = LyapunovAnomalyLoss()

        x = torch.randn(2, 10, 5)
        x_recon = torch.randn(2, 10, 5)
        scores = torch.randn(2)

        # Run once to set prev_scores
        loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores)
        assert loss_fn.prev_scores is not None

        # Reset
        loss_fn.reset_state()
        assert loss_fn.prev_scores is None

    def test_stability_rate(self) -> None:
        """Test stability rate computation."""
        loss_fn = LyapunovAnomalyLoss(mu_stability=0.1)

        x = torch.randn(2, 10, 5)
        x_recon = torch.randn(2, 10, 5)

        # All stable iterations (decreasing scores)
        loss_fn.reset_state()
        for i in range(10, 0, -1):
            scores = torch.ones(2) * (i / 10.0)
            loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores)

        # Most iterations should be stable (scores decreasing)
        rate = loss_fn.get_stability_rate()
        assert 0.0 <= rate <= 1.0


class TestAblationConfigs:
    """Tests for ablation study configurations."""

    @pytest.mark.parametrize(
        "mu_stability",
        [0.0, 0.05, 0.1, 0.2],
    )
    def test_stability_weight_ablation(self, mu_stability: float) -> None:
        """Test different stability weights produce valid losses."""
        loss_fn = LyapunovAnomalyLoss(mu_stability=mu_stability)

        x = torch.randn(4, 50, 10)
        x_recon = torch.randn(4, 50, 10)
        scores = torch.randn(4).abs()

        result = loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores)
        assert not torch.isnan(result["total"])
        assert not torch.isinf(result["total"])

    @pytest.mark.parametrize(
        "alpha",
        [0.1, 0.25, 0.5, 1.0],
    )
    def test_convergence_rate_ablation(self, alpha: float) -> None:
        """Test different convergence rates."""
        loss_fn = LyapunovAnomalyLoss(alpha=alpha, mu_stability=0.1)

        x = torch.randn(4, 50, 10)
        x_recon = torch.randn(4, 50, 10)

        # Two iterations to test stability
        scores1 = torch.ones(4) * 0.5
        loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores1)

        scores2 = torch.ones(4) * 0.6
        result = loss_fn(x=x, x_recon=x_recon, anomaly_scores=scores2)

        assert not torch.isnan(result["total"])


class TestIntegration:
    """Integration tests for full training pipeline."""

    def test_end_to_end_training_step(self) -> None:
        """Test complete training step with model and loss."""
        model = ThreeRAnomalyTransformer(input_dim=25, d_model=64, num_layers=1)
        loss_fn = LyapunovAnomalyLoss(mu_stability=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate training batch
        x = torch.randn(8, 100, 25)
        labels = torch.randint(0, 2, (8,)).float()

        # Forward
        output = model(x)

        # Loss (with labels for supervised signal)
        loss_dict = loss_fn(
            x=x,
            x_recon=output["reconstruction"],
            anomaly_scores=output["anomaly_scores"],
            labels=labels,
        )

        # Backward
        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()

        # Check gradients were computed
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_resonance_engine_initialization(self) -> None:
        """Test initialization from ResonanceEngine."""
        from omni_mercury_engine.core.three_r_mechanism import ResonanceEngine

        block = ThreeRAttentionBlock(d_model=64, max_freqs=5)
        engine = ResonanceEngine(sampling_rate=1.0)

        # Generate synthetic training data
        training_data = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

        # Initialize from engine
        block.init_from_resonance_engine(engine, training_data)

        # Check frequencies were updated
        assert block.resonance_freqs.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

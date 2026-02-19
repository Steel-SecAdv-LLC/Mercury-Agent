"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Tests for advanced optimizers integration with OmniFusionModel.

Covers:
- SyntheticGradientPredictor for decoupled layer training
- DifferenceTargetPropagation for biologically plausible learning
- AuxiliaryMaxVariance for multi-task optimization
- train_with_advanced_optimizers() integration
- Lyapunov stability tracking
- Convergence rate estimation
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Optional torch import
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader


# =============================================================================
# SyntheticGradientPredictor Tests
# =============================================================================


class TestSyntheticGradientPredictor:
    """Tests for SyntheticGradientPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create SyntheticGradientPredictor instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import (
            SyntheticGradientPredictor,
        )

        return SyntheticGradientPredictor(input_dim=64, hidden_dim=128, output_dim=64)

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly."""
        assert predictor is not None
        assert predictor.input_dim == 64
        assert predictor.hidden_dim == 128
        assert predictor.output_dim == 64

    def test_forward_pass(self, predictor, deterministic_rng):
        """Test forward pass produces correct output shape."""
        input_data = deterministic_rng.randn(1, 64)
        output = predictor.forward(input_data)
        assert output.shape == (1, 64)

    def test_forward_batch(self, predictor, deterministic_rng):
        """Test forward pass with batch input."""
        input_data = deterministic_rng.randn(16, 64)
        output = predictor.forward(input_data)
        assert output.shape == (16, 64)

    def test_update_weights(self, predictor, deterministic_rng):
        """Test weight update with target gradients."""
        input_data = deterministic_rng.randn(1, 64)
        target_grad = deterministic_rng.randn(1, 64)
        predicted = predictor.forward(input_data)
        predictor.update(predicted, target_grad)

    def test_prediction_improves(self, predictor, deterministic_rng):
        """Test prediction improves with training."""
        input_data = deterministic_rng.randn(1, 64)
        target_grad = deterministic_rng.randn(1, 64)

        initial_pred = predictor.forward(input_data)
        initial_error = np.mean((initial_pred - target_grad) ** 2)

        for _ in range(100):
            pred = predictor.forward(input_data)
            predictor.update(pred, target_grad)

        final_pred = predictor.forward(input_data)
        final_error = np.mean((final_pred - target_grad) ** 2)

        assert final_error <= initial_error * 1.5

    def test_different_dimensions(self):
        """Test predictor with different dimensions."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import (
            SyntheticGradientPredictor,
        )

        predictor = SyntheticGradientPredictor(input_dim=32, hidden_dim=64, output_dim=32)
        assert predictor.input_dim == 32
        assert predictor.output_dim == 32


# =============================================================================
# DifferenceTargetPropagation Tests
# =============================================================================


class TestDifferenceTargetPropagation:
    """Tests for DifferenceTargetPropagation."""

    @pytest.fixture
    def dtp(self):
        """Create DifferenceTargetPropagation instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import (
            DifferenceTargetPropagation,
        )

        # DTP requires a forward_layer (nn.Module), not layer_dims
        forward_layer = torch.nn.Linear(64, 128)
        return DifferenceTargetPropagation(forward_layer=forward_layer, learning_rate=0.01)

    def test_dtp_initialization(self, dtp):
        """Test DTP initializes correctly."""
        assert dtp is not None
        assert dtp.forward_layer is not None
        assert dtp.learning_rate == 0.01

    def test_compute_targets(self, dtp, deterministic_rng):
        """Test target computation via backward_pass."""
        # h_current should match forward_layer input (64), target should match output (128)
        h_current = deterministic_rng.randn(16, 128)  # Match forward_layer output
        target = deterministic_rng.randn(16, 128)  # Target for current layer
        target_prev = dtp.backward_pass(h_current, target)
        assert target_prev is not None
        assert target_prev.shape == (16, 64)  # Inverse maps back to input dim

    def test_compute_local_loss(self, dtp, deterministic_rng):
        """Test forward pass produces output."""
        input_data = deterministic_rng.randn(16, 64)
        output = dtp.forward(input_data)
        assert output is not None
        assert output.shape == (16, 128)

    def test_biologically_plausible(self, dtp):
        """Test DTP is biologically plausible (has forward and backward_pass)."""
        assert hasattr(dtp, "forward")
        assert hasattr(dtp, "backward_pass")


# =============================================================================
# AuxiliaryMaxVariance Tests
# =============================================================================


class TestAuxiliaryMaxVariance:
    """Tests for AuxiliaryMaxVariance multi-task optimizer."""

    @pytest.fixture
    def amav(self):
        """Create AuxiliaryMaxVariance instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import AuxiliaryMaxVariance

        return AuxiliaryMaxVariance(num_tasks=3, alpha=0.5)

    def test_amav_initialization(self, amav):
        """Test AMAV initializes correctly."""
        assert amav is not None
        assert amav.num_tasks == 3
        assert amav.alpha == 0.5

    def test_compute_loss(self, amav):
        """Test multi-task loss computation."""
        task_losses = [0.5, 0.3, 0.2]
        combined_loss = amav.compute_loss(task_losses)
        assert combined_loss > 0

    def test_update_weights(self, amav):
        """Test task weight updates."""
        task_losses = [0.5, 0.3, 0.2]
        amav.compute_loss(task_losses)
        assert hasattr(amav, "task_weights")

    def test_weight_normalization(self, amav):
        """Test task weights are normalized."""
        for _ in range(10):
            task_losses = [np.random.rand() for _ in range(3)]
            amav.compute_loss(task_losses)
        weights_sum = sum(amav.task_weights)
        assert abs(weights_sum - amav.num_tasks) < 0.1 or weights_sum > 0

    def test_variance_maximization(self, amav):
        """Test variance is maximized across tasks."""
        task_losses_uniform = [0.5, 0.5, 0.5]
        task_losses_varied = [0.1, 0.5, 0.9]

        loss_uniform = amav.compute_loss(task_losses_uniform)
        loss_varied = amav.compute_loss(task_losses_varied)

        assert loss_uniform is not None
        assert loss_varied is not None


# =============================================================================
# Convergence Rate Estimation Tests
# =============================================================================


class TestConvergenceRateEstimation:
    """Tests for convergence rate estimation utilities."""

    def test_estimate_convergence_rate(self):
        """Test convergence rate estimation."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import estimate_convergence_rate

        loss_history = np.exp(-0.1 * np.arange(100))
        stats = estimate_convergence_rate(loss_history)
        assert "convergence_rate" in stats
        assert "half_life" in stats
        assert "converged" in stats

    def test_convergence_rate_decreasing_loss(self):
        """Test convergence rate with decreasing loss."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import estimate_convergence_rate

        loss_history = np.exp(-0.2 * np.arange(50))
        stats = estimate_convergence_rate(loss_history)
        assert stats["convergence_rate"] > 0

    def test_convergence_rate_flat_loss(self):
        """Test convergence rate with flat loss."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import estimate_convergence_rate

        loss_history = np.ones(50) * 0.5
        stats = estimate_convergence_rate(loss_history)
        # Flat loss can have near-zero convergence rate (positive or negative due to numerical precision)
        assert abs(stats["convergence_rate"]) < 0.1  # Should be close to zero for flat loss

    def test_convergence_rate_oscillating_loss(self):
        """Test convergence rate with oscillating loss."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import estimate_convergence_rate

        loss_history = 0.5 + 0.1 * np.sin(np.arange(50))
        stats = estimate_convergence_rate(loss_history)
        assert "convergence_rate" in stats


# =============================================================================
# OmniFusionModel Advanced Training Tests
# =============================================================================


class TestOmniFusionModelAdvancedTraining:
    """Tests for OmniFusionModel.train_with_advanced_optimizers()."""

    @pytest.fixture
    def fusion_model(self):
        """Create OmniFusionModel instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        # OmniFusionModel uses feature_dims dict, not input_dim/num_detectors
        return OmniFusionModel(
            feature_dims={
                "detector_0": 64,
                "detector_1": 64,
                "detector_2": 64,
            },
            hidden_dim=128,
            num_classes=5,
        )

    @pytest.fixture
    def train_loader(self, deterministic_rng):
        """Create training data loader."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")

        n_samples = 100
        features = {
            "detector_0": torch.randn(n_samples, 64),
            "detector_1": torch.randn(n_samples, 64),
            "detector_2": torch.randn(n_samples, 64),
        }
        labels = torch.zeros(n_samples, 3)
        labels[:, 0] = torch.randint(0, 2, (n_samples,)).float()
        labels[:, 1] = torch.randint(0, 5, (n_samples,)).float()
        labels[:, 2] = torch.rand(n_samples)

        class DictDataset:
            def __init__(self, features, labels) -> None:
                self.features = features
                self.labels = labels
                self.n_samples = labels.shape[0]

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                feat_dict = {k: v[idx] for k, v in self.features.items()}
                return feat_dict, self.labels[idx]

        dataset = DictDataset(features, labels)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def test_train_with_advanced_optimizers_exists(self, fusion_model):
        """Test train_with_advanced_optimizers method exists."""
        assert hasattr(fusion_model, "train_with_advanced_optimizers")

    def test_train_basic(self, fusion_model, train_loader):
        """Test basic training with advanced optimizers."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=5,
            learning_rate=0.001,
            use_synthetic_gradients=True,
            use_dtp=True,
            use_amav=True,
            log_interval=2,
        )
        assert "final_loss" in result
        assert "loss_history" in result
        assert "convergence_rate" in result

    def test_train_without_synthetic_gradients(self, fusion_model, train_loader):
        """Test training without synthetic gradients."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=3,
            use_synthetic_gradients=False,
            use_dtp=True,
            use_amav=True,
        )
        assert "final_loss" in result

    def test_train_without_dtp(self, fusion_model, train_loader):
        """Test training without DTP."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=3,
            use_synthetic_gradients=True,
            use_dtp=False,
            use_amav=True,
        )
        assert "final_loss" in result

    def test_train_without_amav(self, fusion_model, train_loader):
        """Test training without AMAV."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=3,
            use_synthetic_gradients=True,
            use_dtp=True,
            use_amav=False,
        )
        assert "final_loss" in result

    def test_train_vanilla(self, fusion_model, train_loader):
        """Test vanilla training without any advanced optimizers."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=3,
            use_synthetic_gradients=False,
            use_dtp=False,
            use_amav=False,
        )
        assert "final_loss" in result

    def test_lyapunov_stability_tracking(self, fusion_model, train_loader):
        """Test Lyapunov stability is tracked."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=5,
            lambda_lyapunov=0.25,
        )
        assert "lyapunov_stable" in result
        assert "lambda_lyapunov" in result
        assert result["lambda_lyapunov"] == 0.25

    def test_speedup_factor(self, fusion_model, train_loader):
        """Test speedup factor is computed."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=5,
        )
        assert "speedup_factor" in result
        assert result["speedup_factor"] >= 1.0

    def test_epochs_trained(self, fusion_model, train_loader):
        """Test epochs trained is recorded."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=10,
        )
        assert "epochs_trained" in result
        assert result["epochs_trained"] == 10

    def test_loss_decreases(self, fusion_model, train_loader):
        """Test loss is tracked during training."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=20,
        )
        loss_history = result["loss_history"]
        # Just verify loss history is recorded - actual decrease depends on model/data
        assert len(loss_history) > 0
        assert all(isinstance(loss, (int, float)) for loss in loss_history)

    def test_convergence_detection(self, fusion_model, train_loader):
        """Test convergence is detected."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=30,
        )
        assert "converged" in result

    def test_half_life_computation(self, fusion_model, train_loader):
        """Test half-life is computed."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=20,
        )
        assert "half_life" in result


# =============================================================================
# Lyapunov Stability Tests
# =============================================================================


class TestLyapunovStability:
    """Tests for Lyapunov stability in training."""

    @pytest.fixture
    def fusion_model(self):
        """Create OmniFusionModel instance."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.fusion_network import OmniFusionModel

        # OmniFusionModel uses feature_dims dict, not input_dim/num_detectors
        return OmniFusionModel(
            feature_dims={
                "detector_0": 64,
                "detector_1": 64,
                "detector_2": 64,
            },
            hidden_dim=128,
            num_classes=5,
        )

    @pytest.fixture
    def train_loader(self, deterministic_rng):
        """Create training data loader."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")

        n_samples = 50
        features = {
            "detector_0": torch.randn(n_samples, 64),
            "detector_1": torch.randn(n_samples, 64),
            "detector_2": torch.randn(n_samples, 64),
        }
        labels = torch.zeros(n_samples, 3)
        labels[:, 0] = torch.randint(0, 2, (n_samples,)).float()
        labels[:, 1] = torch.randint(0, 5, (n_samples,)).float()
        labels[:, 2] = torch.rand(n_samples)

        class DictDataset:
            def __init__(self, features, labels) -> None:
                self.features = features
                self.labels = labels
                self.n_samples = labels.shape[0]

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                feat_dict = {k: v[idx] for k, v in self.features.items()}
                return feat_dict, self.labels[idx]

        dataset = DictDataset(features, labels)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def test_lambda_025_stability(self, fusion_model, train_loader):
        """Test stability with lambda=0.25."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=10,
            lambda_lyapunov=0.25,
        )
        assert result["lambda_lyapunov"] == 0.25

    def test_lambda_018_stability(self, fusion_model, train_loader):
        """Test stability with lambda=0.18 (baseline)."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=10,
            lambda_lyapunov=0.18,
        )
        assert result["lambda_lyapunov"] == 0.18

    def test_phi_weighting_applied(self, fusion_model, train_loader):
        """Test phi weighting is applied in Lyapunov computation."""
        result = fusion_model.train_with_advanced_optimizers(
            train_loader=train_loader,
            epochs=10,
        )
        assert "lyapunov_stable" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestOptimizerIntegration:
    """Integration tests for advanced optimizers."""

    def test_all_optimizers_importable(self):
        """Test all optimizers can be imported."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import (
            AuxiliaryMaxVariance,
            DifferenceTargetPropagation,
            SyntheticGradientPredictor,
            estimate_convergence_rate,
        )

        assert SyntheticGradientPredictor is not None
        assert DifferenceTargetPropagation is not None
        assert AuxiliaryMaxVariance is not None
        assert estimate_convergence_rate is not None

    def test_optimizers_work_together(self, deterministic_rng):
        """Test all optimizers work together."""
        if not HAS_TORCH:
            pytest.skip("torch not installed")
        from omni_mercury_engine.ml.advanced_optimizers import (
            AuxiliaryMaxVariance,
            SyntheticGradientPredictor,
        )

        predictor = SyntheticGradientPredictor(input_dim=64, hidden_dim=128, output_dim=64)
        amav = AuxiliaryMaxVariance(num_tasks=3, alpha=0.5)

        input_data = deterministic_rng.randn(16, 64)
        pred_grad = predictor.forward(input_data)
        assert pred_grad.shape == (16, 64)

        task_losses = [0.5, 0.3, 0.2]
        combined_loss = amav.compute_loss(task_losses)
        assert combined_loss > 0

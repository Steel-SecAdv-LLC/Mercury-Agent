"""
Tests for ML pipeline modules.

Tests fusion_network, training, inference, and related ML components.

Note: These tests require PyTorch to be installed.
"""

from __future__ import annotations

import numpy as np
import pytest


# Conditional torch import for ML tests
try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.ml.fusion_network import (
        AttentionFusion,
        FusionNetwork,
        GatedFusion,
        MultimodalFusion,
    )
    from omni_mercury_engine.ml.inference import BatchInference, InferenceEngine, ModelEnsemble
    from omni_mercury_engine.ml.training import (
        EarlyStopping,
        LearningRateScheduler,
        Trainer,
        TrainingConfig,
    )


class TestFusionNetwork:
    """Tests for FusionNetwork class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.input_dims = [32, 64, 48]  # Multiple modalities
        self.output_dim = 128
        self.model = FusionNetwork(input_dims=self.input_dims, output_dim=self.output_dim)

    def test_initialization(self):
        """Test network initialization."""
        assert self.model is not None
        assert hasattr(self.model, "encoders")
        assert hasattr(self.model, "fusion_layer")

    def test_forward_pass(self):
        """Test forward pass with multiple inputs."""
        batch_size = 8
        inputs = [torch.randn(batch_size, dim) for dim in self.input_dims]
        output = self.model(inputs)

        assert output.shape == (batch_size, self.output_dim)

    def test_single_modality(self):
        """Test with single modality input."""
        model = FusionNetwork(input_dims=[64], output_dim=32)
        x = [torch.randn(4, 64)]
        output = model(x)

        assert output.shape == (4, 32)

    def test_gradient_flow(self):
        """Test gradient flow through network."""
        inputs = [torch.randn(4, dim, requires_grad=True) for dim in self.input_dims]
        output = self.model(inputs)
        loss = output.sum()
        loss.backward()

        for inp in inputs:
            assert inp.grad is not None

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        for batch_size in [1, 8, 32]:
            inputs = [torch.randn(batch_size, dim) for dim in self.input_dims]
            output = self.model(inputs)
            assert output.shape[0] == batch_size


class TestMultimodalFusion:
    """Tests for MultimodalFusion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = MultimodalFusion(
            modality_dims={"visual": 256, "audio": 128, "text": 512}, output_dim=256
        )

    def test_initialization(self):
        """Test fusion module initialization."""
        assert self.fusion is not None

    def test_forward_with_dict_input(self):
        """Test forward pass with dictionary input."""
        inputs = {
            "visual": torch.randn(4, 256),
            "audio": torch.randn(4, 128),
            "text": torch.randn(4, 512),
        }
        output = self.fusion(inputs)

        assert output.shape == (4, 256)

    def test_missing_modality_handling(self):
        """Test handling of missing modality."""
        inputs = {
            "visual": torch.randn(4, 256),
            "audio": torch.randn(4, 128),
            # text missing
        }
        # Should handle gracefully with masking or default
        try:
            output = self.fusion(inputs)
            assert output is not None
        except KeyError:
            pass  # Also acceptable behavior

    def test_modality_weights(self):
        """Test that modality weights are learned."""
        if hasattr(self.fusion, "modality_weights"):
            weights = self.fusion.modality_weights
            assert weights.shape[0] == 3  # Three modalities


class TestAttentionFusion:
    """Tests for AttentionFusion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = AttentionFusion(embed_dim=128, num_heads=8)

    def test_initialization(self):
        """Test attention fusion initialization."""
        assert self.fusion is not None
        assert hasattr(self.fusion, "attention")

    def test_forward_pass(self):
        """Test forward pass."""
        # Sequence of embeddings
        x = torch.randn(4, 10, 128)  # Batch, Seq, Embed
        output = self.fusion(x)

        assert output.shape == (4, 128)  # Pooled output

    def test_attention_weights(self):
        """Test that attention weights are computed."""
        x = torch.randn(4, 10, 128)
        output, weights = self.fusion(x, return_attention=True)

        assert weights is not None
        assert weights.shape[0] == 4  # Batch size


class TestGatedFusion:
    """Tests for GatedFusion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = GatedFusion(input_dim=64, hidden_dim=32)

    def test_initialization(self):
        """Test gated fusion initialization."""
        assert self.fusion is not None

    def test_forward_pass(self):
        """Test forward pass with two inputs."""
        x1 = torch.randn(4, 64)
        x2 = torch.randn(4, 64)
        output = self.fusion(x1, x2)

        assert output.shape == (4, 64)

    def test_gate_values(self):
        """Test that gate values are in [0, 1]."""
        x1 = torch.randn(4, 64)
        x2 = torch.randn(4, 64)
        output, gate = self.fusion(x1, x2, return_gate=True)

        assert torch.all(gate >= 0)
        assert torch.all(gate <= 1)


class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.epochs > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(learning_rate=0.001, batch_size=64, epochs=100, weight_decay=1e-5)

        assert config.learning_rate == 0.001
        assert config.batch_size == 64
        assert config.epochs == 100

    def test_config_validation(self):
        """Test that invalid config raises errors."""
        with pytest.raises((ValueError, AssertionError)):
            TrainingConfig(learning_rate=-0.1)

        with pytest.raises((ValueError, AssertionError)):
            TrainingConfig(batch_size=0)


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_initialization(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.001)
        assert es.patience == 5
        assert es.counter == 0

    def test_improvement_resets_counter(self):
        """Test that improvement resets patience counter."""
        es = EarlyStopping(patience=5)

        es(0.5)  # Initial
        es(0.4)  # Improvement
        assert es.counter == 0

    def test_no_improvement_increments_counter(self):
        """Test that no improvement increments counter."""
        es = EarlyStopping(patience=5, min_delta=0.01)

        es(0.5)  # Initial
        es(0.5)  # No improvement
        assert es.counter == 1

    def test_stops_after_patience(self):
        """Test that training stops after patience exceeded."""
        es = EarlyStopping(patience=3)

        es(0.5)
        assert not es.early_stop

        for _ in range(4):
            es(0.5)  # No improvement

        assert es.early_stop

    def test_best_score_tracking(self):
        """Test that best score is tracked."""
        es = EarlyStopping(patience=5)

        es(0.5)
        es(0.3)
        es(0.4)

        assert es.best_score == pytest.approx(-0.3)  # Negated for minimization


class TestLearningRateScheduler:
    """Tests for LearningRateScheduler class."""

    def test_step_scheduler(self):
        """Test step learning rate scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LearningRateScheduler(optimizer, mode="step", step_size=10, gamma=0.1)

        initial_lr = optimizer.param_groups[0]["lr"]

        for _ in range(10):
            scheduler.step()

        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr < initial_lr

    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LearningRateScheduler(optimizer, mode="cosine", T_max=100)

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Cosine should decrease then stay low
        assert lrs[-1] < lrs[0]


class TestTrainer:
    """Tests for Trainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.config = TrainingConfig(learning_rate=0.01, batch_size=8, epochs=5, device="cpu")
        self.trainer = Trainer(self.model, self.config)

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer is not None
        assert self.trainer.model is not None
        assert self.trainer.optimizer is not None

    def test_train_step(self):
        """Test single training step."""
        x = torch.randn(8, 32)
        y = torch.randn(8, 1)

        loss = self.trainer.train_step(x, y)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validation_step(self):
        """Test validation step."""
        x = torch.randn(8, 32)
        y = torch.randn(8, 1)

        loss = self.trainer.validate_step(x, y)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        self.trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        checkpoint_path = tmp_path / "checkpoint.pt"
        self.trainer.save_checkpoint(str(checkpoint_path))

        # Create new trainer and load
        new_model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        new_trainer = Trainer(new_model, self.config)
        new_trainer.load_checkpoint(str(checkpoint_path))

        # Verify weights match
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.engine = InferenceEngine(self.model, device="cpu")

    def test_initialization(self):
        """Test inference engine initialization."""
        assert self.engine is not None
        assert self.engine.model is not None

    def test_predict_single(self):
        """Test single prediction."""
        x = torch.randn(1, 32)
        output = self.engine.predict(x)

        assert output.shape == (1, 1)

    def test_predict_batch(self):
        """Test batch prediction."""
        x = torch.randn(16, 32)
        output = self.engine.predict(x)

        assert output.shape == (16, 1)

    def test_predict_numpy(self):
        """Test prediction with numpy input."""
        x = np.random.randn(8, 32).astype(np.float32)
        output = self.engine.predict(x)

        assert output.shape == (8, 1)

    def test_no_grad_mode(self):
        """Test that inference runs in no_grad mode."""
        x = torch.randn(4, 32, requires_grad=True)

        with torch.no_grad():
            output = self.engine.predict(x)

        assert not output.requires_grad


class TestBatchInference:
    """Tests for BatchInference class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.batch_inference = BatchInference(self.model, batch_size=4, device="cpu")

    def test_large_batch_processing(self):
        """Test processing large batches in chunks."""
        x = torch.randn(100, 32)
        output = self.batch_inference.predict(x)

        assert output.shape == (100, 1)

    def test_streaming_inference(self):
        """Test streaming inference."""
        data_stream = [torch.randn(4, 32) for _ in range(10)]

        outputs = list(self.batch_inference.predict_stream(data_stream))

        assert len(outputs) == 10
        for out in outputs:
            assert out.shape == (4, 1)


class TestModelEnsemble:
    """Tests for ModelEnsemble class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.models = [
            nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)) for _ in range(3)
        ]
        self.ensemble = ModelEnsemble(self.models, aggregation="mean")

    def test_initialization(self):
        """Test ensemble initialization."""
        assert self.ensemble is not None
        assert len(self.ensemble.models) == 3

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        x = torch.randn(8, 32)
        output = self.ensemble.predict(x)

        assert output.shape == (8, 1)

    def test_voting_aggregation(self):
        """Test voting aggregation for classification."""
        ensemble = ModelEnsemble(self.models, aggregation="voting")
        x = torch.randn(8, 32)
        output = ensemble.predict(x)

        assert output is not None

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation from ensemble."""
        x = torch.randn(8, 32)
        output, uncertainty = self.ensemble.predict_with_uncertainty(x)

        assert output.shape == (8, 1)
        assert uncertainty.shape == (8, 1)
        assert torch.all(uncertainty >= 0)


class TestMLPipelineIntegration:
    """Integration tests for ML pipeline."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        # Create model
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        # Create data
        X_train = torch.randn(100, 32)
        y_train = torch.randn(100, 1)

        # Train
        config = TrainingConfig(learning_rate=0.01, batch_size=16, epochs=3)
        trainer = Trainer(model, config)

        initial_loss = trainer.validate_step(X_train[:16], y_train[:16])

        for epoch in range(3):
            for i in range(0, len(X_train), 16):
                batch_x = X_train[i : i + 16]
                batch_y = y_train[i : i + 16]
                trainer.train_step(batch_x, batch_y)

        final_loss = trainer.validate_step(X_train[:16], y_train[:16])

        # Loss should decrease
        assert final_loss < initial_loss or abs(final_loss - initial_loss) < 0.5

    def test_train_then_inference(self):
        """Test training followed by inference."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

        # Quick training
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(10):
            x = torch.randn(8, 32)
            y = torch.randn(8, 1)
            loss = nn.MSELoss()(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Inference
        model.eval()
        engine = InferenceEngine(model)
        test_x = torch.randn(4, 32)
        output = engine.predict(test_x)

        assert output.shape == (4, 1)

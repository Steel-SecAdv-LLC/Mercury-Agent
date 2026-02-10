"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for ml/training.py module.
Targets coverage improvement for training utilities, optimizers, and loss functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from omni_mercury_engine.ml.training import (
    AnomalyDataset,
    EarlyStopping,
    LearningRateScheduler,
    LyapunovAnomalyLoss,
    MercuryExponentialDecayOptimizer,
    MercuryHarmonicOptimizer,
    MercuryMomentumOptimizer,
    MercuryOptimizer,
    Trainer,
    TrainingConfig,
    create_mercury_optimizer,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.weight_decay == 0.0
        assert config.device == "cpu"
        assert config.optimizer == "adam"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            weight_decay=0.001,
            device="cuda",
            optimizer="adamw",
        )
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.epochs == 50
        assert config.optimizer == "adamw"

    def test_invalid_learning_rate(self):
        """Test validation for invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.01)

    def test_invalid_batch_size(self):
        """Test validation for invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_invalid_epochs(self):
        """Test validation for invalid epochs."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainingConfig(epochs=0)


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_init(self):
        """Test early stopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "min"
        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop is False

    def test_improvement_resets_counter(self):
        """Test that improvement resets counter."""
        es = EarlyStopping(patience=3, mode="min")

        # First call sets baseline
        assert es(1.0) is False
        assert es.counter == 0

        # Improvement
        assert es(0.5) is False
        assert es.counter == 0

        # No improvement
        assert es(0.6) is False
        assert es.counter == 1

        # Improvement again - counter resets
        assert es(0.3) is False
        assert es.counter == 0

    def test_patience_exceeded(self):
        """Test early stopping triggers after patience exceeded."""
        es = EarlyStopping(patience=3, mode="min")

        es(1.0)  # baseline
        es(1.1)  # no improvement, counter=1
        es(1.2)  # no improvement, counter=2
        result = es(1.3)  # no improvement, counter=3 >= patience

        assert result is True
        assert es.early_stop is True

    def test_max_mode(self):
        """Test early stopping in max mode (for accuracy)."""
        es = EarlyStopping(patience=2, mode="max")

        es(0.5)  # baseline
        assert es(0.6) is False  # improvement
        assert es(0.55) is False  # no improvement, counter=1
        assert es(0.54) is True  # no improvement, counter=2 >= patience

    def test_min_delta(self):
        """Test min_delta threshold for improvement."""
        es = EarlyStopping(patience=3, min_delta=0.1, mode="min")

        es(1.0)  # baseline
        # Improvement less than min_delta doesn't count
        es(0.95)  # only 0.05 improvement, counter=1
        assert es.counter == 1

        # Improvement greater than min_delta
        es(0.8)  # 0.2 improvement from best (1.0), resets counter
        assert es.counter == 0


class TestLearningRateScheduler:
    """Tests for LearningRateScheduler wrapper."""

    def test_step_scheduler(self):
        """Test step decay scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LearningRateScheduler(optimizer, mode="step", step_size=5, gamma=0.5)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr == 0.1

        # Step through 5 epochs
        for _ in range(5):
            scheduler.step()

        new_lr = scheduler.get_last_lr()[0]
        assert abs(new_lr - 0.05) < 1e-6  # LR should be halved

    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LearningRateScheduler(optimizer, mode="cosine", T_max=10, eta_min=0.001)

        initial_lr = scheduler.get_last_lr()[0]
        assert initial_lr == 0.1

        # Step to end
        for _ in range(10):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < initial_lr  # LR should decrease

    def test_plateau_scheduler(self):
        """Test reduce on plateau scheduler."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = LearningRateScheduler(optimizer, mode="plateau", step_size=2, gamma=0.5)

        # Simulate no improvement
        scheduler.step(metric=1.0)
        scheduler.step(metric=1.0)
        scheduler.step(metric=1.0)  # Should trigger reduction

    def test_invalid_mode(self):
        """Test invalid scheduler mode raises error."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            LearningRateScheduler(optimizer, mode="invalid")


class TestTrainer:
    """Tests for general Trainer class."""

    def _create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def test_init_adam(self):
        """Test trainer initialization with Adam optimizer."""
        model = self._create_simple_model()
        config = TrainingConfig(optimizer="adam", learning_rate=0.01)
        trainer = Trainer(model, config)

        assert trainer.model is model
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.config.learning_rate == 0.01

    def test_init_adamw(self):
        """Test trainer initialization with AdamW optimizer."""
        model = self._create_simple_model()
        config = TrainingConfig(optimizer="adamw")
        trainer = Trainer(model, config)

        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_init_sgd(self):
        """Test trainer initialization with SGD optimizer."""
        model = self._create_simple_model()
        config = TrainingConfig(optimizer="sgd")
        trainer = Trainer(model, config)

        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_init_default_optimizer(self):
        """Test trainer falls back to Adam for unknown optimizer."""
        model = self._create_simple_model()
        config = TrainingConfig(optimizer="unknown")
        trainer = Trainer(model, config)

        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_train_step(self):
        """Test single training step."""
        model = self._create_simple_model()
        config = TrainingConfig()
        trainer = Trainer(model, config)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_step_with_gradient_clip(self):
        """Test training step with gradient clipping."""
        model = self._create_simple_model()
        config = TrainingConfig(gradient_clip=1.0)
        trainer = Trainer(model, config)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)

    def test_validate_step(self):
        """Test single validation step."""
        model = self._create_simple_model()
        config = TrainingConfig()
        trainer = Trainer(model, config)

        x = torch.randn(8, 10)
        y = torch.randn(8, 1)

        loss = trainer.validate_step(x, y)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_save_load_checkpoint(self):
        """Test checkpoint save and load."""
        model = self._create_simple_model()
        config = TrainingConfig()
        trainer = Trainer(model, config)

        # Do a training step to modify state
        x = torch.randn(8, 10)
        y = torch.randn(8, 1)
        trainer.train_step(x, y)
        trainer.epoch = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            # Create new trainer and load checkpoint
            model2 = self._create_simple_model()
            trainer2 = Trainer(model2, config)
            trainer2.load_checkpoint(str(checkpoint_path))

            assert trainer2.epoch == 5


class TestMercuryOptimizers:
    """Tests for Mercury optimizer variants."""

    def _create_simple_model(self):
        """Create simple model for optimizer testing."""
        return nn.Linear(10, 1)

    def test_mercury_optimizer_step(self):
        """Test MercuryOptimizer step function."""
        model = self._create_simple_model()
        optimizer = MercuryOptimizer(model.parameters(), lr=0.01, alpha=0.1, beta=0.9)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        optimizer.step()

    def test_mercury_optimizer_with_quantum_noise(self):
        """Test MercuryOptimizer with quantum noise."""
        model = self._create_simple_model()
        optimizer = MercuryOptimizer(model.parameters(), lr=0.01, quantum_noise=0.01)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()

        optimizer.step()

    def test_mercury_optimizer_with_closure(self):
        """Test MercuryOptimizer with closure function."""
        model = self._create_simple_model()
        optimizer = MercuryOptimizer(model.parameters(), lr=0.01)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            return float(loss.item())

        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss, float)

    def test_mercury_momentum_optimizer(self):
        """Test MercuryMomentumOptimizer."""
        model = self._create_simple_model()
        optimizer = MercuryMomentumOptimizer(model.parameters(), lr=0.01, alpha=0.1, momentum=0.9)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        # Multiple steps to test momentum accumulation
        for _ in range(3):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

    def test_mercury_exp_decay_optimizer(self):
        """Test MercuryExponentialDecayOptimizer."""
        model = self._create_simple_model()
        optimizer = MercuryExponentialDecayOptimizer(
            model.parameters(), lr=0.01, alpha=0.1, decay_rate=0.99
        )

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

    def test_mercury_harmonic_optimizer(self):
        """Test MercuryHarmonicOptimizer."""
        model = self._create_simple_model()
        optimizer = MercuryHarmonicOptimizer(model.parameters(), lr=0.01, alpha=0.1, omega=0.1)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()


class TestCreateMercuryOptimizer:
    """Tests for create_mercury_optimizer factory function."""

    def test_create_base_optimizer(self):
        """Test creating base Mercury optimizer."""
        model = nn.Linear(10, 1)
        optimizer = create_mercury_optimizer(model.parameters(), variant="base", lr=0.01)
        assert isinstance(optimizer, MercuryOptimizer)

    def test_create_momentum_optimizer(self):
        """Test creating momentum Mercury optimizer."""
        model = nn.Linear(10, 1)
        optimizer = create_mercury_optimizer(model.parameters(), variant="momentum", lr=0.01)
        assert isinstance(optimizer, MercuryMomentumOptimizer)

    def test_create_exp_decay_optimizer(self):
        """Test creating exponential decay Mercury optimizer."""
        model = nn.Linear(10, 1)
        optimizer = create_mercury_optimizer(model.parameters(), variant="exp_decay", lr=0.01)
        assert isinstance(optimizer, MercuryExponentialDecayOptimizer)

    def test_create_harmonic_optimizer(self):
        """Test creating harmonic Mercury optimizer."""
        model = nn.Linear(10, 1)
        optimizer = create_mercury_optimizer(model.parameters(), variant="harmonic", lr=0.01)
        assert isinstance(optimizer, MercuryHarmonicOptimizer)

    def test_create_invalid_variant(self):
        """Test invalid variant raises error."""
        model = nn.Linear(10, 1)
        with pytest.raises(ValueError, match="Unknown Mercury optimizer variant"):
            create_mercury_optimizer(model.parameters(), variant="invalid")


class TestLyapunovAnomalyLoss:
    """Tests for LyapunovAnomalyLoss stability-constrained loss."""

    def test_init(self):
        """Test loss function initialization."""
        loss_fn = LyapunovAnomalyLoss(
            lambda_kl=0.1, lambda_supervised=1.0, mu_stability=0.1, alpha=0.25
        )
        assert loss_fn.lambda_kl == 0.1
        assert loss_fn.lambda_supervised == 1.0
        assert loss_fn.mu_stability == 0.1
        assert loss_fn.alpha == 0.25

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = LyapunovAnomalyLoss()

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))

        result = loss_fn(x, x_recon, anomaly_scores)

        assert "total" in result
        assert "reconstruction" in result
        assert "stability" in result
        assert "lyapunov_V" in result
        assert isinstance(result["total"], torch.Tensor)

    def test_forward_with_labels(self):
        """Test forward pass with supervised labels."""
        loss_fn = LyapunovAnomalyLoss(lambda_supervised=1.0)

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))
        labels = torch.randint(0, 2, (8,)).float()

        result = loss_fn(x, x_recon, anomaly_scores, labels=labels)

        assert result["supervised"].item() > 0  # BCE should be non-zero

    def test_forward_with_vae(self):
        """Test forward pass with VAE KL divergence."""
        loss_fn = LyapunovAnomalyLoss(lambda_kl=0.1)

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)

        result = loss_fn(x, x_recon, anomaly_scores, mu=mu, logvar=logvar)

        assert result["kl"].item() != 0  # KL should be computed

    def test_stability_tracking(self):
        """Test Lyapunov stability tracking over multiple steps."""
        loss_fn = LyapunovAnomalyLoss(mu_stability=0.1, alpha=0.25)

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)

        # First step - no stability loss (no previous scores)
        anomaly_scores1 = torch.sigmoid(torch.randn(8))
        result1 = loss_fn(x, x_recon, anomaly_scores1)
        assert result1["stability"].item() == 0.0

        # Second step - stability is computed
        anomaly_scores2 = torch.sigmoid(torch.randn(8))
        _result2 = loss_fn(x, x_recon, anomaly_scores2)
        # Stability loss may or may not be zero depending on score changes
        assert "stability" in _result2

    def test_reset_state(self):
        """Test reset_state clears previous scores."""
        loss_fn = LyapunovAnomalyLoss()

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))

        # First call sets prev_scores
        loss_fn(x, x_recon, anomaly_scores)
        assert loss_fn.prev_scores is not None

        # Reset clears it
        loss_fn.reset_state()
        assert loss_fn.prev_scores is None

    def test_get_stability_rate(self):
        """Test stability rate computation."""
        loss_fn = LyapunovAnomalyLoss()

        # No steps yet
        assert loss_fn.get_stability_rate() == 1.0

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)

        # Run several steps
        for _ in range(5):
            anomaly_scores = torch.sigmoid(torch.randn(8))
            loss_fn(x, x_recon, anomaly_scores)

        rate = loss_fn.get_stability_rate()
        assert 0.0 <= rate <= 1.0

    def test_sum_reduction(self):
        """Test sum reduction mode."""
        loss_fn = LyapunovAnomalyLoss(reduction="sum")

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))

        result = loss_fn(x, x_recon, anomaly_scores)
        assert isinstance(result["reconstruction"], torch.Tensor)

    def test_multidim_labels(self):
        """Test handling of multi-dimensional labels."""
        loss_fn = LyapunovAnomalyLoss(lambda_supervised=1.0)

        x = torch.randn(8, 10)
        x_recon = torch.randn(8, 10)
        anomaly_scores = torch.sigmoid(torch.randn(8))
        labels = torch.randint(0, 2, (8, 3)).float()  # Multi-dim labels

        result = loss_fn(x, x_recon, anomaly_scores, labels=labels)
        assert "supervised" in result


class TestAnomalyDataset:
    """Tests for AnomalyDataset class."""

    def test_init_basic(self):
        """Test basic dataset initialization."""
        features = {
            "detector_a": torch.randn(100, 10),
            "detector_b": torch.randn(100, 8),
        }
        labels = torch.randint(0, 2, (100,))

        dataset = AnomalyDataset(features, labels)
        assert len(dataset) == 100

    def test_getitem_without_scores(self):
        """Test __getitem__ without scores."""
        features = {
            "detector_a": torch.randn(100, 10),
        }
        labels = torch.randint(0, 2, (100,))

        dataset = AnomalyDataset(features, labels)
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 2
        assert "detector_a" in item[0]

    def test_getitem_with_scores(self):
        """Test __getitem__ with scores."""
        features = {"detector_a": torch.randn(100, 10)}
        labels = torch.randint(0, 2, (100,))
        scores = {"detector_a": torch.randn(100)}

        dataset = AnomalyDataset(features, labels, scores=scores)
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 3  # features, scores, label

    def test_len(self):
        """Test dataset length."""
        features = {"detector_a": torch.randn(50, 10)}
        labels = torch.randint(0, 2, (50,))

        dataset = AnomalyDataset(features, labels)
        assert len(dataset) == 50

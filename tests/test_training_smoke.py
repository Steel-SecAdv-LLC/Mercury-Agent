"""
Mercury Agent ♱
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

from __future__ import annotations

"""
Training module smoke tests to boost coverage
"""

import pytest

# Conditional torch import
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
    from omni_mercury_engine.ml.training import (
        AnomalyDataset,
        AvaExponentialDecayOptimizer,
        AvaHarmonicOptimizer,
        AvaMomentumOptimizer,
        AvaOptimizer,
        FusionTrainer,
        create_ava_optimizer,
    )


def test_anomaly_dataset_initialization():
    """Test AnomalyDataset initialization"""
    features = {
        "statistical": torch.randn(10, 5),
        "temporal": torch.randn(10, 32),
    }
    labels = torch.randint(0, 2, (10,))

    dataset = AnomalyDataset(features, labels)
    assert len(dataset) == 10


def test_anomaly_dataset_getitem():
    """Test AnomalyDataset __getitem__"""
    features = {
        "statistical": torch.randn(10, 5),
        "temporal": torch.randn(10, 32),
    }
    labels = torch.randint(0, 2, (10,))

    dataset = AnomalyDataset(features, labels)
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2


def test_fusion_trainer_initialization():
    """Test FusionTrainer initialization"""
    trainer = FusionTrainer(learning_rate=1e-3)
    assert trainer.learning_rate == 1e-3
    assert trainer.model is not None


def test_fusion_trainer_forward():
    """Test FusionTrainer forward pass"""
    trainer = FusionTrainer()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
    }

    outputs = trainer.forward(features)
    assert outputs is not None
    assert "anomaly_probs" in outputs


def test_fusion_trainer_training_step():
    """Test FusionTrainer training step"""
    trainer = FusionTrainer()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
    }
    labels = torch.randint(0, 2, (4,))
    batch = (features, labels)

    loss = trainer.training_step(batch, 0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_fusion_trainer_validation_step():
    """Test FusionTrainer validation step"""
    trainer = FusionTrainer()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
    }
    labels = torch.randint(0, 2, (4,))
    batch = (features, labels)

    trainer.validation_step(batch, 0)


def test_fusion_trainer_configure_optimizers():
    """Test optimizer configuration"""
    trainer = FusionTrainer()
    optimizers = trainer.configure_optimizers()
    assert optimizers is not None
    assert "optimizer" in optimizers


def test_ava_optimizer_base():
    """Test base Ava optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = AvaOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_optimizer_step():
    """Test Ava optimizer step"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaOptimizer([param], lr=0.001)

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()


def test_ava_momentum_optimizer():
    """Test Ava momentum optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = AvaMomentumOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_exp_decay_optimizer():
    """Test Ava exponential decay optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = AvaExponentialDecayOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_harmonic_optimizer():
    """Test Ava harmonic optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = AvaHarmonicOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_create_ava_optimizer_factory():
    """Test Ava optimizer factory function"""
    params = [torch.randn(10, 10, requires_grad=True)]

    variants = ["base", "momentum", "exp_decay", "harmonic"]
    for variant in variants:
        optimizer = create_ava_optimizer(params, variant=variant, lr=0.001)
        assert optimizer is not None

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training module smoke tests to boost coverage."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

# Probe for torch / lightning without binding them at module import.
# ``TYPE_CHECKING`` keeps mypy resolution stable while the pytestmark
# below skips the suite when either dependency is absent.
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTNING = importlib.util.find_spec("pytorch_lightning") is not None

if TYPE_CHECKING or HAS_TORCH:
    import torch

# Skip all tests in this module if torch or lightning is not available
pytestmark = pytest.mark.skipif(
    not HAS_TORCH or not HAS_LIGHTNING,
    reason="PyTorch or pytorch-lightning not installed",
)

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.ml.training import (
        AnomalyDataset,
        FusionTrainer,
        MercuryExponentialDecayOptimizer,
        MercuryHarmonicOptimizer,
        MercuryMomentumOptimizer,
        MercuryOptimizer,
        create_mercury_optimizer,
    )


def test_anomaly_dataset_initialization() -> None:
    """Test AnomalyDataset initialization"""
    features = {
        "statistical": torch.randn(10, 5),
        "temporal": torch.randn(10, 32),
    }
    labels = torch.randint(0, 2, (10,))

    dataset = AnomalyDataset(features, labels)
    assert len(dataset) == 10


def test_anomaly_dataset_getitem() -> None:
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


def test_fusion_trainer_initialization() -> None:
    """Test FusionTrainer initialization"""
    trainer = FusionTrainer(learning_rate=1e-3)
    assert trainer.learning_rate == 1e-3
    assert trainer.model is not None


def test_fusion_trainer_forward() -> None:
    """Test FusionTrainer forward pass"""
    trainer = FusionTrainer()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
    }

    outputs = trainer.forward(features)
    assert outputs is not None
    assert "anomaly_probs" in outputs


def test_fusion_trainer_training_step() -> None:
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


def test_fusion_trainer_validation_step() -> None:
    """Test FusionTrainer validation step"""
    trainer = FusionTrainer()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
    }
    labels = torch.randint(0, 2, (4,))
    batch = (features, labels)

    trainer.validation_step(batch, 0)


def test_fusion_trainer_configure_optimizers() -> None:
    """Test optimizer configuration"""
    trainer = FusionTrainer()
    optimizers = trainer.configure_optimizers()
    assert optimizers is not None
    assert "optimizer" in optimizers


def test_ava_optimizer_base() -> None:
    """Test base Mercury optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = MercuryOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_optimizer_step() -> None:
    """Test Mercury optimizer step"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryOptimizer([param], lr=0.001)

    loss = (param**2).sum()
    loss.backward()
    optimizer.step()


def test_ava_momentum_optimizer() -> None:
    """Test Ava momentum optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = MercuryMomentumOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_exp_decay_optimizer() -> None:
    """Test Ava exponential decay optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = MercuryExponentialDecayOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_ava_harmonic_optimizer() -> None:
    """Test Ava harmonic optimizer"""
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = MercuryHarmonicOptimizer(params, lr=0.001)
    assert optimizer is not None


def test_create_mercury_optimizer_factory() -> None:
    """Test Mercury optimizer factory function"""
    params = [torch.randn(10, 10, requires_grad=True)]

    variants = ["base", "momentum", "exp_decay", "harmonic"]
    for variant in variants:
        optimizer = create_mercury_optimizer(params, variant=variant, lr=0.001)
        assert optimizer is not None

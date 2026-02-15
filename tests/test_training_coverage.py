"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Additional training tests to boost coverage above 85%
"""

import importlib.util

import pytest

# Conditional torch import
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

HAS_LIGHTNING = importlib.util.find_spec("pytorch_lightning") is not None

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


def test_mercury_optimizer_with_closure():
    """Test MercuryOptimizer with closure function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryOptimizer([param], lr=0.001, alpha=0.2, beta=0.8)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_momentum_optimizer_step():
    """Test MercuryMomentumOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryMomentumOptimizer([param], lr=0.001, alpha=0.2, momentum=0.85)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_momentum_optimizer_with_closure():
    """Test MercuryMomentumOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryMomentumOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_exp_decay_optimizer_step():
    """Test MercuryExponentialDecayOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryExponentialDecayOptimizer([param], lr=0.001, alpha=0.2, decay_rate=0.95)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_exp_decay_optimizer_with_closure():
    """Test MercuryExponentialDecayOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryExponentialDecayOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_harmonic_optimizer_step():
    """Test MercuryHarmonicOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryHarmonicOptimizer([param], lr=0.001, alpha=0.2, omega=0.15)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_harmonic_optimizer_with_closure():
    """Test MercuryHarmonicOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = MercuryHarmonicOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_create_mercury_optimizer_invalid_variant():
    """Test create_mercury_optimizer with invalid variant"""
    params = [torch.randn(10, 10, requires_grad=True)]

    with pytest.raises(ValueError, match="Unknown Mercury optimizer variant"):
        create_mercury_optimizer(params, variant="invalid_variant", lr=0.001)


def test_anomaly_dataset_with_scores():
    """Test AnomalyDataset with scores parameter"""
    features = {
        "statistical": torch.randn(10, 5),
        "temporal": torch.randn(10, 32),
    }
    labels = torch.randint(0, 2, (10,))
    scores = {
        "detector1": torch.randn(10),
        "detector2": torch.randn(10),
    }

    dataset = AnomalyDataset(features, labels, scores)
    item = dataset[0]
    assert len(item) == 3
    assert isinstance(item[2], torch.Tensor)


def test_fusion_trainer_with_ava_base_optimizer():
    """Test FusionTrainer with ava_base optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_base"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], MercuryOptimizer)


def test_fusion_trainer_with_ava_momentum_optimizer():
    """Test FusionTrainer with ava_momentum optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_momentum"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], MercuryMomentumOptimizer)


def test_fusion_trainer_with_ava_exp_decay_optimizer():
    """Test FusionTrainer with ava_exp_decay optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_exp_decay"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], MercuryExponentialDecayOptimizer)


def test_fusion_trainer_with_ava_harmonic_optimizer():
    """Test FusionTrainer with ava_harmonic optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_harmonic"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], MercuryHarmonicOptimizer)

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
Additional training tests to boost coverage above 85%
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
    from omni_anomaly_engine.ml.training import (
        AnomalyDataset,
        AvaExponentialDecayOptimizer,
        AvaHarmonicOptimizer,
        AvaMomentumOptimizer,
        AvaOptimizer,
        FusionTrainer,
        create_ava_optimizer,
    )


def test_ava_optimizer_with_closure():
    """Test AvaOptimizer with closure function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaOptimizer([param], lr=0.001, alpha=0.2, beta=0.8)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_momentum_optimizer_step():
    """Test AvaMomentumOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaMomentumOptimizer([param], lr=0.001, alpha=0.2, momentum=0.85)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_momentum_optimizer_with_closure():
    """Test AvaMomentumOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaMomentumOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_exp_decay_optimizer_step():
    """Test AvaExponentialDecayOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaExponentialDecayOptimizer([param], lr=0.001, alpha=0.2, decay_rate=0.95)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_exp_decay_optimizer_with_closure():
    """Test AvaExponentialDecayOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaExponentialDecayOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_ava_harmonic_optimizer_step():
    """Test AvaHarmonicOptimizer step function"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaHarmonicOptimizer([param], lr=0.001, alpha=0.2, omega=0.15)

    loss = (param**2).sum()
    loss.backward()

    result = optimizer.step()
    assert result is None


def test_ava_harmonic_optimizer_with_closure():
    """Test AvaHarmonicOptimizer with closure"""
    param = torch.randn(10, 10, requires_grad=True)
    optimizer = AvaHarmonicOptimizer([param], lr=0.001)

    def closure():
        optimizer.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss is not None


def test_create_ava_optimizer_invalid_variant():
    """Test create_ava_optimizer with invalid variant"""
    params = [torch.randn(10, 10, requires_grad=True)]

    with pytest.raises(ValueError, match="Unknown Ava optimizer variant"):
        create_ava_optimizer(params, variant="invalid_variant", lr=0.001)


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
    assert isinstance(config["optimizer"], AvaOptimizer)


def test_fusion_trainer_with_ava_momentum_optimizer():
    """Test FusionTrainer with ava_momentum optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_momentum"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], AvaMomentumOptimizer)


def test_fusion_trainer_with_ava_exp_decay_optimizer():
    """Test FusionTrainer with ava_exp_decay optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_exp_decay"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], AvaExponentialDecayOptimizer)


def test_fusion_trainer_with_ava_harmonic_optimizer():
    """Test FusionTrainer with ava_harmonic optimizer"""
    trainer = FusionTrainer()
    trainer.optimizer_type = "ava_harmonic"

    config = trainer.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], AvaHarmonicOptimizer)

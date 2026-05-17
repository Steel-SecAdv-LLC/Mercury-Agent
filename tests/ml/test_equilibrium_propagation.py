"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

import pytest
import torch

from omni_mercury_engine.ml import equilibrium_propagation as ep_mod
from omni_mercury_engine.ml.equilibrium_propagation import (
    EquilibriumPropagationLayer,
    LowPowerAnomalyDetector,
    compute_ep_dynamics_metrics,
)

# ---------------------------------------------------------------------------
# Deleted-API contract
# ---------------------------------------------------------------------------


def test_estimate_power_savings_removed() -> None:
    """The arbitrary-baseline ``estimate_power_savings`` method must not exist."""
    assert not hasattr(LowPowerAnomalyDetector, "estimate_power_savings")


def test_renamed_metric_function() -> None:
    """The renamed dynamics-metric function exists and the old name does not."""
    assert hasattr(ep_mod, "compute_ep_dynamics_metrics")
    assert not hasattr(ep_mod, "compute_neuromorphic_efficiency")


# ---------------------------------------------------------------------------
# EquilibriumPropagationLayer
# ---------------------------------------------------------------------------


def test_layer_constructor_validates_dims() -> None:
    with pytest.raises(ValueError):
        EquilibriumPropagationLayer(input_dim=0, output_dim=4)
    with pytest.raises(ValueError):
        EquilibriumPropagationLayer(input_dim=4, output_dim=-1)


def test_layer_constructor_validates_dt_max_iter() -> None:
    with pytest.raises(ValueError):
        EquilibriumPropagationLayer(input_dim=4, output_dim=2, dt=0.0)
    with pytest.raises(ValueError):
        EquilibriumPropagationLayer(input_dim=4, output_dim=2, max_iter=0)


def test_layer_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError):
        EquilibriumPropagationLayer(input_dim=4, output_dim=2, activation="elu")


def test_layer_forward_shape_and_dtype() -> None:
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=8)
    x = torch.randn(5, 4)
    out = layer(x)
    assert out.shape == (5, 3)
    assert out.dtype == x.dtype


def test_layer_tanh_bounds_output() -> None:
    """A tanh-activated layer's equilibrium state lives in (-1, 1)."""
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, activation="tanh", max_iter=8)
    x = torch.randn(8, 4)
    out = layer(x)
    assert torch.all(out >= -1.0 - 1e-6)
    assert torch.all(out <= 1.0 + 1e-6)


def test_layer_sigmoid_bounds_output() -> None:
    """A sigmoid-activated layer's equilibrium state lives in (0, 1)."""
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, activation="sigmoid", max_iter=8)
    x = torch.randn(8, 4)
    out = layer(x)
    assert torch.all(out >= 0.0 - 1e-6)
    assert torch.all(out <= 1.0 + 1e-6)


def test_layer_records_energy_trajectory() -> None:
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=10)
    layer(torch.randn(4, 4))
    traj = layer.get_energy_trajectory()
    assert 1 <= len(traj) <= 10
    assert all(isinstance(v, float) for v in traj)


def test_layer_lyapunov_stability_on_real_trajectory() -> None:
    """A real forward pass produces a Lyapunov-stable energy trajectory."""
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=10)
    layer(torch.randn(4, 4))
    assert layer.verify_lyapunov_stability() is True


def test_layer_lyapunov_stability_short_trajectory() -> None:
    """A trajectory of length < 2 is trivially stable."""
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=10)
    # Never called forward; energy_history is empty.
    assert layer.verify_lyapunov_stability() is True


def test_layer_lyapunov_detects_violation() -> None:
    """Manually inject a non-monotone trajectory and confirm detection."""
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=10)
    layer.energy_history = [-1.0, -0.5, 5.0]  # large jump > default tolerance
    assert layer.verify_lyapunov_stability() is False


def test_layer_hebbian_noop_without_state() -> None:
    """Without a prior forward pass, ``update_weights_hebbian`` is a no-op."""
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3)
    before_w = layer.weight.detach().clone()
    before_b = layer.bias.detach().clone()
    layer.update_weights_hebbian(torch.randn(4, 4))
    torch.testing.assert_close(layer.weight.detach(), before_w)
    torch.testing.assert_close(layer.bias.detach(), before_b)


def test_layer_hebbian_updates_weights() -> None:
    """After a forward pass, the Hebbian update moves weights."""
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=4)
    x = torch.randn(4, 4)
    layer(x)
    before_w = layer.weight.detach().clone()
    layer.update_weights_hebbian(x, beta=0.1)
    assert not torch.allclose(layer.weight.detach(), before_w)


def test_layer_early_stop_via_tolerance() -> None:
    """A loose tolerance plus a large max_iter should yield short trajectories."""
    torch.manual_seed(0)
    layer = EquilibriumPropagationLayer(input_dim=4, output_dim=3, max_iter=200, tolerance=1.0)
    layer(torch.randn(2, 4))
    # Should terminate well before max_iter under a coarse tolerance.
    assert len(layer.get_energy_trajectory()) < 200


# ---------------------------------------------------------------------------
# LowPowerAnomalyDetector
# ---------------------------------------------------------------------------


def test_detector_default_topology() -> None:
    det = LowPowerAnomalyDetector(input_dim=8)
    # 2 hidden + 1 head
    assert len(det.layers) == 3


def test_detector_forward_returns_anomaly_score() -> None:
    torch.manual_seed(0)
    det = LowPowerAnomalyDetector(input_dim=6, hidden_dims=[8, 4], output_dim=1, max_iter=5)
    out = det(torch.randn(3, 6))
    assert out.shape == (3, 1)
    # Final head is sigmoid -> scores in [0, 1].
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_detector_hebbian_training_advances_weights() -> None:
    torch.manual_seed(0)
    det = LowPowerAnomalyDetector(input_dim=4, hidden_dims=[6], output_dim=1, max_iter=4)
    before = [p.detach().clone() for p in det.parameters()]
    det.train_hebbian(torch.randn(8, 4), learning_rate=0.1)
    after = [p.detach() for p in det.parameters()]
    # At least one parameter must have changed.
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))


def test_detector_verify_all_layers_stable() -> None:
    torch.manual_seed(0)
    det = LowPowerAnomalyDetector(input_dim=4, hidden_dims=[6], output_dim=1, max_iter=4)
    det(torch.randn(4, 4))
    assert det.verify_all_layers_stable() is True


# ---------------------------------------------------------------------------
# compute_ep_dynamics_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_validates_num_trials() -> None:
    det = LowPowerAnomalyDetector(input_dim=4, hidden_dims=[6])
    with pytest.raises(ValueError):
        compute_ep_dynamics_metrics(det, torch.randn(2, 4), num_trials=0)


def test_compute_metrics_returns_expected_keys() -> None:
    torch.manual_seed(0)
    det = LowPowerAnomalyDetector(input_dim=4, hidden_dims=[6], output_dim=1, max_iter=4)
    out = compute_ep_dynamics_metrics(det, torch.randn(8, 4), num_trials=3)
    assert set(out.keys()) == {
        "energy_drop_per_sample",
        "convergence_iterations",
        "stability_ratio",
    }
    assert 0.0 <= out["stability_ratio"] <= 1.0
    assert out["convergence_iterations"] >= 0.0


def test_compute_metrics_with_non_ep_model_returns_zeros() -> None:
    """A non-EP model still returns the documented keys with zero values."""
    not_ep = torch.nn.Linear(4, 1)
    out = compute_ep_dynamics_metrics(not_ep, torch.randn(2, 4), num_trials=2)
    assert out == {
        "energy_drop_per_sample": 0.0,
        "convergence_iterations": 0.0,
        "stability_ratio": 0.0,
    }

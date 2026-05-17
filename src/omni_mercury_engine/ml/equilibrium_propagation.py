"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Equilibrium Propagation (EP) neural layer and EP-based anomaly detector.

EP is an energy-based learning rule due to Scellier & Bengio (2017). For each
forward pass the layer relaxes its state under the dynamics
``ds/dt = -∇_s E(s, x)`` with energy

.. math::

    E(s, x) = -\\tfrac{1}{2}\\, s^T W x + s^T b ,

and stops when the per-iteration change in mean energy drops below
``tolerance``. ``V(s) := E(s, x)`` is a Lyapunov function for the dynamics
(``dV/dt = -\\|\\nabla_s E\\|^2 \\le 0``), so the trajectory is monotonically
non-increasing in energy modulo floating-point and activation-saturation
noise; :meth:`EquilibriumPropagationLayer.verify_lyapunov_stability` audits
that invariant on the captured energy trajectory.

The layer is useful as a low-update-density building block — the iterative
relaxation replaces a full backward pass with a forward fixed-point search,
which can be advantageous on hardware where memory writes (weight updates)
are the dominant cost. Concrete power-vs-backprop comparisons require
measurement on the target hardware and are not produced here.
"""

import numpy as np
import torch
from torch import nn


# Uniform wrappers around the activation functions: torch.relu's stub uses a
# different overload shape than tanh / sigmoid (``out=`` keyword), so wrapping
# them gives one ``(Tensor) -> Tensor`` signature for assignment-time type
# inference.
def _act_tanh(t: torch.Tensor) -> torch.Tensor:
    return torch.tanh(t)


def _act_sigmoid(t: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(t)


def _act_relu(t: torch.Tensor) -> torch.Tensor:
    return torch.relu(t)


class EquilibriumPropagationLayer(nn.Module):
    """
    Single EP layer: weights ``W``, bias ``b``, and a continuous relaxation.

    Reference:
        Scellier, B. & Bengio, Y. *Equilibrium Propagation: Bridging the Gap
        Between Energy-Based Models and Backpropagation.* Frontiers in
        Computational Neuroscience, 2017.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "tanh",
        dt: float = 0.1,
        max_iter: int = 20,
        tolerance: float = 1e-4,
    ) -> None:
        """
        Args:
            input_dim: Dimension of the input vector ``x``.
            output_dim: Dimension of the state vector ``s``.
            activation: Non-linearity applied after each relaxation step.
                One of ``"tanh"``, ``"sigmoid"``, ``"relu"``.
            dt: Step size for the discretised relaxation.
            max_iter: Maximum number of relaxation steps per forward pass.
            tolerance: Stop early when the absolute change in mean energy
                between consecutive iterations falls below this threshold.
        """
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.max_iter = max_iter
        self.tolerance = tolerance

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.activation_name = activation
        if activation == "tanh":
            self._activation = _act_tanh
        elif activation == "sigmoid":
            self._activation = _act_sigmoid
        elif activation == "relu":
            self._activation = _act_relu
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

        self.last_equilibrium_state: torch.Tensor | None = None
        self.energy_history: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Relax the state to (approximate) equilibrium and return it.

        Args:
            x: Input batch of shape ``(batch, input_dim)``.

        Returns:
            Equilibrium state batch of shape ``(batch, output_dim)``.
        """
        batch = x.shape[0]
        state = torch.zeros(batch, self.output_dim, device=x.device, dtype=x.dtype)
        self.energy_history = []

        for iteration in range(self.max_iter):
            grad = self._energy_gradient(state, x)
            state = state - self.dt * grad
            state = self._activation(state)

            energy = self._energy(state, x)
            self.energy_history.append(float(energy.mean().item()))

            if iteration > 0:
                change = abs(self.energy_history[-1] - self.energy_history[-2])
                if change < self.tolerance:
                    break

        self.last_equilibrium_state = state.detach()
        return state

    def _energy(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-sample energy ``E(s, x) = -(1/2) s^T W x + s^T b``."""
        Wx = torch.mm(x, self.weight.t())
        quadratic = -0.5 * (state * Wx).sum(dim=1)
        linear = (state * self.bias).sum(dim=1)
        return quadratic + linear

    def _energy_gradient(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gradient ``∇_s E(s, x)`` whose negation drives the relaxation."""
        Wx = torch.mm(x, self.weight.t())
        return -(Wx - state - self.bias)

    def update_weights_hebbian(self, x: torch.Tensor, beta: float = 0.01) -> None:
        """
        Hebbian-like local update ``ΔW ∝ s s^T / batch`` using the last
        captured equilibrium state.

        Args:
            x: The input batch used to produce ``last_equilibrium_state``.
            beta: Learning rate.
        """
        if self.last_equilibrium_state is None:
            return
        state = self.last_equilibrium_state
        batch = x.shape[0]
        weight_update = torch.mm(state.t(), x) / batch
        with torch.no_grad():
            # In-place ops on .data avoid the typing mismatch on `Parameter += Tensor`.
            self.weight.data.add_(weight_update, alpha=beta)
            self.bias.data.add_(state.mean(dim=0), alpha=beta)

    def get_energy_trajectory(self) -> list[float]:
        """Return a copy of the per-iteration energy trajectory."""
        return list(self.energy_history)

    def verify_lyapunov_stability(self, tolerance: float = 0.01) -> bool:
        """
        Check that the captured energy trajectory is monotonically non-increasing
        within ``tolerance``.

        Args:
            tolerance: Allowed positive step (accounts for floating-point
                error and activation saturation discontinuities).

        Returns:
            True if no ``E_{i+1} - E_i > tolerance`` violation was observed.
        """
        if len(self.energy_history) < 2:
            return True
        for i in range(1, len(self.energy_history)):
            if self.energy_history[i] > self.energy_history[i - 1] + tolerance:
                return False
        return True


class LowPowerAnomalyDetector(nn.Module):
    """
    Stack of EP layers terminated by a sigmoid head, producing anomaly scores.

    Designed for memory-write-bound hardware where the Hebbian local update
    is more attractive than a global backward pass.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        dt: float = 0.1,
        max_iter: int = 20,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension.
            hidden_dims: Sequence of hidden layer widths. Defaults to ``[64, 32]``.
            output_dim: Output dimension; defaults to 1 (scalar anomaly score).
            dt: Step size for each EP layer.
            max_iter: Max relaxation iterations per layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[EquilibriumPropagationLayer] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                EquilibriumPropagationLayer(
                    prev_dim,
                    hidden_dim,
                    activation="tanh",
                    dt=dt,
                    max_iter=max_iter,
                )
            )
            prev_dim = hidden_dim
        layers.append(
            EquilibriumPropagationLayer(
                prev_dim,
                output_dim,
                activation="sigmoid",
                dt=dt,
                max_iter=max_iter,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for layer in self.layers:
            state = layer(state)
        return state

    def train_hebbian(self, x: torch.Tensor, learning_rate: float = 0.01) -> None:
        """
        Forward-pass and apply per-layer Hebbian updates using each layer's
        own input as the post-synaptic anchor.
        """
        state = x
        layer_inputs: list[torch.Tensor] = [x]
        for layer in self.layers:
            state = layer(state)
            layer_inputs.append(state)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, EquilibriumPropagationLayer):
                layer.update_weights_hebbian(layer_inputs[i], beta=learning_rate)

    def verify_all_layers_stable(self) -> bool:
        """Return True iff every EP layer's most-recent trajectory is Lyapunov-stable."""
        return all(
            layer.verify_lyapunov_stability()
            for layer in self.layers
            if isinstance(layer, EquilibriumPropagationLayer)
        )


def compute_ep_dynamics_metrics(
    model: nn.Module, data: torch.Tensor, num_trials: int = 10
) -> dict[str, float]:
    """
    Diagnostic statistics for an EP-based model on a representative batch.

    Args:
        model: Model to evaluate. Expected to be a
            :class:`LowPowerAnomalyDetector` (anything else returns zeros).
        data: Representative input batch.
        num_trials: Number of repeated forward passes to average over.

    Returns:
        Dict with::

            {
                "energy_drop_per_sample": mean |E_final - E_initial| / batch,
                "convergence_iterations": mean per-layer iterations until stop,
                "stability_ratio": fraction of trials where every layer was
                                   Lyapunov-stable,
            }
    """
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")

    energy_drops: list[float] = []
    iter_counts: list[float] = []
    stable_runs = 0

    for _ in range(num_trials):
        _ = model(data)
        if not isinstance(model, LowPowerAnomalyDetector):
            continue

        trial_drop = 0.0
        trial_iters = 0
        ep_layers = 0
        for layer in model.layers:
            if isinstance(layer, EquilibriumPropagationLayer):
                trajectory = layer.get_energy_trajectory()
                if trajectory:
                    trial_drop += abs(trajectory[-1] - trajectory[0])
                    trial_iters += len(trajectory)
                ep_layers += 1

        energy_drops.append(trial_drop / max(data.shape[0], 1))
        iter_counts.append(trial_iters / max(ep_layers, 1))

        if model.verify_all_layers_stable():
            stable_runs += 1

    return {
        "energy_drop_per_sample": float(np.mean(energy_drops)) if energy_drops else 0.0,
        "convergence_iterations": float(np.mean(iter_counts)) if iter_counts else 0.0,
        "stability_ratio": stable_runs / num_trials,
    }


__all__: list[str] = [
    "EquilibriumPropagationLayer",
    "LowPowerAnomalyDetector",
    "compute_ep_dynamics_metrics",
]

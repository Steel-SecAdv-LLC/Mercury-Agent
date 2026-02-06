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

"""
Mercury Equation Optimization Experiments
Runs 10,000+ iterations to find optimal parameters for Mercury optimizers.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.ml.training import (
    MercuryExponentialDecayOptimizer,
    MercuryHarmonicOptimizer,
    MercuryMomentumOptimizer,
    MercuryOptimizer,
)


class SimpleTestModel(nn.Module):  # type: ignore[misc]
    """Simple model for testing optimizer performance."""

    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def run_optimizer_experiment(
    optimizer_class: type,
    params: dict[str, Any],
    num_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Run a single experiment with given optimizer parameters.

    Args:
        optimizer_class: Optimizer class to test
        params: Optimizer parameters (lr, alpha, beta, etc.)
        num_iterations: Number of training iterations
        seed: Random seed for reproducibility

    Returns:
        Dict with convergence metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleTestModel()
    optimizer = optimizer_class(model.parameters(), **params)
    criterion = nn.MSELoss()

    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    losses = []
    start_loss = 0.0

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)

        if iteration == 0:
            start_loss = loss.item()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    final_loss = losses[-1]
    convergence_speed = (start_loss - final_loss) / num_iterations
    min_loss = min(losses)

    return {
        "start_loss": start_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "convergence_speed": convergence_speed,
        "num_iterations": num_iterations,
    }


def grid_search_mercury_base(
    num_experiments: int = 1000,
) -> list[tuple[dict[str, Any], dict[str, float]]]:
    """
    Grid search over MercuryOptimizer parameters.

    Returns:
        List of (params, results) tuples sorted by performance
    """
    print(f"Running {num_experiments} experiments for MercuryOptimizer...")

    lr_values = np.logspace(-4, -1, 10)
    alpha_values = np.linspace(0.01, 0.5, 10)
    beta_values = np.linspace(0.5, 0.99, 10)
    quantum_noise_values = [0.0, 0.001, 0.01, 0.05]

    experiments = []

    for i in range(num_experiments):
        params = {
            "lr": float(np.random.choice(lr_values)),
            "alpha": float(np.random.choice(alpha_values)),
            "beta": float(np.random.choice(beta_values)),
            "quantum_noise": float(np.random.choice(quantum_noise_values)),
        }

        results = run_optimizer_experiment(MercuryOptimizer, params, num_iterations=500, seed=i)

        experiments.append((params, results))

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_experiments} experiments")

    experiments.sort(key=lambda x: x[1]["final_loss"])

    return experiments


def grid_search_mercury_variants(
    num_experiments: int = 500,
) -> dict[str, list[tuple[dict[str, Any], dict[str, float]]]]:
    """
    Grid search over all Mercury optimizer variants.

    Returns:
        Dict mapping variant name to sorted experiments
    """
    variants = {
        "momentum": MercuryMomentumOptimizer,
        "exp_decay": MercuryExponentialDecayOptimizer,
        "harmonic": MercuryHarmonicOptimizer,
    }

    all_results = {}

    for variant_name, optimizer_class in variants.items():
        print(f"\nRunning {num_experiments} experiments for Ava{variant_name.title()}Optimizer...")

        experiments = []

        for i in range(num_experiments):
            if variant_name == "momentum":
                params = {
                    "lr": float(10 ** np.random.uniform(-4, -1)),
                    "alpha": float(np.random.uniform(0.01, 0.5)),
                    "momentum": float(np.random.uniform(0.5, 0.99)),
                }
            elif variant_name == "exp_decay":
                params = {
                    "lr": float(10 ** np.random.uniform(-4, -1)),
                    "alpha": float(np.random.uniform(0.01, 0.5)),
                    "decay_rate": float(np.random.uniform(0.9, 0.999)),
                }
            else:
                params = {
                    "lr": float(10 ** np.random.uniform(-4, -1)),
                    "alpha": float(np.random.uniform(0.01, 0.5)),
                    "omega": float(np.random.uniform(0.01, 0.5)),
                }

            results = run_optimizer_experiment(optimizer_class, params, num_iterations=500, seed=i)

            experiments.append((params, results))

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{num_experiments} experiments")

        experiments.sort(key=lambda x: x[1]["final_loss"])
        all_results[variant_name] = experiments

    return all_results


def main() -> None:
    """Run all Mercury equation experiments and save results."""
    print("=" * 60)
    print("AVA EQUATION OPTIMIZATION EXPERIMENTS")
    print("Target: 10,000+ total iterations")
    print("=" * 60)

    base_experiments = grid_search_mercury_base(num_experiments=5000)

    print(f"\n{'='*60}")
    print("TOP 10 AVAOPTIMIZER CONFIGURATIONS:")
    print(f"{'='*60}")
    for i, (params, results) in enumerate(base_experiments[:10], 1):
        print(f"\n{i}. Final Loss: {results['final_loss']:.6f}")
        print(f"   Parameters: {params}")
        print(f"   Convergence: {results['convergence_speed']:.6e}")

    variant_experiments = grid_search_mercury_variants(num_experiments=2000)

    print(f"\n{'='*60}")
    print("BEST CONFIGURATION PER VARIANT:")
    print(f"{'='*60}")
    for variant_name, experiments in variant_experiments.items():
        best_params, best_results = experiments[0]
        print(f"\n{variant_name.upper()}:")
        print(f"  Final Loss: {best_results['final_loss']:.6f}")
        print(f"  Parameters: {best_params}")
        print(f"  Convergence: {best_results['convergence_speed']:.6e}")

    total_experiments = 5000 + sum(len(exps) for exps in variant_experiments.values())

    results_path = Path(__file__).parent / "mercury_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "base_top_10": [{"params": p, "results": r} for p, r in base_experiments[:10]],
                "variants": {
                    name: [{"params": p, "results": r} for p, r in exps[:5]]
                    for name, exps in variant_experiments.items()
                },
                "total_experiments": total_experiments,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Results saved to: {results_path}")
    print(f"✅ Total experiments: {total_experiments} (exceeds 10,000 target)")


if __name__ == "__main__":
    main()

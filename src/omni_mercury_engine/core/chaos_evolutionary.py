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


"""Chaos-Evolutionary Optimization for Adaptive Hyperparameter Tuning.

Based on: Chaos Game Optimization - A novel metaheuristic algorithm inspired by chaotic dynamics
(Artificial Intelligence Review, 2021: https://link.springer.com/article/10.1007/s10462-020-09867-w)

Implements Chaos Game Optimization (CGO) using fractal configurations and chaos theory
for dynamic hyperparameter tuning in anomaly detection systems.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


if TYPE_CHECKING:
    from collections.abc import Callable


class ChaoticMap:
    """Chaotic map generators for CGO algorithm."""

    @staticmethod
    def logistic_map(x: float, r: float = 4.0) -> float:
        """Logistic chaotic map: x_{n+1} = r * x_n * (1 - x_n).

        Args:
            x: Current value in [0, 1]
            r: Chaos parameter (typically 4.0 for full chaos)

        Returns:
            Next chaotic value
        """
        return r * x * (1 - x)

    @staticmethod
    def tent_map(x: float, mu: float = 2.0) -> float:
        """Tent chaotic map.

        Args:
            x: Current value in [0, 1]
            mu: Chaos parameter (typically 2.0)

        Returns:
            Next chaotic value
        """
        if x < 0.5:
            return mu * x
        else:
            return mu * (1 - x)

    @staticmethod
    def sine_map(x: float, a: float = 2.3) -> float:
        """Sine chaotic map: x_{n+1} = a * sin(pi * x_n).

        Args:
            x: Current value
            a: Chaos parameter

        Returns:
            Next chaotic value
        """
        return float(a * np.sin(np.pi * x) / 4.0)


class ChaosEvolutionOptimizer:
    """Chaos-Evolutionary Optimizer using CGO algorithm."""

    def __init__(
        self, config: dict[str, Any] | None = None, rng: DeterministicRNG | None = None
    ) -> None:
        """Initialize chaos-evolutionary optimizer.

        Args:
            config: Configuration including:
                - population_size: Number of candidate solutions (default: 30)
                - max_iterations: Maximum optimization iterations (default: 100)
                - chaotic_map: Type of chaotic map ('logistic', 'tent', 'sine')
                    (default: 'logistic')
                - alpha: Fractal self-similarity parameter (default: 0.8)
                - beta: Chaos influence parameter (default: 0.2)
            rng: Optional DeterministicRNG instance for reproducibility
        """
        self.config = config or {}
        self.population_size = self.config.get("population_size", 30)
        self.max_iterations = self.config.get("max_iterations", 100)
        self.chaotic_map_type = self.config.get("chaotic_map", "logistic")
        self.alpha = self.config.get("alpha", 0.8)
        self.beta = self.config.get("beta", 0.2)
        self._rng = rng or get_global_rng()

        self.chaotic_map = self._get_chaotic_map()
        self.best_solution: np.ndarray[Any, Any] | None = None
        self.best_fitness: float = np.inf
        self.convergence_history: list[float] = []

    def _get_chaotic_map(self) -> Callable[[float], float]:
        """Get chaotic map function based on configuration."""
        if self.chaotic_map_type == "logistic":
            return ChaoticMap.logistic_map
        elif self.chaotic_map_type == "tent":
            return ChaoticMap.tent_map
        elif self.chaotic_map_type == "sine":
            return ChaoticMap.sine_map
        else:
            return ChaoticMap.logistic_map

    def _initialize_population(
        self, dim: int, bounds: list[tuple[float, float]]
    ) -> np.ndarray[Any, Any]:
        """Initialize population with random solutions.

        Args:
            dim: Problem dimensionality
            bounds: List of (min, max) tuples for each dimension

        Returns:
            Initial population matrix
        """
        population = np.zeros((self.population_size, dim))
        for i in range(dim):
            min_val, max_val = bounds[i]
            population[:, i] = self._rng.rand(self.population_size) * (max_val - min_val) + min_val
        return population

    def _chaos_game_step(
        self,
        position: np.ndarray[Any, Any],
        best_position: np.ndarray[Any, Any],
        chaos_value: float,
        bounds: list[tuple[float, float]],
    ) -> np.ndarray[Any, Any]:
        """Perform one chaos game step for fractal-based position update.

        Based on CGO's fractal self-similarity and chaos game methodology.

        Args:
            position: Current position
            best_position: Global best position
            chaos_value: Chaotic value for perturbation
            bounds: Variable bounds

        Returns:
            Updated position
        """
        dim = len(position)
        new_position = np.zeros(dim)

        for i in range(dim):
            gamma = 2 * chaos_value - 1

            fractal_component = self.alpha * (best_position[i] - position[i])
            chaos_component = self.beta * gamma * (bounds[i][1] - bounds[i][0])

            new_position[i] = position[i] + fractal_component + chaos_component

            new_position[i] = np.clip(new_position[i], bounds[i][0], bounds[i][1])

        return new_position

    def optimize(
        self,
        objective_function: Callable[[np.ndarray[Any, Any]], float],
        dim: int,
        bounds: list[tuple[float, float]],
    ) -> dict[str, Any]:
        """Run chaos-evolutionary optimization.

        Args:
            objective_function: Function to minimize (takes array, returns scalar)
            dim: Problem dimensionality
            bounds: List of (min, max) tuples for each dimension

        Returns:
            Optimization results
        """
        population = self._initialize_population(dim, bounds)

        fitness = np.array([objective_function(ind) for ind in population])
        best_idx = int(np.argmin(fitness))
        self.best_solution = population[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])

        chaos_value = float(self._rng.rand())

        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                chaos_value = self.chaotic_map(chaos_value)

                if self.best_solution is None:
                    self.best_solution = population[i].copy()

                new_position = self._chaos_game_step(
                    population[i], self.best_solution, chaos_value, bounds
                )

                new_fitness = objective_function(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = float(new_fitness)

            self.convergence_history.append(float(self.best_fitness))

            if iteration % 10 == 0:
                chaos_value = self._rng.rand()

        results = {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "convergence_history": self.convergence_history,
            "iterations": self.max_iterations,
            "population_size": self.population_size,
            "chaotic_map": self.chaotic_map_type,
            "method": "Chaos_Evolution_Optimization",
        }

        return results

    def tune_hyperparameters(
        self,
        parameter_space: dict[str, tuple[float, float]],
        evaluation_function: Callable[[dict[str, float]], float],
    ) -> dict[str, Any]:
        """Tune hyperparameters for anomaly detection system.

        Args:
            parameter_space: Dict mapping parameter names to (min, max) bounds
            evaluation_function: Function that takes parameter dict and returns loss

        Returns:
            Tuning results with optimal parameters
        """
        param_names = list(parameter_space.keys())
        bounds = [parameter_space[name] for name in param_names]
        dim = len(param_names)

        def objective_wrapper(x: np.ndarray[Any, Any]) -> float:
            param_dict = {name: x[i] for i, name in enumerate(param_names)}
            return evaluation_function(param_dict)

        results = self.optimize(objective_wrapper, dim, bounds)

        optimal_params = {name: results["best_solution"][i] for i, name in enumerate(param_names)}

        tuning_results = {
            "optimal_parameters": optimal_params,
            "optimal_loss": results["best_fitness"],
            "convergence_history": results["convergence_history"],
            "method": "CGO_Hyperparameter_Tuning",
        }

        return tuning_results

    def generate_creative_hypotheses(
        self,
        base_solution: np.ndarray[Any, Any],
        num_hypotheses: int = 10,
        chaos_intensity: float = 0.1,
    ) -> list[np.ndarray[Any, Any]]:
        """Generate creative hypothesis variations using controlled chaos.

        Inspired by: AI and Human Creativity: Can Chaos Theory Make Machines
        Think Differently (Unite.AI)

        Simulates human-like intuition by introducing chaotic perturbations
        to explore creative solution space.

        Args:
            base_solution: Starting solution
            num_hypotheses: Number of creative variations to generate
            chaos_intensity: Intensity of chaotic perturbations (0-1)

        Returns:
            List of creative hypothesis solutions
        """
        hypotheses = []
        chaos_value = self._rng.rand()

        for _ in range(num_hypotheses):
            chaos_value = self.chaotic_map(chaos_value)
            perturbation = chaos_intensity * (2 * chaos_value - 1)

            hypothesis = base_solution + perturbation * self._rng.randn(*base_solution.shape)
            hypotheses.append(hypothesis)

        return hypotheses

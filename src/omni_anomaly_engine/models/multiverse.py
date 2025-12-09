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
from __future__ import annotations

"""
Multi-Hypothesis Optimization Engine - Parallel Solution Space Exploration

Original implementation for OMNI ♱ AVA neural-symbolic AI archetype.

This engine explores multiple solution pathways simultaneously using ensemble
optimization and multi-dimensional state space exploration to find optimal strategies.
The approach is inspired by population-based optimization methods like genetic
algorithms and particle swarm optimization.
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from omni_anomaly_engine.utils.rng import DeterministicRNG, get_global_rng

_VITALITY_HASH = "V20V11M16V19"


class UniverseState(Enum):
    """States in the multiverse."""

    SUPERPOSITION = "superposition"
    COLLAPSED = "collapsed"
    ENTANGLED = "entangled"
    CONVERGED = "converged"


@dataclass
class Universe:
    """Represents a parallel universe (solution pathway)."""

    universe_id: str
    state_vector: np.ndarray
    probability_amplitude: float
    fitness: float
    state: UniverseState
    timeline: int
    parent_universe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiverseOmniEngine:
    """
    Multi-Hypothesis Optimization Engine - Parallel Solution Exploration.

    Explores multiple solution pathways simultaneously to maximize
    the probability of successful outcomes through ensemble optimization.
    Similar to population-based methods like genetic algorithms.
    """

    def __init__(
        self,
        num_universes: int = 10,
        state_dim: int = 50,
        convergence_threshold: float = 0.95,
        entanglement_strength: float = 0.3,
        rng: DeterministicRNG | None = None,
    ):
        """
        Initialize Multi-Universe Omni Engine.

        Args:
            num_universes: Number of parallel universes to simulate
            state_dim: Dimensionality of state space
            convergence_threshold: Threshold for universe convergence
            entanglement_strength: Strength of entanglement between universes
            rng: Optional DeterministicRNG for reproducibility
        """
        self.num_universes = num_universes
        self.state_dim = state_dim
        self.convergence_threshold = convergence_threshold
        self.entanglement_strength = entanglement_strength
        self._rng = rng or get_global_rng()

        self.universes: dict[str, Universe] = {}
        self.timeline = 0
        self.best_universe: Universe | None = None

        self._initialize_multiverse()

        logging.info("Multi-Universe Omni Engine initialized")
        logging.info(f"Number of parallel universes: {num_universes}")

    def _initialize_multiverse(self) -> None:
        """Initialize the multiverse with random universes."""
        for i in range(self.num_universes):
            universe_id = hashlib.sha256(f"universe_{i}_{time.time()}".encode()).hexdigest()[:16]

            state_vector = self._rng.randn(self.state_dim) * 0.5
            probability_amplitude = 1.0 / self.num_universes

            universe = Universe(
                universe_id=universe_id,
                state_vector=state_vector,
                probability_amplitude=probability_amplitude,
                fitness=0.0,
                state=UniverseState.SUPERPOSITION,
                timeline=0,
                metadata={"generation": 0},
            )

            self.universes[universe_id] = universe

        logging.info(f"Initialized {self.num_universes} parallel universes")

    def evaluate_universe(
        self, universe: Universe, fitness_function: Callable[[np.ndarray[Any, Any]], float]
    ) -> float:
        """
        Evaluate fitness of a universe.

        Args:
            universe: Universe to evaluate
            fitness_function: Function to compute fitness

        Returns:
            Fitness score
        """
        fitness = fitness_function(universe.state_vector)
        universe.fitness = fitness

        if self.best_universe is None or fitness > self.best_universe.fitness:
            self.best_universe = universe

        return fitness

    def combine_hypotheses(self, universes_to_superpose: list[str]) -> Universe:
        """
        Combine multiple solution hypotheses into a weighted average.

        Args:
            universes_to_superpose: List of universe IDs to combine

        Returns:
            New combined universe representing the weighted average
        """
        if len(universes_to_superpose) < 2:
            raise ValueError("Need at least 2 universes for superposition")

        universes = [self.universes[uid] for uid in universes_to_superpose]
        total_amplitude = sum(u.probability_amplitude for u in universes)

        if total_amplitude < 1e-10:
            weights = np.ones(len(universes)) / len(universes)
        else:
            weights = np.array([u.probability_amplitude / total_amplitude for u in universes])

        superposed_state = np.zeros(self.state_dim)
        for i, universe in enumerate(universes):
            superposed_state += weights[i] * universe.state_vector

        universe_id = hashlib.sha256(
            f"superposed_{time.time()}_{self._rng.randint(0, 1000000)}".encode()
        ).hexdigest()[:16]

        new_universe = Universe(
            universe_id=universe_id,
            state_vector=superposed_state,
            probability_amplitude=total_amplitude,
            fitness=0.0,
            state=UniverseState.SUPERPOSITION,
            timeline=self.timeline,
            metadata={
                "type": "superposition",
                "parent_universes": universes_to_superpose,
                "num_parents": len(universes_to_superpose),
            },
        )

        self.universes[universe_id] = new_universe
        return new_universe

    def converge_multiverse(self, fitness_function: Callable[[np.ndarray[Any, Any]], float]) -> Universe:
        """
        Converge the multiverse to the best solution.

        Args:
            fitness_function: Function to evaluate fitness

        Returns:
            Best converged universe
        """
        for universe in self.universes.values():
            self.evaluate_universe(universe, fitness_function)

        total_fitness = sum(u.fitness for u in self.universes.values())

        if total_fitness > 1e-10:
            for universe in self.universes.values():
                universe.probability_amplitude = universe.fitness / total_fitness

        sorted_universes = sorted(self.universes.values(), key=lambda u: u.fitness, reverse=True)

        top_universes = sorted_universes[: max(3, self.num_universes // 3)]
        top_universe_ids = [u.universe_id for u in top_universes]
        converged_universe = self.combine_hypotheses(top_universe_ids)

        converged_universe.state = UniverseState.CONVERGED
        converged_universe.metadata["type"] = "converged"

        self.evaluate_universe(converged_universe, fitness_function)

        logging.info(
            f"Multiverse converged to solution with fitness: " f"{converged_universe.fitness:.4f}"
        )

        return converged_universe

    def extract_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract multiverse features from data for anomaly detection."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        features = []

        for i in range(batch_size):

            def fitness_fn(state: np.ndarray[Any, Any]) -> float:
                data_dim = data[i].shape[0]
                if state.shape[0] > data_dim:
                    state_truncated = state[:data_dim]
                elif state.shape[0] < data_dim:
                    state_truncated = np.pad(state, (0, data_dim - state.shape[0]))
                else:
                    state_truncated = state
                return float(-np.linalg.norm(state_truncated - data[i]))

            converged = self.converge_multiverse(fitness_fn)

            feature_vec = np.concatenate(
                [
                    converged.state_vector[:10],
                    [converged.fitness],
                    [converged.probability_amplitude],
                ]
            )
            features.append(feature_vec)

        return np.array(features).astype(np.float32)

    def predict(self, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Predict anomalies using multiverse optimization."""
        features = self.extract_features(data)

        fitness_scores = features[:, 10]
        anomaly_scores = 1.0 / (1.0 + np.abs(fitness_scores))

        return {
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "multiverse_features": features,
            "best_fitness": float(self.best_universe.fitness) if self.best_universe else 0.0,
        }

    def get_multiverse_report(self) -> dict[str, Any]:
        """Generate comprehensive multiverse report."""
        if not self.universes:
            return {"status": "empty", "message": "No universes in multiverse"}

        state_counts = {}
        for universe in self.universes.values():
            state = universe.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        fitnesses = [u.fitness for u in self.universes.values()]
        amplitudes = [u.probability_amplitude for u in self.universes.values()]

        return {
            "total_universes": len(self.universes),
            "current_timeline": self.timeline,
            "state_distribution": state_counts,
            "best_fitness": self.best_universe.fitness if self.best_universe else 0.0,
            "average_fitness": np.mean(fitnesses),
            "fitness_std": np.std(fitnesses),
            "total_probability": sum(amplitudes),
            "convergence_achieved": (
                self.best_universe.fitness >= self.convergence_threshold
                if self.best_universe
                else False
            ),
            "best_universe_id": (self.best_universe.universe_id if self.best_universe else None),
            "system_version": _VITALITY_HASH,
        }

    def _apply_hierarchical_scaling(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Apply hierarchical base-20 scaling for multi-dimensional exploration.

        Uses base-20 (vigesimal) scaling to create hierarchical representations
        of state vectors, allowing different components to operate at different
        scales for multi-resolution optimization.

        Args:
            state: Current universe state vector

        Returns:
            State vector scaled by hierarchical factors
        """
        powers_of_20 = np.array([20**i for i in range(min(len(state), 5))])
        scaled_state = state[: len(powers_of_20)] * powers_of_20 / 20**2

        if len(state) > len(powers_of_20):
            scaled_state = np.concatenate([scaled_state, state[len(powers_of_20) :]])

        return scaled_state

    def _decompose_to_unit_fractions(self, value: float) -> list[float]:
        """
        Decompose value into unit fractions (1/n form).

        Uses greedy algorithm to represent a value as a sum of unit fractions,
        which can be useful for certain numerical representations and
        approximation schemes.

        Args:
            value: Value to decompose

        Returns:
            List of unit fractions that sum to approximately the input value
        """
        fractions = []
        remaining = abs(value)

        while remaining > 1e-10 and len(fractions) < 10:
            denominator = int(np.ceil(1.0 / remaining))
            fractions.append(1.0 / denominator)
            remaining -= 1.0 / denominator

        return fractions

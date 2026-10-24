# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-Hypothesis Optimization Engine - Parallel Solution Space Exploration.

Original implementation for Mercury Agent neural-symbolic AI archetype.

This engine explores multiple solution pathways simultaneously using ensemble
optimization and multi-dimensional state space exploration to find optimal strategies.
The approach is inspired by population-based optimization methods like genetic
algorithms and particle swarm optimization.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.utils.constants import OmniCodes
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

if TYPE_CHECKING:
    from collections.abc import Callable

# Omni-Code: Omni-Percipient Future (predictive foresight and anticipatory analysis)

# Fixed seed for the default initial universe states. The value is arbitrary
# but immutable: it makes every default-constructed engine's universes (and
# therefore its extracted features) identical across processes. Changing it
# changes what fresh engines extract; checkpoints are unaffected (they
# persist the evolved population).
_UNIVERSE_INIT_SEED: int = 9_148_267

#: Elements of the ``rows x universes x width`` difference tensor built per
#: block of rows in ``extract_features`` (about 32 MB of float64).
_FITNESS_BLOCK_ELEMENTS: int = 1 << 22
_VITALITY_HASH = OmniCodes.OMNI_PERCIPIENT.code


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
    state_vector: np.ndarray[Any, Any]
    probability_amplitude: float
    fitness: float
    state: UniverseState
    timeline: int
    parent_universe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiverseOmniEngine:
    """Multi-Hypothesis Optimization Engine - Parallel Solution Exploration.

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
        """Initialize Multi-Universe Omni Engine.

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
        # Track whether the caller injected an RNG: initial universe states
        # must be instance-independent by default (see _initialize_multiverse)
        # but an explicitly supplied generator keeps full control.
        self._injected_rng = rng is not None
        self._rng = rng or get_global_rng()

        self.universes: dict[str, Universe] = {}
        self.timeline = 0
        self.best_universe: Universe | None = None

        self._initialize_multiverse()

        logging.info("Multi-Universe Omni Engine initialized")
        logging.info(f"Number of parallel universes: {num_universes}")

    def _initialize_multiverse(self) -> None:
        """Initialize the multiverse with deterministically seeded random universes.

        The initial universe states are an *architecture constant*, not a
        per-process lottery: they are drawn from a dedicated fixed-seed
        generator (and the IDs hash only the index), so every instance hosts
        bit-identical initial universes. ``extract_features`` is a pure
        function of those states and the input, so two checkpoint-free
        engines now extract the same multiverse feature group across
        processes; checkpoints additionally persist the evolved population
        (:meth:`get_fitted_state`, ROADMAP row 16). An explicitly injected
        ``rng`` still overrides this for callers that want variation.
        """
        init_rng: Any = (
            self._rng if self._injected_rng else np.random.default_rng(_UNIVERSE_INIT_SEED)
        )
        for i in range(self.num_universes):
            universe_id = hashlib.sha3_256(f"universe_{i}".encode()).hexdigest()[:16]

            state_vector = (
                init_rng.randn(self.state_dim) * 0.5
                if self._injected_rng
                else init_rng.standard_normal(self.state_dim) * 0.5
            )
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
        """Evaluate fitness of a universe.

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
        """Combine multiple solution hypotheses into a weighted average.

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

        universe_id = hashlib.sha3_256(
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

    def converge_multiverse(
        self, fitness_function: Callable[[np.ndarray[Any, Any]], float]
    ) -> Universe:
        """Converge the multiverse to the best solution.

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
        """Extract multiverse features from data for anomaly detection.

        Each row is scored against every universe by distance (state vectors
        truncated or zero-padded to the row's width). The closest
        ``max(3, num_universes // 3)`` universes are blended by probability
        amplitude; the blend's first ten coordinates, its fitness and the
        closest universes' total amplitude are the row's features. A
        universe's fitness is a negative distance, so it is never positive and
        the stored amplitudes always stand. All rows of a block are scored
        against one universe matrix.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size, width = data.shape
        universes = list(self.universes.values())
        states = (
            np.stack([u.state_vector for u in universes])
            if universes
            else np.zeros((0, self.state_dim))
        )
        amplitudes = np.array([u.probability_amplitude for u in universes], dtype=np.float64)
        closest = min(max(3, self.num_universes // 3), len(universes))

        def align(vectors: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            if vectors.shape[1] >= width:
                return vectors[:, :width]
            return np.pad(vectors, ((0, 0), (0, width - vectors.shape[1])))

        aligned = align(states)
        block_rows = max(1, _FITNESS_BLOCK_ELEMENTS // max(1, len(universes) * width))
        features = np.zeros((batch_size, min(10, self.state_dim) + 2), dtype=np.float32)
        best_distance = np.zeros(batch_size)
        best_index = np.zeros(batch_size, dtype=np.intp)
        for start in range(0, batch_size, block_rows):
            rows = slice(start, min(start + block_rows, batch_size))
            batch = data[rows].astype(np.float64)
            distance = np.linalg.norm(aligned[None, :, :] - batch[:, None, :], axis=2)
            # Descending fitness is ascending distance; the stable sort keeps
            # universe order among ties.
            order = np.argsort(distance, axis=1, kind="stable")[:, :closest]
            amplitude = amplitudes[order]
            total_amplitude = amplitude.sum(axis=1)
            uniform = (total_amplitude < 1e-10)[:, None]
            weights = np.where(
                uniform,
                1.0 / max(1, closest),
                amplitude / np.where(uniform[:, 0], 1.0, total_amplitude)[:, None],
            )
            converged = np.einsum("bk,bkd->bd", weights, states[order])
            converged_fitness = -np.linalg.norm(align(converged) - batch, axis=1)
            features[rows] = np.concatenate(
                [converged[:, :10], converged_fitness[:, None], total_amplitude[:, None]], axis=1
            )
            if universes:
                best_index[rows] = np.argmin(distance, axis=1)
                best_distance[rows] = distance[np.arange(len(batch)), best_index[rows]]

        # ``best_universe`` keeps the first row whose best universe beats the
        # best so far, exactly as a row-by-row scan would.
        best_fitness = None if self.best_universe is None else self.best_universe.fitness
        winner: int | None = None
        for row, fitness in enumerate((-best_distance).tolist() if universes else []):
            if best_fitness is None or fitness > best_fitness:
                best_fitness, winner = fitness, row
        if winner is not None and best_fitness is not None:
            source = universes[int(best_index[winner])]
            self.best_universe = Universe(
                universe_id=source.universe_id,
                state_vector=source.state_vector.copy(),
                probability_amplitude=source.probability_amplitude,
                fitness=best_fitness,
                state=source.state,
                timeline=source.timeline,
                parent_universe=source.parent_universe,
                metadata=dict(source.metadata),
            )

        return features

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

        state_counts: dict[str, int] = {}
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
        """Apply hierarchical base-20 scaling for multi-dimensional exploration.

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
        """Decompose value into unit fractions (1/n form).

        Uses greedy algorithm to represent a value as a sum of unit fractions,
        which can be useful for certain numerical representations and
        approximation schemes.

        Args:
            value: Value to decompose

        Returns:
            List of unit fractions that sum to approximately the input value
        """
        fractions: list[float] = []
        remaining = abs(value)

        while remaining > 1e-10 and len(fractions) < 10:
            denominator = int(np.ceil(1.0 / remaining))
            fractions.append(1.0 / denominator)
            remaining -= 1.0 / denominator

        return fractions

    def get_fitted_state(self) -> dict[str, Any] | None:
        """Export the universe population for checkpoint round-tripping.

        The universes' state vectors define the feature transform
        (:meth:`extract_features` scores data against them). Default
        construction is fixed-seed, but the population evolves over the
        engine's lifetime (and callers may inject their own ``rng``), so an
        engine reloading a checkpoint must still restore this exact
        population to reproduce the saving engine's features (ROADMAP
        row 16).

        Returns:
            JSON/tensor-safe mapping of the population.
        """
        return {
            "num_universes": int(self.num_universes),
            "state_dim": int(self.state_dim),
            "universes": [
                {
                    "universe_id": str(u.universe_id),
                    "state_vector": np.asarray(u.state_vector, dtype=np.float64),
                    "probability_amplitude": float(u.probability_amplitude),
                    "fitness": float(u.fitness),
                    "state": u.state.value,
                    "timeline": int(u.timeline),
                    "parent_universe": u.parent_universe,
                }
                for u in self.universes.values()
            ],
        }

    def set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore a population produced by :meth:`get_fitted_state`."""
        self.num_universes = int(state["num_universes"])
        self.state_dim = int(state["state_dim"])
        self.universes = {}
        for entry in state["universes"]:
            universe = Universe(
                universe_id=str(entry["universe_id"]),
                state_vector=np.asarray(entry["state_vector"], dtype=np.float64),
                probability_amplitude=float(entry["probability_amplitude"]),
                fitness=float(entry["fitness"]),
                state=UniverseState(str(entry["state"])),
                timeline=int(entry["timeline"]),
                parent_universe=entry["parent_universe"],
            )
            self.universes[universe.universe_id] = universe

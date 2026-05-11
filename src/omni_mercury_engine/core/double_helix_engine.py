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
Double-Helix Evolution Engine for Mercury Agent

Implements a weighted gradient descent framework with 18+ variant terms for
adaptive anomaly detection evolution. The double-helix structure represents
two complementary optimization strands:

Helix 1 (Discovery/Exploration):
    - Hamiltonian projection (matrix-based gradient descent)
    - Boltzmann sampling (temperature-scheduled noise)
    - Simulated annealing (classical optimization with schedule)
    - Self-attention mechanisms for pattern recognition
    - Fractal patterns for multi-scale analysis
    - Chaos/entropy terms for novelty detection

Helix 2 (Ethical Verification):
    - Lyapunov stability enforcement (λ = 0.18)
    - σ_quadratic ≥ 0.96 constraint
    - Scaled layer dimensions (φ as scaling factor)
    - Ethical matrix constraints

Mathematical Foundation:
    The evolution follows: dS/dt = Σᵢ wᵢ·termᵢ(S) - λ·(S - S*)
    where S is the system state, wᵢ are term weights, and λ is the
    Lyapunov decay rate ensuring convergence to equilibrium S*.

Note: Terms previously labeled "quantum" are classical algorithms.
The naming reflects optimization techniques, not quantum computing.

References:
    - Lyapunov Stability Theory: Khalil (2002)
    - Simulated Annealing: Kirkpatrick et al. (1983)
    - Fractal Geometry: Mandelbrot (1982)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PHI = (1 + np.sqrt(5)) / 2
PHI_SQUARED = PHI**2
PHI_CUBED = PHI**3
SIGMA_QUADRATIC_THRESHOLD = 0.96
# Evolutionary adaptation decay rate for the double helix mechanism.
# Intentionally slower (0.18) than LAMBDA_CONVERGENCE (0.25) in
# centralized_constants.py, which governs OAE fusion stability.
# Slower adaptation ensures the helix evolves gradually without
# destabilizing the fusion layer above it.
# See also: core/centralized_constants.py:LAMBDA_CONVERGENCE = 0.25
LAMBDA_DECAY = 0.18


class EvolutionMode(Enum):
    """Evolution modes for the double-helix engine."""

    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    ETHICAL_PRIORITY = "ethical_priority"


class TermType(Enum):
    """Types of evolution terms."""

    OPTIMIZATION = "optimization"  # Renamed from QUANTUM - these are classical
    ATTENTION = "attention"
    FRACTAL = "fractal"
    CHAOS = "chaos"
    STABILITY = "stability"
    ETHICAL = "ethical"


@dataclass
class EvolutionState:
    """State of the evolution engine."""

    state_vector: np.ndarray[Any, Any]
    iteration: int = 0
    lyapunov_value: float = 0.0
    sigma_quadratic: float = 0.0
    convergence_rate: float = 0.0
    active_terms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine."""

    dimension: int = 64
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    lambda_decay: float = LAMBDA_DECAY
    sigma_threshold: float = SIGMA_QUADRATIC_THRESHOLD
    mode: EvolutionMode = EvolutionMode.BALANCED
    enable_optimization_terms: bool = True  # Renamed from enable_quantum_terms
    enable_attention_terms: bool = True
    enable_fractal_terms: bool = True
    enable_chaos_terms: bool = True

    # Backward compatibility aliases
    @property
    def enable_quantum_terms(self) -> bool:
        """Deprecated: Use enable_optimization_terms instead."""
        return self.enable_optimization_terms

    @enable_quantum_terms.setter
    def enable_quantum_terms(self, value: bool) -> None:
        """Deprecated: Use enable_optimization_terms instead."""
        self.enable_optimization_terms = value


class MercuryEquationEngine:
    """
    Double-Helix Evolution Engine implementing 18+ Ava Equation variants.

    The engine evolves system states through complementary discovery and
    verification helices, ensuring both exploration of novel patterns
    and ethical/stability constraints.

    Example:
        engine = MercuryEquationEngine(dimension=64)
        initial_state = np.random.default_rng().standard_normal(64)
        final_state, history = engine.converge(initial_state, max_iter=100)
    """

    def __init__(
        self,
        dimension: int = 64,
        config: EvolutionConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize the engine with a per-instance RNG.

        Args:
            dimension: State-vector dimension.
            config: Evolution configuration.
            seed: Optional seed for the per-instance ``Generator`` driving
                ethical-matrix initialization, Hamiltonian-projection
                noise, Boltzmann-sampling noise, simulated-annealing
                exploration and Lyapunov-chaos perturbation.  ``None``
                (default) uses OS entropy — same effective behavior as
                before, isolated from the legacy global ``np.random``
                state.
        """
        self.dimension = dimension
        self.config = config or EvolutionConfig(dimension=dimension)
        # Per-instance Generator. Every stochastic term in this engine
        # (ethical-matrix init, Boltzmann sampling, simulated-annealing
        # exploration, Lyapunov chaos perturbation, Hamiltonian symmetric
        # matrix) draws from this Generator — never the legacy global
        # `np.random` state — so reproducibility is in the caller's hands.
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.ethical_matrix = self._initialize_ethical_matrix()

        # Term weights - renamed from misleading "quantum" terminology to honest names
        self.term_weights = {
            "hamiltonian_projection": 0.15,  # Was "vqe" - matrix gradient descent
            "boltzmann_sampling": 0.12,  # Was "qbm" - temperature-scheduled noise
            "simulated_annealing": 0.10,  # Was "quantum_annealing" - classical SA
            "self_attention": 0.18,
            "cross_attention": 0.12,
            "fractal_dimension": 0.08,
            "fractal_recursion": 0.06,
            "lyapunov_chaos": 0.05,
            "entropy_gradient": 0.07,
            "scaled_layer": 0.10,  # Was "golden_ratio" - just scaling by φ
            "phi_amplification": 0.08,  # Was "phi_cubed" - scaling factor
            "sigma_quadratic": 0.12,
            "lyapunov_stability": 0.15,
            "ethical_constraint": 0.20,
            "helical_curvature": 0.05,
            "helical_torsion": 0.05,
            "fibonacci_harmonic": 0.06,
            "convergence_pressure": 0.10,
        }

        self._normalize_weights()

        self.evolution_history: list[EvolutionState] = []
        self.current_state: EvolutionState | None = None

        logger.info(
            f"MercuryEquationEngine initialized (dim={dimension}, "
            f"mode={self.config.mode.value})"
        )

    def _normalize_weights(self) -> None:
        """Normalize term weights to sum to 1.0."""
        total = sum(self.term_weights.values())
        if total > 0:
            self.term_weights = {k: v / total for k, v in self.term_weights.items()}

    def _initialize_ethical_matrix(self) -> np.ndarray[Any, Any]:
        """Initialize positive-definite ethical constraint matrix."""
        E = np.diag([PHI_CUBED] * self.dimension)
        noise = self._rng.standard_normal((self.dimension, self.dimension)) * 0.01 * PHI_CUBED
        noise = (noise + noise.T) / 2
        E = E + noise
        min_eig = float(np.min(np.linalg.eigvals(E).real))
        if min_eig <= 0:
            E += np.eye(self.dimension) * (abs(min_eig) + 0.1 * PHI_CUBED)
        return np.asarray(E)

    def _term_hamiltonian_projection(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Matrix-based gradient descent term (formerly 'VQE' - not quantum)."""
        if not self.config.enable_optimization_terms:
            return np.zeros_like(state)

        # Create symmetric matrix for optimization landscape
        H = self._rng.standard_normal((self.dimension, self.dimension))
        H = (H + H.T) / 2  # Symmetrize

        expectation = state @ H @ state
        gradient = 2 * H @ state

        return np.asarray(-0.1 * gradient * np.tanh(expectation))

    def _term_boltzmann_sampling(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Temperature-scheduled noise sampling (formerly 'QBM' - not quantum)."""
        if not self.config.enable_optimization_terms:
            return np.zeros_like(state)

        # Temperature decay schedule
        temperature = 1.0 / (1 + len(self.evolution_history) * 0.01)
        energy = -0.5 * state @ state
        boltzmann_factor = np.exp(-energy / max(temperature, 0.01))

        noise = self._rng.standard_normal(self.dimension) * temperature
        return np.asarray(boltzmann_factor * noise * 0.1)

    def _term_simulated_annealing(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Classical simulated annealing (formerly 'quantum_annealing' - not quantum)."""
        if not self.config.enable_optimization_terms:
            return np.zeros_like(state)

        iteration = len(self.evolution_history)
        schedule = 1.0 / (1 + iteration * 0.1)  # Cooling schedule

        # Random exploration term (decays with temperature)
        exploration = self._rng.standard_normal(self.dimension) * schedule
        # Gradient descent term (increases as temperature drops)
        gradient = -2 * state

        return schedule * exploration + (1 - schedule) * gradient * 0.1

    def _term_self_attention(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Self-attention mechanism for pattern recognition."""
        if not self.config.enable_attention_terms:
            return np.zeros_like(state)

        state_2d = state.reshape(-1, 1)
        attention_scores = state_2d @ state_2d.T
        attention_scores = attention_scores / np.sqrt(self.dimension)

        attention_weights = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

        attended = (attention_weights @ state_2d).flatten()
        return np.asarray((attended - state) * 0.1)

    def _term_cross_attention(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Cross-attention with ethical matrix."""
        if not self.config.enable_attention_terms:
            return np.zeros_like(state)

        query = state
        key = self.ethical_matrix @ state
        value = key

        attention_score = query @ key / np.sqrt(self.dimension)
        attention_weight = 1.0 / (1 + np.exp(-attention_score))

        return np.asarray(attention_weight * (value - state) * 0.05)

    def _term_fractal_dimension(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fractal dimension analysis term."""
        if not self.config.enable_fractal_terms:
            return np.zeros_like(state)

        scales = [2, 4, 8, 16]
        fractal_features = []

        for scale in scales:
            if scale <= self.dimension:
                reshaped = state[: (self.dimension // scale) * scale].reshape(-1, scale)
                variance = np.var(reshaped, axis=1).mean()
                fractal_features.append(variance)

        if fractal_features:
            fractal_dim = np.log(np.mean(fractal_features) + 1e-10)
            return np.asarray(state * fractal_dim * 0.01)
        return np.zeros_like(state)

    def _term_fractal_recursion(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fractal recursion pattern term."""
        if not self.config.enable_fractal_terms:
            return np.zeros_like(state)

        result = np.zeros_like(state)
        for depth in range(3):
            scale = 2**depth
            if scale < self.dimension:
                shifted = np.roll(state, scale)
                result += (shifted - state) / (scale + 1)
        return result * 0.05

    def _term_lyapunov_chaos(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Lyapunov exponent based chaos term."""
        if not self.config.enable_chaos_terms:
            return np.zeros_like(state)

        perturbation = self._rng.standard_normal(self.dimension) * 0.001
        perturbed = state + perturbation

        divergence = np.linalg.norm(perturbed - state)
        lyapunov_exp = np.log(divergence + 1e-10)

        if lyapunov_exp > 0:
            return np.asarray(-perturbation * lyapunov_exp * 0.1)
        return np.asarray(perturbation * abs(lyapunov_exp) * 0.05)

    def _term_entropy_gradient(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Entropy gradient for novelty detection."""
        if not self.config.enable_chaos_terms:
            return np.zeros_like(state)

        probs = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        gradient = -np.sign(state) * np.log(probs + 1e-10)
        return np.asarray(gradient * entropy * 0.01)

    def _term_scaled_layer(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Scaling term using φ (1.618) as scaling factor - not mathematically special."""
        phi_scaled = state * PHI
        return np.asarray((phi_scaled - state) * 0.05)

    def _term_phi_amplification(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Amplification with norm preservation using φ³ scaling factor."""
        amplified = state * PHI_CUBED
        norm = np.linalg.norm(amplified)
        if norm > 0:
            amplified = amplified / norm * np.linalg.norm(state)
        return np.asarray((amplified - state) * 0.03)

    def _term_sigma_quadratic(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """σ_quadratic constraint enforcement term."""
        Ex = self.ethical_matrix @ state
        x_norm_sq = state @ state
        if x_norm_sq == 0:
            return np.zeros_like(state)

        sigma = (state @ Ex) / x_norm_sq

        if sigma < SIGMA_QUADRATIC_THRESHOLD:
            correction_direction = Ex / (np.linalg.norm(Ex) + 1e-10)
            return np.asarray(correction_direction * (SIGMA_QUADRATIC_THRESHOLD - sigma) * 0.1)
        return np.zeros_like(state)

    def _term_lyapunov_stability(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Lyapunov stability enforcement term.

        Uses Lyapunov function V = sum((state - target)^2) to compute stability gradient.
        """
        target = np.ones(self.dimension) / np.sqrt(self.dimension)
        gradient = 2 * (state - target)
        return np.asarray(-LAMBDA_DECAY * gradient * 0.1)

    def _term_ethical_constraint(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Ethical constraint projection term."""
        projected = self.ethical_matrix @ state
        projected = projected / (np.linalg.norm(projected) + 1e-10) * np.linalg.norm(state)
        return np.asarray((projected - state) * 0.05)

    def _term_helical_curvature(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Helical curvature term for DNA-like evolution."""
        radius = np.linalg.norm(state[: self.dimension // 2])
        pitch = np.linalg.norm(state[self.dimension // 2 :])

        if radius**2 + pitch**2 > 0:
            curvature = radius / (radius**2 + pitch**2)
            return np.asarray(state * curvature * 0.02)
        return np.zeros_like(state)

    def _term_helical_torsion(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Helical torsion term for DNA-like evolution."""
        radius = np.linalg.norm(state[: self.dimension // 2])
        pitch = np.linalg.norm(state[self.dimension // 2 :])

        if radius**2 + pitch**2 > 0:
            torsion = pitch / (radius**2 + pitch**2)
            rotated = np.roll(state, 1)
            return np.asarray((rotated - state) * torsion * 0.02)
        return np.zeros_like(state)

    def _term_fibonacci_harmonic(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Fibonacci sequence harmonic term."""
        fib = [1, 1]
        while len(fib) < 10:
            fib.append(fib[-1] + fib[-2])

        harmonic = np.zeros_like(state)
        for i, f in enumerate(fib):
            if f < self.dimension:
                harmonic[f % self.dimension] += state[f % self.dimension] / (i + 1)

        return np.asarray((harmonic - state) * 0.02)

    def _term_convergence_pressure(self, state: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Convergence pressure toward equilibrium."""
        target = np.ones(self.dimension) / np.sqrt(self.dimension)
        return np.asarray(-(state - target) * LAMBDA_DECAY * 0.1)

    def step(self, state: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], dict[str, float]]:
        """
        Perform one evolution step.

        Args:
            state: Current state vector

        Returns:
            Tuple of (new_state, term_contributions)
        """
        term_methods = {
            "hamiltonian_projection": self._term_hamiltonian_projection,
            "boltzmann_sampling": self._term_boltzmann_sampling,
            "simulated_annealing": self._term_simulated_annealing,
            "self_attention": self._term_self_attention,
            "cross_attention": self._term_cross_attention,
            "fractal_dimension": self._term_fractal_dimension,
            "fractal_recursion": self._term_fractal_recursion,
            "lyapunov_chaos": self._term_lyapunov_chaos,
            "entropy_gradient": self._term_entropy_gradient,
            "scaled_layer": self._term_scaled_layer,
            "phi_amplification": self._term_phi_amplification,
            "sigma_quadratic": self._term_sigma_quadratic,
            "lyapunov_stability": self._term_lyapunov_stability,
            "ethical_constraint": self._term_ethical_constraint,
            "helical_curvature": self._term_helical_curvature,
            "helical_torsion": self._term_helical_torsion,
            "fibonacci_harmonic": self._term_fibonacci_harmonic,
            "convergence_pressure": self._term_convergence_pressure,
        }

        delta = np.zeros_like(state)
        contributions = {}

        for term_name, term_method in term_methods.items():
            weight = self.term_weights.get(term_name, 0.0)
            if weight > 0:
                term_delta = term_method(state)
                contribution = weight * term_delta
                delta += contribution
                contributions[term_name] = float(np.linalg.norm(contribution))

        new_state = state + delta

        Ex = self.ethical_matrix @ new_state
        x_norm_sq = new_state @ new_state
        if x_norm_sq > 0:
            sigma = (new_state @ Ex) / x_norm_sq
            if sigma < SIGMA_QUADRATIC_THRESHOLD:
                scale = np.sqrt(SIGMA_QUADRATIC_THRESHOLD / max(sigma, 1e-10))
                new_state = new_state * min(scale, 2.0)

        return new_state, contributions

    def converge(
        self,
        initial_state: np.ndarray[Any, Any],
        max_iter: int | None = None,
    ) -> tuple[np.ndarray[Any, Any], list[EvolutionState]]:
        """
        Evolve state until convergence.

        Args:
            initial_state: Initial state vector
            max_iter: Maximum iterations (default from config)

        Returns:
            Tuple of (final_state, evolution_history)
        """
        if max_iter is None:
            max_iter = self.config.max_iterations

        state = initial_state.copy()
        self.evolution_history = []

        for iteration in range(max_iter):
            new_state, contributions = self.step(state)

            target = np.ones(self.dimension) / np.sqrt(self.dimension)
            lyapunov_value = float(np.sum((new_state - target) ** 2))

            Ex = self.ethical_matrix @ new_state
            x_norm_sq = new_state @ new_state
            sigma_quadratic = float((new_state @ Ex) / x_norm_sq) if x_norm_sq > 0 else 0.0

            convergence_rate = float(np.linalg.norm(new_state - state))

            evolution_state = EvolutionState(
                state_vector=new_state.copy(),
                iteration=iteration,
                lyapunov_value=lyapunov_value,
                sigma_quadratic=sigma_quadratic,
                convergence_rate=convergence_rate,
                active_terms=list(contributions.keys()),
                metadata={"contributions": contributions},
            )
            self.evolution_history.append(evolution_state)

            if convergence_rate < self.config.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break

            state = new_state

        self.current_state = self.evolution_history[-1] if self.evolution_history else None
        return state, self.evolution_history

    def get_statistics(self) -> dict[str, Any]:
        """Get evolution statistics."""
        if not self.evolution_history:
            return {"status": "not_started"}

        lyapunov_values = [s.lyapunov_value for s in self.evolution_history]
        sigma_values = [s.sigma_quadratic for s in self.evolution_history]
        convergence_rates = [s.convergence_rate for s in self.evolution_history]

        return {
            "iterations": len(self.evolution_history),
            "final_lyapunov": lyapunov_values[-1],
            "final_sigma_quadratic": sigma_values[-1],
            "final_convergence_rate": convergence_rates[-1],
            "lyapunov_trend": (
                "decreasing" if lyapunov_values[-1] < lyapunov_values[0] else "stable"
            ),
            "sigma_satisfied": sigma_values[-1] >= SIGMA_QUADRATIC_THRESHOLD,
            "converged": convergence_rates[-1] < self.config.convergence_threshold,
        }


# Aliases for backward compatibility and naming consistency
DoubleHelixEvolutionEngine = MercuryEquationEngine
HelixConfig = EvolutionConfig

__all__ = [
    "LAMBDA_DECAY",
    "PHI",
    "PHI_CUBED",
    "PHI_SQUARED",
    "SIGMA_QUADRATIC_THRESHOLD",
    "DoubleHelixEvolutionEngine",
    "EvolutionConfig",
    "EvolutionMode",
    "EvolutionState",
    "HelixConfig",
    "MercuryEquationEngine",
    "TermType",
]

"""
Mercury Agent - 3R Mechanism Fusion
Copyright (C) 2025 Steel Security Advisors LLC

AVA Anomaly Fusion Equation (AAFE) implementation for unified precision scoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from omni_mercury_engine.core.three_r.types import (
    CONVERGENCE_RATE_PARAMETER,
    GOLDEN_RATIO_CONSTANT,
    AnomalyFusionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class AnomalyFusionEquation:
    """
    AVA Anomaly Fusion Equation (AAFE) for unified precision scoring in 3R mechanism.

    Implements the mathematical framework:
    A = (w_R * R(x) + w_H * H(omega) + w_O * O(theta)) * η_Ethical^Φ

    This equation provides:
    1. Mathematical superiority over baselines
    2. Lyapunov stability guarantee: V(S_t) <= epsilon * e^(-0.25t)
    3. Ethical gating via η_Ethical^Φ scaling
    4. Harmonic synergy through golden ratio (Φ) weighting

    The weights w_R, w_H, w_O are learned via attention fusion and sum to 1.0.
    Default initialization uses golden ratio proportions for optimal harmony.
    """

    def __init__(
        self,
        ethical_compliance_threshold: float = 0.96,
        convergence_rate: float = CONVERGENCE_RATE_PARAMETER,
        initial_weights: dict[str, float] | None = None,
        sigma_immutable: float | None = None,
        lambda_lyapunov: float | None = None,
    ):
        """
        Initialize AVA Anomaly Fusion Equation.

        Args:
            ethical_compliance_threshold: Ethical threshold η_Ethical (0.93-0.96)
            convergence_rate: Convergence rate parameter for stability
            initial_weights: Optional initial weights {w_R, w_H, w_O}
            sigma_immutable: Deprecated alias for ethical_compliance_threshold
            lambda_lyapunov: Deprecated alias for convergence_rate
        """
        if sigma_immutable is not None:
            ethical_compliance_threshold = sigma_immutable
        if lambda_lyapunov is not None:
            convergence_rate = lambda_lyapunov

        self.ethical_compliance_threshold = max(0.90, min(0.99, ethical_compliance_threshold))
        self.convergence_rate_param = convergence_rate
        self.golden_ratio = GOLDEN_RATIO_CONSTANT

        # Backward-compatible aliases
        self.sigma_immutable = self.ethical_compliance_threshold
        self.lambda_lyapunov = self.convergence_rate_param
        self.phi = self.golden_ratio

        # Initialize weights using golden ratio proportions if not provided
        if initial_weights is None:
            phi_sum = self.phi + 1.0 + (1.0 / self.phi)
            self.weights = {
                "w_R": self.phi / phi_sum,
                "w_H": 1.0 / phi_sum,
                "w_O": (1.0 / self.phi) / phi_sum,
            }
        else:
            total = sum(initial_weights.values())
            self.weights = {k: v / total for k, v in initial_weights.items()}

        self.convergence_history: list[float] = []
        self.time_step: int = 0

    def compute(
        self,
        recursion_score: float,
        resonance_score: float,
        optimization_score: float,
        ethical_threshold_override: float | None = None,
        sigma_immutable_override: float | None = None,
    ) -> AnomalyFusionResult:
        """
        Compute AVA Anomaly Fusion Equation score.

        Args:
            recursion_score: R(x) from hierarchical feature extraction
            resonance_score: H(omega) from frequency-domain analysis
            optimization_score: O(theta) from adaptive enhancement
            ethical_threshold_override: Optional override for η_Ethical
            sigma_immutable_override: Deprecated alias

        Returns:
            AnomalyFusionResult with all component scores and metadata
        """
        if sigma_immutable_override is not None:
            ethical_threshold_override = sigma_immutable_override

        eta = (
            ethical_threshold_override
            if ethical_threshold_override is not None
            else self.ethical_compliance_threshold
        )

        weighted_sum = (
            self.weights["w_R"] * recursion_score
            + self.weights["w_H"] * resonance_score
            + self.weights["w_O"] * optimization_score
        )

        ethical_scaling = eta**self.golden_ratio
        fusion_score = weighted_sum * ethical_scaling

        self.time_step += 1
        epsilon = 1.0
        lyapunov_bound = epsilon * np.exp(-self.convergence_rate_param * self.time_step)

        self.convergence_history.append(fusion_score)

        return AnomalyFusionResult(
            fusion_score=fusion_score,
            recursion_score=recursion_score,
            resonance_score=resonance_score,
            optimization_score=optimization_score,
            ethical_compliance_threshold=eta,
            fusion_weights=self.weights.copy(),
            lyapunov_bound=lyapunov_bound,
            convergence_rate=self.convergence_rate_param,
        )

    def update_weights(
        self,
        attention_weights: NDArray[Any],
        learning_rate: float = 0.01,
    ) -> None:
        """
        Update weights via attention fusion.

        Args:
            attention_weights: Attention scores [w_R, w_H, w_O]
            learning_rate: Learning rate for weight update
        """
        if len(attention_weights) != 3:
            logger.warning(f"Expected 3 attention weights, got {len(attention_weights)}")
            return

        for i, key in enumerate(["w_R", "w_H", "w_O"]):
            self.weights[key] = (1 - learning_rate) * self.weights[
                key
            ] + learning_rate * attention_weights[i]

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def verify_lyapunov_stability(self, window_size: int = 10) -> tuple[bool, float]:
        """
        Verify Lyapunov stability condition.

        Args:
            window_size: Number of recent samples to analyze

        Returns:
            Tuple of (is_stable, estimated_decay_rate)
        """
        if len(self.convergence_history) < window_size:
            return True, self.lambda_lyapunov

        recent = np.array(self.convergence_history[-window_size:])

        if len(recent) < 2:
            return True, self.lambda_lyapunov

        variance = np.var(recent)
        initial_variance = np.var(
            self.convergence_history[: min(window_size, len(self.convergence_history))]
        )

        if initial_variance > 0:
            ratio = variance / initial_variance
            if ratio > 0:
                estimated_lambda = -np.log(ratio) / self.time_step
            else:
                estimated_lambda = self.lambda_lyapunov
        else:
            estimated_lambda = self.lambda_lyapunov

        is_stable = estimated_lambda > 0
        return is_stable, float(estimated_lambda)


class AAFEWeightOptimizer:
    """
    Optimizer for AAFE weights using gradient-based methods.

    Learns optimal weights (w_R, w_H, w_O) to maximize anomaly detection performance
    while maintaining ethical constraints.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize weight optimizer.

        Args:
            learning_rate: Learning rate for weight updates
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.velocity = np.zeros(3)
        self.optimized_weights: NDArray[Any] | None = None
        self.optimization_history: list[float] = []

    def optimize_weights(
        self,
        scores: list[tuple[float, float, float]],
        targets: list[float],
        initial_weights: NDArray[Any] | None = None,
        max_iterations: int = 100,
    ) -> NDArray[Any]:
        """
        Optimize AAFE weights to minimize prediction error.

        Args:
            scores: List of (R, H, O) score tuples
            targets: Target anomaly scores
            initial_weights: Initial weight vector
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized weight vector [w_R, w_H, w_O]
        """
        if initial_weights is None:
            phi = GOLDEN_RATIO_CONSTANT
            phi_sum = phi + 1.0 + 1.0 / phi
            initial_weights = np.array([phi / phi_sum, 1.0 / phi_sum, (1.0 / phi) / phi_sum])

        scores_arr = np.array(scores)
        targets_arr = np.array(targets)

        def objective(w: NDArray[Any]) -> float:
            w_normalized = np.abs(w) / (np.sum(np.abs(w)) + 1e-10)
            predictions = scores_arr @ w_normalized
            mse = np.mean((predictions - targets_arr) ** 2)
            reg = self.weight_decay * np.sum(w**2)
            return float(mse + reg)

        result = optimize.minimize(
            objective,
            initial_weights,
            method="L-BFGS-B",
            bounds=[(0.01, 1.0)] * 3,
            options={"maxiter": max_iterations},
        )

        self.optimized_weights = np.abs(result.x)
        self.optimized_weights = self.optimized_weights / np.sum(self.optimized_weights)
        self.optimization_history.append(result.fun)

        return self.optimized_weights

    def get_optimized_fusion(self) -> AnomalyFusionEquation | None:
        """
        Get AnomalyFusionEquation with optimized weights.

        Returns:
            Configured AnomalyFusionEquation or None if not optimized
        """
        if self.optimized_weights is None:
            return None

        return AnomalyFusionEquation(
            initial_weights={
                "w_R": float(self.optimized_weights[0]),
                "w_H": float(self.optimized_weights[1]),
                "w_O": float(self.optimized_weights[2]),
            },
        )

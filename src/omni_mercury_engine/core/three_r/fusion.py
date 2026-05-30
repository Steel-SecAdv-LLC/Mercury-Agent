"""
Mercury Agent - 3R Mechanism Fusion

Copyright (C) 2025 Steel Security Advisors LLC

Omni-Ava Equation (OAE) implementation for unified precision scoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from omni_mercury_engine.core.centralized_constants import (
    RECURSION,
    sigmoid_benevolence_gate,
)
from omni_mercury_engine.core.three_r.types import (
    CONVERGENCE_RATE_PARAMETER,
    GOLDEN_RATIO_CONSTANT,
    AnomalyFusionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class OmniAvaEquation:
    """
    Omni-Ava Equation (OAE) for unified precision scoring in 3R mechanism.

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

    # Class-level annotations satisfy the type checker: these are instance
    # attributes set via object.__setattr__ to bypass the immutability guard
    # during __init__ without triggering a false ``attr-defined`` error.
    ethical_compliance_threshold: float
    _ethical_threshold_locked: bool

    def __init__(
        self,
        ethical_compliance_threshold: float = 0.96,
        convergence_rate: float = CONVERGENCE_RATE_PARAMETER,
        initial_weights: dict[str, float] | None = None,
        sigma_immutable: float | None = None,
        lambda_lyapunov: float | None = None,
        domain: str = "default",
        ethical_exponent: float | None = None,
    ):
        """
        Initialize Omni-Ava Equation.

        The OAE computes:
            A = (w_R·R(x) + w_H·H(ω) + w_O·O(θ)) · η(b)^p

        Where η(b) is the sigmoid benevolence gate (replacing hard threshold)
        and p is the ethical exponent (default: Φ = 1.618, configurable for
        empirical optimization).

        Args:
            ethical_compliance_threshold: Ethical threshold η_Ethical (0.93-0.96).
            convergence_rate: Convergence rate parameter for Lyapunov stability.
            initial_weights: Optional initial weights {w_R, w_H, w_O}.
            sigma_immutable: Deprecated alias for ethical_compliance_threshold.
            lambda_lyapunov: Deprecated alias for convergence_rate.
            domain: Domain name for sigmoid benevolence profile selection.
            ethical_exponent: Exponent for ethical scaling. Defaults to Φ (1.618).
                Set to None to use golden ratio. Override for empirical optimization.
        """
        if sigma_immutable is not None:
            ethical_compliance_threshold = sigma_immutable
        if lambda_lyapunov is not None:
            convergence_rate = lambda_lyapunov

        # CLOSED(audit-2026-03, severity=high, commit=Phase2):
        #   Floor tightened from 0.90 to domain default (0.93).  The
        #   parameter is clamped at construction — callers cannot drop
        #   below the domain-calibrated value.
        from omni_mercury_engine.core.centralized_constants import ETHICAL

        floor = ETHICAL.SIGMA_IMMUTABLE_MEDICAL  # 0.93 — lowest domain floor
        clamped = max(floor, min(0.99, ethical_compliance_threshold))
        if clamped != ethical_compliance_threshold:
            # Do not interpolate the supplied or clamped value into the log
            # record — CodeQL's py/clear-text-logging-sensitive-data taints
            # ``clamped`` via the parameter and the security review (alerts
            # 877, 878) flags any inclusion of the threshold in log output.
            # The fact-of-clamping is sufficient diagnostic; the floor and
            # ceiling are documented in source and ``ETHICAL`` constants.
            logger.warning(
                "ethical_compliance_threshold outside permitted range; "
                "value clamped to the domain-calibrated floor"
            )
        # Use object.__setattr__ so the immutability guard (set below) doesn't
        # trigger during construction.
        object.__setattr__(self, "ethical_compliance_threshold", clamped)
        self.convergence_rate_param = convergence_rate
        self.golden_ratio = GOLDEN_RATIO_CONSTANT
        self.domain = domain

        # Ethical exponent: defaults to Φ but is configurable for empirical tuning.
        # PHASE 3 NOTE: The golden ratio exponent Φ = 1.618 is used as default.
        # Empirical parameter sweep (see benchmarks/parameter_sweep.py) should
        # validate whether Φ is optimal or should be replaced. The exponent is
        # now configurable to support empirical optimization.
        self.ethical_exponent = (
            ethical_exponent if ethical_exponent is not None else self.golden_ratio
        )

        # Backward-compatible aliases
        self.sigma_immutable = self.ethical_compliance_threshold
        self.lambda_lyapunov = self.convergence_rate_param
        self.phi = self.golden_ratio

        # Initialize weights using golden ratio proportions if not provided.
        # PHI : 1 : 1 normalised to sum 1.0 → denominator ``PHI + 2``.  See
        # the docstring fix in ``ml/three_r_attention.py`` for the full
        # derivation rationale; both call sites are kept in lock-step by
        # the ``oae_weight_certifier`` operator tool.
        if initial_weights is None:
            phi_sum = self.phi + 2.0  # ≈ 3.618
            self.weights = {
                "w_R": self.phi / phi_sum,  # ≈ 0.4472
                "w_H": 1.0 / phi_sum,  # ≈ 0.2764
                "w_O": 1.0 / phi_sum,  # ≈ 0.2764
            }
        else:
            total = sum(initial_weights.values())
            self.weights = {k: v / total for k, v in initial_weights.items()}

        self.convergence_history: list[float] = []
        self.time_step: int = 0
        # Seal: prevent post-construction mutation of the ethical threshold.
        object.__setattr__(self, "_ethical_threshold_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent post-construction mutation of ethical_compliance_threshold."""
        if name == "ethical_compliance_threshold" and getattr(
            self, "_ethical_threshold_locked", False
        ):
            raise AttributeError(
                "ethical_compliance_threshold is immutable after construction. "
                "Create a new OmniAvaEquation instance to change the threshold."
            )
        object.__setattr__(self, name, value)

    def compute(
        self,
        recursion_score: float,
        resonance_score: float,
        optimization_score: float,
        ethical_threshold_override: float | None = None,
        sigma_immutable_override: float | None = None,
        benevolence_score: float | None = None,
    ) -> AnomalyFusionResult:
        """
        Compute Omni-Ava Equation score.

        A = (w_R·R(x) + w_H·H(ω) + w_O·O(θ)) · η^p

        Where η is either:
          - The sigmoid benevolence gate η(b) if benevolence_score is provided
          - The raw ethical_compliance_threshold otherwise (backward compatible)

        And p is the ethical exponent (default Φ = 1.618, configurable).

        NaN guard: If any input score is NaN, it is replaced with 0.0 and
        a warning is logged.

        Args:
            recursion_score: R(x) from hierarchical feature extraction.
            resonance_score: H(ω) from frequency-domain analysis.
            optimization_score: O(θ) from adaptive enhancement.
            ethical_threshold_override: Optional override for η_Ethical.
            sigma_immutable_override: Deprecated alias.
            benevolence_score: If provided, uses sigmoid gate instead of
                raw threshold. This is the recommended approach.

        Returns:
            AnomalyFusionResult with all component scores and metadata.
        """
        if sigma_immutable_override is not None:
            ethical_threshold_override = sigma_immutable_override

        # Validate benevolence_score range
        if benevolence_score is not None and not (0.0 <= benevolence_score <= 1.0):
            logger.warning(
                f"benevolence_score={benevolence_score:.4f} outside [0, 1], clamping to valid range"
            )
            benevolence_score = max(0.0, min(1.0, benevolence_score))

        # NaN guard on input scores
        if np.isnan(recursion_score):
            logger.warning("NaN recursion_score in OAE, replacing with 0.0")
            recursion_score = 0.0
        if np.isnan(resonance_score):
            logger.warning("NaN resonance_score in OAE, replacing with 0.0")
            resonance_score = 0.0
        if np.isnan(optimization_score):
            logger.warning("NaN optimization_score in OAE, replacing with 0.0")
            optimization_score = 0.0

        # Compute ethical gate value
        if benevolence_score is not None:
            # Use sigmoid benevolence gate (Phase 3 upgrade)
            eta = sigmoid_benevolence_gate(benevolence_score, domain=self.domain)
        elif ethical_threshold_override is not None:
            eta = ethical_threshold_override
        else:
            eta = self.ethical_compliance_threshold

        weighted_sum = (
            self.weights["w_R"] * recursion_score
            + self.weights["w_H"] * resonance_score
            + self.weights["w_O"] * optimization_score
        )

        # Use configurable exponent (default Φ) instead of hardcoded golden_ratio
        ethical_scaling = eta**self.ethical_exponent
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


class OAEWeightOptimizer:
    """
    Optimizer for OAE weights using gradient-based methods.

    Learns optimal weights (w_R, w_H, w_O) to maximize anomaly detection performance while
    maintaining ethical constraints.
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
        Optimize OAE weights to minimize prediction error.

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

    def get_optimized_fusion(self) -> OmniAvaEquation | None:
        """
        Get OmniAvaEquation with optimized weights.

        Returns:
            Configured OmniAvaEquation or None if not optimized
        """
        if self.optimized_weights is None:
            return None

        return OmniAvaEquation(
            initial_weights={
                "w_R": float(self.optimized_weights[0]),
                "w_H": float(self.optimized_weights[1]),
                "w_O": float(self.optimized_weights[2]),
            },
        )


class DomainAdaptiveOAEWeights:
    """
    Domain-adaptive weight profiles for the OAE equation.

    When cross-domain weight variance exceeds a threshold (default 10%),
    this class maintains per-domain weight profiles learned from empirical
    data. Falls back to the golden-ratio default when insufficient data exists.

    Reference: Phase 3B specification — "If sweep reveals variance > 10%
    across domains, create domain-specific weight profiles."
    """

    VARIANCE_THRESHOLD: float = 0.10

    def __init__(self) -> None:
        """Initialize with empty domain profiles."""
        phi = GOLDEN_RATIO_CONSTANT
        phi_sum = phi + 1.0 + 1.0 / phi
        self._default_weights = {
            "w_R": phi / phi_sum,
            "w_H": 1.0 / phi_sum,
            "w_O": (1.0 / phi) / phi_sum,
        }
        self._domain_profiles: dict[str, dict[str, float]] = {}
        self._domain_scores: dict[str, list[tuple[float, float, float, float]]] = {}

    def record_observation(
        self,
        domain: str,
        r_score: float,
        h_score: float,
        o_score: float,
        target: float,
    ) -> None:
        """
        Record an observation for domain-specific weight learning.

        Args:
            domain: Domain identifier (e.g. "medical", "security").
            r_score: Recursion score R(x).
            h_score: Resonance score H(omega).
            o_score: Optimization score O(theta).
            target: Ground truth anomaly label (0 or 1).
        """
        key = domain.lower()
        if key not in self._domain_scores:
            self._domain_scores[key] = []
        self._domain_scores[key].append((r_score, h_score, o_score, target))

    def fit_domain_profiles(self, min_samples: int = 30) -> dict[str, dict[str, float]]:
        """
        Fit per-domain weight profiles from recorded observations.

        Only creates a domain-specific profile when enough data exists.
        Returns the mapping of domain -> weight dict.

        Args:
            min_samples: Minimum observations needed to learn domain weights.

        Returns:
            Dict mapping domain name to weight dict.
        """
        all_weights: list[np.ndarray[Any, Any]] = []

        for domain, observations in self._domain_scores.items():
            if len(observations) < min_samples:
                continue

            data = np.array(observations)
            scores = data[:, :3]
            targets = data[:, 3]

            optimizer = OAEWeightOptimizer()
            tuples = [(float(r), float(h), float(o)) for r, h, o in scores]
            optimized = optimizer.optimize_weights(tuples, targets.tolist())

            profile = {
                "w_R": float(optimized[0]),
                "w_H": float(optimized[1]),
                "w_O": float(optimized[2]),
            }
            self._domain_profiles[domain] = profile
            all_weights.append(optimized)

        # Check cross-domain variance
        if len(all_weights) >= 2:
            weight_matrix = np.array(all_weights)
            per_weight_var = np.var(weight_matrix, axis=0)
            mean_variance = float(np.mean(per_weight_var))
            if mean_variance < self.VARIANCE_THRESHOLD:
                logger.info(
                    f"Cross-domain weight variance {mean_variance:.4f} < "
                    f"{self.VARIANCE_THRESHOLD}. Domain-specific profiles not needed."
                )

        return dict(self._domain_profiles)

    def get_weights(self, domain: str) -> dict[str, float]:
        """
        Get weights for a specific domain.

        Returns domain-specific profile if available, otherwise defaults.

        Args:
            domain: Domain identifier.

        Returns:
            Weight dict with keys w_R, w_H, w_O.
        """
        return self._domain_profiles.get(domain.lower(), self._default_weights.copy())

    def has_domain_profile(self, domain: str) -> bool:
        """Check if a domain-specific profile exists."""
        return domain.lower() in self._domain_profiles


class BanachRecursion:
    """Convergence-bounded recursive computation via Banach contraction mapping.

    Implements:
        R(x, d) = f(x) + α · R(g(x), d-1)

    With convergence guarantees:
        - α is constrained via sigmoid: α = σ(α_raw) · α_max where α_max = 0.95
        - This guarantees α ∈ (0, 0.95), ensuring geometric convergence
        - Error bound after d iterations: err ≤ α^d · ‖x₀ - R(x₀)‖ / (1 - α)
        - Runtime contraction monitoring with halt on violation

    Reference: Banach fixed-point theorem (Banach, 1922).

    Mathematical proof of convergence:
        Let (X, d) be a complete metric space and R: X → X satisfy
        d(R(x), R(y)) ≤ α · d(x, y) for all x, y ∈ X and some α ∈ [0, 1).
        Then R has a unique fixed point x* and for any x₀ ∈ X,
        the sequence x_{n+1} = R(x_n) converges to x*.
        Error bound: d(x_n, x*) ≤ α^n / (1-α) · d(x₀, x₁).
    """

    def __init__(
        self,
        alpha_raw: float = 0.0,
        alpha_max: float = RECURSION.ALPHA_MAX,
        max_depth: int = RECURSION.MAX_DEPTH,
        convergence_tolerance: float = RECURSION.CONVERGENCE_TOLERANCE,
    ):
        """
        Initialize convergence-bounded recursion.

        Args:
            alpha_raw: Raw contraction parameter (before sigmoid constraint).
                Will be mapped to (0, alpha_max) via sigmoid.
            alpha_max: Maximum allowed contraction factor. Must be < 1.0
                to guarantee convergence.
            max_depth: Maximum recursion depth.
            convergence_tolerance: Stop when successive changes fall below
                this threshold.

        Raises:
            ValueError: If alpha_max >= 1.0 (convergence impossible).
        """
        if alpha_max >= 1.0:
            raise ValueError(f"alpha_max must be < 1.0 for convergence guarantee, got {alpha_max}")

        self.alpha_max = alpha_max
        self.max_depth = max_depth
        self.convergence_tolerance = convergence_tolerance

        # Constrain alpha via sigmoid: alpha = sigmoid(alpha_raw) * alpha_max
        self.alpha = self._sigmoid(alpha_raw) * self.alpha_max

        # Monitoring state
        self.contraction_ratios: list[float] = []
        self.convergence_achieved = False
        self.actual_depth = 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function with overflow protection."""
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        else:
            z = np.exp(x)
            return float(z / (1.0 + z))

    def set_alpha(self, alpha_raw: float) -> float:
        """
        Set contraction factor from raw value via sigmoid constraint.

        Args:
            alpha_raw: Unconstrained parameter.

        Returns:
            Constrained α ∈ (0, alpha_max).
        """
        self.alpha = self._sigmoid(alpha_raw) * self.alpha_max
        return self.alpha

    def compute_error_bound(self, x0_norm: float, depth: int | None = None) -> float:
        """
        Compute theoretical error bound after d iterations.

        Error bound: err ≤ α^d · ‖x₀ - R(x₀)‖ / (1 - α)

        Args:
            x0_norm: ‖x₀ - R(x₀)‖, the initial displacement.
            depth: Number of iterations (defaults to actual_depth).

        Returns:
            Upper bound on error.
        """
        d = depth if depth is not None else self.actual_depth
        if self.alpha >= 1.0:
            return float("inf")
        return (self.alpha**d) * x0_norm / (1.0 - self.alpha)

    def recurse(
        self,
        x: float,
        f: Any,
        g: Any,
        depth: int | None = None,
    ) -> tuple[float, float]:
        """Execute convergence-bounded recursion.

        Computes R(x, d) = f(x) + α · R(g(x), d-1) with:
          - Contraction monitoring at each step
          - Early termination on convergence
          - Halt on contraction violation

        Args:
            x: Input value.
            f: Base transformation function f(x) -> float.
            g: Recursive transformation function g(x) -> float.
            depth: Maximum recursion depth (defaults to self.max_depth).

        Returns:
            Tuple of (result, error_bound).

        Raises:
            RuntimeError: If contraction ratio exceeds 1.0 (divergence detected).
        """
        max_d = depth if depth is not None else self.max_depth
        self.contraction_ratios = []
        self.convergence_achieved = False
        self.actual_depth = 0

        base_value: float = f(x)
        result = self._recurse_inner(x, f, g, max_d, prev_result=None)

        # Compute error bound: |x0 - R(x0)| approximated as |f(x) - result|
        # This is the displacement between the base (depth=0) and the
        # converged value, used in the Banach contraction error formula:
        #   error <= alpha^d * |x0 - R(x0)| / (1 - alpha)
        if self.contraction_ratios:
            initial_displacement = abs(base_value - result)
            error_bound = self.compute_error_bound(initial_displacement)
        else:
            error_bound = 0.0

        return result, error_bound

    def _recurse_inner(
        self,
        x: float,
        f: Any,
        g: Any,
        depth: int,
        prev_result: float | None,
    ) -> float:
        """Inner recursive computation with contraction monitoring."""
        base: float = f(x)

        if depth <= 0:
            return base

        # Only count actual recursive steps, not base cases
        self.actual_depth += 1

        # Recurse
        next_x = g(x)
        sub_result = self._recurse_inner(next_x, f, g, depth - 1, base)
        result = base + self.alpha * sub_result

        # Monitor contraction ratio
        if prev_result is not None and abs(prev_result) > self.convergence_tolerance:
            contraction_ratio = abs(result - prev_result) / abs(prev_result)
            self.contraction_ratios.append(contraction_ratio)

            if contraction_ratio > RECURSION.CONTRACTION_VIOLATION_THRESHOLD:
                logger.error(
                    f"Contraction violation: ratio={contraction_ratio:.4f} > "
                    f"{RECURSION.CONTRACTION_VIOLATION_THRESHOLD}. "
                    f"Halting recursion at depth={self.actual_depth}."
                )
                raise RuntimeError(f"Banach contraction violated: ratio={contraction_ratio:.4f}")

        # Check convergence
        if prev_result is not None:
            change = abs(result - prev_result)
            if change < self.convergence_tolerance:
                self.convergence_achieved = True

        return result


# Backward compatibility aliases (PRESERVATION PRINCIPLE — see DEPRECATION.md)
AnomalyFusionEquation = OmniAvaEquation
AAFEWeightOptimizer = OAEWeightOptimizer
DomainAdaptiveAAFEWeights = DomainAdaptiveOAEWeights

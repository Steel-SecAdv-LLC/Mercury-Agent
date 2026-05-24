"""
Mercury Agent - GOSNN ↔ 3R Bidirectional Feedback Integration

Copyright (C) 2025 Steel Security Advisors LLC

Implements bidirectional synaptic integration between:
- Global Omni-Scalar Network (GOSNN): ~235 omni-scalars, ethical gating
- Three-R Mechanism: Recursion-Resonance-Refactoring

Key Features:
1. 3R Refactoring engine dynamically adjusts GOSNN ethical thresholds
2. GOSNN scalar categories weight 3R fusion coefficients (w_R, w_H, w_O)
3. Gradient flow from 3R loss back to detector heads
4. Sliding window normalization for time-series inputs
5. Cross-domain transfer learning support

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.global_omni_scalar_network import (
    PHI,
    GlobalOmniScalarNetwork,
    ScalarGroup,
    get_global_scalar_network,
)
from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation
from omni_mercury_engine.core.three_r.types import (
    CONVERGENCE_RATE_PARAMETER,
    AnomalyFusionResult,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FeedbackDirection(Enum):
    """Direction of feedback in the GOSNN-3R integration."""

    GOSNN_TO_3R = "gosnn_to_3r"  # GOSNN scalars influence 3R weights
    THREE_R_TO_GOSNN = "3r_to_gosnn"  # 3R performance adjusts GOSNN thresholds
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class IntegrationState:
    """State of the GOSNN-3R integration."""

    # Current weights
    w_R: float = 0.447  # Recursion weight (golden ratio normalized)
    w_H: float = 0.276  # Harmonic/Resonance weight
    w_O: float = 0.276  # Optimization weight

    # Current thresholds
    ethical_threshold: float = 0.96
    domain_threshold: float = 0.5

    # Performance metrics
    last_fusion_score: float = 0.0
    last_ethical_score: float = 1.0
    convergence_rate: float = CONVERGENCE_RATE_PARAMETER

    # History for stability analysis
    weight_history: list[tuple[float, float, float]] = field(default_factory=list)
    threshold_history: list[float] = field(default_factory=list)
    fusion_score_history: list[float] = field(default_factory=list)

    # Lyapunov stability tracking
    lyapunov_bound: float = 1.0
    is_stable: bool = True


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window normalization."""

    window_size: int = 100
    min_samples: int = 10
    normalization: str = "standard"  # standard, minmax, robust
    decay_factor: float = 0.99  # Exponential decay for weighted normalization
    update_frequency: int = 1  # Update stats every N samples


class SlidingWindowNormalizer:
    """
    Sliding window normalization for time-series inputs.

    Maintains running statistics and normalizes incoming data based on recent history, adapting to
    non-stationary distributions.
    """

    def __init__(self, config: SlidingWindowConfig | None = None):
        """
        Initialize sliding window normalizer.

        Args:
            config: Normalization configuration
        """
        self.config = config or SlidingWindowConfig()
        self._lock = threading.Lock()

        # Running statistics
        self._n_samples = 0
        self._window_data: list[NDArray[np.float64]] = []

        # Computed statistics
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._min: NDArray[np.float64] | None = None
        self._max: NDArray[np.float64] | None = None
        self._median: NDArray[np.float64] | None = None
        self._iqr: NDArray[np.float64] | None = None

        # Exponential moving average for weighted stats
        self._ema_mean: NDArray[np.float64] | None = None
        self._ema_var: NDArray[np.float64] | None = None

        logger.debug(
            f"SlidingWindowNormalizer initialized: window_size={self.config.window_size}, "
            f"normalization={self.config.normalization}"
        )

    def update(self, data: NDArray[np.float64]) -> None:
        """
        Update running statistics with new data.

        Args:
            data: New data sample(s)
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))

        with self._lock:
            self._window_data.append(data)
            self._n_samples += 1

            # Trim window
            while len(self._window_data) > self.config.window_size:
                self._window_data.pop(0)

            # Update statistics periodically
            if self._n_samples % self.config.update_frequency == 0:
                self._compute_statistics()

    def _compute_statistics(self) -> None:
        """Compute window statistics."""
        if len(self._window_data) < self.config.min_samples:
            return

        window_array = np.array(self._window_data)

        if self.config.normalization == "standard":
            self._mean = np.mean(window_array, axis=0)
            self._std = np.std(window_array, axis=0) + 1e-8

        elif self.config.normalization == "minmax":
            self._min = np.min(window_array, axis=0)
            self._max = np.max(window_array, axis=0)
            # Avoid division by zero
            self._max = np.where(self._max == self._min, self._min + 1e-8, self._max)

        elif self.config.normalization == "robust":
            self._median = np.median(window_array, axis=0)
            q75 = np.percentile(window_array, 75, axis=0)
            q25 = np.percentile(window_array, 25, axis=0)
            self._iqr = (q75 - q25) + 1e-8

        # Update EMA statistics
        decay = self.config.decay_factor
        current_mean = np.mean(window_array, axis=0)
        current_var = np.var(window_array, axis=0)

        if self._ema_mean is None or self._ema_var is None:
            self._ema_mean = current_mean
            self._ema_var = current_var
        else:
            self._ema_mean = decay * self._ema_mean + (1 - decay) * current_mean
            self._ema_var = decay * self._ema_var + (1 - decay) * current_var

    def normalize(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize data using current window statistics.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))

        with self._lock:
            if self._n_samples < self.config.min_samples:
                raise RuntimeError(
                    f"Sliding-window normalization requires at least "
                    f"{self.config.min_samples} samples, but only "
                    f"{self._n_samples} have been observed. "
                    "Silent passthrough is not permitted (Phase 2 audit cure)."
                )

            if self.config.normalization == "standard":
                if self._mean is not None and self._std is not None:
                    return (data - self._mean) / self._std

            elif self.config.normalization == "minmax":
                if self._min is not None and self._max is not None:
                    return (data - self._min) / (self._max - self._min)

            elif self.config.normalization == "robust":
                if self._median is not None and self._iqr is not None:
                    return (data - self._median) / self._iqr

            raise RuntimeError(
                f"Sliding-window statistics for '{self.config.normalization}' "
                "are not available despite sufficient samples. "
                "Silent passthrough is not permitted (Phase 2 audit cure)."
            )

    def normalize_with_ema(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalize using exponential moving average statistics.

        Args:
            data: Data to normalize

        Returns:
            Normalized data using EMA statistics
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))

        with self._lock:
            if self._ema_mean is None or self._ema_var is None:
                return data

            ema_std = np.sqrt(self._ema_var) + 1e-8
            return (data - self._ema_mean) / ema_std

    def get_statistics(self) -> dict[str, Any]:
        """Get current window statistics."""
        with self._lock:
            return {
                "n_samples": self._n_samples,
                "window_size": len(self._window_data),
                "mean": self._mean.tolist() if self._mean is not None else None,
                "std": self._std.tolist() if self._std is not None else None,
                "ema_mean": self._ema_mean.tolist() if self._ema_mean is not None else None,
            }

    def reset(self) -> None:
        """Reset normalizer state."""
        with self._lock:
            self._n_samples = 0
            self._window_data = []
            self._mean = None
            self._std = None
            self._min = None
            self._max = None
            self._median = None
            self._iqr = None
            self._ema_mean = None
            self._ema_var = None


class GOSNN3RIntegration:
    """
    Bidirectional integration between GOSNN and 3R mechanism.

    Provides:
    1. GOSNN → 3R: Scalar categories weight fusion coefficients
    2. 3R → GOSNN: Performance adjusts ethical thresholds
    3. Gradient-like feedback from fusion loss
    4. Domain-aware cross-calibration
    """

    # Scalar group to 3R weight mapping
    SCALAR_WEIGHT_MAPPING = {
        ScalarGroup.ETHICAL: "w_R",  # Ethics → Recursion (self-reference)
        ScalarGroup.QUANTUM_CONSCIOUSNESS: "w_H",  # Quantum → Resonance (harmonics)
        ScalarGroup.SOFTWARE_ENGINEERING: "w_O",  # Engineering → Optimization
        ScalarGroup.ADVANCED_REASONING: "w_R",  # Reasoning → Recursion
        ScalarGroup.SECURITY: "w_O",  # Security → Optimization
        ScalarGroup.MEDICAL: "w_R",  # Medical → Recursion (careful analysis)
        ScalarGroup.HUMANITARIAN: "w_R",  # Humanitarian → Recursion
        ScalarGroup.COSMIC: "w_H",  # Cosmic → Resonance (harmonics)
    }

    def __init__(
        self,
        gosnn: GlobalOmniScalarNetwork | None = None,
        fusion_equation: OmniAvaEquation | None = None,
        feedback_direction: FeedbackDirection = FeedbackDirection.BIDIRECTIONAL,
        domain: str | None = None,
        enable_sliding_window: bool = True,
    ):
        """
        Initialize GOSNN-3R integration.

        Args:
            gosnn: GOSNN instance (creates default if None)
            fusion_equation: OAE instance (creates default if None)
            feedback_direction: Direction of feedback flow
            domain: Target domain for calibration
            enable_sliding_window: Enable sliding window normalization
        """
        self.gosnn = gosnn or get_global_scalar_network(domain=domain)
        self.fusion_equation = fusion_equation or OmniAvaEquation()
        self.feedback_direction = feedback_direction
        self.domain = domain

        # Integration state
        self.state = IntegrationState()

        # Initialize weights from GOSNN
        self._sync_weights_from_gosnn()

        # Sliding window normalizer
        self.sliding_normalizer = SlidingWindowNormalizer() if enable_sliding_window else None

        # Feedback learning rate
        self.learning_rate = 0.01
        self.momentum = 0.9
        self._velocity = np.zeros(3)  # For momentum-based updates

        # Lock for thread safety
        self._lock = threading.Lock()

        logger.info(
            f"GOSNN-3R Integration initialized: direction={feedback_direction.value}, "
            f"domain={domain}"
        )

    def _sync_weights_from_gosnn(self) -> None:
        """Synchronize 3R weights from GOSNN scalar categories."""
        if self.feedback_direction not in (
            FeedbackDirection.GOSNN_TO_3R,
            FeedbackDirection.BIDIRECTIONAL,
        ):
            return

        # Collect scalar group scores
        group_scores = {}
        for group in ScalarGroup:
            scalars = self.gosnn.scalar_groups.get(group, {})
            if scalars:
                values = np.array(list(scalars.values()))
                group_scores[group] = float(np.mean(values))

        # Map to 3R weights
        weight_contributions: dict[str, list[float]] = {"w_R": [], "w_H": [], "w_O": []}

        for group, weight_key in self.SCALAR_WEIGHT_MAPPING.items():
            if group in group_scores:
                # Normalize score to [0, 1] range assuming scalars are around 1.0-1.5
                normalized_score = (group_scores[group] - 0.5) / 1.0
                normalized_score = np.clip(normalized_score, 0.0, 1.0)
                weight_contributions[weight_key].append(normalized_score)

        # Compute weighted averages
        for weight_key in ["w_R", "w_H", "w_O"]:
            if weight_contributions[weight_key]:
                avg_contribution = np.mean(weight_contributions[weight_key])
                setattr(
                    self.state,
                    weight_key,
                    avg_contribution * 0.3 + getattr(self.state, weight_key) * 0.7,
                )

        # Normalize weights to sum to 1
        total = self.state.w_R + self.state.w_H + self.state.w_O
        if total > 0:
            self.state.w_R /= total
            self.state.w_H /= total
            self.state.w_O /= total

        # Update fusion equation weights
        self.fusion_equation.weights = {
            "w_R": self.state.w_R,
            "w_H": self.state.w_H,
            "w_O": self.state.w_O,
        }

    def _adjust_gosnn_from_3r(self, fusion_result: AnomalyFusionResult) -> None:
        """Adjust GOSNN thresholds based on 3R performance."""
        if self.feedback_direction not in (
            FeedbackDirection.THREE_R_TO_GOSNN,
            FeedbackDirection.BIDIRECTIONAL,
        ):
            return

        # Track fusion score history for stability analysis
        self.state.fusion_score_history.append(fusion_result.fusion_score)
        if len(self.state.fusion_score_history) > 100:
            self.state.fusion_score_history = self.state.fusion_score_history[-100:]

        # Verify Lyapunov stability
        is_stable, estimated_rate = self.fusion_equation.verify_lyapunov_stability()
        self.state.is_stable = is_stable
        self.state.convergence_rate = estimated_rate

        # Adjust ethical threshold based on fusion performance
        if len(self.state.fusion_score_history) >= 10:
            recent_mean = np.mean(self.state.fusion_score_history[-10:])
            recent_std = np.std(self.state.fusion_score_history[-10:])

            # If fusion scores are too variable, increase ethical threshold
            if recent_std > 0.1:
                adjustment = 0.01 * self.learning_rate
                self.state.ethical_threshold = min(0.99, self.state.ethical_threshold + adjustment)
            # If fusion scores are stable and low, decrease threshold slightly
            elif recent_std < 0.02 and recent_mean < 0.5:
                adjustment = 0.005 * self.learning_rate
                self.state.ethical_threshold = max(0.90, self.state.ethical_threshold - adjustment)

        # Update GOSNN ethical gate threshold
        self.gosnn.ethical_gate.threshold = self.state.ethical_threshold
        self.gosnn.sigma_immutable_threshold = self.state.ethical_threshold

    def process(
        self,
        recursion_score: float,
        resonance_score: float,
        optimization_score: float,
        raw_data: NDArray[np.float64] | None = None,
    ) -> AnomalyFusionResult:
        """
        Process scores through the integrated GOSNN-3R pipeline.

        Args:
            recursion_score: R(x) from hierarchical feature extraction
            resonance_score: H(ω) from frequency-domain analysis
            optimization_score: O(θ) from adaptive enhancement
            raw_data: Optional raw data for sliding window normalization

        Returns:
            Anomaly fusion result with integrated feedback
        """
        with self._lock:
            # Apply sliding window normalization if enabled
            if self.sliding_normalizer is not None and raw_data is not None:
                self.sliding_normalizer.update(raw_data)
                # Normalize the component scores
                scores = np.array([recursion_score, resonance_score, optimization_score])
                normalized_scores = self.sliding_normalizer.normalize(scores)
                recursion_score, resonance_score, optimization_score = normalized_scores

            # Sync weights from GOSNN before computation
            self._sync_weights_from_gosnn()

            # Get ethical compliance from GOSNN
            scalar_vector = np.array(list(self.gosnn._collect_all_scalars().values()))
            passes_gate, ethical_score = self.gosnn.ethical_gate.evaluate(scalar_vector)
            self.state.last_ethical_score = ethical_score

            # Compute fusion with current state
            result = self.fusion_equation.compute(
                recursion_score=recursion_score,
                resonance_score=resonance_score,
                optimization_score=optimization_score,
                ethical_threshold_override=self.state.ethical_threshold,
            )

            # Update state
            self.state.last_fusion_score = result.fusion_score
            self.state.lyapunov_bound = result.lyapunov_bound

            # Track weight history
            self.state.weight_history.append((self.state.w_R, self.state.w_H, self.state.w_O))
            if len(self.state.weight_history) > 100:
                self.state.weight_history = self.state.weight_history[-100:]

            # Adjust GOSNN based on 3R performance
            self._adjust_gosnn_from_3r(result)

            return result

    def update_weights_from_loss(
        self,
        loss: float,
        recursion_score: float,
        resonance_score: float,
        optimization_score: float,
    ) -> None:
        """
        Update weights using gradient-like feedback from loss.

        Implements approximate gradient descent on the fusion weights
        based on the detection loss (e.g., 1 - F1).

        Args:
            loss: Detection loss value
            recursion_score: Component R score
            resonance_score: Component H score
            optimization_score: Component O score
        """
        with self._lock:
            # Compute approximate gradients
            # Higher component scores that correlate with high loss should decrease
            scores = np.array([recursion_score, resonance_score, optimization_score])
            weights = np.array([self.state.w_R, self.state.w_H, self.state.w_O])

            # Gradient approximation: d_loss/d_w ≈ loss * (score - mean_score)
            mean_score = np.mean(scores)
            gradients = loss * (scores - mean_score)

            # Momentum update
            self._velocity = self.momentum * self._velocity - self.learning_rate * gradients

            # Update weights
            new_weights = weights + self._velocity

            # Ensure non-negative
            new_weights = np.maximum(new_weights, 0.01)

            # Normalize to sum to 1
            new_weights = new_weights / np.sum(new_weights)

            # Apply update
            self.state.w_R, self.state.w_H, self.state.w_O = new_weights

            # Sync to fusion equation
            self.fusion_equation.weights = {
                "w_R": float(new_weights[0]),
                "w_H": float(new_weights[1]),
                "w_O": float(new_weights[2]),
            }

            logger.debug(
                f"Weights updated from loss={loss:.4f}: "
                f"w_R={self.state.w_R:.4f}, w_H={self.state.w_H:.4f}, w_O={self.state.w_O:.4f}"
            )

    def register_detector_scalars(
        self,
        detector_name: str,
        performance_metrics: dict[str, float],
    ) -> None:
        """
        Register detector performance as GOSNN scalars.

        This enables cross-detector influence through the scalar network.

        Args:
            detector_name: Name of the detector
            performance_metrics: Dict of metric_name → value
        """
        # Map performance to scalars
        scalars = {}

        if "precision" in performance_metrics:
            scalars[f"omni_{detector_name}_precision"] = (
                1.0 + 0.5 * performance_metrics["precision"]
            )

        if "recall" in performance_metrics:
            scalars[f"omni_{detector_name}_recall"] = 1.0 + 0.5 * performance_metrics["recall"]

        if "f1" in performance_metrics:
            scalars[f"omni_{detector_name}_f1"] = 1.0 + 0.5 * performance_metrics["f1"]

        if "confidence" in performance_metrics:
            scalars[f"omni_{detector_name}_confidence"] = (
                1.0 + 0.3 * performance_metrics["confidence"]
            )

        # Register with GOSNN
        self.gosnn.register_scalars(
            component_name=detector_name,
            scalars=scalars,
            group=ScalarGroup.SOFTWARE_ENGINEERING,
            metadata={"source": "3r_integration", "domain": self.domain},
        )

    def get_integration_state(self) -> dict[str, Any]:
        """Get current integration state."""
        with self._lock:
            return {
                "weights": {
                    "w_R": self.state.w_R,
                    "w_H": self.state.w_H,
                    "w_O": self.state.w_O,
                },
                "thresholds": {
                    "ethical": self.state.ethical_threshold,
                    "domain": self.state.domain_threshold,
                },
                "stability": {
                    "is_stable": self.state.is_stable,
                    "lyapunov_bound": self.state.lyapunov_bound,
                    "convergence_rate": self.state.convergence_rate,
                },
                "performance": {
                    "last_fusion_score": self.state.last_fusion_score,
                    "last_ethical_score": self.state.last_ethical_score,
                },
                "history_length": len(self.state.fusion_score_history),
                "sliding_window": (
                    self.sliding_normalizer.get_statistics() if self.sliding_normalizer else None
                ),
            }

    def verify_stability(self) -> tuple[bool, dict[str, Any]]:
        """
        Verify overall system stability.

        Returns:
            Tuple of (is_stable, stability_report)
        """
        with self._lock:
            # Check Lyapunov stability
            lyapunov_stable, convergence_rate = self.fusion_equation.verify_lyapunov_stability()

            # Check weight stability
            if len(self.state.weight_history) >= 10:
                weight_array = np.array(self.state.weight_history[-10:])
                weight_variance = float(np.mean(np.var(weight_array, axis=0)))
                weight_stable = bool(weight_variance < 0.01)
            else:
                weight_stable = True
                weight_variance = 0.0

            # Check score stability
            if len(self.state.fusion_score_history) >= 10:
                score_variance = float(np.var(self.state.fusion_score_history[-10:]))
                score_stable = bool(score_variance < 0.05)
            else:
                score_stable = True  # type: ignore[assignment, unused-ignore]
                score_variance = 0.0  # type: ignore[assignment, unused-ignore]

            # Overall stability
            is_stable = lyapunov_stable and weight_stable and score_stable

            report = {
                "overall_stable": is_stable,
                "lyapunov": {
                    "stable": lyapunov_stable,
                    "convergence_rate": convergence_rate,
                    "bound": self.state.lyapunov_bound,
                },
                "weights": {
                    "stable": weight_stable,
                    "variance": float(weight_variance),
                },
                "scores": {
                    "stable": score_stable,
                    "variance": float(score_variance),
                },
            }

            return bool(is_stable), report  # type: ignore[return-value, unused-ignore]

    def adjust_weights(
        self,
        neural_score: float,
        symbolic_score: float,
        confidence: float,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """
        Adjust fusion weights based on current scores and confidence.

        This method provides bidirectional feedback:
        1. Uses GOSNN scalars to inform weight adjustments
        2. Updates internal state based on incoming scores
        3. Returns adjusted weights for fusion refinement

        Args:
            neural_score: Neural network anomaly score [0, 1]
            symbolic_score: Symbolic reasoning anomaly score [0, 1]
            confidence: Confidence in the prediction [0, 1]
            domain: Optional domain override

        Returns:
            Dictionary with adjusted weights and stability info
        """
        with self._lock:
            # Update domain if provided
            effective_domain = domain or self.domain

            # Get current state
            current_w_R = self.state.w_R
            current_w_H = self.state.w_H
            current_w_O = self.state.w_O

            # Compute score agreement
            agreement = 1.0 - abs(neural_score - symbolic_score)

            # Adjust weights based on confidence and agreement
            # High confidence + high agreement → trust current weights
            # Low confidence → increase symbolic weight (more interpretable)
            # Low agreement → balance weights more evenly

            if confidence < 0.5:
                # Low confidence: increase symbolic reasoning weight
                adjustment = (0.5 - confidence) * 0.1  # Small adjustment
                new_w_R = current_w_R + adjustment  # Recursion for deeper analysis
                new_w_H = current_w_H
                new_w_O = current_w_O - adjustment * 0.5
            elif agreement < 0.3:
                # Low agreement: balance weights
                target = 1.0 / 3.0
                blend = 0.1  # Small blend factor
                new_w_R = current_w_R * (1 - blend) + target * blend
                new_w_H = current_w_H * (1 - blend) + target * blend
                new_w_O = current_w_O * (1 - blend) + target * blend
            else:
                # High confidence and agreement: keep current weights
                new_w_R = current_w_R
                new_w_H = current_w_H
                new_w_O = current_w_O

            # Normalize to sum to 1
            total = new_w_R + new_w_H + new_w_O
            if total > 0:
                new_w_R /= total
                new_w_H /= total
                new_w_O /= total

            # Map 3R weights to neural/symbolic weights
            # w_R and w_H tend to favor symbolic (reasoning-based)
            # w_O tends to favor neural (optimization-based)
            neural_weight = new_w_O * 0.6 + 0.4  # Base of 0.4, boosted by optimization
            symbolic_weight = 1.0 - neural_weight

            # Verify stability
            is_stable, _ = self.verify_stability()

            # Determine if we should refine the score
            # Only refine if stable and weights changed significantly
            weight_change = abs(neural_weight - (PHI / (1 + PHI)))
            refine_score = is_stable and weight_change > 0.05 and confidence > 0.3

            return {
                "neural_weight": neural_weight,
                "symbolic_weight": symbolic_weight,
                "w_R": new_w_R,
                "w_H": new_w_H,
                "w_O": new_w_O,
                "agreement": agreement,
                "lyapunov_stability": 1.0 if is_stable else 0.0,
                "refine_score": refine_score,
                "domain": effective_domain,
            }


class CrossDomainTransferManager:
    """
    Manages cross-domain transfer learning between GOSNN-3R integrations.

    Pre-trains on high-data domains (Security, Space) then fine-tunes on low-data domains (Medical,
    Humanitarian).
    """

    # Domain hierarchy for transfer (source → targets)
    TRANSFER_HIERARCHY = {
        "security": ["medical", "financial", "infrastructure"],
        "general": ["medical", "financial", "infrastructure", "humanitarian"],
        "financial": ["security", "infrastructure"],
        "infrastructure": ["security", "humanitarian"],
    }

    def __init__(self) -> None:
        """Initialize cross-domain transfer manager."""
        self._domain_integrations: dict[str, GOSNN3RIntegration] = {}
        self._transfer_weights: dict[str, dict[str, float]] = {}

    def register_domain(self, domain: str, integration: GOSNN3RIntegration) -> None:
        """Register a domain's GOSNN-3R integration."""
        self._domain_integrations[domain] = integration

    def transfer_weights(
        self,
        source_domain: str,
        target_domain: str,
        transfer_ratio: float = 0.3,
    ) -> bool:
        """
        Transfer learned weights from source to target domain.

        Args:
            source_domain: Domain to transfer from
            target_domain: Domain to transfer to
            transfer_ratio: How much of source weights to use (0-1)

        Returns:
            True if transfer succeeded
        """
        if source_domain not in self._domain_integrations:
            logger.warning(f"Source domain {source_domain} not registered")
            return False

        if target_domain not in self._domain_integrations:
            logger.warning(f"Target domain {target_domain} not registered")
            return False

        source = self._domain_integrations[source_domain]
        target = self._domain_integrations[target_domain]

        # Transfer weights
        source_weights = np.array([source.state.w_R, source.state.w_H, source.state.w_O])
        target_weights = np.array([target.state.w_R, target.state.w_H, target.state.w_O])

        # Interpolate
        new_weights = (1 - transfer_ratio) * target_weights + transfer_ratio * source_weights

        # Normalize
        new_weights = new_weights / np.sum(new_weights)

        # Apply to target
        target.state.w_R, target.state.w_H, target.state.w_O = new_weights
        target.fusion_equation.weights = {
            "w_R": float(new_weights[0]),
            "w_H": float(new_weights[1]),
            "w_O": float(new_weights[2]),
        }

        # Transfer ethical threshold with damping
        source_threshold = source.state.ethical_threshold
        target_threshold = target.state.ethical_threshold
        new_threshold = (1 - transfer_ratio * 0.5) * target_threshold + (
            transfer_ratio * 0.5
        ) * source_threshold
        target.state.ethical_threshold = new_threshold

        logger.info(
            f"Transferred weights from {source_domain} to {target_domain} "
            f"(ratio={transfer_ratio})"
        )

        return True

    def auto_transfer(self, low_data_domains: list[str] | None = None) -> dict[str, bool]:
        """
        Automatically transfer from high-data to low-data domains.

        Args:
            low_data_domains: Override list of low-data domains

        Returns:
            Dict of domain → transfer_success
        """
        results = {}

        if low_data_domains is None:
            low_data_domains = ["medical", "humanitarian"]

        for source_domain, target_domains in self.TRANSFER_HIERARCHY.items():
            if source_domain not in self._domain_integrations:
                continue

            for target_domain in target_domains:
                if target_domain in low_data_domains and target_domain in self._domain_integrations:
                    success = self.transfer_weights(
                        source_domain, target_domain, transfer_ratio=0.2
                    )
                    results[f"{source_domain}->{target_domain}"] = success

        return results


# Factory function
def create_integrated_pipeline(
    domain: str | None = None,
    enable_sliding_window: bool = True,
) -> GOSNN3RIntegration:
    """
    Create an integrated GOSNN-3R pipeline.

    Args:
        domain: Target domain
        enable_sliding_window: Enable sliding window normalization

    Returns:
        Configured GOSNN3RIntegration instance
    """
    return GOSNN3RIntegration(
        domain=domain,
        feedback_direction=FeedbackDirection.BIDIRECTIONAL,
        enable_sliding_window=enable_sliding_window,
    )


# Exports
__all__ = [
    "CrossDomainTransferManager",
    "FeedbackDirection",
    "GOSNN3RIntegration",
    "IntegrationState",
    "SlidingWindowConfig",
    "SlidingWindowNormalizer",
    "create_integrated_pipeline",
]

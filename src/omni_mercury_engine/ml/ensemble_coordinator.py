"""
Mercury Agent - Advanced Ensemble Coordinator
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Advanced ensemble coordination for hybrid anomaly detection including:
- Adaptive weight learning based on performance feedback
- Cascading detection (efficient -> accurate pipeline)
- Meta-learning for detector selection
- Stacking and blending ensembles
- Dynamic detector activation based on data characteristics
- Uncertainty-aware ensemble fusion
- Cross-validation based weight optimization
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class EnsembleStrategy(StrEnum):
    """Available ensemble strategies."""

    VOTING = "voting"  # Simple majority/weighted voting
    AVERAGING = "averaging"  # Score averaging
    STACKING = "stacking"  # Meta-learner on top
    CASCADING = "cascading"  # Sequential filtering
    BOOSTING = "boosting"  # Boosting-style combination
    DYNAMIC = "dynamic"  # Dynamic strategy selection
    MIXTURE_OF_EXPERTS = "mixture_of_experts"  # Gated mixture


class DetectorState(StrEnum):
    """State of a detector in the ensemble."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    WARMING_UP = "warming_up"
    COOLDOWN = "cooldown"
    FAILED = "failed"


class Detector(Protocol):
    """Protocol for anomaly detectors."""

    def fit(self, data: NDArray[np.float64]) -> Any:
        """Fit the detector."""
        ...

    def detect(self, data: NDArray[np.float64]) -> dict[str, Any]:
        """Detect anomalies."""
        ...


@dataclass
class DetectorMetrics:
    """Performance metrics for a detector."""

    precision: float = 0.5
    recall: float = 0.5
    f1_score: float = 0.5
    auc_roc: float = 0.5
    latency_ms: float = 100.0
    memory_mb: float = 100.0
    samples_processed: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    last_updated: float = 0.0

    def update_from_feedback(
        self,
        predicted: bool,
        actual: bool,
        latency_ms: float,
    ) -> None:
        """Update metrics from feedback."""
        if predicted and actual:
            self.true_positives += 1
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

        self.samples_processed += 1

        # Update derived metrics
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

        # EMA for latency
        alpha = 0.1
        self.latency_ms = alpha * latency_ms + (1 - alpha) * self.latency_ms
        self.last_updated = time.time()


@dataclass
class DetectorEntry:
    """Entry for a detector in the ensemble."""

    name: str
    detector: Detector
    weight: float = 1.0
    state: DetectorState = DetectorState.ACTIVE
    metrics: DetectorMetrics = field(default_factory=DetectorMetrics)
    domain_affinity: dict[str, float] = field(default_factory=dict)
    min_samples: int = 10
    activation_threshold: float = 0.3  # Minimum score to activate


@dataclass
class EnsembleResult:
    """Result from ensemble detection."""

    is_anomaly: NDArray[np.bool_]
    scores: NDArray[np.float64]
    confidence: NDArray[np.float64]
    detector_contributions: dict[str, NDArray[np.float64]]
    active_detectors: list[str]
    strategy_used: str
    ensemble_weights: dict[str, float]
    meta_info: dict[str, Any] = field(default_factory=dict)


class WeightOptimizer(ABC):
    """Abstract base class for weight optimization."""

    @abstractmethod
    def optimize(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        labels: NDArray[np.int64] | None,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Optimize detector weights."""
        pass


class BayesianWeightOptimizer(WeightOptimizer):
    """
    Bayesian weight optimization using Thompson Sampling.

    Maintains Beta distributions for each detector's success rate
    and samples weights proportionally.
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        exploration_bonus: float = 0.1,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            prior_alpha: Prior alpha for Beta distribution
            prior_beta: Prior beta for Beta distribution
            exploration_bonus: Bonus for exploration
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploration_bonus = exploration_bonus
        self._alphas: dict[str, float] = {}
        self._betas: dict[str, float] = {}

    def optimize(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        labels: NDArray[np.int64] | None,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Optimize weights using Thompson Sampling."""
        # Initialize distributions for new detectors
        for name in detector_scores:
            if name not in self._alphas:
                self._alphas[name] = self.prior_alpha
                self._betas[name] = self.prior_beta

        # Update distributions if labels available
        if labels is not None:
            for name, scores in detector_scores.items():
                predictions = scores > 0.5
                # Count successes (correct predictions)
                successes = np.sum(predictions == (labels == 1))
                failures = len(labels) - successes

                self._alphas[name] += successes * 0.1
                self._betas[name] += failures * 0.1

        # Sample from posterior distributions
        weights = {}
        for name in detector_scores:
            sampled = np.random.beta(self._alphas[name], self._betas[name])
            # Add exploration bonus for less-used detectors
            uncertainty = np.sqrt(
                self._alphas[name]
                * self._betas[name]
                / (
                    (self._alphas[name] + self._betas[name]) ** 2
                    * (self._alphas[name] + self._betas[name] + 1)
                )
            )
            weights[name] = sampled + self.exploration_bonus * uncertainty

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class GradientWeightOptimizer(WeightOptimizer):
    """
    Gradient-based weight optimization using online learning.

    Uses exponential gradient updates for soft-max weights.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        regularization: float = 0.01,
    ):
        """
        Initialize gradient optimizer.

        Args:
            learning_rate: Learning rate for updates
            momentum: Momentum coefficient
            regularization: L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.regularization = regularization
        self._velocities: dict[str, float] = {}

    def optimize(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        labels: NDArray[np.int64] | None,
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Optimize weights using gradient descent on log-loss."""
        if labels is None:
            return current_weights

        weights = dict(current_weights)
        n_samples = len(labels)

        # Initialize velocities for new detectors
        for name in detector_scores:
            if name not in self._velocities:
                self._velocities[name] = 0.0
            if name not in weights:
                weights[name] = 1.0 / len(detector_scores)

        # Compute ensemble prediction
        ensemble_scores = np.zeros(n_samples)
        total_weight = sum(weights.values())

        for name, scores in detector_scores.items():
            w = weights.get(name, 0.0) / total_weight
            ensemble_scores += w * scores

        # Compute gradients for each detector
        epsilon = 1e-7
        ensemble_scores = np.clip(ensemble_scores, epsilon, 1 - epsilon)

        # Binary cross-entropy gradient
        error = ensemble_scores - labels

        for name, scores in detector_scores.items():
            # Gradient of loss w.r.t. this detector's weight
            grad = np.mean(error * scores) + self.regularization * weights[name]

            # Momentum update
            self._velocities[name] = (
                self.momentum * self._velocities[name] + (1 - self.momentum) * grad  # type: ignore[assignment, unused-ignore]
            )

            # Update weight using exponential gradient
            log_weight = np.log(weights[name] + epsilon)
            log_weight -= self.learning_rate * self._velocities[name]
            weights[name] = np.exp(log_weight)

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class MetaLearner:
    """
    Meta-learner for detector selection and combination.

    Uses data characteristics to predict optimal detector configuration.
    """

    def __init__(
        self,
        n_features: int = 10,
        hidden_dim: int = 32,
    ):
        """
        Initialize meta-learner.

        Args:
            n_features: Number of meta-features
            hidden_dim: Hidden layer dimension
        """
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self._fitted = False

        # Simple linear meta-learner (can be replaced with neural network)
        self._feature_weights: NDArray[np.float64] | None = None
        self._bias: float = 0.0

    def extract_meta_features(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract meta-features from data for detector selection."""
        features = []

        # Statistical features
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(stats.skew(data.flatten()))
        features.append(stats.kurtosis(data.flatten()))

        # Percentiles
        features.append(np.percentile(data, 25))
        features.append(np.percentile(data, 75))

        # Range and scale
        features.append(np.max(data) - np.min(data))
        features.append(np.median(np.abs(data - np.median(data))))  # MAD

        # Dimensionality features
        if data.ndim > 1:
            features.append(data.shape[1])  # n_features
            # Correlation strength
            if data.shape[1] > 1:
                corr = np.corrcoef(data.T)
                features.append(np.mean(np.abs(corr[np.triu_indices(len(corr), 1)])))  # type: ignore[arg-type, index, unused-ignore]
            else:
                features.append(0.0)  # type: ignore[arg-type, unused-ignore]
        else:
            features.append(1)  # type: ignore[arg-type, unused-ignore]
            features.append(0.0)  # type: ignore[arg-type, unused-ignore]

        return np.array(features[: self.n_features])

    def predict_weights(
        self,
        data: NDArray[np.float64],
        detector_names: list[str],
    ) -> dict[str, float]:
        """Predict optimal detector weights based on data characteristics."""
        meta_features = self.extract_meta_features(data)

        if not self._fitted or self._feature_weights is None:
            # Uniform weights if not fitted
            return {name: 1.0 / len(detector_names) for name in detector_names}

        # Simple linear prediction
        n_detectors = len(detector_names)
        raw_weights = np.zeros(n_detectors)

        for i, name in enumerate(detector_names):
            raw_weights[i] = np.dot(meta_features, self._feature_weights[i]) + self._bias

        # Softmax normalization
        exp_weights = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_weights / exp_weights.sum()

        return dict(zip(detector_names, weights))

    def fit(
        self,
        data_batches: list[NDArray[np.float64]],
        optimal_weights: list[dict[str, float]],
        detector_names: list[str],
    ) -> None:
        """Fit meta-learner from historical data."""
        if not data_batches or not optimal_weights:
            return

        n_detectors = len(detector_names)
        # n_batches used implicitly in array operations

        # Extract meta-features for all batches
        X = np.array([self.extract_meta_features(d) for d in data_batches])
        Y = np.array(
            [[w.get(name, 1.0 / n_detectors) for name in detector_names] for w in optimal_weights]
        )

        # Simple linear regression for each detector
        self._feature_weights = np.zeros((n_detectors, self.n_features))

        for i in range(n_detectors):
            try:
                # Ridge regression
                lambda_reg = 0.1
                XtX = X.T @ X + lambda_reg * np.eye(self.n_features)
                XtY = X.T @ Y[:, i]
                self._feature_weights[i] = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                self._feature_weights[i] = np.zeros(self.n_features)

        self._fitted = True


class CascadingPipeline:
    """
    Cascading detection pipeline for efficiency.

    Runs cheap/fast detectors first, only using expensive detectors
    when initial detectors are uncertain.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        max_stages: int = 3,
    ):
        """
        Initialize cascading pipeline.

        Args:
            uncertainty_threshold: Score range considered uncertain
            max_stages: Maximum cascade stages
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.max_stages = max_stages
        self._stages: list[list[tuple[str, DetectorEntry]]] = []

    def add_stage(self, detectors: list[tuple[str, DetectorEntry]]) -> None:
        """Add a stage to the cascade."""
        if len(self._stages) < self.max_stages:
            self._stages.append(detectors)

    def detect(
        self,
        data: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], list[str]]:
        """
        Run cascading detection.

        Returns:
            Tuple of (scores, list of detectors used)
        """
        n_samples = len(data)
        final_scores = np.full(n_samples, np.nan)
        detectors_used: list[str] = []
        uncertain_mask = np.ones(n_samples, dtype=bool)

        for stage_idx, stage in enumerate(self._stages):
            if not np.any(uncertain_mask):
                break

            # Get uncertain samples
            uncertain_indices = np.where(uncertain_mask)[0]
            uncertain_data = data[uncertain_indices]

            if len(uncertain_data) == 0:
                break

            # Run detectors in this stage
            stage_scores = np.zeros(len(uncertain_indices))
            stage_weights = np.zeros(len(stage))

            for det_idx, (name, entry) in enumerate(stage):
                if entry.state != DetectorState.ACTIVE:
                    continue

                try:
                    result = entry.detector.detect(uncertain_data)
                    scores = np.asarray(result.get("scores", [0.0]))

                    if len(scores) == len(uncertain_indices):
                        stage_scores += entry.weight * scores
                        stage_weights[det_idx] = entry.weight
                        detectors_used.append(name)

                except Exception as e:
                    logger.warning(f"Detector {name} failed: {e}")

            # Normalize by total weight
            total_weight = stage_weights.sum()
            if total_weight > 0:
                stage_scores /= total_weight

            # Update final scores for confident predictions
            confident_mask = (stage_scores < 0.5 - self.uncertainty_threshold) | (
                stage_scores > 0.5 + self.uncertainty_threshold
            )

            # Map back to original indices
            confident_original = uncertain_indices[confident_mask]
            final_scores[confident_original] = stage_scores[confident_mask]

            # Update uncertain mask
            uncertain_mask[confident_original] = False

        # Fill remaining uncertain with last stage scores
        if np.any(np.isnan(final_scores)):
            final_scores[np.isnan(final_scores)] = 0.5

        return final_scores, list(set(detectors_used))


class EnsembleCoordinator:
    """
    Advanced ensemble coordinator for hybrid anomaly detection.

    Features:
    - Multi-strategy ensemble fusion
    - Adaptive weight learning
    - Meta-learning for detector selection
    - Cascading detection pipelines
    - Dynamic detector activation
    - Uncertainty-aware fusion
    """

    def __init__(
        self,
        strategy: EnsembleStrategy = EnsembleStrategy.DYNAMIC,
        weight_optimizer: WeightOptimizer | None = None,
        enable_meta_learning: bool = True,
        enable_cascading: bool = True,
        feedback_window: int = 1000,
    ):
        """
        Initialize ensemble coordinator.

        Args:
            strategy: Ensemble combination strategy
            weight_optimizer: Weight optimization method
            enable_meta_learning: Enable meta-learning for detector selection
            enable_cascading: Enable cascading detection
            feedback_window: Window for feedback-based learning
        """
        self.strategy = strategy
        self.weight_optimizer = weight_optimizer or BayesianWeightOptimizer()
        self.enable_meta_learning = enable_meta_learning
        self.enable_cascading = enable_cascading
        self.feedback_window = feedback_window

        self._detectors: dict[str, DetectorEntry] = {}
        self._detector_order: list[str] = []  # Order by cost

        self._meta_learner: MetaLearner | None = None
        if enable_meta_learning:
            self._meta_learner = MetaLearner()

        self._cascade: CascadingPipeline | None = None
        if enable_cascading:
            self._cascade = CascadingPipeline()

        self._feedback_buffer: deque[tuple[NDArray, NDArray]] = deque(maxlen=feedback_window)  # type: ignore[type-arg, unused-ignore]
        self._lock = threading.Lock()

        # Strategy-specific state
        self._stacking_meta_model: Any = None
        self._mixture_gates: dict[str, float] = {}

    def register_detector(
        self,
        name: str,
        detector: Detector,
        weight: float = 1.0,
        cost_tier: int = 1,  # 1=cheap, 3=expensive
        domain_affinity: dict[str, float] | None = None,
    ) -> None:
        """
        Register a detector with the ensemble.

        Args:
            name: Unique detector name
            detector: Detector instance
            weight: Initial weight
            cost_tier: Cost tier for cascading (1-3)
            domain_affinity: Affinity scores for different domains
        """
        entry = DetectorEntry(
            name=name,
            detector=detector,
            weight=weight,
            domain_affinity=domain_affinity or {},
        )

        with self._lock:
            self._detectors[name] = entry

            # Insert in order by cost tier
            inserted = False
            for i, existing_name in enumerate(self._detector_order):
                existing = self._detectors.get(existing_name)
                if existing and cost_tier < existing.metrics.latency_ms / 100:
                    self._detector_order.insert(i, name)
                    inserted = True
                    break
            if not inserted:
                self._detector_order.append(name)

            # Update cascade stages
            if self._cascade:
                self._rebuild_cascade()

    def _rebuild_cascade(self) -> None:
        """Rebuild cascading pipeline from detectors."""
        if not self._cascade:
            return

        self._cascade = CascadingPipeline()

        # Group detectors by cost tier
        tiers: dict[int, list[tuple[str, DetectorEntry]]] = {1: [], 2: [], 3: []}

        for name in self._detector_order:
            entry = self._detectors[name]
            # Estimate tier from latency
            if entry.metrics.latency_ms < 50:
                tier = 1
            elif entry.metrics.latency_ms < 200:
                tier = 2
            else:
                tier = 3
            tiers[tier].append((name, entry))

        # Add stages
        for tier in [1, 2, 3]:
            if tiers[tier]:
                self._cascade.add_stage(tiers[tier])

    def fit(self, data: NDArray[np.float64]) -> EnsembleCoordinator:
        """Fit all detectors in the ensemble."""
        for name, entry in self._detectors.items():
            try:
                entry.detector.fit(data)
                entry.state = DetectorState.ACTIVE
            except Exception as e:
                logger.error(f"Failed to fit detector {name}: {e}")
                entry.state = DetectorState.FAILED

        return self

    def detect(
        self,
        data: NDArray[np.float64],
        domain: str | None = None,
    ) -> EnsembleResult:
        """
        Run ensemble detection.

        Args:
            data: Input data array
            domain: Optional domain hint for detector selection

        Returns:
            EnsembleResult with combined predictions
        """
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = len(data)

        # Get detector weights
        weights = self._get_weights(data, domain)

        # Select strategy
        if self.strategy == EnsembleStrategy.CASCADING and self._cascade:
            scores, used_detectors = self._cascade.detect(data)
            detector_scores = dict.fromkeys(used_detectors, scores)
        else:
            # Run all active detectors
            detector_scores = self._run_detectors(data)
            used_detectors = list(detector_scores.keys())

        if not detector_scores:
            # No detectors available
            return EnsembleResult(
                is_anomaly=np.zeros(n_samples, dtype=bool),
                scores=np.zeros(n_samples),
                confidence=np.zeros(n_samples),
                detector_contributions={},
                active_detectors=[],
                strategy_used=self.strategy.value,
                ensemble_weights={},
            )

        # Combine predictions based on strategy
        if self.strategy == EnsembleStrategy.VOTING:
            combined_scores, confidence = self._voting_fusion(detector_scores, weights)
        elif self.strategy == EnsembleStrategy.STACKING:
            combined_scores, confidence = self._stacking_fusion(detector_scores, data)
        elif self.strategy == EnsembleStrategy.MIXTURE_OF_EXPERTS:
            combined_scores, confidence = self._moe_fusion(detector_scores, data, weights)
        elif self.strategy == EnsembleStrategy.BOOSTING:
            combined_scores, confidence = self._boosting_fusion(detector_scores, weights)
        else:  # AVERAGING, DYNAMIC
            combined_scores, confidence = self._weighted_average_fusion(detector_scores, weights)

        # Determine anomalies
        threshold = 0.5
        is_anomaly = combined_scores > threshold

        return EnsembleResult(
            is_anomaly=is_anomaly,
            scores=combined_scores,
            confidence=confidence,
            detector_contributions=detector_scores,
            active_detectors=used_detectors,
            strategy_used=self.strategy.value,
            ensemble_weights=weights,
            meta_info={
                "n_detectors": len(used_detectors),
                "threshold": threshold,
            },
        )

    def _get_weights(
        self,
        data: NDArray[np.float64],
        domain: str | None,
    ) -> dict[str, float]:
        """Get detector weights based on meta-learning and domain."""
        detector_names = [
            name for name, entry in self._detectors.items() if entry.state == DetectorState.ACTIVE
        ]

        if not detector_names:
            return {}

        # Start with base weights
        weights = {name: self._detectors[name].weight for name in detector_names}

        # Apply domain affinity if specified
        if domain:
            for name in detector_names:
                affinity = self._detectors[name].domain_affinity.get(domain, 1.0)
                weights[name] *= affinity

        # Use meta-learner if available
        if self._meta_learner and self.enable_meta_learning:
            meta_weights = self._meta_learner.predict_weights(data, detector_names)
            # Blend with base weights
            for name in detector_names:
                weights[name] = 0.7 * weights[name] + 0.3 * meta_weights.get(name, weights[name])

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _run_detectors(
        self,
        data: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Run all active detectors and collect scores."""
        results = {}

        for name, entry in self._detectors.items():
            if entry.state != DetectorState.ACTIVE:
                continue

            start_time = time.time()

            try:
                result = entry.detector.detect(data)
                scores = np.asarray(result.get("scores", np.zeros(len(data))))

                if len(scores) != len(data):
                    logger.warning(f"Detector {name} returned wrong number of scores")
                    continue

                results[name] = scores

                # Update latency
                latency = (time.time() - start_time) * 1000
                entry.metrics.latency_ms = 0.1 * latency + 0.9 * entry.metrics.latency_ms

            except Exception as e:
                logger.error(f"Detector {name} failed: {e}")
                entry.state = DetectorState.FAILED

        return results

    def _weighted_average_fusion(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        weights: dict[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Weighted average fusion with uncertainty estimation."""
        n_samples = len(next(iter(detector_scores.values())))
        combined = np.zeros(n_samples)
        total_weight = 0.0

        all_scores = []
        all_weights = []

        for name, scores in detector_scores.items():
            w = weights.get(name, 1.0 / len(detector_scores))
            combined += w * scores
            total_weight += w
            all_scores.append(scores)
            all_weights.append(w)

        if total_weight > 0:
            combined /= total_weight

        # Confidence from agreement (low variance = high confidence)
        scores_array = np.array(all_scores)
        variance = np.var(scores_array, axis=0)
        confidence = 1.0 / (1.0 + variance)  # Higher agreement = higher confidence

        return combined, confidence

    def _voting_fusion(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        weights: dict[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Weighted voting fusion."""
        n_samples = len(next(iter(detector_scores.values())))
        votes = np.zeros(n_samples)
        total_weight = 0.0

        for name, scores in detector_scores.items():
            w = weights.get(name, 1.0 / len(detector_scores))
            predictions = (scores > 0.5).astype(float)
            votes += w * predictions
            total_weight += w

        if total_weight > 0:
            vote_ratio = votes / total_weight
        else:
            vote_ratio = np.zeros(n_samples)

        # Confidence from vote margin
        confidence = np.abs(vote_ratio - 0.5) * 2

        return vote_ratio, confidence

    def _stacking_fusion(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        data: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stacking fusion with meta-learner."""
        # Stack detector outputs as features
        stacked_features = np.column_stack(list(detector_scores.values()))

        if self._stacking_meta_model is None:
            # Use simple weighted average as fallback
            weights = {name: 1.0 / len(detector_scores) for name in detector_scores}
            return self._weighted_average_fusion(detector_scores, weights)

        try:
            # Use meta-model for prediction
            meta_scores = self._stacking_meta_model.predict_proba(stacked_features)[:, 1]
            confidence = np.abs(meta_scores - 0.5) * 2
            return meta_scores, confidence
        except Exception:
            # Fallback to weighted average
            weights = {name: 1.0 / len(detector_scores) for name in detector_scores}
            return self._weighted_average_fusion(detector_scores, weights)

    def _moe_fusion(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        data: NDArray[np.float64],
        weights: dict[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Mixture of Experts fusion with gating network."""
        n_samples = len(data)
        n_detectors = len(detector_scores)

        if n_detectors == 0:
            return np.zeros(n_samples), np.zeros(n_samples)

        # Compute gating weights based on data characteristics
        # Simple approach: use data statistics for gating
        gate_input = np.column_stack(
            [
                np.mean(data, axis=1),
                np.std(data, axis=1),
                np.max(data, axis=1),
                np.min(data, axis=1),
            ]
        )

        # Softmax gating (simplified - in production use learned gates)
        gate_logits = np.zeros((n_samples, n_detectors))
        detector_names = list(detector_scores.keys())

        for i, name in enumerate(detector_names):
            # Base weight + data-dependent component
            base = weights.get(name, 1.0)
            gate_logits[:, i] = base + 0.1 * np.mean(gate_input, axis=1)

        # Softmax
        gate_exp = np.exp(gate_logits - np.max(gate_logits, axis=1, keepdims=True))
        gate_weights = gate_exp / gate_exp.sum(axis=1, keepdims=True)

        # Weighted combination
        combined = np.zeros(n_samples)
        for i, (name, scores) in enumerate(detector_scores.items()):
            combined += gate_weights[:, i] * scores

        # Confidence from gate certainty
        confidence = np.max(gate_weights, axis=1)

        return combined, confidence

    def _boosting_fusion(
        self,
        detector_scores: dict[str, NDArray[np.float64]],
        weights: dict[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Boosting-style sequential fusion."""
        n_samples = len(next(iter(detector_scores.values())))
        combined = np.zeros(n_samples)
        sample_weights = np.ones(n_samples) / n_samples

        # Sort detectors by weight (importance)
        sorted_detectors = sorted(
            detector_scores.items(),
            key=lambda x: weights.get(x[0], 0),
            reverse=True,
        )

        for name, scores in sorted_detectors:
            w = weights.get(name, 1.0)

            # Add weighted contribution
            combined += w * np.log((scores + 1e-7) / (1 - scores + 1e-7))

            # Update sample weights (focus on hard examples)
            errors = np.abs((scores > 0.5).astype(float) - sample_weights)
            sample_weights = sample_weights * np.exp(errors)
            sample_weights /= sample_weights.sum()

        # Convert from log-odds to probability
        combined_prob = 1 / (1 + np.exp(-combined))

        # Confidence from margin
        confidence = np.abs(combined_prob - 0.5) * 2

        return combined_prob, confidence

    def provide_feedback(
        self,
        predictions: NDArray[np.bool_],
        labels: NDArray[np.int64],
    ) -> None:
        """
        Provide feedback for online learning.

        Args:
            predictions: Predicted anomaly flags
            labels: True labels (1=anomaly, 0=normal)
        """
        # Store in feedback buffer
        self._feedback_buffer.append((predictions, labels))

        # Update detector metrics
        for name, entry in self._detectors.items():
            if entry.state == DetectorState.ACTIVE:
                for pred, label in zip(predictions, labels):
                    entry.metrics.update_from_feedback(pred, label == 1, 0)

        # Periodically update weights
        if len(self._feedback_buffer) >= 100:
            self._update_weights_from_feedback()

    def _update_weights_from_feedback(self) -> None:
        """Update detector weights from accumulated feedback."""
        if not self._feedback_buffer:
            return

        # Feedback buffer contains (predictions, labels) tuples
        # These are used for weight optimization in production scenarios
        # For now, use a simple performance-based update
        for name, entry in self._detectors.items():
            if entry.metrics.f1_score > 0:
                entry.weight = entry.metrics.f1_score

        # Normalize weights
        total = sum(entry.weight for entry in self._detectors.values())
        if total > 0:
            for entry in self._detectors.values():
                entry.weight /= total

    def get_detector_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all detectors."""
        return {
            name: {
                "state": entry.state.value,
                "weight": entry.weight,
                "precision": entry.metrics.precision,
                "recall": entry.metrics.recall,
                "f1_score": entry.metrics.f1_score,
                "latency_ms": entry.metrics.latency_ms,
                "samples_processed": entry.metrics.samples_processed,
            }
            for name, entry in self._detectors.items()
        }

    def get_ensemble_summary(self) -> dict[str, Any]:
        """Get ensemble summary statistics."""
        active = sum(1 for e in self._detectors.values() if e.state == DetectorState.ACTIVE)

        return {
            "strategy": self.strategy.value,
            "total_detectors": len(self._detectors),
            "active_detectors": active,
            "feedback_buffer_size": len(self._feedback_buffer),
            "meta_learning_enabled": self.enable_meta_learning,
            "cascading_enabled": self.enable_cascading,
            "average_f1": (
                np.mean(
                    [
                        e.metrics.f1_score
                        for e in self._detectors.values()
                        if e.state == DetectorState.ACTIVE
                    ]
                )
                if active > 0
                else 0.0
            ),
        }


# Factory function
def create_ensemble_coordinator(
    strategy: str = "dynamic",
    **kwargs: Any,
) -> EnsembleCoordinator:
    """
    Factory function to create ensemble coordinator.

    Args:
        strategy: Ensemble strategy name
        **kwargs: Additional configuration

    Returns:
        Configured EnsembleCoordinator
    """
    strategy_enum = EnsembleStrategy(strategy.lower())

    return EnsembleCoordinator(
        strategy=strategy_enum,
        enable_meta_learning=kwargs.get("enable_meta_learning", True),
        enable_cascading=kwargs.get("enable_cascading", True),
        feedback_window=kwargs.get("feedback_window", 1000),
    )


# Exports
__all__ = [
    "BayesianWeightOptimizer",
    "CascadingPipeline",
    "DetectorEntry",
    "DetectorMetrics",
    "DetectorState",
    "EnsembleCoordinator",
    "EnsembleResult",
    "EnsembleStrategy",
    "GradientWeightOptimizer",
    "MetaLearner",
    "WeightOptimizer",
    "create_ensemble_coordinator",
]

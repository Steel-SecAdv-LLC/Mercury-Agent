"""
Mercury Agent - Adaptive Per-Domain Thresholding System
Copyright (C) 2025 Steel Security Advisory LLC

Advanced thresholding system with per-domain optimization:
- Domain-specific threshold calibration
- Platt scaling for probability calibration
- Isotonic regression calibration
- Dynamic threshold adjustment based on domain characteristics
- Domain ensemble weighting optimizer

This module implements the strategic recommendations for adaptive thresholding
to improve domain competence across Medical, Financial, and Infrastructure.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

from omni_mercury_engine.core.score_calibration import (
    CalibrationDiagnostics,
    CalibrationMethod,
    ScoreCalibrationManager,
    ScoreDiagnostics,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Supported domain types for threshold calibration."""

    MEDICAL = "medical"
    FINANCIAL = "financial"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    HUMANITARIAN = "humanitarian"
    GENERAL = "general"


@dataclass
class DomainThresholdConfig:
    """Configuration for domain-specific thresholding."""

    domain: DomainType

    # Base threshold (before calibration)
    base_threshold: float = 0.5

    # Contamination estimate (expected anomaly ratio)
    contamination: float = 0.05

    # Preferred calibration method
    calibration_method: CalibrationMethod = CalibrationMethod.AUTO

    # Enable probability calibration (Platt/Isotonic)
    enable_probability_calibration: bool = True

    # Priority: precision vs recall tradeoff
    # Higher values favor precision (fewer false positives)
    precision_priority: float = 0.5

    # Domain-specific ethical threshold (sigma_immutable)
    ethical_threshold: float = 0.96

    # Minimum/maximum threshold bounds
    min_threshold: float = 0.01
    max_threshold: float = 0.99

    # Historical performance weights for adaptive adjustment
    history_weight: float = 0.3

    # Sliding window size for dynamic adjustment
    window_size: int = 1000


@dataclass
class DomainCalibrationResult:
    """Result from domain-specific calibration."""

    domain: DomainType
    threshold: float
    calibrated_scores: NDArray[np.float64]
    predictions: NDArray[np.bool_]
    diagnostics: CalibrationDiagnostics
    calibration_method: str
    probability_calibration: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


# Domain-specific default configurations
DOMAIN_DEFAULTS: dict[DomainType, dict[str, Any]] = {
    DomainType.MEDICAL: {
        "contamination": 0.03,  # Lower - false negatives are critical
        "precision_priority": 0.3,  # Favor recall - don't miss anomalies
        "ethical_threshold": 0.93,  # Lower ethical threshold for medical
        "calibration_method": CalibrationMethod.OPTIMAL_F1,
    },
    DomainType.FINANCIAL: {
        "contamination": 0.05,
        "precision_priority": 0.6,  # Balance, slightly favor precision
        "ethical_threshold": 0.96,
        "calibration_method": CalibrationMethod.AUTO,
    },
    DomainType.INFRASTRUCTURE: {
        "contamination": 0.02,  # Very low - infrastructure anomalies rare but critical
        "precision_priority": 0.4,  # Slightly favor recall
        "ethical_threshold": 0.995,  # Highest for infrastructure
        "calibration_method": CalibrationMethod.GAUSSIAN_MIXTURE,
    },
    DomainType.SECURITY: {
        "contamination": 0.10,  # Higher - many potential threats
        "precision_priority": 0.5,  # Balanced
        "ethical_threshold": 0.96,
        "calibration_method": CalibrationMethod.MAD,
    },
    DomainType.HUMANITARIAN: {
        "contamination": 0.05,
        "precision_priority": 0.3,  # Favor recall - don't miss crises
        "ethical_threshold": 0.95,
        "calibration_method": CalibrationMethod.AUTO,
    },
    DomainType.GENERAL: {
        "contamination": 0.05,
        "precision_priority": 0.5,
        "ethical_threshold": 0.96,
        "calibration_method": CalibrationMethod.AUTO,
    },
}


class PlattScalingCalibrator:
    """
    Platt scaling for probability calibration.

    Fits a sigmoid function to map raw scores to calibrated probabilities:
    P(y=1|s) = 1 / (1 + exp(A*s + B))

    This is particularly effective for neural network and SVM outputs.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize Platt scaling calibrator.

        Args:
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.A: float = -1.0
        self.B: float = 0.0
        self._fitted = False

    def fit(self, scores: NDArray[np.float64], labels: NDArray[np.int32]) -> PlattScalingCalibrator:
        """
        Fit Platt scaling parameters.

        Uses cross-entropy loss optimization to find optimal A, B parameters.

        Args:
            scores: Raw anomaly scores
            labels: Binary labels (1=anomaly, 0=normal)

        Returns:
            self for method chaining
        """
        scores = np.asarray(scores).flatten()
        labels = np.asarray(labels).flatten()

        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")

        n = len(scores)
        if n < 10:
            logger.warning("Too few samples for Platt scaling, using defaults")
            self._fitted = True
            return self

        # Target probabilities with smoothing (Platt's improvement)
        n_pos = np.sum(labels == 1)
        n_neg = n - n_pos

        if n_pos == 0 or n_neg == 0:
            logger.warning("Single-class labels, Platt scaling not applicable")
            self._fitted = True
            return self

        # Smoothed targets
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)
        targets = np.where(labels == 1, t_pos, t_neg)

        # Optimize A, B using cross-entropy loss
        def objective(params: NDArray[np.float64]) -> float:
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * scores + b))
            # Cross-entropy loss with clipping for numerical stability
            p = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))
            return loss  # type: ignore[no-any-return]

        # Initialize with reasonable values
        x0 = np.array([-1.0, 0.0])

        result = optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.A = float(result.x[0])
        self.B = float(result.x[1])
        self._fitted = True

        logger.debug(f"Platt scaling fitted: A={self.A:.4f}, B={self.B:.4f}")
        return self

    def calibrate(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calibrate scores to probabilities.

        Args:
            scores: Raw anomaly scores

        Returns:
            Calibrated probabilities in [0, 1]
        """
        if not self._fitted:
            raise ValueError("PlattScalingCalibrator must be fitted before calibration")

        scores = np.asarray(scores).flatten()
        calibrated = 1.0 / (1.0 + np.exp(self.A * scores + self.B))
        return calibrated

    def get_params(self) -> dict[str, float]:
        """Get fitted parameters."""
        return {"A": self.A, "B": self.B}


class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration.

    Fits a non-decreasing function to map scores to probabilities.
    More flexible than Platt scaling but requires more data.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Initialize isotonic calibrator.

        Args:
            out_of_bounds: How to handle out-of-bounds values ('clip' or 'nan')
        """
        self.out_of_bounds = out_of_bounds
        self._fitted = False
        self._calibration_map: NDArray[np.float64] | None = None
        self._score_bins: NDArray[np.float64] | None = None

    def fit(self, scores: NDArray[np.float64], labels: NDArray[np.int32]) -> IsotonicCalibrator:
        """
        Fit isotonic regression calibration.

        Uses Pool Adjacent Violators Algorithm (PAVA).

        Args:
            scores: Raw anomaly scores
            labels: Binary labels (1=anomaly, 0=normal)

        Returns:
            self for method chaining
        """
        scores = np.asarray(scores).flatten()
        labels = np.asarray(labels).flatten()

        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")

        # Sort by scores
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Pool Adjacent Violators Algorithm (PAVA)
        n = len(sorted_scores)
        weights = np.ones(n)

        # Initialize calibrated values with labels
        calibrated = sorted_labels.astype(float).copy()

        # PAVA iterations
        while True:
            violations = []
            for i in range(n - 1):
                if calibrated[i] > calibrated[i + 1]:
                    violations.append(i)

            if not violations:
                break

            for i in violations:
                # Pool adjacent blocks
                j = i + 1
                while j < n and calibrated[i] > calibrated[j]:
                    j += 1

                # Average over pooled region
                pooled_value = np.average(calibrated[i:j], weights=weights[i:j])
                calibrated[i:j] = pooled_value

        # Create calibration mapping
        unique_scores, unique_indices = np.unique(sorted_scores, return_index=True)
        unique_calibrated = calibrated[unique_indices]

        self._score_bins = unique_scores
        self._calibration_map = unique_calibrated
        self._fitted = True

        logger.debug(f"Isotonic calibration fitted with {len(unique_scores)} bins")
        return self

    def calibrate(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calibrate scores to probabilities.

        Args:
            scores: Raw anomaly scores

        Returns:
            Calibrated probabilities in [0, 1]
        """
        if not self._fitted:
            raise ValueError("IsotonicCalibrator must be fitted before calibration")

        scores = np.asarray(scores).flatten()
        calibrated = np.interp(scores, self._score_bins, self._calibration_map)  # type: ignore[arg-type, unused-ignore]

        if self.out_of_bounds == "clip":
            calibrated = np.clip(calibrated, 0.0, 1.0)

        return calibrated


class CalibrationEnsemble:
    """
    Ensemble of calibration methods for robust probability estimation.

    Combines Platt scaling and isotonic regression with weighted averaging.
    """

    def __init__(self, platt_weight: float = 0.5):
        """
        Initialize calibration ensemble.

        Args:
            platt_weight: Weight for Platt scaling (isotonic gets 1-weight)
        """
        self.platt_weight = platt_weight
        self.platt = PlattScalingCalibrator()
        self.isotonic = IsotonicCalibrator()
        self._fitted = False
        self.best_method: str = "ensemble"

    def fit(self, scores: NDArray[np.float64], labels: NDArray[np.int32]) -> CalibrationEnsemble:
        """
        Fit both calibrators.

        Args:
            scores: Raw anomaly scores
            labels: Binary labels

        Returns:
            self for method chaining
        """
        scores = np.asarray(scores).flatten()
        labels = np.asarray(labels).flatten()

        # Fit both methods
        try:
            self.platt.fit(scores, labels)
        except Exception as e:
            logger.warning(f"Platt scaling fit failed: {e}")
            self.platt_weight = 0.0

        try:
            self.isotonic.fit(scores, labels)
        except Exception as e:
            logger.warning(f"Isotonic fit failed: {e}")
            self.platt_weight = 1.0

        # Select best method based on cross-validation
        if len(scores) > 20:
            self._select_best_method(scores, labels)

        self._fitted = True
        return self

    def _select_best_method(self, scores: NDArray[np.float64], labels: NDArray[np.int32]) -> None:
        """Select best calibration method via cross-validation."""
        try:
            from sklearn.model_selection import KFold
        except ImportError:
            logger.warning(
                "sklearn not available — skipping cross-validation method selection, "
                "using default ensemble calibration"
            )
            return

        kfold = KFold(n_splits=min(5, len(scores) // 4), shuffle=True, random_state=42)

        platt_losses = []
        isotonic_losses = []
        ensemble_losses = []

        for train_idx, val_idx in kfold.split(scores):
            train_scores, val_scores = scores[train_idx], scores[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Fit on train
            platt_temp = PlattScalingCalibrator()
            isotonic_temp = IsotonicCalibrator()

            try:
                platt_temp.fit(train_scores, train_labels)
                platt_cal = platt_temp.calibrate(val_scores)
                platt_loss = self._brier_score(platt_cal, val_labels)
                platt_losses.append(platt_loss)
            except (ValueError, RuntimeError) as e:
                # Platt scaling may fail with degenerate data
                logger.debug(f"Platt calibration fold failed: {e}")
                platt_losses.append(1.0)

            try:
                isotonic_temp.fit(train_scores, train_labels)
                isotonic_cal = isotonic_temp.calibrate(val_scores)
                isotonic_loss = self._brier_score(isotonic_cal, val_labels)
                isotonic_losses.append(isotonic_loss)
            except (ValueError, RuntimeError) as e:
                # Isotonic regression may fail with insufficient unique values
                logger.debug(f"Isotonic calibration fold failed: {e}")
                isotonic_losses.append(1.0)

            # Ensemble
            try:
                ensemble_cal = (
                    self.platt_weight * platt_cal + (1 - self.platt_weight) * isotonic_cal
                )
                ensemble_loss = self._brier_score(ensemble_cal, val_labels)
                ensemble_losses.append(ensemble_loss)
            except (NameError, ValueError) as e:
                # Ensemble fails if constituent calibrators failed
                logger.debug(f"Ensemble calibration fold failed: {e}")
                ensemble_losses.append(1.0)

        # Select method with lowest average Brier score
        avg_losses = {
            "platt": np.mean(platt_losses),
            "isotonic": np.mean(isotonic_losses),
            "ensemble": np.mean(ensemble_losses),
        }

        self.best_method = min(avg_losses, key=lambda k: avg_losses.get(k, float("inf")))
        logger.debug(f"Selected calibration method: {self.best_method} (losses: {avg_losses})")

    def _brier_score(self, probs: NDArray[np.float64], labels: NDArray[np.int32]) -> float:
        """Compute Brier score (lower is better)."""
        return float(np.mean((probs - labels) ** 2))

    def calibrate(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calibrate scores using the best method.

        Args:
            scores: Raw anomaly scores

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise ValueError("CalibrationEnsemble must be fitted before calibration")

        scores = np.asarray(scores).flatten()

        if self.best_method == "platt":
            return self.platt.calibrate(scores)
        elif self.best_method == "isotonic":
            return self.isotonic.calibrate(scores)
        else:
            # Ensemble
            platt_cal = self.platt.calibrate(scores)
            isotonic_cal = self.isotonic.calibrate(scores)
            return self.platt_weight * platt_cal + (1 - self.platt_weight) * isotonic_cal


class AdaptiveDomainThresholdManager:
    """
    Adaptive per-domain thresholding manager.

    Provides domain-specific threshold calibration with:
    - Automatic method selection based on domain
    - Platt scaling/isotonic regression probability calibration
    - Dynamic threshold adjustment based on performance history
    - Precision/recall tradeoff control per domain
    """

    def __init__(self, domain: DomainType | str):
        """
        Initialize adaptive threshold manager.

        Args:
            domain: Target domain
        """
        if isinstance(domain, str):
            domain = DomainType(domain.lower())

        self.domain = domain

        # Load domain defaults
        defaults = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS[DomainType.GENERAL])
        self.config = DomainThresholdConfig(domain=domain, **defaults)

        # Calibration components
        self.base_calibrator = ScoreCalibrationManager(
            contamination=self.config.contamination,
            method=self.config.calibration_method,
        )
        self.probability_calibrator: CalibrationEnsemble | None = None

        # History tracking for adaptive adjustment
        self._threshold_history: list[float] = []
        self._performance_history: list[dict[str, float]] = []
        self._score_history: list[NDArray[np.float64]] = []

        self._fitted = False
        self._current_threshold = self.config.base_threshold

        logger.info(
            f"AdaptiveDomainThresholdManager initialized for {domain.value} "
            f"(contamination={self.config.contamination}, "
            f"precision_priority={self.config.precision_priority})"
        )

    def fit(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
    ) -> AdaptiveDomainThresholdManager:
        """
        Fit the threshold manager on training data.

        Args:
            scores: Anomaly scores
            labels: Optional ground truth labels for supervised calibration

        Returns:
            self for method chaining
        """
        scores = np.asarray(scores).flatten()

        # Fit base calibrator
        result = self.base_calibrator.calibrate(scores, labels)
        self._current_threshold = result.threshold

        # Fit probability calibrator if labels available and enabled
        if labels is not None and self.config.enable_probability_calibration:
            labels = np.asarray(labels).flatten()
            self.probability_calibrator = CalibrationEnsemble()
            self.probability_calibrator.fit(scores, labels)

        # Apply precision/recall tradeoff adjustment
        self._current_threshold = self._adjust_for_priority(self._current_threshold, scores, labels)

        # Enforce bounds
        self._current_threshold = np.clip(
            self._current_threshold,
            self.config.min_threshold,
            self.config.max_threshold,
        )

        self._threshold_history.append(self._current_threshold)
        self._fitted = True

        logger.info(
            f"Domain threshold fitted: {self._current_threshold:.4f} "
            f"(method: {result.method.value})"
        )

        return self

    def _adjust_for_priority(
        self,
        threshold: float,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None,
    ) -> float:
        """
        Adjust threshold based on precision/recall priority.

        Args:
            threshold: Base threshold
            scores: Anomaly scores
            labels: Optional labels

        Returns:
            Adjusted threshold
        """
        priority = self.config.precision_priority

        if labels is None:
            # Without labels, adjust based on percentile
            if priority > 0.5:
                # Favor precision - increase threshold
                adjustment = (priority - 0.5) * 0.2  # Max 10% increase
                new_percentile = np.percentile(scores, (1 - self.config.contamination) * 100)
                threshold = threshold + adjustment * (new_percentile - threshold)
            else:
                # Favor recall - decrease threshold
                adjustment = (0.5 - priority) * 0.2  # Max 10% decrease
                new_percentile = np.percentile(scores, (1 - self.config.contamination * 2) * 100)
                threshold = threshold - adjustment * (threshold - new_percentile)
        else:
            # With labels, optimize for weighted F-beta score
            beta = (1 - priority) / (priority + 1e-10)  # Higher priority -> lower beta
            beta = np.clip(beta, 0.1, 10.0)

            best_threshold = threshold
            best_score = 0.0

            for t in np.percentile(scores, np.linspace(0, 100, 50)):
                predictions = scores > t
                tp = np.sum((labels == 1) & predictions)
                fp = np.sum((labels == 0) & predictions)
                fn = np.sum((labels == 1) & ~predictions)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                if precision + recall > 0:
                    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                    if f_beta > best_score:
                        best_score = f_beta
                        best_threshold = t

            threshold = best_threshold

        return float(threshold)

    def calibrate(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32] | None = None,
    ) -> DomainCalibrationResult:
        """
        Calibrate scores and compute optimal threshold.

        Args:
            scores: Raw anomaly scores
            labels: Optional ground truth labels

        Returns:
            Domain calibration result
        """
        scores = np.asarray(scores).flatten()

        # Apply probability calibration if available
        calibrated_scores = scores.copy()
        prob_calibration_applied = False

        if self.probability_calibrator is not None:
            try:
                calibrated_scores = self.probability_calibrator.calibrate(scores)
                prob_calibration_applied = True
            except Exception as e:
                logger.warning(f"Probability calibration failed: {e}")

        # Use current threshold or re-calibrate if labels available
        if labels is not None:
            result = self.base_calibrator.calibrate(calibrated_scores, labels)
            threshold = result.threshold
            threshold = self._adjust_for_priority(threshold, calibrated_scores, labels)
            threshold = np.clip(threshold, self.config.min_threshold, self.config.max_threshold)
        else:
            threshold = self._current_threshold if self._fitted else self.config.base_threshold

        # Generate predictions
        predictions = calibrated_scores > threshold

        # Generate diagnostics
        diagnostics = ScoreDiagnostics.analyze(
            calibrated_scores, threshold, labels, f"domain_{self.domain.value}"
        )

        # Adaptive threshold adjustment based on history
        if self._fitted and len(self._score_history) > 0:
            threshold = self._adaptive_adjustment(threshold, calibrated_scores)

        # Track history
        self._score_history.append(calibrated_scores[-self.config.window_size :])
        if len(self._score_history) > 10:
            self._score_history = self._score_history[-10:]

        return DomainCalibrationResult(
            domain=self.domain,
            threshold=threshold,
            calibrated_scores=calibrated_scores,
            predictions=predictions,
            diagnostics=diagnostics,
            calibration_method=self.base_calibrator.method.value,
            probability_calibration=prob_calibration_applied,
            confidence=self._compute_confidence(calibrated_scores, threshold),
            metadata={
                "precision_priority": self.config.precision_priority,
                "contamination": self.config.contamination,
                "ethical_threshold": self.config.ethical_threshold,
            },
        )

    def _adaptive_adjustment(self, threshold: float, current_scores: NDArray[np.float64]) -> float:
        """
        Adaptively adjust threshold based on score distribution drift.

        Args:
            threshold: Current threshold
            current_scores: Current batch of scores

        Returns:
            Adjusted threshold
        """
        if len(self._score_history) < 2:
            return threshold

        # Compute distribution shift
        historical_scores = np.concatenate(self._score_history[:-1])
        current_mean = np.mean(current_scores)
        historical_mean = np.mean(historical_scores)

        current_std = np.std(current_scores)
        historical_std = np.std(historical_scores) + 1e-10

        # Z-score of distribution shift
        mean_shift = (current_mean - historical_mean) / historical_std
        std_ratio = current_std / historical_std

        # Adjust threshold based on shift
        adjustment = 0.0

        if abs(mean_shift) > 1.0:
            # Significant mean shift - adjust threshold proportionally
            adjustment += mean_shift * historical_std * self.config.history_weight  # type: ignore[assignment, unused-ignore]

        if std_ratio > 1.5 or std_ratio < 0.67:
            # Significant variance change - be more conservative
            if std_ratio > 1.5:
                adjustment += 0.05 * self.config.history_weight  # Increase threshold
            else:
                adjustment -= 0.05 * self.config.history_weight  # Decrease threshold

        new_threshold = threshold + adjustment
        new_threshold = np.clip(new_threshold, self.config.min_threshold, self.config.max_threshold)

        if abs(adjustment) > 0.01:
            logger.debug(
                f"Adaptive threshold adjustment for {self.domain.value}: "
                f"{threshold:.4f} -> {new_threshold:.4f} (shift={mean_shift:.2f})"
            )

        return float(new_threshold)

    def _compute_confidence(self, scores: NDArray[np.float64], threshold: float) -> float:
        """Compute confidence in the calibration."""
        n = len(scores)

        # Factors affecting confidence
        sample_factor = min(1.0, n / 1000)  # More samples = higher confidence

        # Score spread around threshold
        near_threshold = np.sum(np.abs(scores - threshold) < 0.1) / (n + 1e-10)
        spread_factor = 1.0 - near_threshold  # Less ambiguous scores = higher confidence

        # History consistency
        if len(self._threshold_history) > 2:
            threshold_std = np.std(self._threshold_history[-5:])
            history_factor = 1.0 / (1.0 + threshold_std * 10)
        else:
            history_factor = 0.5  # type: ignore[assignment, unused-ignore]

        confidence = 0.4 * sample_factor + 0.3 * spread_factor + 0.3 * history_factor
        return float(np.clip(confidence, 0.0, 1.0))

    def get_threshold(
        self,
        score: float | None = None,
        confidence: float | None = None,
    ) -> float | dict[str, Any]:
        """
        Get current calibrated threshold, optionally with adaptive adjustment.

        When called with no arguments, returns just the threshold (backward compatible).
        When called with score and/or confidence, returns a dict with adaptive info.

        Args:
            score: Optional anomaly score for adaptive threshold adjustment
            confidence: Optional confidence level for threshold tuning

        Returns:
            float if no args provided, dict with threshold info otherwise
        """
        if score is None and confidence is None:
            # Backward compatible: just return the threshold
            return self._current_threshold

        # Adaptive threshold adjustment based on score and confidence
        base_threshold = self._current_threshold

        # Adjust threshold based on confidence
        # Low confidence → be more conservative (higher threshold)
        if confidence is not None:
            confidence_factor = 1.0 + (1.0 - confidence) * 0.1
            adjusted_threshold = base_threshold * confidence_factor
        else:
            adjusted_threshold = base_threshold

        # Enforce bounds
        adjusted_threshold = float(
            np.clip(
                adjusted_threshold,
                self.config.min_threshold,
                self.config.max_threshold,
            )
        )

        # Calibrate score if probability calibrator is available
        calibrated_score = None
        if self.probability_calibrator is not None and score is not None:
            try:
                calibrated_score = float(
                    self.probability_calibrator.calibrate(np.array([score]))[0]
                )
            except (ValueError, IndexError, TypeError) as e:
                # Calibration failed - fall back to raw score
                logger.debug(f"Score calibration fallback: {type(e).__name__}")
                calibrated_score = score

        return {
            "threshold": adjusted_threshold,
            "base_threshold": base_threshold,
            "calibrated_score": calibrated_score,
            "confidence_adjusted": confidence is not None,
            "domain": self.domain.value,
        }

    def update_performance(
        self,
        true_labels: NDArray[np.int32],
        predictions: NDArray[np.bool_],
    ) -> dict[str, float]:
        """
        Update performance history for adaptive adjustment.

        Args:
            true_labels: Ground truth labels
            predictions: Model predictions

        Returns:
            Performance metrics
        """
        true_labels = np.asarray(true_labels).flatten()
        predictions = np.asarray(predictions).flatten()

        # Compute metrics
        tp = np.sum((true_labels == 1) & predictions)
        fp = np.sum((true_labels == 0) & predictions)
        fn = np.sum((true_labels == 1) & ~predictions)
        tn = np.sum((true_labels == 0) & ~predictions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "threshold": self._current_threshold,
        }

        self._performance_history.append(metrics)
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]

        return metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of historical performance."""
        if not self._performance_history:
            return {"n_records": 0}

        f1_scores = [p["f1"] for p in self._performance_history]
        precision_scores = [p["precision"] for p in self._performance_history]
        recall_scores = [p["recall"] for p in self._performance_history]

        return {
            "n_records": len(self._performance_history),
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "f1_trend": (
                float(np.mean(f1_scores[-5:]) - np.mean(f1_scores[:5]))
                if len(f1_scores) >= 10
                else 0.0
            ),
            "precision_mean": float(np.mean(precision_scores)),
            "recall_mean": float(np.mean(recall_scores)),
            "current_threshold": self._current_threshold,
            "threshold_history": self._threshold_history[-10:],
        }


class DomainEnsembleWeightOptimizer:
    """
    Domain-specific ensemble weighting optimizer.

    Learns optimal weights for combining multiple detectors per domain,
    extending the AAFEWeightOptimizer with domain awareness.
    """

    def __init__(
        self,
        domain: DomainType | str,
        n_detectors: int = 3,
        golden_ratio_init: bool = True,
    ):
        """
        Initialize domain ensemble weight optimizer.

        Args:
            domain: Target domain
            n_detectors: Number of detectors to weight
            golden_ratio_init: Use golden ratio for initial weights
        """
        if isinstance(domain, str):
            domain = DomainType(domain.lower())

        self.domain = domain
        self.n_detectors = n_detectors
        self.golden_ratio = 1.618033988749895

        # Initialize weights
        if golden_ratio_init and n_detectors == 3:
            phi_sum = self.golden_ratio + 1.0 + 1.0 / self.golden_ratio
            self.weights = np.array(
                [
                    self.golden_ratio / phi_sum,
                    1.0 / phi_sum,
                    (1.0 / self.golden_ratio) / phi_sum,
                ]
            )
        else:
            self.weights = np.ones(n_detectors) / n_detectors

        self.optimized_weights: NDArray[np.float64] | None = None
        self.optimization_history: list[dict[str, Any]] = []

        # Domain-specific parameters
        defaults = DOMAIN_DEFAULTS.get(domain, DOMAIN_DEFAULTS[DomainType.GENERAL])
        self.precision_priority = defaults.get("precision_priority", 0.5)

    def optimize(
        self,
        detector_scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Optimize detector weights for the domain.

        Args:
            detector_scores: Shape (n_samples, n_detectors) - scores from each detector
            labels: Binary labels (1=anomaly, 0=normal)
            max_iterations: Maximum optimization iterations

        Returns:
            Optimization result with weights and metrics
        """
        detector_scores = np.asarray(detector_scores)
        labels = np.asarray(labels).flatten()

        if detector_scores.ndim != 2:
            raise ValueError("detector_scores must be 2D (n_samples, n_detectors)")

        if detector_scores.shape[1] != self.n_detectors:
            self.n_detectors = detector_scores.shape[1]
            self.weights = np.ones(self.n_detectors) / self.n_detectors

        # Compute baseline F1 with initial weights
        baseline_scores = detector_scores @ self.weights
        baseline_f1 = self._compute_f1(baseline_scores, labels)

        # Optimization objective
        def objective(w: NDArray[np.float64]) -> float:
            # Normalize weights
            w_normalized = np.abs(w) / (np.sum(np.abs(w)) + 1e-10)
            combined_scores = detector_scores @ w_normalized

            # Compute F-beta score based on precision priority
            beta = (1 - self.precision_priority) / (self.precision_priority + 1e-10)
            beta = np.clip(beta, 0.1, 10.0)

            f_beta = self._compute_f_beta(combined_scores, labels, beta)
            return -f_beta  # Negative for minimization

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: all weights >= 0
        bounds = [(0.01, 1.0)] * self.n_detectors

        # Optimize
        result = optimize.minimize(
            objective,
            self.weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iterations, "ftol": 1e-6},
        )

        self.optimized_weights = np.abs(result.x)
        self.optimized_weights = self.optimized_weights / np.sum(self.optimized_weights)

        # Compute optimized F1
        optimized_scores = detector_scores @ self.optimized_weights
        optimized_f1 = self._compute_f1(optimized_scores, labels)

        optimization_result = {
            "domain": self.domain.value,
            "initial_weights": self.weights.tolist(),
            "optimized_weights": self.optimized_weights.tolist(),
            "baseline_f1": baseline_f1,
            "optimized_f1": optimized_f1,
            "f1_improvement": (optimized_f1 - baseline_f1) / (baseline_f1 + 1e-10),
            "precision_priority": self.precision_priority,
            "convergence": {
                "success": result.success,
                "iterations": result.nit,
                "message": result.message,
            },
        }

        self.optimization_history.append(optimization_result)
        logger.info(
            f"Domain weight optimization for {self.domain.value}: "
            f"F1 {baseline_f1:.4f} -> {optimized_f1:.4f}"
        )

        return optimization_result

    def _compute_f1(self, scores: NDArray[np.float64], labels: NDArray[np.int32]) -> float:
        """Compute F1 score at optimal threshold."""
        threshold = np.percentile(scores, 95)  # Top 5% as anomalies
        predictions = scores > threshold

        tp = np.sum((labels == 1) & predictions)
        fp = np.sum((labels == 0) & predictions)
        fn = np.sum((labels == 1) & ~predictions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _compute_f_beta(
        self,
        scores: NDArray[np.float64],
        labels: NDArray[np.int32],
        beta: float,
    ) -> float:
        """Compute F-beta score at optimal threshold."""
        threshold = np.percentile(scores, 95)
        predictions = scores > threshold

        tp = np.sum((labels == 1) & predictions)
        fp = np.sum((labels == 0) & predictions)
        fn = np.sum((labels == 1) & ~predictions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    def combine_scores(self, detector_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Combine detector scores using optimized weights.

        Args:
            detector_scores: Shape (n_samples, n_detectors)

        Returns:
            Combined scores
        """
        weights = self.optimized_weights if self.optimized_weights is not None else self.weights
        return detector_scores @ weights


# Factory function for easy creation
def create_domain_threshold_manager(
    domain: DomainType | str,
) -> AdaptiveDomainThresholdManager:
    """
    Create an adaptive threshold manager for the specified domain.

    Args:
        domain: Target domain

    Returns:
        Configured threshold manager
    """
    return AdaptiveDomainThresholdManager(domain)


# Convenience exports
__all__ = [
    "DOMAIN_DEFAULTS",
    "AdaptiveDomainThresholdManager",
    "CalibrationEnsemble",
    "DomainCalibrationResult",
    "DomainEnsembleWeightOptimizer",
    "DomainThresholdConfig",
    "DomainType",
    "IsotonicCalibrator",
    "PlattScalingCalibrator",
    "create_domain_threshold_manager",
]

"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Uncertainty Quantification Module

Provides rigorous uncertainty estimation for neuro-symbolic AI:
- Epistemic uncertainty: Model uncertainty (reducible with more data)
- Aleatoric uncertainty: Data uncertainty (irreducible)
- Confidence calibration
- Uncertainty-aware decision making

Research Sources:
- DARPA ANSR: Trustworthy AI requires quantified uncertainty
- Bayesian Deep Learning: Epistemic uncertainty estimation
- Conformal Prediction: Distribution-free confidence
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty."""

    EPISTEMIC = "epistemic"  # Model uncertainty - what we don't know
    ALEATORIC = "aleatoric"  # Data uncertainty - inherent randomness
    TOTAL = "total"  # Combined uncertainty


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""

    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class UncertaintyEstimate:
    """Complete uncertainty estimate for a prediction."""

    prediction: float
    epistemic: float  # Model uncertainty
    aleatoric: float  # Data uncertainty
    total: float  # Combined
    confidence: float  # Calibrated confidence
    confidence_interval: tuple[float, float]
    calibration_error: float
    is_reliable: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.total,
            "confidence": self.confidence,
            "ci": self.confidence_interval,
            "calibration_error": self.calibration_error,
            "reliable": self.is_reliable,
            "explanation": self.explanation,
        }


@dataclass
class CalibrationResult:
    """Result of calibration assessment."""

    expected_confidence: list[float]
    observed_accuracy: list[float]
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    is_calibrated: bool
    reliability_diagram: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "calibrated": self.is_calibrated,
            "diagram": self.reliability_diagram,
        }


class UncertaintyQuantifier:
    """
    Uncertainty Quantification Engine.

    Provides rigorous uncertainty estimation following DARPA ANSR
    requirements for trustworthy AI:

    1. Epistemic Uncertainty: What the model doesn't know
       - High with limited training data
       - High in out-of-distribution regions
       - Reducible with more data

    2. Aleatoric Uncertainty: Inherent data randomness
       - Independent of model
       - Cannot be reduced with more data
       - Represents measurement noise, stochasticity

    3. Calibration: Confidence should match accuracy
       - A 90% confident prediction should be correct 90% of the time
       - Essential for trustworthy decision-making
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        n_monte_carlo: int = 30,
        calibration_bins: int = 10,
        reliability_threshold: float = 0.1,
    ):
        """
        Initialize Uncertainty Quantifier.

        Args:
            n_monte_carlo: Number of MC samples for epistemic estimation
            calibration_bins: Number of bins for calibration
            reliability_threshold: Max ECE for reliable predictions
        """
        self.n_monte_carlo = n_monte_carlo
        self.calibration_bins = calibration_bins
        self.reliability_threshold = reliability_threshold

        # Calibration history
        self._predictions: list[float] = []
        self._confidences: list[float] = []
        self._outcomes: list[bool] = []

        # Statistics
        self._stats = {
            "estimates_computed": 0,
            "calibrations_performed": 0,
            "avg_epistemic": 0.0,
            "avg_aleatoric": 0.0,
        }

        logger.info(f"UncertaintyQuantifier initialized (MC={n_monte_carlo})")

    def estimate_uncertainty(
        self,
        predictions: np.ndarray,
        prediction_function: Callable[[np.ndarray], np.ndarray] | None = None,
        input_data: np.ndarray | None = None,
        return_samples: bool = False,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for a prediction.

        Uses Monte Carlo sampling if prediction function provided,
        otherwise estimates from prediction variance.

        Args:
            predictions: Model predictions (can be single or multiple samples)
            prediction_function: Function to generate more predictions
            input_data: Input data for prediction function
            return_samples: Whether to return MC samples

        Returns:
            Complete uncertainty estimate
        """
        self._stats["estimates_computed"] += 1

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Get multiple predictions for uncertainty estimation
        if prediction_function is not None and input_data is not None:
            mc_predictions = self._monte_carlo_sampling(
                prediction_function, input_data, self.n_monte_carlo
            )
        else:
            mc_predictions = predictions

        # Epistemic uncertainty: variance across model predictions
        if mc_predictions.shape[0] > 1:
            epistemic = float(np.var(mc_predictions, axis=0).mean())
        else:
            epistemic = 0.1  # Default if no MC sampling

        # Aleatoric uncertainty: estimated from prediction spread
        aleatoric = self._estimate_aleatoric(mc_predictions)

        # Total uncertainty
        total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

        # Mean prediction
        prediction = float(mc_predictions.mean())

        # Confidence interval (using total uncertainty)
        ci_low = prediction - 1.96 * np.sqrt(total)
        ci_high = prediction + 1.96 * np.sqrt(total)

        # Calibrated confidence
        confidence = self._compute_confidence(epistemic, aleatoric)
        calibration_error = self._estimate_calibration_error()

        # Determine reliability
        is_reliable = (
            calibration_error < self.reliability_threshold
            and epistemic < 0.5
            and confidence > 0.5
        )

        # Generate explanation
        explanation = self._generate_explanation(
            epistemic, aleatoric, confidence, is_reliable
        )

        # Update running averages
        self._stats["avg_epistemic"] = (
            0.9 * self._stats["avg_epistemic"] + 0.1 * epistemic
        )
        self._stats["avg_aleatoric"] = (
            0.9 * self._stats["avg_aleatoric"] + 0.1 * aleatoric
        )

        return UncertaintyEstimate(
            prediction=prediction,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence,
            confidence_interval=(ci_low, ci_high),
            calibration_error=calibration_error,
            is_reliable=is_reliable,
            explanation=explanation,
        )

    def estimate_epistemic(
        self,
        prediction_function: Callable[[np.ndarray], np.ndarray],
        input_data: np.ndarray,
    ) -> float:
        """
        Estimate epistemic (model) uncertainty using MC dropout or ensemble.

        Args:
            prediction_function: Function to generate predictions
            input_data: Input data

        Returns:
            Epistemic uncertainty estimate
        """
        mc_predictions = self._monte_carlo_sampling(
            prediction_function, input_data, self.n_monte_carlo
        )
        return float(np.var(mc_predictions, axis=0).mean())

    def estimate_aleatoric(
        self,
        data: np.ndarray,
        window_size: int = 10,
    ) -> float:
        """
        Estimate aleatoric (data) uncertainty.

        Uses local variance to estimate inherent data noise.

        Args:
            data: Data samples
            window_size: Window for local variance estimation

        Returns:
            Aleatoric uncertainty estimate
        """
        if len(data) < window_size:
            return float(np.std(data))

        # Rolling variance
        variances = []
        for i in range(len(data) - window_size + 1):
            window = data[i : i + window_size]
            variances.append(np.var(window))

        return float(np.mean(variances))

    def calibrate(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        outcomes: np.ndarray,
    ) -> CalibrationResult:
        """
        Assess and compute calibration of predictions.

        Args:
            predictions: Model predictions
            confidences: Confidence scores
            outcomes: Actual outcomes (binary)

        Returns:
            Calibration assessment
        """
        self._stats["calibrations_performed"] += 1

        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        expected_conf = []
        observed_acc = []
        bin_counts = []

        for i in range(self.calibration_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            bin_count = mask.sum()

            if bin_count > 0:
                expected_conf.append(confidences[mask].mean())
                observed_acc.append(outcomes[mask].mean())
                bin_counts.append(bin_count)
            else:
                expected_conf.append(bin_centers[i])
                observed_acc.append(bin_centers[i])
                bin_counts.append(0)

        # Expected Calibration Error
        bin_counts_arr = np.array(bin_counts)
        expected_conf_arr = np.array(expected_conf)
        observed_acc_arr = np.array(observed_acc)

        total_samples = bin_counts_arr.sum()
        if total_samples > 0:
            ece = np.sum(
                bin_counts_arr * np.abs(expected_conf_arr - observed_acc_arr)
            ) / total_samples
        else:
            ece = 0.0

        # Maximum Calibration Error
        mce = float(np.max(np.abs(expected_conf_arr - observed_acc_arr)))

        # Store for future calibration
        self._predictions.extend(predictions.tolist())
        self._confidences.extend(confidences.tolist())
        self._outcomes.extend(outcomes.tolist())

        return CalibrationResult(
            expected_confidence=expected_conf,
            observed_accuracy=observed_acc,
            ece=float(ece),
            mce=mce,
            is_calibrated=ece < self.reliability_threshold,
            reliability_diagram={
                "expected": expected_conf,
                "observed": observed_acc,
                "counts": bin_counts,
            },
        )

    def uncertainty_aware_decision(
        self,
        uncertainty: UncertaintyEstimate,
        action_threshold: float = 0.5,
        uncertainty_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Make uncertainty-aware decisions.

        Args:
            uncertainty: Uncertainty estimate
            action_threshold: Threshold for taking action
            uncertainty_threshold: Max uncertainty for confident action

        Returns:
            Decision with explanation
        """
        decision = {
            "should_act": False,
            "should_defer": False,
            "should_collect_more_data": False,
            "action": "wait",
            "reason": "",
        }

        if uncertainty.total > uncertainty_threshold:
            # High uncertainty - need more data
            decision["should_collect_more_data"] = True
            decision["action"] = "collect_more_data"
            decision["reason"] = f"High uncertainty ({uncertainty.total:.2f}) - need more information"

            if uncertainty.epistemic > uncertainty.aleatoric:
                decision["reason"] += " (model uncertainty dominates - more training data needed)"
            else:
                decision["reason"] += " (data uncertainty dominates - inherent variability)"

        elif not uncertainty.is_reliable:
            # Unreliable prediction - defer to human
            decision["should_defer"] = True
            decision["action"] = "defer_to_human"
            decision["reason"] = f"Prediction may not be reliable (calibration error: {uncertainty.calibration_error:.2f})"

        elif uncertainty.prediction > action_threshold and uncertainty.confidence > 0.7:
            # Confident positive prediction
            decision["should_act"] = True
            decision["action"] = "take_action"
            decision["reason"] = f"Confident prediction ({uncertainty.confidence:.0%}) above threshold"

        else:
            # Confident negative or borderline
            decision["action"] = "monitor"
            decision["reason"] = f"Prediction ({uncertainty.prediction:.2f}) with confidence {uncertainty.confidence:.0%}"

        return decision

    def decompose_uncertainty(
        self,
        predictions_ensemble: np.ndarray,
    ) -> dict[str, float]:
        """
        Decompose total uncertainty into epistemic and aleatoric components.

        Args:
            predictions_ensemble: Ensemble of predictions (models x samples x outputs)

        Returns:
            Decomposed uncertainties
        """
        if predictions_ensemble.ndim < 2:
            return {"epistemic": 0.0, "aleatoric": 0.0, "total": float(np.std(predictions_ensemble))}

        # Epistemic: variance of means
        model_means = predictions_ensemble.mean(axis=1)
        epistemic = float(np.var(model_means))

        # Aleatoric: mean of variances
        model_vars = predictions_ensemble.var(axis=1)
        aleatoric = float(np.mean(model_vars))

        # Total
        total = np.sqrt(epistemic + aleatoric)

        return {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "epistemic_ratio": epistemic / (epistemic + aleatoric) if (epistemic + aleatoric) > 0 else 0.5,
        }

    def conformal_prediction(
        self,
        calibration_scores: np.ndarray,
        test_score: float,
        alpha: float = 0.1,
    ) -> dict[str, Any]:
        """
        Compute conformal prediction interval.

        Provides distribution-free coverage guarantee.

        Args:
            calibration_scores: Nonconformity scores from calibration set
            test_score: Score for test point
            alpha: Significance level (1-alpha = coverage)

        Returns:
            Conformal prediction result
        """
        n = len(calibration_scores)

        # Compute quantile
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(1.0, quantile_level)

        threshold = np.quantile(calibration_scores, quantile_level)

        is_in_set = test_score <= threshold

        return {
            "in_prediction_set": is_in_set,
            "threshold": float(threshold),
            "test_score": test_score,
            "coverage_guarantee": 1 - alpha,
            "calibration_size": n,
        }

    def _monte_carlo_sampling(
        self,
        prediction_function: Callable,
        input_data: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Generate Monte Carlo samples for uncertainty estimation."""
        samples = []
        for _ in range(n_samples):
            try:
                pred = prediction_function(input_data)
                samples.append(pred)
            except Exception:
                continue

        if not samples:
            return input_data  # Fallback

        return np.array(samples)

    def _estimate_aleatoric(self, predictions: np.ndarray) -> float:
        """Estimate aleatoric uncertainty from predictions."""
        if predictions.ndim == 1:
            return float(np.std(predictions) * 0.5)

        # Use prediction range as proxy for aleatoric
        pred_range = predictions.max() - predictions.min()
        return float(pred_range / (2 * self.PHI))

    def _compute_confidence(
        self,
        epistemic: float,
        aleatoric: float,
    ) -> float:
        """Compute calibrated confidence from uncertainties."""
        total = np.sqrt(epistemic ** 2 + aleatoric ** 2)

        # Transform uncertainty to confidence
        # Lower uncertainty = higher confidence
        confidence = np.exp(-total * self.PHI)
        return float(np.clip(confidence, 0.1, 0.99))

    def _estimate_calibration_error(self) -> float:
        """Estimate current calibration error from history."""
        if len(self._outcomes) < 10:
            return 0.15  # Default for limited data

        # Simple binned ECE
        confidences = np.array(self._confidences[-100:])
        outcomes = np.array(self._outcomes[-100:])

        bin_edges = np.linspace(0, 1, 6)
        ece = 0.0

        for i in range(len(bin_edges) - 1):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                avg_conf = confidences[mask].mean()
                avg_acc = outcomes[mask].mean()
                ece += mask.sum() * abs(avg_conf - avg_acc)

        return float(ece / len(confidences))

    def _generate_explanation(
        self,
        epistemic: float,
        aleatoric: float,
        confidence: float,
        is_reliable: bool,
    ) -> str:
        """Generate human-readable explanation of uncertainty."""
        parts = []

        # Overall assessment
        if is_reliable:
            parts.append(f"Prediction is reliable (confidence: {confidence:.0%}).")
        else:
            parts.append(f"Prediction has limited reliability (confidence: {confidence:.0%}).")

        # Uncertainty breakdown
        if epistemic > aleatoric:
            parts.append(
                f"Model uncertainty ({epistemic:.2f}) exceeds data uncertainty ({aleatoric:.2f}). "
                "More training data could improve predictions."
            )
        else:
            parts.append(
                f"Data uncertainty ({aleatoric:.2f}) exceeds model uncertainty ({epistemic:.2f}). "
                "Variability is inherent to this domain."
            )

        return " ".join(parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get quantifier statistics."""
        return {
            **self._stats,
            "prediction_history_size": len(self._predictions),
            "current_calibration_error": self._estimate_calibration_error(),
        }

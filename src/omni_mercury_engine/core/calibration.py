"""
Mercury Agent - Probability Calibration Module
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Implements calibration methods to align confidence scores with true error rates:
- Platt Scaling (logistic regression on scores)
- Isotonic Regression (non-parametric monotonic calibration)
- Temperature Scaling (single-parameter neural network calibration)
- Reliability diagrams for visualization
- Target: Brier score < 0.05 for well-calibrated predictions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""

    method: str
    brier_before: float
    brier_after: float
    ece_before: float  # Expected Calibration Error
    ece_after: float
    mce_before: float  # Maximum Calibration Error
    mce_after: float
    improvement_percent: float
    reliability_curve: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "method": self.method,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "mce_before": self.mce_before,
            "mce_after": self.mce_after,
            "improvement_percent": self.improvement_percent,
            "meets_target": self.brier_after < 0.05,
        }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and accuracy
    across probability bins, weighted by bin size.

    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Expected Calibration Error (lower is better, 0 is perfect)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask) / len(y_prob)
            ece += bin_size * np.abs(bin_acc - bin_conf)

    return ece


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum gap between predicted confidence and accuracy
    across all probability bins.

    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Maximum Calibration Error (lower is better, 0 is perfect)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            mce = max(mce, np.abs(bin_acc - bin_conf))

    return mce


class PlattScaling:
    """
    Platt Scaling calibration using logistic regression.

    Fits a logistic regression model to map raw scores/probabilities
    to calibrated probabilities. Works well when the uncalibrated
    outputs are sigmoidal or when the calibration curve is S-shaped.

    Reference: Platt (1999) "Probabilistic Outputs for SVMs"
    """

    def __init__(self, solver: str = "lbfgs", max_iter: int = 1000):
        """
        Initialize Platt scaling.

        Args:
            solver: Solver for logistic regression
            max_iter: Maximum iterations for convergence
        """
        self.model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=1e10,  # Minimal regularization
        )
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> PlattScaling:
        """
        Fit Platt scaling calibrator.

        Args:
            y_prob: Uncalibrated probability predictions (n_samples,)
            y_true: Binary ground truth labels (n_samples,)

        Returns:
            Self for method chaining
        """
        # Reshape for sklearn
        X = y_prob.reshape(-1, 1)

        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present, calibration skipped")
            self._fitted = False
            return self

        self.model.fit(X, y_true)
        self._fitted = True

        logger.debug(
            f"PlattScaling fitted: coef={self.model.coef_[0, 0]:.4f}, "
            f"intercept={self.model.intercept_[0]:.4f}"
        )
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling calibration.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            return y_prob

        X = y_prob.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibration:
    """
    Isotonic regression calibration (non-parametric).

    Fits a stepwise non-decreasing function to map raw scores
    to calibrated probabilities. More flexible than Platt scaling
    but may overfit with small datasets.

    Reference: Zadrozny & Elkan (2002) "Transforming Classifier Scores"
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Initialize isotonic calibration.

        Args:
            out_of_bounds: How to handle out-of-bounds values ("clip", "nan")
        """
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds=out_of_bounds,
        )
        self._fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibration:
        """
        Fit isotonic regression calibrator.

        Args:
            y_prob: Uncalibrated probability predictions
            y_true: Binary ground truth labels

        Returns:
            Self for method chaining
        """
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present, calibration skipped")
            self._fitted = False
            return self

        self.model.fit(y_prob, y_true)
        self._fitted = True

        logger.debug("IsotonicCalibration fitted")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            return y_prob

        return self.model.predict(y_prob)


class TemperatureScaling:
    """
    Temperature Scaling calibration (single-parameter).

    Divides logits by a learned temperature parameter T before
    applying softmax. Simple but effective for neural networks.
    Optimal T is found by minimizing negative log-likelihood (NLL).

    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """

    def __init__(self, max_iter: int = 100, lr: float = 0.01):
        """
        Initialize temperature scaling.

        Args:
            max_iter: Maximum optimization iterations
            lr: Learning rate for gradient descent
        """
        self.temperature = 1.0
        self.max_iter = max_iter
        self.lr = lr
        self._fitted = False

    def _logit(self, p: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        return 1 / (1 + np.exp(-z))

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> TemperatureScaling:
        """
        Fit temperature parameter by minimizing NLL.

        Args:
            y_prob: Uncalibrated probability predictions
            y_true: Binary ground truth labels

        Returns:
            Self for method chaining
        """
        if len(np.unique(y_true)) < 2:
            self._fitted = False
            return self

        logits = self._logit(y_prob)

        # Grid search for optimal temperature
        best_nll = float("inf")
        best_T = 1.0

        for T in np.logspace(-1, 1, 50):  # 0.1 to 10
            scaled_probs = self._sigmoid(logits / T)
            nll = log_loss(y_true, scaled_probs)

            if nll < best_nll:
                best_nll = nll
                best_T = T

        self.temperature = best_T
        self._fitted = True

        logger.debug(f"TemperatureScaling fitted: T={self.temperature:.4f}")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            return y_prob

        logits = self._logit(y_prob)
        return self._sigmoid(logits / self.temperature)


class CalibrationEnsemble:
    """
    Ensemble of calibration methods with automatic selection.

    Combines multiple calibration approaches and selects the best
    based on validation performance (Brier score).
    """

    def __init__(self):
        """Initialize calibration ensemble."""
        self.calibrators = {
            "platt": PlattScaling(),
            "isotonic": IsotonicCalibration(),
            "temperature": TemperatureScaling(),
        }
        self.best_method: str | None = None
        self._fitted = False

    def fit(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        validation_split: float = 0.3,
    ) -> CalibrationEnsemble:
        """
        Fit all calibrators and select best.

        Args:
            y_prob: Uncalibrated probabilities
            y_true: Binary ground truth labels
            validation_split: Fraction for validation

        Returns:
            Self for method chaining
        """
        n = len(y_prob)
        n_val = int(n * validation_split)

        # Split for validation
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[n_val:], idx[:n_val]

        y_prob_train = y_prob[train_idx]
        y_true_train = y_true[train_idx]
        y_prob_val = y_prob[val_idx]
        y_true_val = y_true[val_idx]

        best_brier = float("inf")

        for name, calibrator in self.calibrators.items():
            calibrator.fit(y_prob_train, y_true_train)
            calibrated = calibrator.calibrate(y_prob_val)

            try:
                brier = brier_score_loss(y_true_val, calibrated)
            except ValueError:
                brier = float("inf")

            if brier < best_brier:
                best_brier = brier
                self.best_method = name

        # Refit best on all data
        if self.best_method:
            self.calibrators[self.best_method].fit(y_prob, y_true)

        self._fitted = True
        logger.info(f"CalibrationEnsemble selected: {self.best_method} (Brier={best_brier:.4f})")
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply best calibration method.

        Args:
            y_prob: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted or not self.best_method:
            return y_prob

        return self.calibrators[self.best_method].calibrate(y_prob)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    method: str = "Unknown",
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Evaluate calibration improvement.

    Args:
        y_true: Binary ground truth labels
        y_prob_uncalibrated: Uncalibrated probabilities
        y_prob_calibrated: Calibrated probabilities
        method: Name of calibration method used
        n_bins: Number of bins for reliability diagram

    Returns:
        CalibrationResult with before/after metrics
    """
    # Brier scores
    brier_before = brier_score_loss(y_true, y_prob_uncalibrated)
    brier_after = brier_score_loss(y_true, y_prob_calibrated)

    # ECE/MCE
    ece_before = compute_ece(y_true, y_prob_uncalibrated, n_bins)
    ece_after = compute_ece(y_true, y_prob_calibrated, n_bins)
    mce_before = compute_mce(y_true, y_prob_uncalibrated, n_bins)
    mce_after = compute_mce(y_true, y_prob_calibrated, n_bins)

    # Improvement
    improvement = (brier_before - brier_after) / brier_before * 100 if brier_before > 0 else 0.0

    # Reliability curves
    try:
        prob_true_before, prob_pred_before = calibration_curve(
            y_true, y_prob_uncalibrated, n_bins=n_bins, strategy="uniform"
        )
        prob_true_after, prob_pred_after = calibration_curve(
            y_true, y_prob_calibrated, n_bins=n_bins, strategy="uniform"
        )
        reliability_curve = {
            "before_true": prob_true_before,
            "before_pred": prob_pred_before,
            "after_true": prob_true_after,
            "after_pred": prob_pred_after,
        }
    except ValueError:
        reliability_curve = {}

    result = CalibrationResult(
        method=method,
        brier_before=brier_before,
        brier_after=brier_after,
        ece_before=ece_before,
        ece_after=ece_after,
        mce_before=mce_before,
        mce_after=mce_after,
        improvement_percent=improvement,
        reliability_curve=reliability_curve,
    )

    logger.info(
        f"Calibration ({method}): Brier {brier_before:.4f} -> {brier_after:.4f} "
        f"({improvement:+.1f}%), ECE {ece_before:.4f} -> {ece_after:.4f}"
    )

    return result


def calibrate_detector(
    detector: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "auto",
) -> tuple[Any, CalibrationResult | None]:
    """
    Calibrate a detector's probability outputs.

    Args:
        detector: Fitted anomaly detector with predict_proba
        X_cal: Calibration set features
        y_cal: Calibration set labels
        method: Calibration method ("platt", "isotonic", "temperature", "auto")

    Returns:
        Tuple of (calibrator, CalibrationResult)
    """
    # Get uncalibrated predictions
    try:
        y_prob = detector.predict_proba(X_cal)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
    except (AttributeError, NotImplementedError):
        logger.warning("Detector does not support predict_proba, skipping calibration")
        return None, None

    # Select calibrator
    if method == "auto":
        calibrator = CalibrationEnsemble()
    elif method == "platt":
        calibrator = PlattScaling()
    elif method == "isotonic":
        calibrator = IsotonicCalibration()
    elif method == "temperature":
        calibrator = TemperatureScaling()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Fit and evaluate
    calibrator.fit(y_prob, y_cal)
    y_prob_calibrated = calibrator.calibrate(y_prob)

    result = evaluate_calibration(
        y_cal,
        y_prob,
        y_prob_calibrated,
        method=method if method != "auto" else getattr(calibrator, "best_method", "auto"),
    )

    return calibrator, result

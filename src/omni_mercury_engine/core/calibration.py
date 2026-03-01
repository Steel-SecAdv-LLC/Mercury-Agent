"""
Mercury Agent - Probability Calibration Module
Copyright (C) 2025 Steel Security Advisors LLC

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
        try:
            from omni_mercury_engine.ml._native_utils import (
                NativeLogisticRegression as LogisticRegression,
            )
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

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
        # Reshape for calibration model input
        X = y_prob.reshape(-1, 1)

        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class present, calibration skipped")
            self._fitted = False
            return self

        self.model.fit(X, y_true)
        self._fitted = True

        assert self.model.coef_ is not None
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
        return np.asarray(self.model.predict_proba(X)[:, 1])  # type: ignore[no-any-return, unused-ignore]


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
        try:
            from omni_mercury_engine.ml._native_utils import (
                NativeIsotonicRegression as IsotonicRegression,
            )
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

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

        return np.asarray(self.model.predict(y_prob))  # type: ignore[no-any-return, unused-ignore]


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
        return np.asarray(np.log(p / (1 - p)))  # type: ignore[no-any-return, unused-ignore]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        return np.asarray(1 / (1 + np.exp(-z)))  # type: ignore[no-any-return, unused-ignore]

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

        try:
            from omni_mercury_engine.ml._native_utils import native_log_loss as log_loss
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

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

    def __init__(self) -> None:
        """Initialize calibration ensemble."""
        self.calibrators: dict[str, PlattScaling | IsotonicCalibration | TemperatureScaling] = {
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
                from omni_mercury_engine.ml._native_utils import (
                    native_brier_score_loss as brier_score_loss,
                )
            except ImportError as e:
                raise ImportError(
                    "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
                ) from e

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
    try:
        from omni_mercury_engine.ml._native_utils import (
            native_brier_score_loss as brier_score_loss,
            native_calibration_curve as calibration_curve,
        )
    except ImportError as e:
        raise ImportError(
            "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
        ) from e

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

    Implements a robust fallback cascade to obtain probability scores from
    any detector type, then applies calibration to improve probability estimates.

    Args:
        detector: Fitted anomaly detector (supports predict_proba, decision_function,
            score_samples, or predict).
        X_cal: Calibration set features.
        y_cal: Calibration set labels.
        method: Calibration method ("platt", "isotonic", "temperature", "auto").

    Returns:
        Tuple of (calibrator, CalibrationResult). Returns (None, None) only if
        all score extraction methods fail.

    Example:
        >>> calibrator, result = calibrate_detector(detector, X_cal, y_cal)
    """
    # Get uncalibrated predictions with robust fallback
    y_prob, score_source = _extract_calibration_scores(detector, X_cal)

    if y_prob is None:
        logger.warning("Could not extract scores from detector, skipping calibration")
        return None, None

    logger.info(f"Extracted scores using {score_source} for calibration")

    # Select calibrator
    calibrator: CalibrationEnsemble | PlattScaling | IsotonicCalibration | TemperatureScaling
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


def _extract_calibration_scores(
    detector: Any,
    X: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    """Extract probability-like scores from any detector for calibration.

    Uses a multi-strategy fallback cascade:
    1. predict_proba - Standard probability output (preferred)
    2. decision_function - Margin scores with sigmoid transformation
    3. score_samples - Density scores with min-max normalization
    4. predict - Binary predictions with synthetic probabilities
    5. Statistical scoring - Feature-based anomaly estimation

    Args:
        detector: Fitted anomaly detector.
        X: Calibration data.

    Returns:
        Tuple of (scores array or None, source method name).
    """
    # Strategy 1: predict_proba (preferred)
    try:
        y_prob = detector.predict_proba(X)
        if y_prob.ndim == 2:
            return y_prob[:, 1], "predict_proba"
        return y_prob, "predict_proba"
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 2: decision_function with sigmoid transformation
    try:
        decision = detector.decision_function(X)
        # Apply sigmoid to convert margins to pseudo-probabilities
        y_prob = 1.0 / (1.0 + np.exp(-decision))
        return y_prob, "decision_function"
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 3: score_samples with min-max normalization
    try:
        scores = detector.score_samples(X)
        # Invert and normalize (higher score_samples = more normal)
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s > min_s:
            y_prob = 1.0 - (scores - min_s) / (max_s - min_s)
        else:
            y_prob = np.full(len(scores), 0.5)
        return y_prob, "score_samples"
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 4: Binary predict with synthetic probabilities
    try:
        predictions = detector.predict(X)
        return _synthesize_probabilities_from_predictions(detector, predictions, X), "predict"
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 5: Statistical scoring fallback
    try:
        y_prob = _compute_statistical_scores_for_calibration(X)
        return y_prob, "statistical"
    except Exception as e:
        logger.warning(f"Statistical scoring failed: {e}")

    return None, "none"


def _synthesize_probabilities_from_predictions(
    detector: Any,
    predictions: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Synthesize continuous probabilities from binary predictions.

    Creates differentiated probability scores for samples within
    each class based on feature characteristics.

    Args:
        detector: The fitted detector (for extracting any stored info).
        predictions: Binary predictions array.
        X: Input features.

    Returns:
        Synthetic probability scores in [0, 1].
    """
    len(predictions)

    # Determine anomaly convention
    if hasattr(detector, "contamination"):
        # scikit-learn convention: -1 = anomaly, 1 = normal
        is_anomaly = predictions == -1
    else:
        # Common convention: 1 = anomaly, 0 = normal
        is_anomaly = predictions == 1

    # Base probabilities
    base_probs = np.where(is_anomaly, 0.75, 0.25)

    # Add variation based on feature distances
    feature_variation = _compute_feature_variation(X)
    anomaly_variation = feature_variation * 0.2
    normal_variation = (1 - feature_variation) * 0.2

    # Apply variation
    probs = np.where(
        is_anomaly,
        base_probs + anomaly_variation,
        base_probs - normal_variation,
    )

    return np.clip(probs, 0.01, 0.99)


def _compute_feature_variation(X: np.ndarray) -> np.ndarray:
    """Compute per-sample feature variation for probability differentiation."""
    # Use standardized feature distances from center
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    z_scores = np.abs((X - mean) / std)
    avg_z = np.mean(z_scores, axis=1)

    # Normalize to [0, 1]
    min_z, max_z = np.min(avg_z), np.max(avg_z)
    if max_z > min_z:
        return np.asarray((avg_z - min_z) / (max_z - min_z))  # type: ignore[no-any-return, unused-ignore]
    return np.full(len(avg_z), 0.5)


def _compute_statistical_scores_for_calibration(X: np.ndarray) -> np.ndarray:
    """Compute statistical anomaly scores when detector provides no scoring.

    Combines multiple statistical methods:
    - Mahalanobis-like distance (using diagonal covariance)
    - k-NN density estimation
    - Percentile extremity

    Args:
        X: Input data.

    Returns:
        Statistical anomaly scores in [0, 1].
    """
    n_samples, n_features = X.shape

    # Method 1: Z-score based distance
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    z_scores = np.abs((X - mean) / std)
    z_distance = np.mean(z_scores, axis=1)
    z_anomaly = 1 - np.exp(-z_distance / 3)

    # Method 2: Simplified local density (5-NN)
    k = min(5, n_samples - 1)
    if k >= 1:
        try:
            from scipy.spatial.distance import cdist

            distances = cdist(X, X, metric="euclidean")
            np.fill_diagonal(distances, np.inf)
            knn_distances = np.partition(distances, k, axis=1)[:, :k]
            avg_knn = np.mean(knn_distances, axis=1)
            max_dist = np.max(avg_knn)
            density_anomaly = avg_knn / max_dist if max_dist > 0 else np.zeros(n_samples)
        except ImportError:
            density_anomaly = np.zeros(n_samples)
    else:
        density_anomaly = np.zeros(n_samples)

    # Method 3: Percentile extremity
    extremity = np.zeros(n_samples)
    for j in range(n_features):
        ranks = np.argsort(np.argsort(X[:, j]))
        pct = ranks / (n_samples - 1) if n_samples > 1 else np.zeros(n_samples)
        extremity += np.abs(pct - 0.5) * 2
    pct_anomaly = extremity / n_features

    # Weighted ensemble
    scores = 0.4 * z_anomaly + 0.35 * density_anomaly + 0.25 * pct_anomaly

    return np.asarray(np.clip(scores, 0.0, 1.0))  # type: ignore[no-any-return, unused-ignore]

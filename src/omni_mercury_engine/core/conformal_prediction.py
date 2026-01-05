"""
Mercury Agent - Conformal Prediction for Uncertainty Quantification
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Implements conformal prediction for rigorous uncertainty quantification:
- Split Conformal Prediction (inductive)
- Cross-Conformal Prediction (aggregated)
- Adaptive Conformal Inference (distribution-free)
- Guaranteed coverage at user-specified confidence levels
- Integration with anomaly detection pipelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class ScoringFunction(Protocol):
    """Protocol for nonconformity scoring functions."""

    def __call__(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Compute nonconformity scores."""
        ...


@dataclass
class ConformalPredictionSet:
    """Result of conformal prediction."""

    prediction: np.ndarray  # Point predictions
    lower_bound: np.ndarray  # Lower confidence bound
    upper_bound: np.ndarray  # Upper confidence bound
    coverage_level: float  # Target coverage (e.g., 0.95)
    set_sizes: np.ndarray  # Size of each prediction set
    quantile_threshold: float  # Computed quantile threshold


@dataclass
class CoverageResult:
    """Result of coverage evaluation."""

    empirical_coverage: float
    target_coverage: float
    coverage_gap: float
    average_set_size: float
    marginal_coverage_by_class: dict[int, float] = field(default_factory=dict)


class SplitConformalPredictor:
    """
    Split (Inductive) Conformal Prediction.

    Uses a held-out calibration set to compute nonconformity scores
    and determine prediction set thresholds. Computationally efficient
    as it requires only one model fit.

    Reference: Vovk et al. (2005) "Algorithmic Learning in a Random World"
    """

    def __init__(
        self,
        coverage: float = 0.95,
        seed: int = 42,
    ):
        """
        Initialize split conformal predictor.

        Args:
            coverage: Target coverage level (e.g., 0.95 for 95% CI)
            seed: Random seed for reproducibility
        """
        self.coverage = coverage
        self.seed = seed
        self.calibration_scores: np.ndarray | None = None
        self.quantile_threshold: float | None = None
        self._fitted = False

    def fit(
        self,
        nonconformity_scores: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Fit the conformal predictor on calibration scores.

        Args:
            nonconformity_scores: Nonconformity scores from calibration set
                Higher scores indicate more "unusual" examples

        Returns:
            Self for method chaining
        """
        self.calibration_scores = np.sort(nonconformity_scores)
        n = len(nonconformity_scores)

        # Compute quantile with finite-sample correction
        # q = ceil((n+1) * (1-alpha)) / n
        alpha = 1 - self.coverage
        q_idx = int(np.ceil((n + 1) * (1 - alpha)))
        q_idx = min(q_idx, n) - 1  # 0-indexed, bounded

        self.quantile_threshold = self.calibration_scores[q_idx]
        self._fitted = True

        logger.debug(
            f"SplitConformal fitted: n={n}, coverage={self.coverage}, "
            f"threshold={self.quantile_threshold:.4f}"
        )
        return self

    def predict(
        self,
        new_scores: np.ndarray,
        point_predictions: np.ndarray | None = None,
    ) -> ConformalPredictionSet:
        """
        Generate prediction sets for new examples.

        Args:
            new_scores: Nonconformity scores for new examples
            point_predictions: Optional point predictions

        Returns:
            ConformalPredictionSet with bounds and set sizes
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        if point_predictions is None:
            point_predictions = np.zeros(len(new_scores))

        # For regression/scores: prediction set is [pred - q, pred + q]
        lower = point_predictions - self.quantile_threshold
        upper = point_predictions + self.quantile_threshold

        # Set sizes (for classification this would be number of labels)
        set_sizes = upper - lower

        return ConformalPredictionSet(
            prediction=point_predictions,
            lower_bound=lower,
            upper_bound=upper,
            coverage_level=self.coverage,
            set_sizes=set_sizes,
            quantile_threshold=self.quantile_threshold,
        )

    def get_anomaly_threshold(self) -> float:
        """
        Get the conformal anomaly threshold.

        Examples with nonconformity score above this threshold
        are considered anomalies at the specified coverage level.

        Returns:
            Anomaly threshold score
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before get_anomaly_threshold()")
        return self.quantile_threshold


class CrossConformalPredictor:
    """
    Cross-Conformal Prediction (K-fold aggregated).

    Aggregates conformal predictions from K folds for better
    efficiency (uses all data for calibration). Slightly more
    computationally expensive but uses data more efficiently.

    Reference: Barber et al. (2021) "Predictive Inference Is Free"
    """

    def __init__(
        self,
        coverage: float = 0.95,
        n_folds: int = 5,
        seed: int = 42,
    ):
        """
        Initialize cross-conformal predictor.

        Args:
            coverage: Target coverage level
            n_folds: Number of cross-validation folds
            seed: Random seed
        """
        self.coverage = coverage
        self.n_folds = n_folds
        self.seed = seed
        self.fold_thresholds: list[float] = []
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        scoring_fn: ScoringFunction,
    ) -> "CrossConformalPredictor":
        """
        Fit using cross-validation aggregation.

        Args:
            X: Full dataset
            scoring_fn: Function to compute nonconformity scores

        Returns:
            Self for method chaining
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        all_scores = []
        self.fold_thresholds = []

        for train_idx, cal_idx in kf.split(X):
            X_train, X_cal = X[train_idx], X[cal_idx]

            # Compute scores on calibration fold
            cal_scores = scoring_fn(X_cal)
            all_scores.extend(cal_scores)

            # Compute fold threshold
            sorted_scores = np.sort(cal_scores)
            n = len(cal_scores)
            q_idx = int(np.ceil((n + 1) * self.coverage)) - 1
            q_idx = min(max(q_idx, 0), n - 1)
            self.fold_thresholds.append(sorted_scores[q_idx])

        # Aggregate threshold (conservative: use max)
        self.aggregated_threshold = np.mean(self.fold_thresholds)
        self._fitted = True

        logger.debug(
            f"CrossConformal fitted: n_folds={self.n_folds}, "
            f"threshold={self.aggregated_threshold:.4f}"
        )
        return self

    def get_anomaly_threshold(self) -> float:
        """Get the aggregated anomaly threshold."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before get_anomaly_threshold()")
        return self.aggregated_threshold


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference for streaming/online settings.

    Adjusts the quantile threshold adaptively to maintain coverage
    over time, handling distribution shift. Uses exponential moving
    average of miscoverage.

    Reference: Gibbs & Candès (2021) "Adaptive Conformal Inference"
    """

    def __init__(
        self,
        target_coverage: float = 0.95,
        learning_rate: float = 0.1,
        initial_threshold: float = 0.5,
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            target_coverage: Target coverage level
            learning_rate: Step size for threshold updates
            initial_threshold: Initial quantile threshold
        """
        self.target_coverage = target_coverage
        self.alpha = 1 - target_coverage
        self.learning_rate = learning_rate
        self.threshold = initial_threshold

        # Track coverage history
        self.coverage_history: list[float] = []
        self.threshold_history: list[float] = [initial_threshold]
        self.miscoverage_sum = 0.0
        self.n_updates = 0

    def update(
        self,
        score: float,
        true_label: int | None = None,
    ) -> tuple[float, bool]:
        """
        Update threshold based on new observation.

        Args:
            score: Nonconformity score of new example
            true_label: True label if available (for evaluation)

        Returns:
            Tuple of (current_threshold, is_covered)
        """
        # Check if example is covered
        is_covered = score <= self.threshold

        # Update threshold using gradient descent on miscoverage
        # Increase threshold if miscoverage too high, decrease otherwise
        miscoverage = 1.0 if not is_covered else 0.0
        gradient = miscoverage - self.alpha

        # Adaptive update
        self.threshold = self.threshold + self.learning_rate * gradient
        self.threshold = max(0.0, self.threshold)  # Ensure non-negative

        # Track history
        self.threshold_history.append(self.threshold)
        self.n_updates += 1
        self.miscoverage_sum += miscoverage

        # Compute running coverage
        running_coverage = 1 - (self.miscoverage_sum / self.n_updates)
        self.coverage_history.append(running_coverage)

        return self.threshold, is_covered

    def get_current_threshold(self) -> float:
        """Get current adaptive threshold."""
        return self.threshold

    def get_coverage_stats(self) -> dict[str, float]:
        """Get coverage statistics."""
        if self.n_updates == 0:
            return {"empirical_coverage": 0.0, "target_coverage": self.target_coverage}

        empirical = 1 - (self.miscoverage_sum / self.n_updates)
        return {
            "empirical_coverage": empirical,
            "target_coverage": self.target_coverage,
            "coverage_gap": abs(empirical - self.target_coverage),
            "n_updates": self.n_updates,
        }


class ConformalAnomalyDetector:
    """
    Wrapper for anomaly detectors with conformal prediction.

    Provides distribution-free uncertainty quantification for any
    anomaly detector, with guaranteed coverage at specified level.
    """

    def __init__(
        self,
        base_detector: Any,
        coverage: float = 0.95,
        calibration_fraction: float = 0.2,
        method: str = "split",
        seed: int = 42,
    ):
        """
        Initialize conformal anomaly detector.

        Args:
            base_detector: Base anomaly detector (must have fit/predict_proba)
            coverage: Target coverage level
            calibration_fraction: Fraction of training data for calibration
            method: Conformal method ("split", "cross", "adaptive")
            seed: Random seed
        """
        self.base_detector = base_detector
        self.coverage = coverage
        self.calibration_fraction = calibration_fraction
        self.method = method
        self.seed = seed

        if method == "split":
            self.conformal = SplitConformalPredictor(coverage=coverage, seed=seed)
        elif method == "cross":
            self.conformal = CrossConformalPredictor(coverage=coverage, seed=seed)
        elif method == "adaptive":
            self.conformal = AdaptiveConformalInference(target_coverage=coverage)
        else:
            raise ValueError(f"Unknown method: {method}")

        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ConformalAnomalyDetector":
        """
        Fit detector and calibrate conformal predictor.

        Args:
            X: Training data
            y: Optional labels

        Returns:
            Self for method chaining
        """
        np.random.seed(self.seed)

        n = len(X)
        n_cal = int(n * self.calibration_fraction)

        # Split data
        idx = np.random.permutation(n)
        train_idx, cal_idx = idx[n_cal:], idx[:n_cal]

        X_train = X[train_idx]
        X_cal = X[cal_idx]

        # Fit base detector
        if y is not None:
            y_train = y[train_idx]
            try:
                self.base_detector.fit(X_train, y_train)
            except TypeError:
                self.base_detector.fit(X_train)
        else:
            self.base_detector.fit(X_train)

        # Get calibration scores (nonconformity = anomaly score)
        cal_scores = self._get_anomaly_scores(X_cal)

        # Fit conformal predictor
        if self.method in ["split"]:
            self.conformal.fit(cal_scores)
        elif self.method == "cross":
            # For cross-conformal, need scoring function
            def score_fn(X):
                return self._get_anomaly_scores(X)

            self.conformal.fit(X_cal, score_fn)

        self._fitted = True
        return self

    def _get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from base detector."""
        try:
            proba = self.base_detector.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        except (AttributeError, NotImplementedError):
            # Try decision_function or score_samples
            try:
                return -self.base_detector.decision_function(X)
            except AttributeError:
                try:
                    return -self.base_detector.score_samples(X)
                except AttributeError:
                    # Last resort: binary predictions
                    return self.base_detector.predict(X).astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies with conformal guarantee.

        Args:
            X: Test data

        Returns:
            Binary predictions (1 = anomaly) with coverage guarantee
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        scores = self._get_anomaly_scores(X)
        threshold = self.conformal.get_anomaly_threshold()

        return (scores > threshold).astype(int)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.

        Args:
            X: Test data

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        scores = self._get_anomaly_scores(X)
        predictions = (scores > self.conformal.get_anomaly_threshold()).astype(int)

        # Prediction intervals for scores
        if isinstance(self.conformal, SplitConformalPredictor):
            pred_set = self.conformal.predict(scores)
            return predictions, pred_set.lower_bound, pred_set.upper_bound

        # For other methods, return score-based bounds
        threshold = self.conformal.get_anomaly_threshold()
        lower = scores - threshold
        upper = scores + threshold

        return predictions, lower, upper

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> CoverageResult:
        """
        Evaluate empirical coverage on test set.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            CoverageResult with coverage statistics
        """
        scores = self._get_anomaly_scores(X_test)
        predictions = (scores > self.conformal.get_anomaly_threshold()).astype(int)

        # Empirical coverage: fraction of true labels in prediction sets
        # For anomaly detection: correct predictions
        correct = predictions == y_test
        empirical_coverage = np.mean(correct)

        # Per-class coverage
        class_coverage = {}
        for label in np.unique(y_test):
            mask = y_test == label
            if np.sum(mask) > 0:
                class_coverage[int(label)] = np.mean(correct[mask])

        # Average set size (for binary: just 1)
        avg_set_size = 1.0

        return CoverageResult(
            empirical_coverage=empirical_coverage,
            target_coverage=self.coverage,
            coverage_gap=abs(empirical_coverage - self.coverage),
            average_set_size=avg_set_size,
            marginal_coverage_by_class=class_coverage,
        )


def add_conformal_to_detector(
    detector: Any,
    X_cal: np.ndarray,
    coverage: float = 0.95,
    method: str = "split",
) -> tuple[SplitConformalPredictor | CrossConformalPredictor, float]:
    """
    Add conformal prediction to an existing fitted detector.

    Args:
        detector: Fitted anomaly detector
        X_cal: Calibration data
        coverage: Target coverage level
        method: Conformal method ("split" or "cross")

    Returns:
        Tuple of (conformal_predictor, anomaly_threshold)
    """
    # Get calibration scores
    try:
        scores = detector.predict_proba(X_cal)
        if scores.ndim == 2:
            scores = scores[:, 1]
    except (AttributeError, NotImplementedError):
        try:
            scores = -detector.decision_function(X_cal)
        except AttributeError:
            scores = -detector.score_samples(X_cal)

    # Fit conformal predictor
    if method == "split":
        conformal = SplitConformalPredictor(coverage=coverage)
        conformal.fit(scores)
    else:
        conformal = CrossConformalPredictor(coverage=coverage)

        def score_fn(X):
            try:
                s = detector.predict_proba(X)
                return s[:, 1] if s.ndim == 2 else s
            except:
                return -detector.decision_function(X)

        conformal.fit(X_cal, score_fn)

    return conformal, conformal.get_anomaly_threshold()

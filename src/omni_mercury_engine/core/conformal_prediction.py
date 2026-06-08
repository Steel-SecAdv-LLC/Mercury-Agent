"""
Mercury Agent - Conformal Prediction for Uncertainty Quantification

Copyright (C) 2025 Steel Security Advisors LLC

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

logger = logging.getLogger(__name__)


class ScoringFunction(Protocol):
    """Protocol for nonconformity scoring functions."""

    def __call__(
        self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any] | None = None
    ) -> np.ndarray[Any, Any]:
        """Compute nonconformity scores."""
        ...


@dataclass
class ConformalPredictionSet:
    """Result of conformal prediction."""

    prediction: np.ndarray[Any, Any]  # Point predictions
    lower_bound: np.ndarray[Any, Any]  # Lower confidence bound
    upper_bound: np.ndarray[Any, Any]  # Upper confidence bound
    coverage_level: float  # Target coverage (e.g., 0.95)
    set_sizes: np.ndarray[Any, Any]  # Size of each prediction set
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
        self.calibration_scores: np.ndarray[Any, Any] | None = None
        self.quantile_threshold: float | None = None
        self._fitted = False

    def fit(
        self,
        nonconformity_scores: np.ndarray[Any, Any],
    ) -> SplitConformalPredictor:
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
        new_scores: np.ndarray[Any, Any],
        point_predictions: np.ndarray[Any, Any] | None = None,
    ) -> ConformalPredictionSet:
        """
        Generate prediction sets for new examples.

        Args:
            new_scores: Nonconformity scores for new examples
            point_predictions: Optional point predictions

        Returns:
            ConformalPredictionSet with bounds and set sizes
        """
        if not self._fitted or self.quantile_threshold is None:
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
        if not self._fitted or self.quantile_threshold is None:
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
        X: np.ndarray[Any, Any],
        scoring_fn: ScoringFunction,
    ) -> CrossConformalPredictor:
        """
        Fit using cross-validation aggregation.

        Args:
            X: Full dataset
            scoring_fn: Function to compute nonconformity scores

        Returns:
            Self for method chaining
        """
        try:
            from omni_mercury_engine.ml.mercury_ml import KFold
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        all_scores: list[float] = []  # type: ignore[var-annotated, unused-ignore]
        self.fold_thresholds = []

        for train_idx, cal_idx in kf.split(X):
            X_cal = X[cal_idx]

            # Compute scores on calibration fold
            cal_scores = scoring_fn(X_cal)
            all_scores.extend(cal_scores)

            # Compute fold threshold
            sorted_scores = np.sort(cal_scores)
            n = len(cal_scores)
            q_idx = int(np.ceil((n + 1) * self.coverage)) - 1
            q_idx = min(max(q_idx, 0), n - 1)
            self.fold_thresholds.append(sorted_scores[q_idx])

        # Aggregate threshold: use maximum across folds for conservative
        # coverage guarantee (ensures at least target coverage in each fold).
        self.aggregated_threshold = float(np.max(self.fold_thresholds))
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
        return float(self.aggregated_threshold)


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

    Provides distribution-free uncertainty quantification for any anomaly detector, with guaranteed
    coverage at specified level.
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

        # Type annotation for conformal predictor
        self.conformal: (
            SplitConformalPredictor | CrossConformalPredictor | AdaptiveConformalInference
        )
        if method == "split":
            self.conformal = SplitConformalPredictor(coverage=coverage, seed=seed)
        elif method == "cross":
            self.conformal = CrossConformalPredictor(coverage=coverage, seed=seed)
        elif method == "adaptive":
            self.conformal = AdaptiveConformalInference(target_coverage=coverage)
        else:
            raise ValueError(f"Unknown method: {method}")

        self._fitted = False

    def fit(
        self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any] | None = None
    ) -> ConformalAnomalyDetector:
        """
        Fit detector and calibrate conformal predictor.

        Args:
            X: Training data
            y: Optional labels

        Returns:
            Self for method chaining
        """
        rng = np.random.default_rng(self.seed)

        n = len(X)
        n_cal = int(n * self.calibration_fraction)

        # Split data
        idx = rng.permutation(n)
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
        if isinstance(self.conformal, SplitConformalPredictor):
            self.conformal.fit(cal_scores)
        elif isinstance(self.conformal, CrossConformalPredictor):
            # For cross-conformal, need scoring function
            def score_fn(
                X_input: np.ndarray[Any, Any], y: np.ndarray[Any, Any] | None = None
            ) -> np.ndarray[Any, Any]:
                return self._get_anomaly_scores(X_input)

            self.conformal.fit(X_cal, score_fn)  # type: ignore[arg-type]

        self._fitted = True
        return self

    def _get_anomaly_scores(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Get anomaly scores from base detector with robust fallback cascade.

        Implements a multi-strategy fallback mechanism to obtain continuous
        anomaly scores from any detector, regardless of its API.

        Strategy cascade:
        1. predict_proba - Standard probability output
        2. decision_function - SVM/linear model margins
        3. score_samples - Density-based log-likelihood
        4. predict + distance fusion - Binary with synthetic scores
        5. Ensemble scoring - Multiple feature-based estimates

        Args:
            X: Input data array of shape (n_samples, n_features).

        Returns:
            Continuous anomaly scores in [0, 1] range.
        """
        # Strategy 1: predict_proba (preferred)
        try:
            proba = self.base_detector.predict_proba(X)
            if proba.ndim == 2:
                return np.asarray(proba[:, 1])  # type: ignore[no-any-return, unused-ignore]
            return np.asarray(proba)  # type: ignore[no-any-return, unused-ignore]
        except (AttributeError, NotImplementedError):
            pass

        # Strategy 2: decision_function (SVM, LinearSVC, etc.)
        try:
            decision = self.base_detector.decision_function(X)
            # Normalize to [0, 1] using sigmoid (clip to prevent exp overflow)
            decision = np.clip(decision, -500, 500)
            return np.asarray(1.0 / (1.0 + np.exp(-decision)))  # type: ignore[no-any-return, unused-ignore]
        except (AttributeError, NotImplementedError):
            pass

        # Strategy 3: score_samples (Isolation Forest, LOF, etc.)
        try:
            scores = self.base_detector.score_samples(X)
            # score_samples is typically log-likelihood, higher is more normal
            # Invert and normalize to [0, 1]
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                normalized = 1.0 - (scores - min_score) / (max_score - min_score)
                return np.asarray(normalized)  # type: ignore[no-any-return, unused-ignore]
            return np.full(len(scores), 0.5)
        except (AttributeError, NotImplementedError):
            pass

        # Strategy 4: Binary prediction with synthetic probability scores
        try:
            predictions = self.base_detector.predict(X)

            # If detector has labels_ (e.g., LOF), use for context
            if hasattr(self.base_detector, "negative_outlier_factor_"):
                # LOF stores this for training data - use as reference
                nof = np.abs(self.base_detector.negative_outlier_factor_)
                np.median(nof)
                # Estimate anomaly scores based on prediction
                scores = np.where(predictions == -1, 0.7, 0.3)
            elif hasattr(self.base_detector, "offset_"):
                # Isolation Forest has offset
                scores = np.where(predictions == 1, 0.7, 0.3)
            else:
                # Generic binary to continuous conversion
                scores = predictions.astype(float)
                # Add small noise for differentiation
                # Independent ``Generator`` driven by ``self.seed`` so the
                # binary→continuous score smoothing is reproducible without
                # touching the legacy global ``np.random`` state.
                scores = (
                    0.3 + 0.4 * scores + 0.1 * np.random.default_rng(self.seed).random(len(scores))
                )
                scores = np.clip(scores, 0.0, 1.0)

            return np.asarray(scores)  # type: ignore[no-any-return, unused-ignore]
        except (AttributeError, NotImplementedError):
            pass

        # Strategy 5: Ensemble scoring from feature statistics
        return self._compute_ensemble_anomaly_scores(X)

    def _compute_ensemble_anomaly_scores(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Compute anomaly scores using ensemble of statistical methods.

        Uses multiple lightweight statistical approaches to estimate
        anomaly likelihood when detector doesn't provide scores directly.

        Methods combined:
        - Z-score based (Mahalanobis-inspired for univariate)
        - Local density estimation (k-NN approximation)
        - Feature-wise percentile extremity

        Args:
            X: Input data array.

        Returns:
            Aggregated anomaly scores.
        """
        n_samples, n_features = X.shape

        # Method 1: Z-score extremity
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-10
        z_scores = np.abs((X - mean) / std)
        z_anomaly = np.mean(z_scores, axis=1)  # Average across features
        z_anomaly = 1 - np.exp(-z_anomaly / 3)  # Normalize to [0, 1]

        # Method 2: Local density (simplified k-NN)
        k = min(5, n_samples - 1)
        if k >= 1:
            from scipy.spatial.distance import cdist

            distances = cdist(X, X, metric="euclidean")
            np.fill_diagonal(distances, np.inf)  # Exclude self
            knn_distances = np.partition(distances, k, axis=1)[:, :k]
            avg_knn_dist = np.mean(knn_distances, axis=1)
            # Normalize: larger distance = more anomalous
            max_dist = np.max(avg_knn_dist)
            if max_dist > 0:
                density_anomaly = avg_knn_dist / max_dist
            else:
                density_anomaly = np.zeros(n_samples)
        else:
            density_anomaly = np.zeros(n_samples)

        # Method 3: Percentile extremity
        percentile_scores = np.zeros(n_samples)
        for j in range(n_features):
            col = X[:, j]
            ranks = np.argsort(np.argsort(col))  # Rank each value
            pct = ranks / (n_samples - 1) if n_samples > 1 else np.zeros(n_samples)
            # Extreme percentiles = more anomalous
            extremity = np.abs(pct - 0.5) * 2
            percentile_scores += extremity
        percentile_anomaly = percentile_scores / n_features

        # Ensemble fusion (weighted average)
        ensemble_scores = 0.4 * z_anomaly + 0.35 * density_anomaly + 0.25 * percentile_anomaly

        return np.asarray(np.clip(ensemble_scores, 0.0, 1.0))  # type: ignore[no-any-return, unused-ignore]

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
        if isinstance(self.conformal, AdaptiveConformalInference):
            threshold = self.conformal.get_current_threshold()
        else:
            threshold = self.conformal.get_anomaly_threshold()

        return (scores > threshold).astype(int)

    def predict_with_uncertainty(
        self,
        X: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
        if isinstance(self.conformal, AdaptiveConformalInference):
            threshold = self.conformal.get_current_threshold()
        else:
            threshold = self.conformal.get_anomaly_threshold()
        predictions = (scores > threshold).astype(int)

        # Prediction intervals for scores
        if isinstance(self.conformal, SplitConformalPredictor):
            pred_set = self.conformal.predict(scores)
            return predictions, pred_set.lower_bound, pred_set.upper_bound

        # For other methods, return score-based bounds (threshold already computed above)
        lower = scores - threshold
        upper = scores + threshold

        return predictions, lower, upper

    def evaluate_coverage(
        self,
        X_test: np.ndarray[Any, Any],
        y_test: np.ndarray[Any, Any],
    ) -> CoverageResult:
        """
        Evaluate prediction accuracy on test set.

        NOTE: Despite the name, this method measures overall classification
        accuracy (predictions == y_test), NOT the conformal coverage guarantee.
        For the true conformal guarantee (fraction of test nonconformity scores
        at or below the calibration quantile threshold), use
        ``SplitConformalPredictor`` or ``CrossConformalPredictor`` directly
        and measure score-based coverage. See
        ``benchmarks/calibration_validation.py::measure_score_based_coverage()``
        for the correct methodology.

        For heavily imbalanced anomaly detection datasets, accuracy is
        dominated by the majority (normal) class and can be misleadingly
        high even with poor anomaly detection.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            CoverageResult with prediction accuracy statistics
        """
        scores = self._get_anomaly_scores(X_test)
        if isinstance(self.conformal, AdaptiveConformalInference):
            threshold = self.conformal.get_current_threshold()
        else:
            threshold = self.conformal.get_anomaly_threshold()
        predictions = (scores > threshold).astype(int)

        # Prediction accuracy: fraction of correct binary predictions
        # NOTE: This is NOT the conformal coverage guarantee
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


class MondrianConformalPredictor:
    """
    Mondrian Conformal Prediction for label-conditional coverage.

    Provides per-group (e.g., per-domain, per-class) coverage guarantees
    instead of just marginal coverage. This ensures that coverage is
    balanced across subpopulations, preventing under-coverage of rare
    groups — critical for fair anomaly detection.

    Reference: Vovk et al. (2005) "Algorithmic Learning in a Random World",
    Chapter 8: Mondrian Conformal Prediction.
    """

    def __init__(
        self,
        coverage: float = 0.95,
        seed: int = 42,
    ):
        """
        Initialize Mondrian conformal predictor.

        Args:
            coverage: Target coverage level per group.
            seed: Random seed for reproducibility.
        """
        self.coverage = coverage
        self.seed = seed
        self._group_predictors: dict[int | str, SplitConformalPredictor] = {}
        self._fitted = False

    def fit(
        self,
        nonconformity_scores: np.ndarray[Any, Any],
        groups: np.ndarray[Any, Any],
    ) -> MondrianConformalPredictor:
        """
        Fit per-group conformal predictors.

        Args:
            nonconformity_scores: Nonconformity scores from calibration set.
            groups: Group labels for each calibration example (int or str).

        Returns:
            Self for method chaining.
        """
        unique_groups = np.unique(groups)
        self._group_predictors = {}

        for group in unique_groups:
            mask = groups == group
            group_scores = nonconformity_scores[mask]
            if len(group_scores) < 2:
                logger.warning(
                    f"MondrianConformal: group {group} has only "
                    f"{len(group_scores)} samples — using global threshold."
                )
                continue

            predictor = SplitConformalPredictor(coverage=self.coverage, seed=self.seed)
            predictor.fit(group_scores)
            self._group_predictors[group] = predictor

        # Global fallback for groups with insufficient data
        self._global_predictor = SplitConformalPredictor(coverage=self.coverage, seed=self.seed)
        self._global_predictor.fit(nonconformity_scores)

        self._fitted = True
        logger.debug(
            f"MondrianConformal fitted: {len(self._group_predictors)} groups, "
            f"coverage={self.coverage}"
        )
        return self

    def get_anomaly_threshold(self, group: int | str | None = None) -> float:
        """
        Get anomaly threshold for a specific group.

        Args:
            group: Group identifier. If None, returns global threshold.

        Returns:
            Anomaly threshold for the specified group.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before get_anomaly_threshold()")

        if group is not None and group in self._group_predictors:
            return self._group_predictors[group].get_anomaly_threshold()

        return self._global_predictor.get_anomaly_threshold()

    def predict(
        self,
        scores: np.ndarray[Any, Any],
        groups: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        Predict anomalies with per-group coverage guarantees.

        Args:
            scores: Nonconformity scores for test examples.
            groups: Group labels for each test example.

        Returns:
            Binary predictions with per-group coverage guarantee.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        predictions = np.zeros(len(scores), dtype=int)
        for i, (score, group) in enumerate(zip(scores, groups)):
            threshold = self.get_anomaly_threshold(group)
            predictions[i] = int(score > threshold)
        return predictions

    def predict_with_uncertainty(
        self,
        scores: np.ndarray[Any, Any],
        group_ids: np.ndarray[Any, Any],
        alpha: float = 0.1,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Predict anomalies with per-group confidence intervals.

        Returns predictions along with lower and upper confidence bounds
        on the anomaly scores, providing ``(1 - alpha)`` confidence
        intervals computed from per-group nonconformity score quantiles.

        Args:
            scores: Nonconformity scores for test examples.
            group_ids: Group labels for each test example.
            alpha: Significance level (default 0.1 for 90% CI).

        Returns:
            Tuple of ``(predictions, lower_bound, upper_bound)`` where:
              - ``predictions``: Binary anomaly predictions (int array).
              - ``lower_bound``: Lower confidence bound on scores.
              - ``upper_bound``: Upper confidence bound on scores.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_with_uncertainty()")

        predictions = self.predict(scores, group_ids)
        n = len(scores)
        lower = np.zeros(n, dtype=np.float64)
        upper = np.ones(n, dtype=np.float64)

        for i, (score, group) in enumerate(zip(scores, group_ids)):
            predictor = self._group_predictors.get(group, self._global_predictor)
            cal_scores = predictor.calibration_scores
            if cal_scores is not None and len(cal_scores) > 0:
                q_lo = float(np.quantile(cal_scores, alpha / 2))
                q_hi = float(np.quantile(cal_scores, 1 - alpha / 2))
                # Confidence interval: how the score relates to calibration
                half_width = (q_hi - q_lo) / 2.0
                lower[i] = max(0.0, score - half_width)
                upper[i] = min(1.0, score + half_width)
            else:
                lower[i] = max(0.0, score - 0.1)
                upper[i] = min(1.0, score + 0.1)

        return predictions, lower, upper

    def evaluate_group_coverage(
        self,
        scores: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        groups: np.ndarray[Any, Any],
    ) -> dict[str, float | dict[int | str, float]]:
        """
        Evaluate per-group empirical coverage.

        Args:
            scores: Nonconformity scores.
            labels: True labels (0/1).
            groups: Group identifiers.

        Returns:
            Dict with overall and per-group coverage stats.
        """
        predictions = self.predict(scores, groups)
        correct = predictions == labels

        per_group: dict[int | str, float] = {}
        for group in np.unique(groups):
            mask = groups == group
            if np.sum(mask) > 0:
                per_group[group] = float(np.mean(correct[mask]))

        return {
            "overall_coverage": float(np.mean(correct)),
            "target_coverage": self.coverage,
            "per_group_coverage": per_group,
            "worst_group_coverage": min(per_group.values()) if per_group else 0.0,
            "coverage_gap": (
                max(abs(v - self.coverage) for v in per_group.values()) if per_group else 0.0
            ),
        }


class ConformalCalibrationBridge:
    """
    Bridge between conformal prediction and the calibration pipeline.

    Integrates conformal uncertainty quantification into the threshold calibration process,
    providing distribution-free coverage guarantees on top of calibrated thresholds.
    """

    def __init__(
        self,
        base_coverage: float = 0.95,
        adaptive_lr: float = 0.05,
    ):
        """
        Initialize calibration bridge.

        Args:
            base_coverage: Target coverage level.
            adaptive_lr: Learning rate for adaptive threshold.
        """
        self.split_predictor = SplitConformalPredictor(coverage=base_coverage)
        self.adaptive_predictor = AdaptiveConformalInference(
            target_coverage=base_coverage,
            learning_rate=adaptive_lr,
        )
        self.mondrian_predictor = MondrianConformalPredictor(coverage=base_coverage)
        self._calibrated = False

    def calibrate(
        self,
        calibration_scores: np.ndarray[Any, Any],
        groups: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, float]:
        """
        Calibrate all conformal predictors.

        Args:
            calibration_scores: Nonconformity scores from calibration data.
            groups: Optional group labels for Mondrian prediction.

        Returns:
            Dict of threshold values from each method.
        """
        self.split_predictor.fit(calibration_scores)

        if groups is not None:
            self.mondrian_predictor.fit(calibration_scores, groups)

        self._calibrated = True

        result = {
            "split_threshold": self.split_predictor.get_anomaly_threshold(),
            "adaptive_threshold": self.adaptive_predictor.get_current_threshold(),
        }

        if groups is not None:
            unique_groups = np.unique(groups)
            for g in unique_groups:
                result[f"mondrian_{g}_threshold"] = self.mondrian_predictor.get_anomaly_threshold(g)

        return result

    def update_adaptive(self, score: float, true_label: int | None = None) -> tuple[float, bool]:
        """
        Update adaptive conformal threshold with new observation.

        Args:
            score: New nonconformity score.
            true_label: Optional true label.

        Returns:
            Tuple of (current_threshold, is_covered).
        """
        return self.adaptive_predictor.update(score, true_label)


@dataclass
class BinaryPredictionSet:
    """Conformal label prediction sets over ``{0 = normal, 1 = anomaly}``.

    For each sample the set is the subset of labels the classifier cannot
    confidently rule out at the target coverage. ``set_size`` is therefore in
    ``{0, 1, 2}``: a singleton ``{1}``/``{0}`` is a confident anomaly/normal
    call, ``{0, 1}`` flags genuine uncertainty (abstain), and ``{}`` flags an
    atypical point neither class explains well.

    Attributes:
        contains_normal: ``(n,)`` bool -- label 0 is in the set.
        contains_anomaly: ``(n,)`` bool -- label 1 is in the set.
        set_size: ``(n,)`` int in ``{0, 1, 2}``.
        probability: ``(n,)`` calibrated ``P(anomaly)`` used to build the sets.
        coverage_level: Target coverage (e.g. 0.9).
    """

    contains_normal: np.ndarray[Any, Any]
    contains_anomaly: np.ndarray[Any, Any]
    set_size: np.ndarray[Any, Any]
    probability: np.ndarray[Any, Any]
    coverage_level: float

    def label_sets(self) -> list[list[int]]:
        """Return the per-sample label set as a list of sorted label lists."""
        sets: list[list[int]] = []
        for has0, has1 in zip(self.contains_normal, self.contains_anomaly):
            labels = [lbl for lbl, has in ((0, bool(has0)), (1, bool(has1))) if has]
            sets.append(labels)
        return sets


class BinaryConformalClassifier:
    """Class-conditional (Mondrian) split-conformal classifier for anomaly detection.

    Turns calibrated anomaly probabilities into label prediction sets with a
    distribution-free coverage guarantee. Uses the Least-Ambiguous-set (LAC)
    nonconformity score ``s(x, y) = 1 - p_hat(y | x)`` and calibrates a separate
    quantile per class, so coverage holds **per class** (Mondrian) and hence
    marginally -- the right notion for imbalanced anomaly data, where a single
    marginal threshold would be swamped by the normal class.

    Each class threshold reuses :class:`SplitConformalPredictor` (PR #242), so
    the finite-sample quantile is identical to the rest of the conformal stack.
    A point is admitted to label ``y`` iff ``1 - p_y <= q_y``.

    Reference: Sadinle, Lei & Wasserman (2019), "Least Ambiguous Set-Valued
    Classifiers With Bounded Error Levels".

    Args:
        coverage: Target per-class coverage (e.g. 0.9 for 90%).
        seed: Random seed forwarded to the per-class predictors.
    """

    def __init__(self, coverage: float = 0.9, seed: int = 42) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self.seed = seed
        self._predictors: dict[int, SplitConformalPredictor] = {}
        self._thresholds: dict[int, float] = {}
        self._fitted = False

    @staticmethod
    def _class_probability(probs: np.ndarray[Any, Any], label: int) -> np.ndarray[Any, Any]:
        """Return ``p_hat(label | x)`` for binary label ``0`` or ``1``."""
        p1 = np.asarray(probs, dtype=float).ravel()
        return p1 if label == 1 else 1.0 - p1

    def fit(
        self, probabilities: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> BinaryConformalClassifier:
        """Calibrate per-class thresholds on held-out calibrated probabilities.

        Args:
            probabilities: ``(n_cal,)`` calibrated ``P(anomaly)`` on the
                calibration split (e.g. from temperature-scaled fusion output).
            labels: ``(n_cal,)`` binary ground-truth labels (1 = anomaly).

        Returns:
            Self for chaining.
        """
        probs = np.asarray(probabilities, dtype=float).ravel()
        y = np.asarray(labels).astype(int).ravel()
        if probs.shape != y.shape:
            raise ValueError("probabilities and labels must have the same length")

        for label in (0, 1):
            mask = y == label
            if not np.any(mask):
                # No calibration data for this class: include it unconditionally
                # (threshold 1.0) so the coverage guarantee stays conservative
                # rather than silently dropping the class.
                self._thresholds[label] = 1.0
                continue
            nonconformity = 1.0 - self._class_probability(probs[mask], label)
            predictor = SplitConformalPredictor(coverage=self.coverage, seed=self.seed)
            predictor.fit(nonconformity)
            self._predictors[label] = predictor
            self._thresholds[label] = predictor.get_anomaly_threshold()

        self._fitted = True
        return self

    def anomaly_score_threshold(self) -> float:
        """Single anomaly operating point implied by the class-1 LAC quantile.

        A point is admitted to the anomaly label iff its nonconformity
        ``1 - p_anomaly <= q_1``, i.e. ``p_anomaly >= 1 - q_1``.  This returns
        ``1 - q_1`` so a caller that needs one threshold (rather than a full
        prediction set) can flag ``score >= threshold`` with the conformal
        class-1 coverage guarantee.  Returns ``inf`` when class 1 was never
        calibrated (no positive in the calibration split), so nothing is
        flagged rather than everything.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before anomaly_score_threshold()")
        q1 = self._thresholds.get(1)
        if q1 is None or q1 >= 1.0:
            return float("inf")
        return float(1.0 - q1)

    def predict(self, probabilities: np.ndarray[Any, Any]) -> BinaryPredictionSet:
        """Build conformal label sets for new calibrated probabilities.

        Args:
            probabilities: ``(n,)`` calibrated ``P(anomaly)``.

        Returns:
            A :class:`BinaryPredictionSet`.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        probs = np.asarray(probabilities, dtype=float).ravel()
        contains: dict[int, np.ndarray[Any, Any]] = {}
        for label in (0, 1):
            nonconformity = 1.0 - self._class_probability(probs, label)
            contains[label] = nonconformity <= self._thresholds[label] + 1e-12

        set_size = contains[0].astype(int) + contains[1].astype(int)
        return BinaryPredictionSet(
            contains_normal=contains[0],
            contains_anomaly=contains[1],
            set_size=set_size,
            probability=probs,
            coverage_level=self.coverage,
        )

    def coverage_report(
        self, probabilities: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Measure the *true* empirical coverage guarantee on a labelled set.

        Unlike accuracy, this reports the fraction of points whose prediction
        set contains the true label -- the quantity the conformal guarantee
        bounds below by ``coverage``.

        Returns:
            Dict with marginal/per-class empirical coverage, average set size,
            and the abstention/empty-set rates.
        """
        pred = self.predict(probabilities)
        y = np.asarray(labels).astype(int).ravel()
        covered = np.where(y == 1, pred.contains_anomaly, pred.contains_normal)

        per_class: dict[int, float] = {}
        for label in (0, 1):
            mask = y == label
            if np.any(mask):
                per_class[label] = float(np.mean(covered[mask]))

        return {
            "target_coverage": self.coverage,
            "empirical_coverage": float(np.mean(covered)) if len(y) else float("nan"),
            "coverage_by_class": per_class,
            "average_set_size": float(np.mean(pred.set_size)),
            "abstain_rate": float(np.mean(pred.set_size == 2)),
            "empty_rate": float(np.mean(pred.set_size == 0)),
            "thresholds": dict(self._thresholds),
        }


def add_conformal_to_detector(
    detector: Any,
    X_cal: np.ndarray[Any, Any],
    coverage: float = 0.95,
    method: str = "split",
) -> tuple[SplitConformalPredictor | CrossConformalPredictor, float]:
    """
    Add conformal prediction to an existing fitted detector.

    Implements a robust fallback cascade to extract anomaly scores from any
    detector type, enabling conformal prediction regardless of detector API.

    Args:
        detector: Fitted anomaly detector (any scikit-learn compatible).
        X_cal: Calibration data for conformal threshold estimation.
        coverage: Target coverage level (default 0.95 = 95% coverage).
        method: Conformal method ("split" or "cross").

    Returns:
        Tuple of (conformal_predictor, anomaly_threshold).

    Raises:
        ValueError: If no valid scoring method can be found.

    Example:
        >>> from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
        >>> detector = MercuryAnomalyDetector()
        >>> detector.fit(X_train)
        >>> conformal, threshold = add_conformal_to_detector(detector, X_cal)
    """
    # Get calibration scores with robust fallback cascade
    scores = _extract_detector_scores(detector, X_cal)

    # Fit conformal predictor
    conformal: SplitConformalPredictor | CrossConformalPredictor
    if method == "split":
        conformal = SplitConformalPredictor(coverage=coverage)
        conformal.fit(scores)
    else:
        cross_conformal = CrossConformalPredictor(coverage=coverage)

        def score_fn(
            X_input: np.ndarray[Any, Any], y: np.ndarray[Any, Any] | None = None
        ) -> np.ndarray[Any, Any]:
            return _extract_detector_scores(detector, X_input)

        cross_conformal.fit(X_cal, score_fn)  # type: ignore[arg-type]
        conformal = cross_conformal

    return conformal, conformal.get_anomaly_threshold()


def _extract_detector_scores(detector: Any, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Extract continuous anomaly scores from any detector.

    Uses a multi-strategy fallback cascade to obtain scores:
    1. predict_proba - Standard probability output
    2. decision_function - Margin-based scores
    3. score_samples - Density/likelihood scores
    4. predict - Binary with synthesized probabilities
    5. Ensemble fallback - Statistical scoring

    Args:
        detector: Fitted anomaly detector.
        X: Input data.

    Returns:
        Continuous anomaly scores (higher = more anomalous).
    """
    # Strategy 1: predict_proba
    try:
        proba = detector.predict_proba(X)
        if proba.ndim == 2:
            return np.asarray(proba[:, 1])  # type: ignore[no-any-return, unused-ignore]
        return np.asarray(proba)  # type: ignore[no-any-return, unused-ignore]
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 2: decision_function (negate if needed for anomaly)
    try:
        decision = detector.decision_function(X)
        # For most detectors, negative = anomaly, so negate
        return np.asarray(1.0 / (1.0 + np.exp(decision)))  # type: ignore[no-any-return, unused-ignore]
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 3: score_samples (log-likelihood based)
    try:
        scores = detector.score_samples(X)
        # Higher score_samples = more normal, invert for anomaly
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s > min_s:
            return np.asarray(1.0 - (scores - min_s) / (max_s - min_s))  # type: ignore[no-any-return, unused-ignore]
        return np.full(len(scores), 0.5)
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 4: Binary predict with synthesized continuous scores
    try:
        predictions = detector.predict(X)

        # Handle sklearn convention (-1 = anomaly)
        if hasattr(detector, "contamination"):
            # Isolation Forest / LOF convention
            scores = np.where(predictions == -1, 0.75, 0.25)
        else:
            # Generic: 1 = anomaly
            scores = np.where(predictions == 1, 0.75, 0.25)

        # Add small variation based on features for ranking
        feature_std = np.std(X, axis=1)
        feature_std_norm = (feature_std - np.min(feature_std)) / (
            np.max(feature_std) - np.min(feature_std) + 1e-10
        )
        scores = scores + 0.1 * (feature_std_norm - 0.5)
        return np.asarray(np.clip(scores, 0.0, 1.0))  # type: ignore[no-any-return, unused-ignore]
    except (AttributeError, NotImplementedError):
        pass

    # Strategy 5: Ensemble statistical fallback
    return _compute_statistical_anomaly_scores(X)


def _compute_statistical_anomaly_scores(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Compute anomaly scores using ensemble of statistical methods.

    Fallback when detector doesn't provide any scoring method.

    Args:
        X: Input data.

    Returns:
        Statistical anomaly scores in [0, 1].
    """
    n_samples, n_features = X.shape

    # Method 1: Z-score extremity
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    z_scores = np.abs((X - mean) / std)
    z_anomaly = np.mean(z_scores, axis=1)
    z_anomaly = 1 - np.exp(-z_anomaly / 3)

    # Method 2: Simplified LOF-like (k-NN distance)
    k = min(5, n_samples - 1)
    if k >= 1:
        try:
            from scipy.spatial.distance import cdist

            distances = cdist(X, X, metric="euclidean")
            np.fill_diagonal(distances, np.inf)
            knn_distances = np.partition(distances, k, axis=1)[:, :k]
            avg_knn_dist = np.mean(knn_distances, axis=1)
            max_dist = np.max(avg_knn_dist)
            density_anomaly = avg_knn_dist / max_dist if max_dist > 0 else np.zeros(n_samples)
        except ImportError:
            density_anomaly = np.zeros(n_samples)
    else:
        density_anomaly = np.zeros(n_samples)

    # Method 3: Percentile extremity
    percentile_scores = np.zeros(n_samples)
    for j in range(n_features):
        col = X[:, j]
        ranks = np.argsort(np.argsort(col))
        pct = ranks / (n_samples - 1) if n_samples > 1 else np.zeros(n_samples)
        extremity = np.abs(pct - 0.5) * 2
        percentile_scores += extremity
    percentile_anomaly = percentile_scores / n_features

    # Weighted ensemble
    ensemble_scores = 0.4 * z_anomaly + 0.35 * density_anomaly + 0.25 * percentile_anomaly

    return np.asarray(np.clip(ensemble_scores, 0.0, 1.0))  # type: ignore[no-any-return, unused-ignore]

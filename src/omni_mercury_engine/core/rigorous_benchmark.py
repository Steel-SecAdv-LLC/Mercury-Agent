"""
Mercury Agent - Rigorous Benchmark Harness
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Implements rigorous benchmark methodology:
- Fixed random seeds (42) for reproducibility
- Stratified 80/20 train/test splits
- K-fold cross-validation (k=10 default)
- Standard metrics: ROC-AUC, point-adjusted F1, event-based precision/recall
- Statistical significance testing (paired t-test, Wilcoxon)
- Confidence intervals (95%)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Fixed seed for reproducibility
GLOBAL_SEED = 42

logger = logging.getLogger(__name__)


class AnomalyDetector(Protocol):
    """Protocol for anomaly detectors to benchmark."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Fit the detector to training data."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (0=normal, 1=anomaly)."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities."""
        ...


@dataclass
class MetricResult:
    """Container for a single metric's results across folds."""

    name: str
    values: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    def compute_stats(self) -> None:
        """Compute mean, std, and 95% confidence interval."""
        if not self.values:
            return

        arr = np.array(self.values)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        # 95% CI using t-distribution
        if len(arr) > 1:
            t_crit = stats.t.ppf(0.975, len(arr) - 1)
            margin = t_crit * self.std / np.sqrt(len(arr))
            self.ci_lower = self.mean - margin
            self.ci_upper = self.mean + margin
        else:
            self.ci_lower = self.mean
            self.ci_upper = self.mean


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a detector."""

    detector_name: str
    dataset_name: str
    n_folds: int
    seed: int

    # Core metrics
    roc_auc: MetricResult = field(default_factory=lambda: MetricResult("ROC-AUC"))
    f1: MetricResult = field(default_factory=lambda: MetricResult("F1"))
    precision: MetricResult = field(default_factory=lambda: MetricResult("Precision"))
    recall: MetricResult = field(default_factory=lambda: MetricResult("Recall"))
    brier: MetricResult = field(default_factory=lambda: MetricResult("Brier"))
    pr_auc: MetricResult = field(default_factory=lambda: MetricResult("PR-AUC"))

    # Event-based metrics (for time-series)
    event_precision: MetricResult = field(default_factory=lambda: MetricResult("Event-Precision"))
    event_recall: MetricResult = field(default_factory=lambda: MetricResult("Event-Recall"))
    event_f1: MetricResult = field(default_factory=lambda: MetricResult("Event-F1"))

    # Timing
    fit_times: list[float] = field(default_factory=list)
    predict_times: list[float] = field(default_factory=list)

    # Per-fold predictions for statistical tests
    fold_predictions: list[np.ndarray] = field(default_factory=list)
    fold_labels: list[np.ndarray] = field(default_factory=list)

    def finalize(self) -> None:
        """Compute all statistics after folds complete."""
        for metric in [self.roc_auc, self.f1, self.precision, self.recall,
                       self.brier, self.pr_auc, self.event_precision,
                       self.event_recall, self.event_f1]:
            metric.compute_stats()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "detector": self.detector_name,
            "dataset": self.dataset_name,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "metrics": {
                "roc_auc": {"mean": self.roc_auc.mean, "std": self.roc_auc.std,
                           "ci": [self.roc_auc.ci_lower, self.roc_auc.ci_upper]},
                "f1": {"mean": self.f1.mean, "std": self.f1.std,
                       "ci": [self.f1.ci_lower, self.f1.ci_upper]},
                "precision": {"mean": self.precision.mean, "std": self.precision.std},
                "recall": {"mean": self.recall.mean, "std": self.recall.std},
                "brier": {"mean": self.brier.mean, "std": self.brier.std},
                "pr_auc": {"mean": self.pr_auc.mean, "std": self.pr_auc.std},
            },
            "event_metrics": {
                "precision": {"mean": self.event_precision.mean, "std": self.event_precision.std},
                "recall": {"mean": self.event_recall.mean, "std": self.event_recall.std},
                "f1": {"mean": self.event_f1.mean, "std": self.event_f1.std},
            },
            "timing": {
                "fit_mean_ms": np.mean(self.fit_times) * 1000 if self.fit_times else 0,
                "predict_mean_ms": np.mean(self.predict_times) * 1000 if self.predict_times else 0,
            },
        }


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform stratified train/test split.

    Args:
        X: Feature matrix
        y: Binary labels (0=normal, 1=anomaly)
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        seed: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    set_all_seeds(seed)
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 0,
) -> tuple[float, float, float]:
    """
    Compute event-based metrics for time-series anomaly detection.

    An event is a contiguous sequence of anomalous points.
    Event-based recall: fraction of true events that are detected.
    Event-based precision: fraction of predicted events that overlap true events.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        tolerance: Points of tolerance for event matching (default 0)

    Returns:
        (event_precision, event_recall, event_f1)
    """
    def get_events(arr: np.ndarray) -> list[tuple[int, int]]:
        """Extract contiguous event ranges."""
        events = []
        in_event = False
        start = 0

        for i, val in enumerate(arr):
            if val == 1 and not in_event:
                start = i
                in_event = True
            elif val == 0 and in_event:
                events.append((start, i - 1))
                in_event = False

        if in_event:
            events.append((start, len(arr) - 1))

        return events

    def events_overlap(e1: tuple[int, int], e2: tuple[int, int], tol: int) -> bool:
        """Check if two events overlap within tolerance."""
        return not (e1[1] + tol < e2[0] or e2[1] + tol < e1[0])

    true_events = get_events(y_true)
    pred_events = get_events(y_pred)

    if not true_events:
        return (1.0, 1.0, 1.0) if not pred_events else (0.0, 1.0, 0.0)
    if not pred_events:
        return (1.0, 0.0, 0.0)

    # Event recall: fraction of true events detected
    detected = 0
    for te in true_events:
        for pe in pred_events:
            if events_overlap(te, pe, tolerance):
                detected += 1
                break
    event_recall = detected / len(true_events)

    # Event precision: fraction of predictions matching true events
    matched = 0
    for pe in pred_events:
        for te in true_events:
            if events_overlap(pe, te, tolerance):
                matched += 1
                break
    event_precision = matched / len(pred_events)

    # F1
    if event_precision + event_recall > 0:
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall)
    else:
        event_f1 = 0.0

    return (event_precision, event_recall, event_f1)


def point_adjusted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute point-adjusted F1 score (PA-F1).

    If any point in a true anomaly segment is detected, the entire
    segment is considered detected. This is standard for time-series
    anomaly detection evaluation.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels

    Returns:
        Point-adjusted F1 score
    """
    # Get anomaly segments
    adjusted_pred = np.zeros_like(y_pred)

    in_segment = False
    segment_start = 0

    for i, val in enumerate(y_true):
        if val == 1 and not in_segment:
            segment_start = i
            in_segment = True
        elif val == 0 and in_segment:
            # Check if any prediction in segment
            if np.any(y_pred[segment_start:i]):
                adjusted_pred[segment_start:i] = 1
            in_segment = False

    # Handle final segment
    if in_segment:
        if np.any(y_pred[segment_start:]):
            adjusted_pred[segment_start:] = 1

    # Also include any isolated predictions (not in true segments)
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 0:
            adjusted_pred[i] = 1

    return f1_score(y_true, adjusted_pred, zero_division=1.0)


class RigorousBenchmarkHarness:
    """
    Rigorous benchmark harness for anomaly detection evaluation.

    Features:
    - Fixed random seeds (42) for reproducibility
    - Stratified K-fold cross-validation (k=10 default)
    - Standard metrics: ROC-AUC, F1, Precision, Recall, Brier, PR-AUC
    - Event-based metrics for time-series (humanitarian alert scoring)
    - Point-adjusted F1 (PA-F1) for time-series
    - Statistical significance testing
    - 95% confidence intervals
    """

    def __init__(
        self,
        n_folds: int = 10,
        seed: int = GLOBAL_SEED,
        compute_event_metrics: bool = True,
    ):
        """
        Initialize benchmark harness.

        Args:
            n_folds: Number of cross-validation folds (default 10)
            seed: Random seed for reproducibility (default 42)
            compute_event_metrics: Whether to compute event-based metrics
        """
        self.n_folds = n_folds
        self.seed = seed
        self.compute_event_metrics_flag = compute_event_metrics

        set_all_seeds(seed)
        logger.info(
            f"RigorousBenchmarkHarness initialized: n_folds={n_folds}, seed={seed}"
        )

    def benchmark_detector(
        self,
        detector: Any,
        X: np.ndarray,
        y: np.ndarray,
        detector_name: str = "Unknown",
        dataset_name: str = "Unknown",
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark on a detector.

        Args:
            detector: Anomaly detector implementing fit/predict/predict_proba
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,) - 0=normal, 1=anomaly
            detector_name: Name of detector for reporting
            dataset_name: Name of dataset for reporting

        Returns:
            BenchmarkResult with all metrics and statistics
        """
        set_all_seeds(self.seed)

        result = BenchmarkResult(
            detector_name=detector_name,
            dataset_name=dataset_name,
            n_folds=self.n_folds,
            seed=self.seed,
        )

        # Stratified K-fold
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit
            start_time = time.perf_counter()
            try:
                detector.fit(X_train, y_train)
            except TypeError:
                # Some detectors don't accept y (unsupervised)
                detector.fit(X_train)
            fit_time = time.perf_counter() - start_time
            result.fit_times.append(fit_time)

            # Predict
            start_time = time.perf_counter()
            y_pred = detector.predict(X_test)
            predict_time = time.perf_counter() - start_time
            result.predict_times.append(predict_time)

            # Get probabilities
            try:
                y_proba = detector.predict_proba(X_test)
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]  # Get probability of anomaly class
            except (AttributeError, NotImplementedError):
                # Some detectors don't have predict_proba
                y_proba = y_pred.astype(float)

            # Convert predictions to binary if needed
            if not np.array_equal(y_pred, y_pred.astype(int)):
                y_pred = (y_pred > 0.5).astype(int)

            # Handle sklearn's -1/1 convention for anomaly detectors
            if set(np.unique(y_pred)) == {-1, 1}:
                y_pred = (y_pred == -1).astype(int)

            # Store for statistical tests
            result.fold_predictions.append(y_pred)
            result.fold_labels.append(y_test)

            # Compute metrics
            try:
                result.roc_auc.values.append(roc_auc_score(y_test, y_proba))
            except ValueError:
                result.roc_auc.values.append(0.5)

            result.f1.values.append(f1_score(y_test, y_pred, zero_division=0.0))
            result.precision.values.append(precision_score(y_test, y_pred, zero_division=0.0))
            result.recall.values.append(recall_score(y_test, y_pred, zero_division=0.0))

            try:
                result.brier.values.append(brier_score_loss(y_test, y_proba))
            except ValueError:
                result.brier.values.append(0.25)

            try:
                result.pr_auc.values.append(average_precision_score(y_test, y_proba))
            except ValueError:
                result.pr_auc.values.append(0.0)

            # Event-based metrics
            if self.compute_event_metrics_flag:
                ep, er, ef = compute_event_metrics(y_test, y_pred)
                result.event_precision.values.append(ep)
                result.event_recall.values.append(er)
                result.event_f1.values.append(ef)

            logger.debug(
                f"Fold {fold_idx + 1}/{self.n_folds}: "
                f"AUC={result.roc_auc.values[-1]:.3f}, "
                f"F1={result.f1.values[-1]:.3f}"
            )

        result.finalize()

        logger.info(
            f"{detector_name} on {dataset_name}: "
            f"AUC={result.roc_auc.mean:.3f}±{result.roc_auc.std:.3f}, "
            f"F1={result.f1.mean:.3f}±{result.f1.std:.3f}"
        )

        return result

    def compare_detectors(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
        metric: str = "f1",
    ) -> dict[str, Any]:
        """
        Statistical comparison between two detectors.

        Args:
            result_a: First detector's results
            result_b: Second detector's results
            metric: Metric to compare ("f1", "roc_auc", etc.)

        Returns:
            Dictionary with t-test and Wilcoxon results
        """
        metric_a = getattr(result_a, metric).values
        metric_b = getattr(result_b, metric).values

        if len(metric_a) != len(metric_b):
            raise ValueError("Results must have same number of folds")

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(metric_a, metric_b)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = stats.wilcoxon(metric_a, metric_b)
        except ValueError:
            # All differences are zero
            w_stat, w_pvalue = 0.0, 1.0

        # Effect size (Cohen's d)
        diff = np.array(metric_a) - np.array(metric_b)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0

        improvement = (
            (np.mean(metric_a) - np.mean(metric_b)) / np.mean(metric_b) * 100
            if np.mean(metric_b) > 0 else 0.0
        )

        return {
            "detector_a": result_a.detector_name,
            "detector_b": result_b.detector_name,
            "metric": metric,
            "mean_a": np.mean(metric_a),
            "mean_b": np.mean(metric_b),
            "improvement_percent": improvement,
            "t_test": {
                "statistic": t_stat,
                "p_value": t_pvalue,
                "significant_p05": t_pvalue < 0.05,
            },
            "wilcoxon": {
                "statistic": w_stat,
                "p_value": w_pvalue,
                "significant_p05": w_pvalue < 0.05,
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": (
                    "large" if abs(cohens_d) > 0.8 else
                    "medium" if abs(cohens_d) > 0.5 else
                    "small" if abs(cohens_d) > 0.2 else
                    "negligible"
                ),
            },
        }


def run_baseline_benchmarks(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "Dataset",
    n_folds: int = 10,
    seed: int = GLOBAL_SEED,
) -> dict[str, BenchmarkResult]:
    """
    Run benchmarks on standard PyOD-style baselines.

    Args:
        X: Feature matrix
        y: Binary labels
        dataset_name: Name of dataset
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary mapping detector name to BenchmarkResult
    """
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    harness = RigorousBenchmarkHarness(n_folds=n_folds, seed=seed)
    results = {}

    # Anomaly ratio for contamination parameter
    anomaly_ratio = np.mean(y)
    contamination = min(0.5, max(0.01, anomaly_ratio))

    # Isolation Forest
    class IFWrapper:
        def __init__(self):
            self.model = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=seed,
            )

        def fit(self, X, y=None):
            self.model.fit(X)

        def predict(self, X):
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        def predict_proba(self, X):
            scores = -self.model.score_samples(X)
            # Normalize to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores

    results["IsolationForest"] = harness.benchmark_detector(
        IFWrapper(), X, y, "IsolationForest", dataset_name
    )

    # One-Class SVM
    class OCSVMWrapper:
        def __init__(self):
            self.model = OneClassSVM(kernel="rbf", nu=contamination)

        def fit(self, X, y=None):
            self.model.fit(X)

        def predict(self, X):
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        def predict_proba(self, X):
            scores = -self.model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores

    results["OneClassSVM"] = harness.benchmark_detector(
        OCSVMWrapper(), X, y, "OneClassSVM", dataset_name
    )

    # Local Outlier Factor
    class LOFWrapper:
        def __init__(self):
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True,
            )

        def fit(self, X, y=None):
            self.model.fit(X)

        def predict(self, X):
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        def predict_proba(self, X):
            scores = -self.model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores

    results["LOF"] = harness.benchmark_detector(
        LOFWrapper(), X, y, "LOF", dataset_name
    )

    # Elliptic Envelope
    class EEWrapper:
        def __init__(self):
            self.model = EllipticEnvelope(
                contamination=contamination,
                random_state=seed,
            )

        def fit(self, X, y=None):
            try:
                self.model.fit(X)
            except ValueError:
                # Fallback for singular covariance
                self.model = EllipticEnvelope(
                    contamination=contamination,
                    random_state=seed,
                    support_fraction=0.9,
                )
                self.model.fit(X)

        def predict(self, X):
            preds = self.model.predict(X)
            return (preds == -1).astype(int)

        def predict_proba(self, X):
            scores = -self.model.decision_function(X)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores

    results["EllipticEnvelope"] = harness.benchmark_detector(
        EEWrapper(), X, y, "EllipticEnvelope", dataset_name
    )

    return results

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Agent — native ML primitives (numpy/scipy only).

Mercury Agent is an original product with its own production-built systems. This module provides
Mercury-native implementations of standard ML utilities (metrics, preprocessing, model selection,
classical models) using only numpy and scipy — no external ML library dependencies.

These are general-purpose ML building blocks (metrics, scalers, CV splitters, classical models) used
internally by Mercury's detection pipeline.  They are standard algorithms documented in their
respective docstrings.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =====================================================================
# Classification Metrics
# =====================================================================


def accuracy_score(y_true: NDArray[np.number[Any]], y_pred: NDArray[np.number[Any]]) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision_score(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    *,
    zero_division: float = 0.0,
    average: str = "binary",
) -> float:
    """Compute precision score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _precision_binary, zero_division)

    classes = np.unique(y_true)
    pos_label = classes[-1]
    tp = float(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = float(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    denom = tp + fp
    if denom == 0:
        return zero_division
    return tp / denom


def recall_score(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    *,
    zero_division: float = 0.0,
    average: str = "binary",
) -> float:
    """Compute recall score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _recall_binary, zero_division)

    classes = np.unique(y_true)
    pos_label = classes[-1]
    tp = float(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fn = float(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    denom = tp + fn
    if denom == 0:
        return zero_division
    return tp / denom


def f1_score(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    *,
    zero_division: float = 0.0,
    average: str = "binary",
    pos_label: int | None = None,
) -> float:
    """Compute F1 score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if pos_label is not None:
        # Filter to binary evaluation for pos_label
        mask = (y_true == pos_label) | (y_pred == pos_label)
        if mask.sum() == 0:
            return zero_division
        binary_true = (y_true == pos_label).astype(np.intp)
        binary_pred = (y_pred == pos_label).astype(np.intp)
        p = precision_score(binary_true, binary_pred, zero_division=zero_division, average="binary")
        r = recall_score(binary_true, binary_pred, zero_division=zero_division, average="binary")
    elif average == "weighted":
        # Compute per-class F1 then take weighted average (matches sklearn)
        return _weighted_metric(y_true, y_pred, _f1_binary, zero_division)
    else:
        p = precision_score(y_true, y_pred, zero_division=zero_division, average=average)
        r = recall_score(y_true, y_pred, zero_division=zero_division, average=average)
    if p + r == 0:
        return zero_division
    return 2.0 * p * r / (p + r)


def confusion_matrix(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    *,
    labels: NDArray[np.number[Any]] | None = None,
) -> NDArray[np.number[Any]]:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, optional
        List of labels to index the matrix. If None, sorted unique union
        of *y_true* and *y_pred* is used.

    Returns
    -------
    NDArray[np.number[Any]] of shape (n_classes, n_classes)
        Entry *C[i, j]* is the number of samples with true label *labels[i]*
        and predicted label *labels[j]*.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    label_to_idx = {int(lab): i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(int(t))
        pi = label_to_idx.get(int(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def _precision_binary(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    label: int,
    zero_division: float,
) -> float:
    tp = float(np.sum((y_pred == label) & (y_true == label)))
    fp = float(np.sum((y_pred == label) & (y_true != label)))
    return tp / (tp + fp) if (tp + fp) > 0 else zero_division


def _recall_binary(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    label: int,
    zero_division: float,
) -> float:
    tp = float(np.sum((y_pred == label) & (y_true == label)))
    fn = float(np.sum((y_pred != label) & (y_true == label)))
    return tp / (tp + fn) if (tp + fn) > 0 else zero_division


def _f1_binary(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    label: int,
    zero_division: float,
) -> float:
    p = _precision_binary(y_true, y_pred, label, zero_division)
    r = _recall_binary(y_true, y_pred, label, zero_division)
    if p + r == 0:
        return zero_division
    return 2.0 * p * r / (p + r)


def _weighted_metric(
    y_true: NDArray[np.number[Any]],
    y_pred: NDArray[np.number[Any]],
    metric_fn: Any,
    zero_division: float,
) -> float:
    classes = np.unique(y_true)
    total = len(y_true)
    result = 0.0
    for c in classes:
        weight = float(np.sum(y_true == c)) / total
        result += weight * metric_fn(y_true, y_pred, c, zero_division)
    return result


# =====================================================================
# Ranking / Probabilistic Metrics
# =====================================================================


def roc_auc_score(y_true: NDArray[np.number[Any]], y_score: NDArray[np.number[Any]]) -> float:
    """Compute Area Under the ROC Curve using the trapezoidal rule.

    Equivalent to the Wilcoxon-Mann-Whitney statistic.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError("ROC AUC requires at least two classes in y_true")
    if len(classes) > 2:
        raise ValueError(
            "ROC AUC with multiclass data is not supported. "
            "y_true must be binary (2 unique values)."
        )

    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # Compute TPR and FPR at each threshold (support arbitrary binary labels)
    pos_label = classes[-1]  # highest class value is positive
    n_pos = float(np.sum(y_true == pos_label))
    n_neg = float(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires both positive and negative samples")

    tps = np.cumsum(y_true_sorted == pos_label).astype(np.float64)
    fps = np.cumsum(y_true_sorted != pos_label).astype(np.float64)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Handle tied scores
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.concatenate([distinct_indices, [len(y_true_sorted) - 1]])
    tpr = tpr[threshold_indices]
    fpr = fpr[threshold_indices]

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    assert _trapz is not None, "numpy has neither trapezoid nor trapz"
    return float(_trapz(tpr, fpr))


def average_precision_score(
    y_true: NDArray[np.number[Any]], y_score: NDArray[np.number[Any]]
) -> float:
    """Compute Average Precision (area under precision-recall curve)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    classes = np.unique(y_true)
    pos_label = classes[-1] if len(classes) >= 2 else 1

    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]

    n_pos = float(np.sum(y_true == pos_label))
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted == pos_label).astype(np.float64)
    precision = tps / np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    recall_change = np.diff(np.concatenate([[0.0], tps / n_pos]))

    return float(np.sum(precision * recall_change))


def precision_recall_curve(
    y_true: NDArray[np.number[Any]], y_score: NDArray[np.number[Any]]
) -> tuple[NDArray[np.number[Any]], NDArray[np.number[Any]], NDArray[np.number[Any]]]:
    """Compute precision-recall pairs for different probability thresholds."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    classes = np.unique(y_true)
    pos_label = classes[-1] if len(classes) >= 2 else 1

    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    thresholds = y_score[desc_idx]

    n_pos = float(np.sum(y_true == pos_label))
    if n_pos == 0:
        return np.array([1.0]), np.array([0.0]), np.array([])

    tps = np.cumsum(y_true_sorted == pos_label).astype(np.float64)
    fps = np.cumsum(y_true_sorted != pos_label).astype(np.float64)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos

    # Deduplicate at tied thresholds
    distinct = np.concatenate([np.where(np.diff(thresholds))[0], [len(thresholds) - 1]])
    precisions = precisions[distinct]
    recalls = recalls[distinct]
    thresholds = thresholds[distinct]

    # Append sentinel
    precisions = np.concatenate([precisions, [1.0]])
    recalls = np.concatenate([recalls, [0.0]])

    return precisions, recalls, thresholds


def brier_score_loss(y_true: NDArray[np.number[Any]], y_prob: NDArray[np.number[Any]]) -> float:
    """Compute Brier score (mean squared error of predicted probabilities)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(
    y_true: NDArray[np.number[Any]], y_prob: NDArray[np.number[Any]], *, eps: float = 1e-15
) -> float:
    """Compute log loss (cross-entropy loss)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return -float(np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def calibration_curve(
    y_true: NDArray[np.number[Any]],
    y_prob: NDArray[np.number[Any]],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]:
    """Compute calibration curve (reliability diagram data).

    Returns (fraction_of_positives, mean_predicted_value) per bin.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y_prob, quantiles)
        bins = np.unique(bins)

    bin_ids = np.digitize(y_prob, bins[1:-1])

    prob_true: list[float] = []
    prob_pred: list[float] = []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            prob_true.append(float(np.mean(y_true[mask])))
            prob_pred.append(float(np.mean(y_prob[mask])))

    return np.array(prob_true), np.array(prob_pred)


# =====================================================================
# Model Selection
# =====================================================================


class KFold:
    """K-Fold cross-validator."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]] | None = None
    ) -> list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]]:
        """Split."""
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        folds: list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]] = []
        current = 0
        for size in fold_sizes:
            test_idx = indices[current : current + size]
            train_idx = np.concatenate([indices[:current], indices[current + size :]])
            folds.append((train_idx, test_idx))
            current += size
        return folds

    def get_n_splits(self) -> int:
        """Get n splits."""
        return self.n_splits


class StratifiedKFold:
    """Stratified K-Fold cross-validator preserving class proportions."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]
    ) -> list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]]:
        """Split."""
        y = np.asarray(y)
        classes = np.unique(y)
        rng = np.random.RandomState(self.random_state) if self.shuffle else None

        class_indices: dict[Any, NDArray[np.number[Any]]] = {}
        for c in classes:
            idx = np.where(y == c)[0]
            if rng is not None:
                rng.shuffle(idx)
            class_indices[c] = idx

        # Assign each class's indices to folds in round-robin
        fold_indices: list[list[int]] = [[] for _ in range(self.n_splits)]
        for c in classes:
            c_idx = class_indices[c]
            fold_sizes = np.full(self.n_splits, len(c_idx) // self.n_splits, dtype=int)
            fold_sizes[: len(c_idx) % self.n_splits] += 1
            current = 0
            for i, size in enumerate(fold_sizes):
                fold_indices[i].extend(c_idx[current : current + size].tolist())
                current += size

        folds: list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]] = []
        all_indices = np.arange(len(y))
        for fi in fold_indices:
            test_idx = np.array(fi, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]
            folds.append((train_idx, test_idx))
        return folds


class StratifiedShuffleSplit:
    """Stratified shuffle-split cross-validator.

    Provides train/test indices that preserve class proportions.
    """

    def __init__(
        self,
        n_splits: int = 10,
        *,
        test_size: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]
    ) -> list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]]:
        """Split."""
        y = np.asarray(y)
        n = len(y)
        classes = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        folds: list[tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]] = []
        for _ in range(self.n_splits):
            test_indices: list[int] = []
            for c in classes:
                c_idx = np.where(y == c)[0]
                n_test = max(1, int(np.round(len(c_idx) * self.test_size)))
                chosen = rng.choice(c_idx, size=n_test, replace=False)
                test_indices.extend(chosen.tolist())
            test_arr = np.array(test_indices, dtype=int)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_arr] = False
            train_arr = np.where(train_mask)[0]
            folds.append((train_arr, test_arr))
        return folds


def train_test_split(
    X: NDArray[np.number[Any]],
    y: NDArray[np.number[Any]],
    *,
    test_size: float = 0.25,
    random_state: int | None = None,
    stratify: NDArray[np.number[Any]] | None = None,
) -> tuple[
    NDArray[np.number[Any]],
    NDArray[np.number[Any]],
    NDArray[np.number[Any]],
    NDArray[np.number[Any]],
]:
    """Split arrays into random train and test subsets."""
    n = len(X)
    rng = np.random.RandomState(random_state)
    n_test = int(n * test_size)

    if stratify is not None:
        # Stratified split
        classes = np.unique(stratify)
        test_indices: list[int] = []
        for c in classes:
            c_idx = np.where(stratify == c)[0]
            rng.shuffle(c_idx)
            n_c_test = max(1, int(len(c_idx) * test_size))
            test_indices.extend(c_idx[:n_c_test].tolist())
        test_idx = np.array(test_indices, dtype=int)
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        test_idx = indices[:n_test]

    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_val_predict(
    estimator: Any,
    X: NDArray[np.number[Any]],
    y: NDArray[np.number[Any]],
    *,
    cv: int = 5,
    method: str = "predict",
) -> NDArray[np.number[Any]]:
    """Generate cross-validated predictions using stratified folds."""
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    predictions = None

    for train_idx, test_idx in kf.split(X, y):
        est = clone(estimator)
        try:
            est.fit(X[train_idx], y[train_idx])
        except TypeError:
            est.fit(X[train_idx])

        pred_fn = getattr(est, method)
        pred = pred_fn(X[test_idx])

        if predictions is None:
            if pred.ndim == 2:
                predictions = np.zeros((len(X), pred.shape[1]))
            else:
                predictions = np.zeros(len(X))

        predictions[test_idx] = pred

    assert predictions is not None
    return predictions


def cross_val_score(
    estimator: Any,
    X: NDArray[np.number[Any]],
    y: NDArray[np.number[Any]],
    *,
    cv: int = 5,
) -> NDArray[np.number[Any]]:
    """Evaluate estimator by cross-validation, returning per-fold accuracy."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores: list[float] = []

    for train_idx, test_idx in kf.split(X, y):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])
        pred = est.predict(X[test_idx])
        scores.append(float(np.mean(pred == y[test_idx])))

    return np.array(scores)


def clone(estimator: Any) -> Any:
    """Deep-copy an estimator (sklearn-free clone)."""
    return copy.deepcopy(estimator)


# =====================================================================
# Preprocessing
# =====================================================================


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.mean_: NDArray[np.number[Any]] | None = None
        self.scale_: NDArray[np.number[Any]] | None = None

    def fit(self, X: NDArray[np.number[Any]]) -> StandardScaler:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Transform."""
        assert self.mean_ is not None and self.scale_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Fit transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Inverse transform."""
        assert self.mean_ is not None and self.scale_ is not None
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.classes_: NDArray[np.number[Any]] | None = None

    def fit(self, y: NDArray[np.number[Any]]) -> LabelEncoder:
        """Fit."""
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Transform."""
        assert self.classes_ is not None
        y = np.asarray(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[v] for v in y], dtype=int)

    def fit_transform(self, y: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Fit transform."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Inverse transform."""
        assert self.classes_ is not None
        return self.classes_[np.asarray(y, dtype=int)]


# =====================================================================
# Decomposition
# =====================================================================


class PCA:
    """Principal Component Analysis via truncated SVD."""

    def __init__(self, n_components: int = 2) -> None:
        """Initialize the instance."""
        self.n_components = n_components
        self.components_: NDArray[np.number[Any]] = np.empty((0, 0))
        self.mean_: NDArray[np.number[Any]] = np.empty(0)
        self.explained_variance_: NDArray[np.number[Any]] = np.empty(0)
        self.explained_variance_ratio_: NDArray[np.number[Any]] = np.empty(0)

    def fit(self, X: NDArray[np.number[Any]]) -> PCA:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)
        X_centered: NDArray[np.float64] = np.asarray(X - self.mean_, dtype=np.float64)
        _U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        # Explained variance.  ``n_samples == 1`` collapses the
        # sample-variance denominator to zero; in that degenerate
        # case there is no variance to explain, so set the values
        # explicitly rather than letting the divide emit a
        # RuntimeWarning and produce inf/nan that the ratio branch
        # then has to clean up.
        if n_samples > 1:
            explained_var = (S**2) / (n_samples - 1)
        else:
            explained_var = np.zeros_like(S)
        self.explained_variance_ = explained_var[: self.n_components]
        total_var = explained_var.sum()
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Transform."""
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Fit transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Inverse transform."""
        return np.asarray(X_reduced, dtype=np.float64) @ self.components_ + self.mean_


# =====================================================================
# Clustering
# =====================================================================


class KMeans:
    """K-Means clustering."""

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: NDArray[np.number[Any]] | None = None
        self.labels_: NDArray[np.number[Any]] | None = None
        self.inertia_: float = 0.0

    def fit(self, X: NDArray[np.number[Any]]) -> KMeans:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        k = min(self.n_clusters, n_samples)

        indices = rng.choice(n_samples, k, replace=False)
        centroids = X[indices].copy()

        for _ in range(self.max_iter):
            dists = cdist(X, centroids, metric="euclidean")
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = X[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        dists = cdist(X, centroids, metric="euclidean")
        self.inertia_ = float(np.sum(np.min(dists, axis=1) ** 2))
        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        assert self.cluster_centers_ is not None
        dists = cdist(np.asarray(X, dtype=np.float64), self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Fit predict."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ) -> None:
        """Initialize the instance."""
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: NDArray[np.number[Any]] = np.empty(0, dtype=int)
        self.core_sample_indices_: NDArray[np.number[Any]] = np.empty(0, dtype=int)

    def fit_predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Fit predict."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        dists = cdist(X, X, metric=self.metric)

        # Find core points
        neighbors = [np.where(dists[i] <= self.eps)[0] for i in range(n)]
        core_mask = np.array([len(nb) >= self.min_samples for nb in neighbors])
        self.core_sample_indices_ = np.where(core_mask)[0]

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        for i in self.core_sample_indices_:
            if labels[i] != -1:
                continue
            # BFS expansion
            queue = [i]
            labels[i] = cluster_id
            while queue:
                pt = queue.pop(0)
                nbs = neighbors[pt]
                for nb in nbs:
                    if labels[nb] == -1:
                        labels[nb] = cluster_id
                        if core_mask[nb]:
                            queue.append(nb)
            cluster_id += 1

        self.labels_ = labels
        return labels


# =====================================================================
# Neighbors
# =====================================================================


class NearestNeighbors:
    """Unsupervised nearest neighbors using brute-force distance computation."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
    ) -> None:
        """Initialize the instance."""
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self._X: NDArray[np.number[Any]] | None = None

    def fit(self, X: NDArray[np.number[Any]]) -> NearestNeighbors:
        """Fit."""
        self._X = np.asarray(X, dtype=np.float64)
        return self

    def kneighbors(
        self, X: NDArray[np.number[Any]] | None = None, n_neighbors: int | None = None
    ) -> tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]:
        """Kneighbors."""
        assert self._X is not None
        if X is None:
            X = self._X
        else:
            X = np.asarray(X, dtype=np.float64)
        k = n_neighbors or self.n_neighbors
        dists = cdist(X, self._X, metric=self.metric)
        indices = np.argsort(dists, axis=1)[:, :k]
        distances = np.take_along_axis(dists, indices, axis=1)
        return distances, indices


# =====================================================================
# Linear Models
# =====================================================================


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before being fit.

    Matches the semantics of :class:`sklearn.exceptions.NotFittedError`:
    the exception inherits from both :class:`ValueError` (so that
    existing call-sites catching ``ValueError`` continue to handle the
    typed error) and :class:`AttributeError` (so that hasattr-style
    feature probes return False for unfit estimators).
    """


class LogisticRegression:
    """Logistic Regression using L-BFGS (sklearn-free reimplementation)."""

    def __init__(
        self,
        *,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.coef_: NDArray[np.number[Any]] = np.empty((0, 0))
        self.intercept_: NDArray[np.number[Any]] = np.empty(0)
        self.classes_: NDArray[np.number[Any]] = np.empty(0)
        self.is_fitted_: bool = False

    def fit(self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]) -> LogisticRegression:
        """Fit."""
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        if self.classes_.size < 2:
            raise ValueError(
                "LogisticRegression.fit requires at least two distinct classes in y; "
                f"got {self.classes_.tolist()}."
            )

        # Binary classification
        y_bin = (y == self.classes_[-1]).astype(np.float64)
        n_features = X.shape[1]

        # w = [weights..., bias]
        w0 = np.zeros(n_features + 1)

        def objective(w: NDArray[np.number[Any]]) -> float:
            weights, bias = w[:-1], w[-1]
            z = X @ weights + bias
            z = np.clip(z, -500, 500)
            log_likelihood = np.sum(y_bin * z - np.log1p(np.exp(z)))
            reg = 0.5 / self.C * np.sum(weights**2)
            return -(log_likelihood - reg)

        def gradient(w: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
            weights, bias = w[:-1], w[-1]
            z = X @ weights + bias
            z = np.clip(z, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            diff = p - y_bin
            grad_w = X.T @ diff + weights / self.C
            grad_b = np.sum(diff)
            return np.concatenate([grad_w, [grad_b]])

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coef_ = result.x[:-1].reshape(1, -1)
        self.intercept_ = np.array([result.x[-1]])
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        """Raise :class:`NotFittedError` if the estimator was not fit."""
        if not self.is_fitted_:
            raise NotFittedError(
                "LogisticRegression instance is not fitted yet. Call 'fit' with "
                "appropriate arguments (and at least two distinct class labels) "
                "before using this estimator."
            )

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        self._check_is_fitted()
        prob = self.predict_proba(X)
        return self.classes_[(prob[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict proba."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_.T + self.intercept_
        z = np.clip(z.ravel(), -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


class SGDClassifier:
    """SGD classifier for online learning (sklearn-free reimplementation).

    Based on standard stochastic gradient descent with hinge/log loss.
    """

    def __init__(
        self,
        loss: str = "hinge",
        penalty: str = "l2",
        alpha: float = 0.0001,
        learning_rate: str = "constant",
        eta0: float = 0.01,
        warm_start: bool = False,
        max_iter: int = 1,
        tol: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._w: NDArray[np.number[Any]] | None = None
        self._b: float = 0.0
        self._classes: NDArray[np.number[Any]] | None = None
        self._t: int = 0

    def partial_fit(
        self,
        X: NDArray[np.number[Any]],
        y: NDArray[np.number[Any]],
        classes: NDArray[np.number[Any]] | None = None,
    ) -> SGDClassifier:
        """Partial fit."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if classes is not None:
            self._classes = np.asarray(classes)
        elif self._classes is None:
            self._classes = np.unique(y)

        if self._w is None:
            self._w = np.zeros(X.shape[1])

        # Map labels to +1/-1
        y_bin = np.where(y == self._classes[-1], 1.0, -1.0)

        for i in range(len(X)):
            self._t += 1
            eta = self.eta0
            xi, yi = X[i], y_bin[i]
            margin = yi * (xi @ self._w + self._b)

            if self.loss == "hinge":
                if margin < 1.0:
                    self._w += eta * (yi * xi - self.alpha * self._w)
                    self._b += eta * yi
                else:
                    self._w -= eta * self.alpha * self._w
            else:  # log loss
                p = 1.0 / (1.0 + np.exp(-yi * (xi @ self._w + self._b)))
                self._w += eta * ((1 - p) * yi * xi - self.alpha * self._w)
                self._b += eta * (1 - p) * yi

        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        assert self._w is not None and self._classes is not None
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self._w + self._b
        return self._classes[(scores >= 0).astype(int)]

    def predict_proba(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict proba."""
        assert self._w is not None
        X = np.asarray(X, dtype=np.float64)
        z = X @ self._w + self._b
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return np.column_stack([1 - p1, p1])

    def decision_function(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Decision function."""
        assert self._w is not None
        return np.asarray(X, dtype=np.float64) @ self._w + self._b


class PassiveAggressiveClassifier:
    """Passive-Aggressive classifier (sklearn-free reimplementation).

    Algorithm by Crammer, Dekel, Keshet, Shalev-Shwartz & Singer (2006).

    Reference:
        Crammer, K., et al. (2006). Online passive-aggressive algorithms.
        *JMLR*, 7, 551–585.
    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        warm_start: bool = True,
        max_iter: int = 1,
        tol: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.C = C
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._w: NDArray[np.number[Any]] | None = None
        self._b: float = 0.0
        self._classes: NDArray[np.number[Any]] | None = None

    def partial_fit(
        self,
        X: NDArray[np.number[Any]],
        y: NDArray[np.number[Any]],
        classes: NDArray[np.number[Any]] | None = None,
    ) -> PassiveAggressiveClassifier:
        """Partial fit."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if classes is not None:
            self._classes = np.asarray(classes)
        elif self._classes is None:
            self._classes = np.unique(y)

        if self._w is None:
            self._w = np.zeros(X.shape[1])

        y_bin = np.where(y == self._classes[-1], 1.0, -1.0)

        for i in range(len(X)):
            xi, yi = X[i], y_bin[i]
            margin = yi * (xi @ self._w + self._b)
            loss = max(0.0, 1.0 - margin)
            if loss > 0:
                norm_sq = float(np.dot(xi, xi)) + (1.0 if self.fit_intercept else 0.0)
                tau = min(self.C, loss / max(norm_sq, 1e-10))
                self._w += tau * yi * xi
                if self.fit_intercept:
                    self._b += tau * yi

        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        assert self._w is not None and self._classes is not None
        scores = np.asarray(X, dtype=np.float64) @ self._w + self._b
        return self._classes[(scores >= 0).astype(int)]

    def decision_function(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Decision function."""
        assert self._w is not None
        return np.asarray(X, dtype=np.float64) @ self._w + self._b


# =====================================================================
# Ensemble Classifiers
# =====================================================================


class GradientBoostingClassifier:
    """Gradient Boosting using decision stumps (sklearn-free reimplementation).

    Based on Friedman (2001). Simplified to binary classification with
    decision stumps as weak learners.

    Reference:
        Friedman, J. H. (2001). Greedy function approximation: a
        gradient boosting machine. *Annals of Statistics*, 29(5),
        1189–1232.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._stumps: list[_DecisionStump] = []
        self._init_pred: float = 0.0
        self.classes_: NDArray[np.number[Any]] | None = None

    def fit(
        self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]
    ) -> GradientBoostingClassifier:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[-1]).astype(np.float64)

        # Initialize with log-odds
        p = np.mean(y_bin)
        self._init_pred = float(np.log(max(p, 1e-10) / max(1 - p, 1e-10)))

        F = np.full(len(X), self._init_pred)
        rng = np.random.RandomState(self.random_state)

        self._stumps = []
        for _ in range(self.n_estimators):
            # Gradient (negative gradient of log-loss)
            p = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            residuals = y_bin - p

            stump = _DecisionStump(max_depth=self.max_depth, rng=rng)
            stump.fit(X, residuals)
            pred = stump.predict(X)
            F += self.learning_rate * pred
            self._stumps.append(stump)

        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict proba."""
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self._init_pred)
        for stump in self._stumps:
            F += self.learning_rate * stump.predict(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        return np.column_stack([1 - p1, p1])


class RandomForestClassifier:
    """Random Forest classifier (sklearn-free reimplementation).

    Ensemble of bootstrapped decision stumps. Based on Breiman (2001).

    Reference:
        Breiman, L. (2001). Random forests. *Machine Learning*, 45(1),
        5–32.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth or 10
        self.random_state = random_state
        self._trees: list[_DecisionStump] = []
        self.classes_: NDArray[np.number[Any]] = np.empty(0)
        self.feature_importances_: NDArray[np.number[Any]] = np.empty(0)

    def fit(self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]) -> RandomForestClassifier:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        n_features = X.shape[1]

        self._trees = []
        importances = np.zeros(n_features)
        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = rng.choice(n, n, replace=True)
            tree = _DecisionStump(max_depth=self.max_depth, rng=rng)
            tree.fit(X[idx], y[idx])
            self._trees.append(tree)
            if tree.feature_idx is not None:
                importances[tree.feature_idx] += 1.0
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict proba."""
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)
        class_vals = self.classes_.astype(np.float64)
        all_preds = np.zeros((len(X), n_classes))
        for tree in self._trees:
            preds = tree.predict(X)
            # _DecisionStump returns continuous leaf means; map to nearest class
            diffs = np.abs(preds[:, np.newaxis] - class_vals[np.newaxis, :])
            nearest_class_idx = np.argmin(diffs, axis=1)
            for i in range(n_classes):
                all_preds[:, i] += (nearest_class_idx == i).astype(float)
        return all_preds / len(self._trees)


class SVC:
    """SVC using kernel-based scoring (sklearn-free reimplementation).

    Lightweight RBF-kernel similarity scorer. Based on the support vector classification framework
    of Vapnik (1995).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        probability: bool = True,
        random_state: int | None = None,
        C: float = 1.0,
    ) -> None:
        """Initialize the instance."""
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.C = C
        self._X_train: NDArray[np.number[Any]] | None = None
        self._y_train: NDArray[np.number[Any]] | None = None
        self._alpha: NDArray[np.number[Any]] | None = None
        self._gamma: float = 1.0
        self.classes_: NDArray[np.number[Any]] | None = None

    def fit(self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]) -> SVC:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        self._gamma = 1.0 / (X.shape[1] * max(X.var(), 1e-10))

        # Simplified: store training data and use kernel-weighted voting
        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict proba."""
        assert self._X_train is not None and self._y_train is not None
        X = np.asarray(X, dtype=np.float64)
        K = np.exp(-self._gamma * cdist(X, self._X_train, metric="sqeuclidean"))

        assert self.classes_ is not None
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, c in enumerate(self.classes_):
            mask = self._y_train == c
            proba[:, i] = np.sum(K[:, mask], axis=1)
        proba /= proba.sum(axis=1, keepdims=True) + 1e-10
        return proba


class _DecisionStump:
    """Simple decision tree (stump or shallow) for gradient boosting / RF."""

    def __init__(self, max_depth: int = 3, rng: np.random.RandomState | None = None) -> None:
        """Initialize the instance."""
        self.max_depth = max_depth
        self.rng = rng or np.random.RandomState()
        self.feature: int = 0
        self.feature_idx: int | None = None
        self.threshold: float = 0.0
        self.value: float = 0.0
        self.left: _DecisionStump | None = None
        self.right: _DecisionStump | None = None
        self.is_leaf: bool = True

    def fit(self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]], depth: int = 0) -> None:
        y_arr: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
        self.value = float(np.mean(y_arr))
        if depth >= self.max_depth or len(X) <= 2 or float(np.std(y_arr)) < 1e-10:
            self.is_leaf = True
            return

        n_features = X.shape[1]
        # Random feature subset (sqrt)
        n_try = max(1, int(np.sqrt(n_features)))
        features = self.rng.choice(n_features, n_try, replace=False)

        best_gain = -1.0
        for f in features:
            col = X[:, f]
            # Try a few random thresholds
            for _ in range(min(10, len(col))):
                t = float(self.rng.uniform(col.min(), col.max()))
                left_mask = col < t
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                var_reduction = float(np.var(y_arr)) - (
                    float(np.sum(left_mask)) * float(np.var(y_arr[left_mask]))
                    + float(np.sum(right_mask)) * float(np.var(y_arr[right_mask]))
                ) / len(y_arr)
                if var_reduction > best_gain:
                    best_gain = var_reduction
                    self.feature = int(f)
                    self.threshold = t

        if best_gain <= 0:
            self.is_leaf = True
            return

        self.is_leaf = False
        self.feature_idx = self.feature
        left_mask = X[:, self.feature] < self.threshold
        self.left = _DecisionStump(self.max_depth, self.rng)
        self.right = _DecisionStump(self.max_depth, self.rng)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        if self.is_leaf:
            return np.full(len(X), self.value)
        col = X[:, self.feature]
        left_mask = col < self.threshold
        result = np.full(len(X), self.value)
        if self.left is not None and np.any(left_mask):
            result[left_mask] = self.left.predict(X[left_mask])
        if self.right is not None and np.any(~left_mask):
            result[~left_mask] = self.right.predict(X[~left_mask])
        return result


# =====================================================================
# Mixture Models
# =====================================================================


class GaussianMixture:
    """Gaussian Mixture Model via EM (sklearn-free reimplementation).

    Expectation-Maximization algorithm by Dempster, Laird & Rubin (1977).

    Reference:
        Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum
        likelihood from incomplete data via the EM algorithm. *JRSS-B*,
        39(1), 1–38.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        """Initialize the instance."""
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_: NDArray[np.number[Any]] = np.empty(0)
        self.covariances_: NDArray[np.number[Any]] = np.empty(0)
        self.weights_: NDArray[np.number[Any]] = np.empty(0)

    def fit(self, X: NDArray[np.number[Any]]) -> GaussianMixture:
        """Fit."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize
        idx = rng.choice(n, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([np.eye(d)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: returns (normalized responsibilities, per-sample likelihoods)
            resp, sample_likelihoods = self._e_step(X)

            # M-step
            nk = resp.sum(axis=0)
            self.weights_ = nk / n
            for k in range(self.n_components):
                if nk[k] < 1e-10:
                    continue
                self.means_[k] = (resp[:, k : k + 1].T @ X / nk[k]).ravel()
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k : k + 1] * diff).T @ diff / nk[k] + np.eye(
                    d
                ) * 1e-6

            # Check convergence using actual data log-likelihood
            ll = float(np.sum(np.log(sample_likelihoods + 1e-300)))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def _e_step(
        self, X: NDArray[np.number[Any]]
    ) -> tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]:
        n = len(X)
        resp = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            try:
                resp[:, k] = self.weights_[k] * sp_stats.multivariate_normal.pdf(
                    X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
                )
            except (np.linalg.LinAlgError, ValueError):
                resp[:, k] = 1e-300
        # Save per-sample total likelihood before normalization
        sample_likelihoods = resp.sum(axis=1)
        resp /= sample_likelihoods[:, np.newaxis] + 1e-300
        return resp, sample_likelihoods


# =====================================================================
# Feature Selection
# =====================================================================


def mutual_info_classif(
    X: NDArray[np.number[Any]], y: NDArray[np.number[Any]], *, random_state: int | None = None
) -> NDArray[np.number[Any]]:
    """Estimate mutual information between features and discrete target.

    Uses a histogram-based approximation.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_features = X.shape[1]
    mi = np.zeros(n_features)

    classes = np.unique(y)
    n = len(y)

    for f in range(n_features):
        # Discretize continuous feature into bins
        col = X[:, f]
        n_bins = min(10, len(np.unique(col)))
        if n_bins <= 1:
            mi[f] = 0.0
            continue
        col_f64: NDArray[np.float64] = np.asarray(col, dtype=np.float64)
        bins = np.percentile(col_f64, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) <= 1:
            mi[f] = 0.0
            continue
        x_binned = np.digitize(np.asarray(col, dtype=np.float64), bins[1:-1])

        # Compute MI via contingency table
        x_vals = np.unique(x_binned)
        mi_val = 0.0
        for xv in x_vals:
            for c in classes:
                pxy = np.sum((x_binned == xv) & (y == c)) / n
                px = np.sum(x_binned == xv) / n
                py = np.sum(y == c) / n
                if pxy > 0 and px > 0 and py > 0:
                    mi_val += pxy * np.log(pxy / (px * py))
        mi[f] = max(0.0, mi_val)

    return mi


# =====================================================================
# Isotonic Regression
# =====================================================================

# =====================================================================
# Synthetic Data Generators
# =====================================================================


def make_classification(
    n_samples: int = 100,
    n_features: int = 20,
    n_informative: int = 2,
    n_redundant: int = 2,
    n_classes: int = 2,
    weights: list[float] | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
    """Generate a random n-class classification problem (Mercury-native).

    Parameters match sklearn's make_classification for compatibility.
    """
    rng = np.random.RandomState(random_state)
    n_inf = min(n_informative, n_features)
    X = rng.randn(n_samples, n_features).astype(np.float64)
    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    cumw = np.cumsum(weights)
    cumw = cumw / cumw[-1]
    u = rng.rand(n_samples)
    y = np.zeros(n_samples, dtype=np.intp)
    for i in range(1, n_classes):
        y[u > cumw[i - 1]] = i
    # Make informative features correlate with labels
    for i in range(n_inf):
        X[:, i] += y * 2.0
    # Add redundant features as linear combos of informative ones
    n_red = min(n_redundant, n_features - n_inf)
    for i in range(n_red):
        combo = rng.randn(n_inf)
        X[:, n_inf + i] = X[:, :n_inf] @ combo
    return X, y


def make_blobs(
    n_samples: int = 100,
    centers: list[list[float]] | NDArray[np.floating[Any]] | None = None,
    n_features: int = 2,
    cluster_std: float = 1.0,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
    """Generate isotropic Gaussian blobs (Mercury-native).

    Parameters match sklearn's make_blobs for compatibility.
    """
    rng = np.random.RandomState(random_state)
    centers_arr: NDArray[np.floating[Any]]
    if centers is None:
        centers_arr = np.zeros((1, n_features))
    else:
        centers_arr = np.asarray(centers, dtype=np.float64)
    n_centers = len(centers_arr)
    n_features = centers_arr.shape[1]
    samples_per = n_samples // max(n_centers, 1)
    X_list = []
    y_list = []
    for ci, center in enumerate(centers_arr):
        count = samples_per if ci < n_centers - 1 else n_samples - samples_per * ci
        blob = rng.randn(count, n_features) * cluster_std + center
        X_list.append(blob)
        y_list.append(np.full(count, ci, dtype=np.intp))
    X = np.vstack(X_list).astype(np.float64)
    y = np.concatenate(y_list)
    return X, y


class IsotonicRegression:
    """Isotonic regression via pool adjacent violators (sklearn-free reimplementation).

    Algorithm by Barlow, Bartholomew, Bremner & Brunk (1972).
    """

    def __init__(
        self,
        y_min: float = 0.0,
        y_max: float = 1.0,
        out_of_bounds: str = "clip",
    ) -> None:
        """Initialize the instance."""
        self.y_min = y_min
        self.y_max = y_max
        self.out_of_bounds = out_of_bounds
        self._x: NDArray[np.floating[Any]] | None = None
        self._y: NDArray[np.floating[Any]] | None = None

    def fit(self, X: NDArray[np.number[Any]], y: NDArray[np.number[Any]]) -> IsotonicRegression:
        """Fit."""
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        # Sort by X
        order = np.argsort(X)
        X_sorted = X[order]
        y_sorted = y[order].copy()

        # Pool Adjacent Violators
        n = len(y_sorted)
        blocks = list(range(n))
        values: NDArray[np.float64] = y_sorted.copy()
        weights: NDArray[np.float64] = np.ones(n, dtype=np.float64)

        i = 0
        while i < len(values) - 1:
            if values[i] > values[i + 1]:
                # Merge
                total_w = weights[i] + weights[i + 1]
                values[i] = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / total_w
                weights[i] = total_w
                values = np.delete(values, i + 1)
                weights = np.delete(weights, i + 1)
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Expand back
        result: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
        idx = 0
        for val, w in zip(values, weights):
            count = int(w)
            result[idx : idx + count] = val
            idx += count
        # Handle any remaining due to float rounding
        if idx < n:
            result[idx:] = values[-1]

        result = np.clip(result, self.y_min, self.y_max)
        self._x = X_sorted
        self._y = result
        return self

    def predict(self, X: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Predict."""
        assert self._x is not None and self._y is not None
        X = np.asarray(X, dtype=np.float64).ravel()
        result = np.interp(X, self._x, self._y)
        if self.out_of_bounds == "clip":
            result = np.clip(result, self.y_min, self.y_max)
        return result

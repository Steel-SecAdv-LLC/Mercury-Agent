"""Mercury-native ML primitives — zero sklearn dependency.

This module provides Mercury's own implementations of common ML utilities,
metrics, model selection, preprocessing, and anomaly detection components.
Every function here uses only numpy and scipy (standard numerical libs).

NO sklearn. If Mercury fails, it fails. Period.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# =====================================================================
# Classification Metrics
# =====================================================================


def accuracy_score(y_true: NDArray, y_pred: NDArray) -> float:
    """Compute classification accuracy."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    zero_division: float = 0.0,
    average: str = "binary",
) -> float:
    """Compute precision score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _precision_binary, zero_division)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    denom = tp + fp
    if denom == 0:
        return zero_division
    return tp / denom


def recall_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    zero_division: float = 0.0,
    average: str = "binary",
) -> float:
    """Compute recall score."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _recall_binary, zero_division)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    denom = tp + fn
    if denom == 0:
        return zero_division
    return tp / denom


def f1_score(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    zero_division: float = 0.0,
    average: str = "binary",
) -> float:
    """Compute F1 score."""
    p = precision_score(y_true, y_pred, zero_division=zero_division, average=average)
    r = recall_score(y_true, y_pred, zero_division=zero_division, average=average)
    if p + r == 0:
        return zero_division
    return 2.0 * p * r / (p + r)


def confusion_matrix(
    y_true: NDArray, y_pred: NDArray, *, labels: NDArray | None = None
) -> NDArray:
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
    NDArray of shape (n_classes, n_classes)
        Entry *C[i, j]* is the number of samples with true label *labels[i]*
        and predicted label *labels[j]*.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    label_to_idx = {int(l): i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(int(t))
        pi = label_to_idx.get(int(p))
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def _precision_binary(
    y_true: NDArray, y_pred: NDArray, label: int, zero_division: float
) -> float:
    tp = float(np.sum((y_pred == label) & (y_true == label)))
    fp = float(np.sum((y_pred == label) & (y_true != label)))
    return tp / (tp + fp) if (tp + fp) > 0 else zero_division


def _recall_binary(
    y_true: NDArray, y_pred: NDArray, label: int, zero_division: float
) -> float:
    tp = float(np.sum((y_pred == label) & (y_true == label)))
    fn = float(np.sum((y_pred != label) & (y_true == label)))
    return tp / (tp + fn) if (tp + fn) > 0 else zero_division


def _weighted_metric(
    y_true: NDArray,
    y_pred: NDArray,
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


def roc_auc_score(y_true: NDArray, y_score: NDArray) -> float:
    """Compute Area Under the ROC Curve using the trapezoidal rule.

    Equivalent to the Wilcoxon-Mann-Whitney statistic.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError("ROC AUC requires at least two classes in y_true")

    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # Compute TPR and FPR at each threshold
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC requires both positive and negative samples")

    tps = np.cumsum(y_true_sorted == 1).astype(np.float64)
    fps = np.cumsum(y_true_sorted == 0).astype(np.float64)
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

    return float(np.trapz(tpr, fpr))


def average_precision_score(y_true: NDArray, y_score: NDArray) -> float:
    """Compute Average Precision (area under precision-recall curve)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]

    n_pos = float(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted == 1).astype(np.float64)
    precision = tps / np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)
    recall_change = np.diff(np.concatenate([[0.0], tps / n_pos]))

    return float(np.sum(precision * recall_change))


def precision_recall_curve(
    y_true: NDArray, y_score: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute precision-recall pairs for different probability thresholds."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    thresholds = y_score[desc_idx]

    n_pos = float(np.sum(y_true == 1))
    if n_pos == 0:
        return np.array([1.0]), np.array([0.0]), np.array([])

    tps = np.cumsum(y_true_sorted == 1).astype(np.float64)
    fps = np.cumsum(y_true_sorted == 0).astype(np.float64)
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


def brier_score_loss(y_true: NDArray, y_prob: NDArray) -> float:
    """Compute Brier score (mean squared error of predicted probabilities)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: NDArray, y_prob: NDArray, *, eps: float = 1e-15) -> float:
    """Compute log loss (cross-entropy loss)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return -float(np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def calibration_curve(
    y_true: NDArray,
    y_prob: NDArray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[NDArray, NDArray]:
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
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: NDArray, y: NDArray | None = None
    ) -> list[tuple[NDArray, NDArray]]:
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        folds: list[tuple[NDArray, NDArray]] = []
        current = 0
        for size in fold_sizes:
            test_idx = indices[current : current + size]
            train_idx = np.concatenate([indices[:current], indices[current + size :]])
            folds.append((train_idx, test_idx))
            current += size
        return folds

    def get_n_splits(self) -> int:
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
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: NDArray, y: NDArray
    ) -> list[tuple[NDArray, NDArray]]:
        y = np.asarray(y)
        classes = np.unique(y)
        rng = np.random.RandomState(self.random_state) if self.shuffle else None

        class_indices: dict[Any, NDArray] = {}
        for c in classes:
            idx = np.where(y == c)[0]
            if rng is not None:
                rng.shuffle(idx)
            class_indices[c] = idx

        # Assign each class's indices to folds in round-robin
        fold_indices: list[list[int]] = [[] for _ in range(self.n_splits)]
        for c in classes:
            idx = class_indices[c]
            fold_sizes = np.full(self.n_splits, len(idx) // self.n_splits, dtype=int)
            fold_sizes[: len(idx) % self.n_splits] += 1
            current = 0
            for i, size in enumerate(fold_sizes):
                fold_indices[i].extend(idx[current : current + size].tolist())
                current += size

        folds: list[tuple[NDArray, NDArray]] = []
        all_indices = np.arange(len(y))
        for fi in fold_indices:
            test_idx = np.array(fi, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]
            folds.append((train_idx, test_idx))
        return folds


def train_test_split(
    X: NDArray,
    y: NDArray,
    *,
    test_size: float = 0.25,
    random_state: int | None = None,
    stratify: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
    X: NDArray,
    y: NDArray,
    *,
    cv: int = 5,
    method: str = "predict",
) -> NDArray:
    """Generate cross-validated predictions."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
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
    X: NDArray,
    y: NDArray,
    *,
    cv: int = 5,
) -> NDArray:
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
    """Deep-copy an estimator (Mercury-native replacement for sklearn.base.clone)."""
    return copy.deepcopy(estimator)


# =====================================================================
# Preprocessing
# =====================================================================


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self) -> None:
        self.mean_: NDArray | None = None
        self.scale_: NDArray | None = None

    def fit(self, X: NDArray) -> StandardScaler:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: NDArray) -> NDArray:
        assert self.mean_ is not None and self.scale_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: NDArray) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: NDArray) -> NDArray:
        assert self.mean_ is not None and self.scale_ is not None
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1."""

    def __init__(self) -> None:
        self.classes_: NDArray | None = None

    def fit(self, y: NDArray) -> LabelEncoder:
        self.classes_ = np.unique(y)
        return self

    def transform(self, y: NDArray) -> NDArray:
        assert self.classes_ is not None
        y = np.asarray(y)
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return np.array([mapping[v] for v in y], dtype=int)

    def fit_transform(self, y: NDArray) -> NDArray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: NDArray) -> NDArray:
        assert self.classes_ is not None
        return self.classes_[np.asarray(y, dtype=int)]


# =====================================================================
# Decomposition
# =====================================================================


class PCA:
    """Principal Component Analysis via truncated SVD."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: NDArray | None = None
        self.mean_: NDArray | None = None
        self.explained_variance_: NDArray | None = None
        self.explained_variance_ratio_: NDArray | None = None

    def fit(self, X: NDArray) -> PCA:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        _U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        # Explained variance
        explained_var = (S ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_var[: self.n_components]
        total_var = explained_var.sum()
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self

    def transform(self, X: NDArray) -> NDArray:
        assert self.mean_ is not None and self.components_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: NDArray) -> NDArray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: NDArray) -> NDArray:
        assert self.mean_ is not None and self.components_ is not None
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
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_: NDArray | None = None
        self.labels_: NDArray | None = None
        self.inertia_: float = 0.0

    def fit(self, X: NDArray) -> KMeans:
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

    def predict(self, X: NDArray) -> NDArray:
        assert self.cluster_centers_ is not None
        dists = cdist(np.asarray(X, dtype=np.float64), self.cluster_centers_)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: NDArray) -> NDArray:
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
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: NDArray | None = None
        self.core_sample_indices_: NDArray | None = None

    def fit_predict(self, X: NDArray) -> NDArray:
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
                    elif labels[nb] == -1:
                        labels[nb] = cluster_id
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
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self._X: NDArray | None = None

    def fit(self, X: NDArray) -> NearestNeighbors:
        self._X = np.asarray(X, dtype=np.float64)
        return self

    def kneighbors(
        self, X: NDArray | None = None, n_neighbors: int | None = None
    ) -> tuple[NDArray, NDArray]:
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
# Anomaly Detection
# =====================================================================


class IsolationForest:
    """Mercury-native Isolation Forest for anomaly detection.

    Uses random recursive partitioning — anomalies are isolated in
    fewer splits on average, yielding shorter path lengths.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float = 0.1,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._trees: list[_ITree] = []
        self._n_samples: int = 0
        self._offset: float = 0.0

    def fit(self, X: NDArray, y: Any = None) -> IsolationForest:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self._n_samples = n_samples

        if self.max_samples == "auto":
            subsample_size = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            subsample_size = int(n_samples * self.max_samples)
        else:
            subsample_size = min(int(self.max_samples), n_samples)

        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))
        rng = np.random.RandomState(self.random_state)

        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            tree = _ITree(max_depth=max_depth, rng=rng)
            tree.fit(X[idx])
            self._trees.append(tree)

        # Set offset for decision_function (threshold at contamination)
        scores = self._raw_score(X)
        self._offset = float(np.percentile(scores, 100 * self.contamination))

        return self

    def _raw_score(self, X: NDArray) -> NDArray:
        """Compute raw anomaly scores (higher = more anomalous)."""
        avg_path = np.zeros(len(X))
        for tree in self._trees:
            avg_path += tree.path_lengths(X)
        avg_path /= len(self._trees)

        # Normalize by expected path length c(n)
        c_n = _expected_path_length(self._n_samples)
        scores = 2.0 ** (-avg_path / max(c_n, 1e-10))
        return scores

    def decision_function(self, X: NDArray) -> NDArray:
        """Compute decision function (negative = more anomalous, like sklearn)."""
        X = np.asarray(X, dtype=np.float64)
        return -(self._raw_score(X) - self._offset)

    def predict(self, X: NDArray) -> NDArray:
        """Predict: -1 for anomalies, 1 for inliers."""
        scores = self.decision_function(X)
        return np.where(scores < 0, -1, 1)

    def score_samples(self, X: NDArray) -> NDArray:
        """Return anomaly scores (negative = more anomalous)."""
        return -self._raw_score(np.asarray(X, dtype=np.float64))


class _ITree:
    """Single Isolation Tree node."""

    def __init__(self, max_depth: int, rng: np.random.RandomState) -> None:
        self.max_depth = max_depth
        self.rng = rng
        self.feature: int = 0
        self.threshold: float = 0.0
        self.left: _ITree | None = None
        self.right: _ITree | None = None
        self.size: int = 0
        self.is_leaf: bool = True

    def fit(self, X: NDArray, depth: int = 0) -> None:
        self.size = len(X)
        if depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            return

        n_features = X.shape[1]
        self.feature = int(self.rng.randint(0, n_features))
        col = X[:, self.feature]
        col_min, col_max = col.min(), col.max()

        if col_min == col_max:
            self.is_leaf = True
            return

        self.threshold = float(self.rng.uniform(col_min, col_max))
        self.is_leaf = False

        left_mask = col < self.threshold
        right_mask = ~left_mask

        self.left = _ITree(self.max_depth, self.rng)
        self.right = _ITree(self.max_depth, self.rng)
        self.left.fit(X[left_mask], depth + 1)
        self.right.fit(X[right_mask], depth + 1)

    def path_lengths(self, X: NDArray, depth: int = 0) -> NDArray:
        if self.is_leaf:
            return np.full(len(X), depth + _expected_path_length(self.size))

        col = X[:, self.feature]
        left_mask = col < self.threshold
        right_mask = ~left_mask

        result = np.zeros(len(X))
        if self.left is not None and np.any(left_mask):
            result[left_mask] = self.left.path_lengths(X[left_mask], depth + 1)
        if self.right is not None and np.any(right_mask):
            result[right_mask] = self.right.path_lengths(X[right_mask], depth + 1)

        return result


def _expected_path_length(n: int) -> float:
    """Expected path length for unsuccessful search in BST (harmonic number)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n


class LocalOutlierFactor:
    """Mercury-native Local Outlier Factor for anomaly detection."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "euclidean",
        p: int = 2,
        novelty: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.p = p
        self.novelty = novelty
        self.n_jobs = n_jobs
        self._X_train: NDArray | None = None
        self._lrd: NDArray | None = None
        self._k_distances: NDArray | None = None
        self._k_indices: NDArray | None = None
        self.offset_: float = 0.0
        self.negative_outlier_factor_: NDArray | None = None

    def fit(self, X: NDArray, y: Any = None) -> LocalOutlierFactor:
        X = np.asarray(X, dtype=np.float64)
        self._X_train = X
        n = len(X)
        k = min(self.n_neighbors, n - 1)

        # Compute pairwise distances
        if self.metric == "minkowski":
            dists = cdist(X, X, metric="minkowski", p=self.p)
        else:
            dists = cdist(X, X, metric=self.metric)

        # Set self-distance to infinity
        np.fill_diagonal(dists, np.inf)

        # k-nearest neighbors
        self._k_indices = np.argsort(dists, axis=1)[:, :k]
        self._k_distances = np.take_along_axis(dists, self._k_indices, axis=1)

        # Local reachability density
        self._lrd = self._compute_lrd(dists, self._k_indices, self._k_distances)

        # LOF scores for training data
        lof_scores = self._compute_lof(self._k_indices, self._lrd)
        self.negative_outlier_factor_ = -lof_scores

        # Offset for decision function
        self.offset_ = -float(np.percentile(lof_scores, 100 * (1 - self.contamination)))
        return self

    def _compute_lrd(
        self, dists: NDArray, k_indices: NDArray, k_distances: NDArray
    ) -> NDArray:
        """Compute local reachability density."""
        n = len(k_indices)
        k = k_indices.shape[1]
        lrd = np.zeros(n)

        for i in range(n):
            reach_dists = np.maximum(
                k_distances[k_indices[i], -1],  # k-distance of neighbors
                dists[i, k_indices[i]],
            )
            mean_reach = np.mean(reach_dists)
            lrd[i] = 1.0 / max(mean_reach, 1e-10)

        return lrd

    def _compute_lof(self, k_indices: NDArray, lrd: NDArray) -> NDArray:
        """Compute LOF scores."""
        n = len(k_indices)
        lof = np.zeros(n)
        for i in range(n):
            neighbor_lrd = lrd[k_indices[i]]
            lof[i] = np.mean(neighbor_lrd) / max(lrd[i], 1e-10)
        return lof

    def decision_function(self, X: NDArray) -> NDArray:
        """Compute decision function for new samples (shift by offset)."""
        X = np.asarray(X, dtype=np.float64)
        assert self._X_train is not None and self._lrd is not None
        lof_scores = self._score_samples(X)
        return -lof_scores - self.offset_

    def predict(self, X: NDArray) -> NDArray:
        """Predict: -1 for anomalies, 1 for inliers."""
        scores = self.decision_function(X)
        return np.where(scores < 0, -1, 1)

    def score_samples(self, X: NDArray) -> NDArray:
        """Return opposite of LOF scores (higher = more normal)."""
        return -self._score_samples(np.asarray(X, dtype=np.float64))

    def _score_samples(self, X: NDArray) -> NDArray:
        """Compute raw LOF scores for new data."""
        assert self._X_train is not None and self._lrd is not None and self._k_distances is not None
        k = self._k_distances.shape[1]

        if self.metric == "minkowski":
            dists = cdist(X, self._X_train, metric="minkowski", p=self.p)
        else:
            dists = cdist(X, self._X_train, metric=self.metric)

        k_idx = np.argsort(dists, axis=1)[:, :k]
        k_dists = np.take_along_axis(dists, k_idx, axis=1)

        lof = np.zeros(len(X))
        for i in range(len(X)):
            reach_dists = np.maximum(
                self._k_distances[k_idx[i], -1],
                dists[i, k_idx[i]],
            )
            lrd_new = 1.0 / max(np.mean(reach_dists), 1e-10)
            neighbor_lrd = self._lrd[k_idx[i]]
            lof[i] = np.mean(neighbor_lrd) / max(lrd_new, 1e-10)

        return lof


class EllipticEnvelope:
    """Mercury-native EllipticEnvelope for outlier detection via Mahalanobis distance."""

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = None,
        support_fraction: float | None = None,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.support_fraction = support_fraction
        self._mean: NDArray | None = None
        self._cov_inv: NDArray | None = None
        self._threshold: float = 0.0

    def fit(self, X: NDArray, y: Any = None) -> EllipticEnvelope:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if self.support_fraction is not None:
            n_support = max(d + 1, int(n * self.support_fraction))
            self._mean, self._cov_inv = self._robust_estimate(X, n_support)
        else:
            self._mean = X.mean(axis=0)
            cov = np.cov(X, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            cov += np.eye(d) * 1e-6  # regularize
            self._cov_inv = np.linalg.inv(cov)

        distances = self._mahalanobis(X)
        self._threshold = float(np.percentile(distances, 100 * (1 - self.contamination)))
        return self

    def _robust_estimate(
        self, X: NDArray, n_support: int
    ) -> tuple[NDArray, NDArray]:
        """Simple robust location/scatter estimation using C-step."""
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape
        n_support = min(n_support, n)

        # Initialize with random subset
        idx = rng.choice(n, n_support, replace=False)
        subset = X[idx]
        mean = subset.mean(axis=0)
        cov = np.cov(subset, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(d) * 1e-6

        # C-step iterations
        for _ in range(30):
            cov_inv = np.linalg.inv(cov)
            diff = X - mean
            dists = np.sum(diff @ cov_inv * diff, axis=1)
            idx = np.argsort(dists)[:n_support]
            subset = X[idx]
            new_mean = subset.mean(axis=0)
            new_cov = np.cov(subset, rowvar=False)
            if new_cov.ndim == 0:
                new_cov = np.array([[new_cov]])
            new_cov += np.eye(d) * 1e-6

            if np.linalg.norm(new_mean - mean) < 1e-8:
                break
            mean, cov = new_mean, new_cov

        return mean, np.linalg.inv(cov)

    def _mahalanobis(self, X: NDArray) -> NDArray:
        assert self._mean is not None and self._cov_inv is not None
        diff = X - self._mean
        return np.sqrt(np.sum(diff @ self._cov_inv * diff, axis=1))

    def decision_function(self, X: NDArray) -> NDArray:
        """Negative Mahalanobis distance (higher = more normal)."""
        return -self._mahalanobis(np.asarray(X, dtype=np.float64))

    def predict(self, X: NDArray) -> NDArray:
        """Predict: -1 for anomalies, 1 for inliers."""
        distances = self._mahalanobis(np.asarray(X, dtype=np.float64))
        return np.where(distances > self._threshold, -1, 1)

    def mahalanobis(self, X: NDArray) -> NDArray:
        """Compute Mahalanobis distances."""
        return self._mahalanobis(np.asarray(X, dtype=np.float64))


class MinCovDet:
    """Mercury-native Minimum Covariance Determinant estimator."""

    def __init__(
        self,
        support_fraction: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.location_: NDArray | None = None
        self.covariance_: NDArray | None = None
        self._cov_inv: NDArray | None = None

    def fit(self, X: NDArray) -> MinCovDet:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        sf = self.support_fraction or max((n + d + 1) / (2 * n), 0.5)
        n_support = max(d + 1, int(n * sf))

        ee = EllipticEnvelope(support_fraction=sf, random_state=self.random_state)
        ee.fit(X)
        self.location_ = ee._mean
        assert ee._cov_inv is not None
        self.covariance_ = np.linalg.inv(ee._cov_inv)
        self._cov_inv = ee._cov_inv
        return self

    def mahalanobis(self, X: NDArray) -> NDArray:
        assert self.location_ is not None and self._cov_inv is not None
        diff = np.asarray(X, dtype=np.float64) - self.location_
        return np.sum(diff @ self._cov_inv * diff, axis=1)


class OneClassSVM:
    """Mercury-native One-Class SVM using RBF kernel approximation.

    Uses kernel approximation + linear scoring for efficiency.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: str | float = "scale",
        random_state: int | None = None,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.random_state = random_state
        self._center: NDArray | None = None
        self._scale: float = 1.0
        self._threshold: float = 0.0
        self._X_train: NDArray | None = None

    def fit(self, X: NDArray, y: Any = None) -> OneClassSVM:
        X = np.asarray(X, dtype=np.float64)
        self._X_train = X
        self._center = X.mean(axis=0)
        if self.gamma == "scale":
            variance = X.var()
            self._scale = 1.0 / (X.shape[1] * max(variance, 1e-10))
        else:
            self._scale = float(self.gamma)

        scores = self._compute_scores(X)
        self._threshold = float(np.percentile(scores, 100 * self.nu))
        return self

    def _compute_scores(self, X: NDArray) -> NDArray:
        """Compute distance-based anomaly scores using RBF kernel."""
        assert self._X_train is not None
        # Average RBF kernel similarity to training data
        dists_sq = cdist(X, self._X_train, metric="sqeuclidean")
        K = np.exp(-self._scale * dists_sq)
        return -np.mean(K, axis=1)  # Negative = more anomalous

    def decision_function(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        return -(self._compute_scores(X) - self._threshold)

    def predict(self, X: NDArray) -> NDArray:
        scores = self.decision_function(X)
        return np.where(scores < 0, -1, 1)


# =====================================================================
# Linear Models
# =====================================================================


class LogisticRegression:
    """Mercury-native Logistic Regression using L-BFGS (via scipy)."""

    def __init__(
        self,
        *,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state
        self.coef_: NDArray | None = None
        self.intercept_: NDArray | None = None
        self.classes_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> LogisticRegression:
        from scipy.optimize import minimize

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        # Binary classification
        y_bin = (y == self.classes_[-1]).astype(np.float64)
        n_features = X.shape[1]

        # w = [weights..., bias]
        w0 = np.zeros(n_features + 1)

        def objective(w: NDArray) -> float:
            weights, bias = w[:-1], w[-1]
            z = X @ weights + bias
            z = np.clip(z, -500, 500)
            log_likelihood = np.sum(y_bin * z - np.log1p(np.exp(z)))
            reg = 0.5 / self.C * np.sum(weights ** 2)
            return -(log_likelihood - reg)

        def gradient(w: NDArray) -> NDArray:
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
        return self

    def predict(self, X: NDArray) -> NDArray:
        prob = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[(prob[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: NDArray) -> NDArray:
        assert self.coef_ is not None and self.intercept_ is not None
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.coef_.T + self.intercept_
        z = np.clip(z.ravel(), -500, 500)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


class SGDClassifier:
    """Mercury-native SGD classifier for online learning."""

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
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._w: NDArray | None = None
        self._b: float = 0.0
        self._classes: NDArray | None = None
        self._t: int = 0

    def partial_fit(
        self, X: NDArray, y: NDArray, classes: NDArray | None = None
    ) -> SGDClassifier:
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

    def predict(self, X: NDArray) -> NDArray:
        assert self._w is not None and self._classes is not None
        X = np.asarray(X, dtype=np.float64)
        scores = X @ self._w + self._b
        return self._classes[(scores >= 0).astype(int)]

    def predict_proba(self, X: NDArray) -> NDArray:
        assert self._w is not None
        X = np.asarray(X, dtype=np.float64)
        z = X @ self._w + self._b
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return np.column_stack([1 - p1, p1])

    def decision_function(self, X: NDArray) -> NDArray:
        assert self._w is not None
        return np.asarray(X, dtype=np.float64) @ self._w + self._b


class PassiveAggressiveClassifier:
    """Mercury-native Passive-Aggressive classifier for online learning."""

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        warm_start: bool = True,
        max_iter: int = 1,
        tol: float | None = None,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._w: NDArray | None = None
        self._b: float = 0.0
        self._classes: NDArray | None = None

    def partial_fit(
        self, X: NDArray, y: NDArray, classes: NDArray | None = None
    ) -> PassiveAggressiveClassifier:
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

    def predict(self, X: NDArray) -> NDArray:
        assert self._w is not None and self._classes is not None
        scores = np.asarray(X, dtype=np.float64) @ self._w + self._b
        return self._classes[(scores >= 0).astype(int)]

    def decision_function(self, X: NDArray) -> NDArray:
        assert self._w is not None
        return np.asarray(X, dtype=np.float64) @ self._w + self._b


# =====================================================================
# Ensemble Classifiers
# =====================================================================


class GradientBoostingClassifier:
    """Mercury-native Gradient Boosting using decision stumps.

    Simplified but functional GBM for binary classification.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._stumps: list[_DecisionStump] = []
        self._init_pred: float = 0.0
        self.classes_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> GradientBoostingClassifier:
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

    def predict(self, X: NDArray) -> NDArray:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self._init_pred)
        for stump in self._stumps:
            F += self.learning_rate * stump.predict(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        return np.column_stack([1 - p1, p1])


class RandomForestClassifier:
    """Mercury-native Random Forest for classification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth or 10
        self.random_state = random_state
        self._trees: list[_DecisionStump] = []
        self.classes_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> RandomForestClassifier:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        n = len(X)

        self._trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sample
            idx = rng.choice(n, n, replace=True)
            tree = _DecisionStump(max_depth=self.max_depth, rng=rng)
            tree.fit(X[idx], y[idx])
            self._trees.append(tree)
        return self

    def predict(self, X: NDArray) -> NDArray:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=np.float64)
        assert self.classes_ is not None
        n_classes = len(self.classes_)
        all_preds = np.zeros((len(X), n_classes))
        for tree in self._trees:
            preds = tree.predict(X)
            for i, c in enumerate(self.classes_):
                all_preds[:, i] += (preds == c).astype(float)
        return all_preds / len(self._trees)


class SVC:
    """Mercury-native SVC using kernel-based scoring.

    Lightweight implementation suitable for moderate-scale problems.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        probability: bool = True,
        random_state: int | None = None,
        C: float = 1.0,
    ) -> None:
        self.kernel = kernel
        self.probability = probability
        self.random_state = random_state
        self.C = C
        self._X_train: NDArray | None = None
        self._y_train: NDArray | None = None
        self._alpha: NDArray | None = None
        self._gamma: float = 1.0
        self.classes_: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> SVC:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        self._gamma = 1.0 / (X.shape[1] * max(X.var(), 1e-10))

        # Simplified: store training data and use kernel-weighted voting
        return self

    def predict(self, X: NDArray) -> NDArray:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
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

    def __init__(
        self, max_depth: int = 3, rng: np.random.RandomState | None = None
    ) -> None:
        self.max_depth = max_depth
        self.rng = rng or np.random.RandomState()
        self.feature: int = 0
        self.threshold: float = 0.0
        self.value: float = 0.0
        self.left: _DecisionStump | None = None
        self.right: _DecisionStump | None = None
        self.is_leaf: bool = True

    def fit(self, X: NDArray, y: NDArray, depth: int = 0) -> None:
        self.value = float(np.mean(y))
        if depth >= self.max_depth or len(X) <= 2 or np.std(y) < 1e-10:
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
                var_reduction = (
                    np.var(y)
                    - (np.sum(left_mask) * np.var(y[left_mask])
                       + np.sum(right_mask) * np.var(y[right_mask]))
                    / len(y)
                )
                if var_reduction > best_gain:
                    best_gain = var_reduction
                    self.feature = int(f)
                    self.threshold = t

        if best_gain <= 0:
            self.is_leaf = True
            return

        self.is_leaf = False
        left_mask = X[:, self.feature] < self.threshold
        self.left = _DecisionStump(self.max_depth, self.rng)
        self.right = _DecisionStump(self.max_depth, self.rng)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)
        self.right.fit(X[~left_mask], y[~left_mask], depth + 1)

    def predict(self, X: NDArray) -> NDArray:
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
    """Mercury-native Gaussian Mixture Model using EM algorithm."""

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_: NDArray | None = None
        self.covariances_: NDArray | None = None
        self.weights_: NDArray | None = None

    def fit(self, X: NDArray) -> GaussianMixture:
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
            # E-step
            resp = self._e_step(X)

            # M-step
            nk = resp.sum(axis=0)
            self.weights_ = nk / n
            for k in range(self.n_components):
                if nk[k] < 1e-10:
                    continue
                self.means_[k] = (resp[:, k : k + 1].T @ X / nk[k]).ravel()
                diff = X - self.means_[k]
                self.covariances_[k] = (
                    (resp[:, k : k + 1] * diff).T @ diff / nk[k]
                    + np.eye(d) * 1e-6
                )

            # Check convergence
            ll = np.sum(np.log(np.sum(resp, axis=1) + 1e-300))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def _e_step(self, X: NDArray) -> NDArray:
        n = len(X)
        resp = np.zeros((n, self.n_components))
        assert self.means_ is not None and self.covariances_ is not None
        for k in range(self.n_components):
            try:
                resp[:, k] = self.weights_[k] * sp_stats.multivariate_normal.pdf(
                    X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
                )
            except (np.linalg.LinAlgError, ValueError):
                resp[:, k] = 1e-300
        resp /= resp.sum(axis=1, keepdims=True) + 1e-300
        return resp


# =====================================================================
# Feature Selection
# =====================================================================


def mutual_info_classif(
    X: NDArray, y: NDArray, *, random_state: int | None = None
) -> NDArray:
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
        bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) <= 1:
            mi[f] = 0.0
            continue
        x_binned = np.digitize(col, bins[1:-1])

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


class IsotonicRegression:
    """Mercury-native isotonic regression using the pool adjacent violators algorithm."""

    def __init__(
        self,
        y_min: float = 0.0,
        y_max: float = 1.0,
        out_of_bounds: str = "clip",
    ) -> None:
        self.y_min = y_min
        self.y_max = y_max
        self.out_of_bounds = out_of_bounds
        self._x: NDArray | None = None
        self._y: NDArray | None = None

    def fit(self, X: NDArray, y: NDArray) -> IsotonicRegression:
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        # Sort by X
        order = np.argsort(X)
        X_sorted = X[order]
        y_sorted = y[order].copy()

        # Pool Adjacent Violators
        n = len(y_sorted)
        blocks = list(range(n))
        values = y_sorted.copy()
        weights = np.ones(n)

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
        result = np.zeros(n)
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

    def predict(self, X: NDArray) -> NDArray:
        assert self._x is not None and self._y is not None
        X = np.asarray(X, dtype=np.float64).ravel()
        result = np.interp(X, self._x, self._y)
        if self.out_of_bounds == "clip":
            result = np.clip(result, self.y_min, self.y_max)
        return result

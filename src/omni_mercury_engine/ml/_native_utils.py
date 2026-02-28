"""
Mercury Agent - Native ML Utilities (sklearn-free)
Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0)

Drop-in replacements for sklearn utilities used across Mercury.
All implementations use only numpy and scipy.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def native_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUC-ROC (sklearn-compatible threshold indexing)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        raise ValueError("Only one class present in y_true.")
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_s = y_true[desc_idx]
    y_score_s = y_score[desc_idx]
    # Use last position of each tied-score group (sklearn convention)
    distinct_value_indices = np.where(np.diff(y_score_s))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true_s) - 1]
    tps = np.cumsum(y_true_s)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    fpr = fps / fps[-1] if fps[-1] > 0 else fps
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(tpr, fpr))


def native_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under precision-recall curve, sklearn-compatible)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_true_s = y_true[desc_idx]
    y_score_s = y_score[desc_idx]
    # Use last position of each tied-score group
    distinct_value_indices = np.where(np.diff(y_score_s))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true_s) - 1]
    tps = np.cumsum(y_true_s)[threshold_idxs]
    total_pos = tps[-1] if tps[-1] > 0 else 1.0
    precision = tps / (threshold_idxs + 1)
    recall = tps / total_pos
    recall_change = np.diff(np.concatenate([[0], recall]))
    return float(np.sum(precision * recall_change))


def native_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def native_precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _binary_precision, zero_division)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    return tp / (tp + fp) if (tp + fp) > 0 else zero_division


def native_recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if average == "weighted":
        return _weighted_metric(y_true, y_pred, _binary_recall, zero_division)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    return tp / (tp + fn) if (tp + fn) > 0 else zero_division


def native_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    p = native_precision_score(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=zero_division
    )
    r = native_recall_score(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=zero_division
    )
    return 2 * p * r / (p + r) if (p + r) > 0 else zero_division


def _binary_precision(y_true: np.ndarray, y_pred: np.ndarray, cls: int, zd: float) -> float:
    tp = int(np.sum((y_pred == cls) & (y_true == cls)))
    fp = int(np.sum((y_pred == cls) & (y_true != cls)))
    return tp / (tp + fp) if (tp + fp) > 0 else zd


def _binary_recall(y_true: np.ndarray, y_pred: np.ndarray, cls: int, zd: float) -> float:
    tp = int(np.sum((y_pred == cls) & (y_true == cls)))
    fn = int(np.sum((y_pred != cls) & (y_true == cls)))
    return tp / (tp + fn) if (tp + fn) > 0 else zd


def _weighted_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn: Any,
    zd: float,
) -> float:
    classes, counts = np.unique(y_true, return_counts=True)
    total = counts.sum()
    val = 0.0
    for cls, cnt in zip(classes, counts):
        val += fn(y_true, y_pred, cls, zd) * cnt / total
    return val


def native_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def native_brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_prob) ** 2))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class NativeStandardScaler:
    """Z-score normalization with fit / transform / fit_transform."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NativeStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Must call fit first"
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class NativeLabelEncoder:
    """Integer-encode arbitrary labels."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._mapping: dict[Any, int] = {}

    def fit(self, y: np.ndarray) -> "NativeLabelEncoder":
        self.classes_ = np.unique(y)
        self._mapping = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.array([self._mapping[v] for v in np.asarray(y).ravel()], dtype=np.int64)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        assert self.classes_ is not None, "Must call fit first"
        y = np.asarray(y).ravel()
        return self.classes_[y]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class NativeKFold:
    """Basic k-fold cross-validation split."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

    def split(self, X: np.ndarray, y: np.ndarray | None = None):
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            test = indices[current : current + fold_size]
            train = np.concatenate([indices[:current], indices[current + fold_size :]])
            yield train, test
            current += fold_size


class NativeStratifiedKFold:
    """Stratified k-fold preserving class proportions."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_state)

    def split(self, X: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        if self.shuffle:
            for c in classes:
                self.rng.shuffle(class_indices[c])
        folds: list[list[int]] = [[] for _ in range(self.n_splits)]
        for c in classes:
            idx = class_indices[c]
            for i, ix in enumerate(idx):
                folds[i % self.n_splits].append(ix)
        for fold_idx in range(self.n_splits):
            test = np.array(folds[fold_idx])
            train = np.concatenate([np.array(folds[j]) for j in range(self.n_splits) if j != fold_idx])
            yield train, test

    def get_n_splits(self) -> int:
        return self.n_splits


def native_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split, optionally stratified."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    n_test = int(n * test_size)

    if stratify is not None:
        train_idx: list[int] = []
        test_idx: list[int] = []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify == cls)[0]
            rng.shuffle(cls_idx)
            n_cls_test = max(1, int(len(cls_idx) * test_size))
            test_idx.extend(cls_idx[:n_cls_test])
            train_idx.extend(cls_idx[n_cls_test:])
        train_idx_arr = np.array(train_idx)
        test_idx_arr = np.array(test_idx)
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        test_idx_arr = indices[:n_test]
        train_idx_arr = indices[n_test:]

    return X[train_idx_arr], X[test_idx_arr], y[train_idx_arr], y[test_idx_arr]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NativeLogisticRegression:
    """L2-regularized logistic regression via gradient descent."""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        learning_rate: float = 0.01,
        random_state: int = 42,
        tol: float = 1e-6,
        solver: str = "lbfgs",
        **kwargs: Any,
    ):
        self.max_iter = max_iter
        self.C = C
        self.lr = learning_rate
        self.tol = tol
        self.solver = solver  # accepted for sklearn API compat
        self.rng = np.random.RandomState(random_state)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.classes_: np.ndarray | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeLogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.classes_ = np.unique(y)
        n, d = X.shape
        w = self.rng.randn(d) * 0.01
        b = 0.0
        reg = 1.0 / (self.C + 1e-12)

        for _ in range(self.max_iter):
            z = X @ w + b
            pred = self._sigmoid(z)
            error = pred - y
            grad_w = (X.T @ error) / n + reg * w
            grad_b = np.mean(error)
            w -= self.lr * grad_w
            b -= self.lr * grad_b
            if np.linalg.norm(grad_w) < self.tol:
                break
        # Store in sklearn-compatible shapes: coef_ is (1, d), intercept_ is (1,)
        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def _w(self) -> np.ndarray:
        assert self.coef_ is not None
        return self.coef_.ravel()

    def _b(self) -> float:
        return float(self.intercept_[0]) if isinstance(self.intercept_, np.ndarray) else float(self.intercept_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        z = X @ self._w() + self._b()
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self._w() + self._b()


class NativeSGDClassifier:
    """Mini-batch SGD classifier with partial_fit support."""

    def __init__(
        self,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        learning_rate: str = "constant",
        eta0: float = 0.01,
        warm_start: bool = True,
        max_iter: int = 1,
        tol: float | None = None,
        random_state: int = 42,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.eta0 = eta0
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._fitted = False

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> "NativeSGDClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.coef_ is None:
            self.coef_ = self.rng.randn(X.shape[1]) * 0.01
            self.intercept_ = 0.0

        for _ in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            if self.loss in ("log_loss", "log"):
                pred = self._sigmoid(z)
                error = pred - y
            else:  # hinge
                margin = y * z
                error = np.where(margin < 1, -y, 0.0)

            grad_w = (X.T @ error) / len(X)
            if self.penalty == "l2":
                grad_w += self.alpha * self.coef_
            elif self.penalty == "l1":
                grad_w += self.alpha * np.sign(self.coef_)

            self.coef_ -= self.eta0 * grad_w
            self.intercept_ -= self.eta0 * np.mean(error)

        self._fitted = True
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeSGDClassifier":
        if not self.warm_start:
            self.coef_ = None
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.coef_ is not None
        z = X @ self.coef_ + self.intercept_
        return (z >= 0).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.coef_ is not None
        z = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.coef_ is not None
        return X @ self.coef_ + self.intercept_


class NativePassiveAggressiveClassifier:
    """PA-I classifier (Crammer et al. 2006) with partial_fit."""

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        warm_start: bool = True,
        max_iter: int = 1,
        tol: float | None = None,
        random_state: int = 42,
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> "NativePassiveAggressiveClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        # Convert {0,1} to {-1,+1}
        y_signed = 2 * y - 1

        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0

        for _ in range(self.max_iter):
            for i in range(len(X)):
                x_i = X[i]
                margin = y_signed[i] * (x_i @ self.coef_ + self.intercept_)
                loss = max(0, 1 - margin)
                if loss > 0:
                    sq_norm = x_i @ x_i + (1.0 if self.fit_intercept else 0.0)
                    tau = min(self.C, loss / (sq_norm + 1e-12))
                    self.coef_ += tau * y_signed[i] * x_i
                    if self.fit_intercept:
                        self.intercept_ += tau * y_signed[i]

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativePassiveAggressiveClassifier":
        if not self.warm_start:
            self.coef_ = None
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.coef_ is not None
        z = X @ self.coef_ + self.intercept_
        return (z >= 0).astype(np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert self.coef_ is not None
        return X @ self.coef_ + self.intercept_


class NativeGradientBoostingClassifier:
    """Stump-based gradient boosting classifier (simplified)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = min(max_depth, 3)  # cap for simplicity
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(random_state)
        self.stumps: list[tuple[int, float, float, float]] = []
        self.init_pred: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeGradientBoostingClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        p = np.mean(y)
        self.init_pred = float(np.log((p + 1e-12) / (1 - p + 1e-12)))
        F = np.full(len(y), self.init_pred)
        self.stumps = []

        for _ in range(self.n_estimators):
            prob = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
            residuals = y - prob
            # Find best stump
            best_feature, best_threshold, best_left, best_right = self._find_best_stump(X, residuals)
            self.stumps.append((best_feature, best_threshold, best_left, best_right))
            mask = X[:, best_feature] <= best_threshold
            F[mask] += self.learning_rate * best_left
            F[~mask] += self.learning_rate * best_right

        return self

    def _find_best_stump(
        self, X: np.ndarray, residuals: np.ndarray
    ) -> tuple[int, float, float, float]:
        best_score = float("inf")
        best = (0, 0.0, 0.0, 0.0)
        n_features = X.shape[1]
        # Subsample features for speed
        n_try = min(n_features, max(5, int(np.sqrt(n_features))))
        features = self.rng.choice(n_features, n_try, replace=False)

        for feat in features:
            col = X[:, feat]
            thresholds = np.percentile(col, [10, 25, 50, 75, 90])
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue
                left_val = float(np.mean(residuals[left_mask]))
                right_val = float(np.mean(residuals[right_mask]))
                pred = np.where(left_mask, left_val, right_val)
                score = float(np.sum((residuals - pred) ** 2))
                if score < best_score:
                    best_score = score
                    best = (int(feat), float(thr), left_val, right_val)
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.init_pred)
        for feat, thr, left_val, right_val in self.stumps:
            mask = X[:, feat] <= thr
            F[mask] += self.learning_rate * left_val
            F[~mask] += self.learning_rate * right_val
        p1 = 1.0 / (1.0 + np.exp(-np.clip(F, -500, 500)))
        return np.column_stack([1 - p1, p1])


class NativePCA:
    """PCA via SVD decomposition."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NativePCA":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        explained_var = (S**2) / (len(X) - 1)
        total_var = explained_var.sum()
        self.explained_variance_ratio_ = explained_var[: self.n_components] / (total_var + 1e-12)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class NativeKMeans:
    """K-means clustering (Lloyd's algorithm)."""

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, random_state: int = 42, n_init: int = 3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.rng = np.random.RandomState(random_state)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NativeKMeans":
        X = np.asarray(X, dtype=np.float64)
        best_inertia = float("inf")
        for _ in range(self.n_init):
            centers = X[self.rng.choice(len(X), self.n_clusters, replace=False)]
            for _ in range(self.max_iter):
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([
                    X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                    for k in range(self.n_clusters)
                ])
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
            inertia = sum(np.sum((X[labels == k] - centers[k]) ** 2) for k in range(self.n_clusters))
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers.copy()
                self.labels_ = labels.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.cluster_centers_ is not None
        X = np.asarray(X, dtype=np.float64)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_  # type: ignore[return-value]


class NativeGaussianMixture:
    """Gaussian Mixture Model via EM algorithm."""

    def __init__(self, n_components: int = 2, max_iter: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.rng = np.random.RandomState(random_state)
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NativeGaussianMixture":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        # Initialize with random points
        idx = self.rng.choice(n, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([np.eye(d)] * self.n_components)
        self.weights_ = np.ones(self.n_components) / self.n_components

        for _ in range(self.max_iter):
            # E-step
            resp = self._compute_responsibilities(X)
            # M-step
            Nk = resp.sum(axis=0) + 1e-10
            self.weights_ = Nk / n
            for k in range(self.n_components):
                self.means_[k] = resp[:, k] @ X / Nk[k]
                diff = X - self.means_[k]
                self.covariances_[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
                self.covariances_[k] += 1e-6 * np.eye(d)  # regularize

        return self

    def _compute_responsibilities(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        log_resp = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = self._log_gaussian(X, self.means_[k], self.covariances_[k])
            log_resp[:, k] += np.log(self.weights_[k] + 1e-12)
        # Log-sum-exp normalization
        max_log = log_resp.max(axis=1, keepdims=True)
        log_resp -= max_log
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-12
        return resp

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        d = len(mean)
        diff = X - mean
        try:
            cov_inv = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
        except np.linalg.LinAlgError:
            cov_inv = np.eye(d)
            log_det = 0.0
        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)

    def predict(self, X: np.ndarray) -> np.ndarray:
        resp = self._compute_responsibilities(X)
        return np.argmax(resp, axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood per sample."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        log_probs = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            log_probs[:, k] = self._log_gaussian(X, self.means_[k], self.covariances_[k])
            log_probs[:, k] += np.log(self.weights_[k] + 1e-12)
        max_log = log_probs.max(axis=1, keepdims=True)
        return (max_log.ravel() + np.log(np.sum(np.exp(log_probs - max_log), axis=1)))


class NativeIsotonicRegression:
    """Isotonic regression via pool-adjacent-violators algorithm."""

    def __init__(self, out_of_bounds: str = "clip", y_min: float | None = None, y_max: float | None = None, **kwargs: Any):
        self.out_of_bounds = out_of_bounds
        self.y_min = y_min
        self.y_max = y_max
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeIsotonicRegression":
        X = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        order = np.argsort(X)
        X, y = X[order], y[order]
        # Pool adjacent violators
        blocks: list[list[int]] = [[i] for i in range(len(y))]
        values = list(y)
        i = 0
        while i < len(values) - 1:
            if values[i] > values[i + 1]:
                blocks[i] = blocks[i] + blocks[i + 1]
                values[i] = np.mean(y[blocks[i]])
                blocks.pop(i + 1)
                values.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        result_y = np.empty(len(y))
        for block, val in zip(blocks, values):
            for idx in block:
                result_y[idx] = val
        self._x = X
        self._y = result_y
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._x is not None and self._y is not None
        X = np.asarray(X, dtype=np.float64).ravel()
        result = np.interp(X, self._x, self._y)
        if self.out_of_bounds == "clip":
            lo = self._y[0] if self.y_min is None else self.y_min
            hi = self._y[-1] if self.y_max is None else self.y_max
            result = np.clip(result, lo, hi)
        if self.y_min is not None:
            result = np.maximum(result, self.y_min)
        if self.y_max is not None:
            result = np.minimum(result, self.y_max)
        return result


class NativeNearestNeighbors:
    """Brute-force nearest neighbors."""

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean", algorithm: str = "auto"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._data: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NativeNearestNeighbors":
        self._data = np.asarray(X, dtype=np.float64)
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        assert self._data is not None
        X = np.asarray(X, dtype=np.float64)
        k = n_neighbors or self.n_neighbors
        from scipy.spatial.distance import cdist

        dists = cdist(X, self._data, metric=self.metric)
        idx = np.argsort(dists, axis=1)[:, :k]
        distances = np.take_along_axis(dists, idx, axis=1)
        return distances, idx


def native_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve (reliability diagram data)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_true: list[float] = []
    bin_pred: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_true.append(float(np.mean(y_true[mask])))
            bin_pred.append(float(np.mean(y_prob[mask])))
    return np.array(bin_true), np.array(bin_pred)


def native_mutual_info_classif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Approximate mutual information via histogram binning."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n_features = X.shape[1]
    mi = np.zeros(n_features)
    classes = np.unique(y)
    n = len(y)
    n_bins = max(5, int(np.sqrt(n)))

    for feat_idx in range(n_features):
        col = X[:, feat_idx]
        bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
        digitized = np.digitize(col, bins) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)

        # Joint and marginal counts
        p_xy = np.zeros((n_bins, len(classes)))
        for i, cls in enumerate(classes):
            for b in range(n_bins):
                p_xy[b, i] = np.sum((digitized == b) & (y == cls))
        p_xy = p_xy / (n + 1e-12)
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        for b in range(n_bins):
            for i in range(len(classes)):
                if p_xy[b, i] > 1e-12 and p_x[b] > 1e-12 and p_y[i] > 1e-12:
                    mi[feat_idx] += p_xy[b, i] * np.log(p_xy[b, i] / (p_x[b] * p_y[i]))

    return np.maximum(mi, 0)


def native_cross_val_predict(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    method: str = "predict_proba",
) -> np.ndarray:
    """Cross-validated predictions."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    kf = NativeKFold(n_splits=cv, shuffle=True, random_state=42)
    out = np.zeros((len(X), 2) if method == "predict_proba" else len(X))

    for train_idx, test_idx in kf.split(X, y):
        cloned = native_clone(model)
        cloned.fit(X[train_idx], y[train_idx])
        pred_fn = getattr(cloned, method)
        out[test_idx] = pred_fn(X[test_idx])

    return out


def native_cross_val_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> np.ndarray:
    """Cross-validated scores."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    kf = NativeKFold(n_splits=cv, shuffle=True, random_state=42)
    scores: list[float] = []

    for train_idx, test_idx in kf.split(X, y):
        cloned = native_clone(model)
        cloned.fit(X[train_idx], y[train_idx])
        preds = cloned.predict(X[test_idx])
        if scoring == "accuracy":
            scores.append(native_accuracy_score(y[test_idx], preds))
        elif scoring == "f1":
            scores.append(native_f1_score(y[test_idx], preds))
        else:
            scores.append(native_accuracy_score(y[test_idx], preds))

    return np.array(scores)


def native_clone(estimator: Any) -> Any:
    """Clone an estimator by re-instantiating with same parameters."""
    try:
        if hasattr(estimator, "get_params"):
            params = estimator.get_params(deep=False)
            return estimator.__class__(**params)
    except Exception:
        pass
    return copy.deepcopy(estimator)

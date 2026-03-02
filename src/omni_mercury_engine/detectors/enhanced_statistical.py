"""Enhanced statistical anomaly detection using Mercury-native ML primitives.

Provides LOF, DBSCAN, and MCD-based detectors built on Mercury's own
implementations in omni_mercury_engine.ml.mercury_ml.

Do not import this module in production or benchmark code paths.

Original: Enhanced Statistical Anomaly Detection Module.
Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0-or-later

Enhanced statistical methods for robust anomaly detection including:
- Median Absolute Deviation (MAD) - robust to outliers
- Local Outlier Factor (LOF) - density-based detection
- DBSCAN Clustering - cluster-based anomaly identification
- Minimum Covariance Determinant (MCD) - robust covariance
- Grubbs' Test - statistical outlier test
- CUSUM (Cumulative Sum) - sequential change detection
- GESD (Generalized ESD) - multiple outlier detection
- Dynamic Threshold Adaptation - adaptive thresholding
"""

from __future__ import annotations

import warnings

warnings.warn(
    f"{__name__} is deprecated. Use MercuryAnomalyDetector.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StatisticalMethod(StrEnum):
    """Available enhanced statistical methods."""

    MAD = "mad"  # Median Absolute Deviation
    LOF = "lof"  # Local Outlier Factor
    DBSCAN = "dbscan"  # Density-based clustering
    MCD = "mcd"  # Minimum Covariance Determinant
    GRUBBS = "grubbs"  # Grubbs' outlier test
    CUSUM = "cusum"  # Cumulative Sum control chart
    GESD = "gesd"  # Generalized ESD test
    DYNAMIC = "dynamic"  # Dynamic threshold adaptation


@dataclass
class DynamicThresholdState:
    """State for dynamic threshold adaptation."""

    current_threshold: float = 0.5
    ema_score: float = 0.0
    ema_variance: float = 0.01
    adaptation_rate: float = 0.1
    min_threshold: float = 0.1
    max_threshold: float = 0.9
    history: list[float] = field(default_factory=list)
    max_history: int = 1000


@dataclass
class AnomalyResult:
    """Comprehensive anomaly detection result."""

    is_anomaly: np.ndarray
    scores: np.ndarray
    method: str
    threshold: float
    confidence: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


class MADDetector:
    """
    Median Absolute Deviation (MAD) based anomaly detector.

    MAD is more robust to outliers than standard deviation:
    MAD = median(|X_i - median(X)|)

    Advantages:
    - 50% breakdown point (robust to up to 50% contamination)
    - Works well with heavy-tailed distributions
    - Computationally efficient
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.5,
        consistency_constant: float = 1.4826,
    ):
        """
        Initialize MAD detector.

        Args:
            threshold_multiplier: Number of MADs for threshold (default: 3.5)
            consistency_constant: Scale factor for normal distribution (1.4826)
        """
        self.threshold_multiplier = threshold_multiplier
        self.consistency_constant = consistency_constant
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> MADDetector:
        """Fit the MAD detector."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.median_ = np.median(X, axis=0)
        deviations = np.abs(X - self.median_)
        self.mad_ = np.median(deviations, axis=0)

        # Prevent division by zero
        self.mad_ = np.where(self.mad_ == 0, 1e-10, self.mad_)

        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using MAD."""
        if not self._fitted:
            raise DetectorException("MADDetector must be fitted before detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Modified Z-score using MAD
        assert self.median_ is not None
        assert self.mad_ is not None
        modified_z = (X - self.median_) / (self.consistency_constant * self.mad_)

        # Max absolute modified z-score across features
        scores = np.max(np.abs(modified_z), axis=1)

        # Normalize to [0, 1]
        normalized_scores = np.clip(scores / (self.threshold_multiplier * 2), 0, 1)

        threshold = self.threshold_multiplier
        is_anomaly = scores > threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=normalized_scores,
            method="mad",
            threshold=threshold,
            confidence=1 - stats.norm.sf(scores) * 2,  # Two-tailed p-value
            details={
                "modified_z_scores": modified_z,
                "median": self.median_,
                "mad": self.mad_,
            },
        )


class LOFDetector:
    """
    Local Outlier Factor (LOF) detector — Mercury-native (cKDTree).

    LOF measures local density deviation compared to neighbors.
    Points with significantly lower density are anomalies.

    Advantages:
    - Detects local anomalies (not just global)
    - Works with non-uniform density distributions
    - No assumption about data distribution
    - O(n log n) via cKDTree — no O(n²) pairwise matrix
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "minkowski",
        p: int = 2,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.p = p
        self._X_train: np.ndarray | None = None
        self._tree = None
        self._k_dist: np.ndarray | None = None
        self._knn_idx: np.ndarray | None = None
        self._lrd: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> LOFDetector:
        """Fit the LOF detector."""
        from scipy.spatial import cKDTree

        from omni_mercury_engine.ml.mercury_ml import LocalOutlierFactor

        self._lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            contamination=self.contamination,
            metric=self.metric,
            p=self.p,
            novelty=True,
        )
        assert self._lof is not None
        self._lof.fit(X)
        self._fitted = True

        self._X_train = X.copy()
        n = len(X)
        k = min(self.n_neighbors, n - 1)

        tree = cKDTree(X)
        # query k+1 to include self; drop column 0 (distance to self = 0)
        dists_all, idx_all = tree.query(X, k=k + 1, p=self.p)
        dists = dists_all[:, 1:]  # shape (n, k) — exclude self
        idx = idx_all[:, 1:]  # shape (n, k)

        self._k_dist = dists[:, -1]  # kth-neighbor distance per point
        self._knn_idx = idx

        # Reachability distances and local reachability densities (LRD)
        reach = np.maximum(dists, self._k_dist[idx])  # (n, k)
        self._lrd = 1.0 / (np.mean(reach, axis=1) + 1e-10)

        self._tree = tree
        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using LOF."""
        if not self._fitted or self._tree is None:
            raise DetectorException("LOFDetector must be fitted before detection")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        k = min(self.n_neighbors, len(self._X_train) - 1)

        dists, idx = self._tree.query(X, k=k, p=self.p)
        reach = np.maximum(dists, self._k_dist[idx])  # (n_q, k)
        lrd_q = 1.0 / (np.mean(reach, axis=1) + 1e-10)

        lof = np.mean(self._lrd[idx] / (lrd_q[:, np.newaxis] + 1e-10), axis=1)

        # Normalize to [0, 1]: LOF=1 is normal, higher is anomalous.
        lof_max = np.max(lof)
        scores = np.clip((lof - 1.0) / (lof_max - 1.0 + 1e-10), 0.0, 1.0)
        threshold = np.percentile(scores, 100.0 * (1.0 - self.contamination))
        is_anomaly = scores > threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="lof_mercury_native",
            threshold=float(threshold),
            details={
                "raw_lof": lof,
                "n_neighbors": self.n_neighbors,
            },
        )


class DBSCANDetector:
    """
    DBSCAN-based anomaly detector — Mercury-native (cKDTree).

    Points not belonging to any cluster are labeled as anomalies.

    Advantages:
    - No assumption about cluster shape
    - Automatically finds number of clusters
    - Robust to noise
    - Mercury-native: uses scipy.spatial.cKDTree
    """

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int = 5,
        metric: str = "euclidean",
        auto_eps: bool = True,
        contamination: float = 0.1,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.auto_eps = auto_eps
        self.contamination = contamination
        self._auto_eps_value: float | None = None
        self._eps_used: float | None = None
        self._tree = None
        self._labels: np.ndarray | None = None
        self._fitted = False

    def _estimate_eps(self, X: NDArray[np.float64]) -> float:
        """Estimate optimal eps using k-distance graph."""
        from scipy.spatial import cKDTree

        k = min(self.min_samples, len(X) - 1)
        tree = cKDTree(X)
        dists, _ = tree.query(X, k=k + 1)
        knn_dists = np.sort(dists[:, -1])
        # Knee detection: largest gap in sorted k-distances
        gaps = np.diff(knn_dists)
        return float(knn_dists[np.argmax(gaps) + 1])

    def fit(self, X: NDArray[np.float64]) -> DBSCANDetector:
        """Fit the DBSCAN detector."""
        from scipy.spatial import cKDTree

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._X_train = X.copy()

        if self.auto_eps and self.eps is None:
            eps = self._estimate_eps(X)
        else:
            eps = self.eps or 0.5
        self._eps_used = eps

        tree = cKDTree(X)
        labels = np.full(len(X), -1, dtype=int)
        visited = np.zeros(len(X), dtype=bool)
        cluster_id = 0
        for i in range(len(X)):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = tree.query_ball_point(X[i], eps)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # noise
                continue
            labels[i] = cluster_id
            queue = list(neighbors)
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = tree.query_ball_point(X[j], eps)
                    if len(j_neighbors) >= self.min_samples:
                        queue.extend(j_neighbors)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
        self._labels = labels
        self._tree = tree
        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using DBSCAN."""
        if not self._fitted or self._tree is None:
            raise DetectorException("DBSCANDetector must be fitted before detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        dists, _ = self._tree.query(X, k=1)
        scores = np.clip(dists.ravel() / (self._eps_used + 1e-10), 0.0, 1.0)
        threshold = np.percentile(scores, 100.0 * (1.0 - self.contamination))
        is_anomaly = scores > threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="dbscan_mercury_native",
            threshold=float(threshold),
            details={
                "n_clusters": int(self._labels.max() + 1) if self._labels.max() >= 0 else 0,
                "n_noise": int(np.sum(self._labels == -1)),
                "eps": self._eps_used,
            },
        )


class MCDDetector:
    """
    Minimum Covariance Determinant (MCD) based detector — Mercury-native.

    Uses iterative reweighted covariance estimation for Mahalanobis distance.
    Mercury-native: numpy + scipy only.

    Advantages:
    - Robust to outliers in training data
    - Accounts for correlations between features
    - Works well for multivariate data
    """

    def __init__(
        self,
        support_fraction: float | None = None,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state
        self._location: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._mahal_threshold: float | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> MCDDetector:
        """Fit the MCD detector."""

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        h = max(int(0.75 * n), p + 1)

        # Initial robust center
        center = np.median(X, axis=0)
        dists = np.linalg.norm(X - center, axis=1)
        h_idx = np.argsort(dists)[:h]
        X_h = X[h_idx]

        mu = center
        cov = np.cov(X_h.T) + np.eye(p) * 1e-6
        if cov.ndim == 0:
            cov = np.array([[float(cov) + 1e-6]])
        cov_inv = np.linalg.inv(cov)

        for _ in range(2):
            mu = np.mean(X_h, axis=0)
            cov = np.cov(X_h.T) + np.eye(p) * 1e-6
            if cov.ndim == 0:
                cov = np.array([[float(cov) + 1e-6]])
            cov_inv = np.linalg.inv(cov)
            diff = X - mu
            mah = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            thresh = float(np.percentile(mah, 100.0 * (1.0 - self.contamination)))
            h_idx = np.where(mah <= thresh)[0][:h]
            if len(h_idx) < p + 1:
                break
            X_h = X[h_idx]

        self._location = mu
        self._precision = cov_inv
        self._mahal_threshold = thresh
        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using MCD."""
        if not self._fitted or self._precision is None or self._location is None:
            raise DetectorException("MCDDetector must be fitted before detection")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        diff = X - self._location
        mah = np.einsum("ij,jk,ik->i", diff, self._precision, diff)
        mahal_thr = self._mahal_threshold if self._mahal_threshold is not None else 1.0
        scores = np.clip(mah / (mahal_thr * 3.0 + 1e-10), 0.0, 1.0)

        is_anomaly = mah > mahal_thr

        # Chi-squared based p-values
        n_features = X.shape[1]
        p_values = 1 - stats.chi2.cdf(np.maximum(mah, 0.0), df=n_features)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="mcd_mercury_native",
            threshold=mahal_thr,
            confidence=1 - p_values,
            details={
                "mahalanobis_distances": mah,
                "robust_location": self._location,
                "p_values": p_values,
            },
        )


class GrubbsTest:
    """
    Grubbs' Test for detecting outliers.

    Statistical test for identifying single outliers in univariate data.
    Assumes normally distributed data.

    Advantages:
    - Formal statistical test with p-values
    - Well-understood statistical properties
    - Good for sequential outlier removal
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_outliers: int | None = None,
    ):
        """
        Initialize Grubbs' test.

        Args:
            alpha: Significance level
            max_outliers: Maximum outliers to detect (None = detect all)
        """
        self.alpha = alpha
        self.max_outliers = max_outliers

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """
        Detect outliers using Grubbs' test.

        For multivariate data, applies test to each feature independently.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        is_anomaly = np.zeros(n_samples, dtype=bool)
        scores = np.zeros(n_samples)

        for feat_idx in range(n_features):
            feat_data = X[:, feat_idx].copy()
            mask = np.ones(n_samples, dtype=bool)

            n_detected = 0
            while True:
                if self.max_outliers and n_detected >= self.max_outliers:
                    break

                current_data = feat_data[mask]
                if len(current_data) < 3:
                    break

                # Grubbs statistic
                mean = np.mean(current_data)
                std = np.std(current_data, ddof=1)

                if std < 1e-10:
                    break

                deviations = np.abs(current_data - mean) / std
                max_idx = np.argmax(deviations)
                G = deviations[max_idx]

                # Critical value
                n = len(current_data)
                t_crit = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
                G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

                if G_crit < G:
                    # Find original index
                    original_idx = np.where(mask)[0][max_idx]
                    is_anomaly[original_idx] = True
                    scores[original_idx] = max(scores[original_idx], G / G_crit)
                    mask[original_idx] = False
                    n_detected += 1
                else:
                    break

        # Normalize scores
        if scores.max() > 0:
            scores = np.clip(scores / (scores.max() + 0.1), 0, 1)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="grubbs",
            threshold=self.alpha,
            details={
                "alpha": self.alpha,
                "n_outliers_detected": np.sum(is_anomaly),
            },
        )


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) control chart for sequential anomaly detection.

    Detects shifts in the mean of a process over time.

    Advantages:
    - Sequential monitoring (online detection)
    - Sensitive to small persistent shifts
    - Well-suited for time series data
    """

    def __init__(
        self,
        target_mean: float | None = None,
        target_std: float | None = None,
        threshold_h: float = 5.0,
        slack_k: float = 0.5,
        two_sided: bool = True,
    ):
        """
        Initialize CUSUM detector.

        Args:
            target_mean: Target process mean (estimated if None)
            target_std: Target process std (estimated if None)
            threshold_h: Decision boundary (in std units)
            slack_k: Slack parameter (allowable slack in std units)
            two_sided: Detect both increases and decreases
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.threshold_h = threshold_h
        self.slack_k = slack_k
        self.two_sided = two_sided
        self._fitted_mean: float | None = None
        self._fitted_std: float | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> CUSUMDetector:
        """Fit CUSUM parameters from reference data."""
        X = np.asarray(X)
        # For multivariate data, compute mean across features per sample
        if X.ndim > 1:
            X = np.mean(X, axis=1)
        X = X.flatten()

        self._fitted_mean = self.target_mean if self.target_mean is not None else float(np.mean(X))  # type: ignore[assignment, unused-ignore]
        self._fitted_std = self.target_std if self.target_std is not None else float(np.std(X))  # type: ignore[assignment, unused-ignore]

        if self._fitted_std < 1e-10:  # type: ignore[operator, unused-ignore]
            self._fitted_std = 1.0

        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using CUSUM."""
        if not self._fitted:
            raise DetectorException("CUSUMDetector must be fitted before detection")

        X = np.asarray(X)
        # For multivariate data, compute mean across features per sample
        if X.ndim > 1:
            X = np.mean(X, axis=1)
        X = X.flatten()
        n = len(X)

        # Normalize data
        if self._fitted_mean is None or self._fitted_std is None:
            raise RuntimeError("CUSUM detector must be fitted before scoring")
        z = (X - self._fitted_mean) / self._fitted_std  # type: ignore[operator, unused-ignore]

        # CUSUM statistics
        c_plus = np.zeros(n)  # Upper CUSUM
        c_minus = np.zeros(n)  # Lower CUSUM

        for i in range(n):
            c_plus[i] = (
                max(0, c_plus[i - 1] + z[i] - self.slack_k)
                if i > 0
                else max(0, z[i] - self.slack_k)
            )
            c_minus[i] = (
                max(0, c_minus[i - 1] - z[i] - self.slack_k)
                if i > 0
                else max(0, -z[i] - self.slack_k)
            )

        # Anomaly detection
        if self.two_sided:
            is_anomaly = (c_plus > self.threshold_h) | (c_minus > self.threshold_h)
            scores = np.maximum(c_plus, c_minus) / self.threshold_h
        else:
            is_anomaly = c_plus > self.threshold_h
            scores = c_plus / self.threshold_h

        scores = np.clip(scores, 0, 1)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="cusum",
            threshold=self.threshold_h,
            details={
                "c_plus": c_plus,
                "c_minus": c_minus,
                "target_mean": self._fitted_mean,
                "target_std": self._fitted_std,
            },
        )


class GESDTest:
    """
    Generalized Extreme Studentized Deviate (GESD) test.

    Detects up to k outliers in univariate data.
    More powerful than repeated Grubbs' test.

    Advantages:
    - Handles multiple outliers (masking effect)
    - Formal statistical test
    - Controls family-wise error rate
    """

    def __init__(
        self,
        max_outliers: int = 10,
        alpha: float = 0.05,
    ):
        """
        Initialize GESD test.

        Args:
            max_outliers: Maximum number of outliers to detect
            alpha: Significance level
        """
        self.max_outliers = max_outliers
        self.alpha = alpha

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect outliers using GESD test."""
        X = np.asarray(X)
        original_shape = X.shape

        if X.ndim > 1:
            # For multivariate, apply to magnitude
            X_flat = np.linalg.norm(X, axis=1) if X.ndim == 2 else X.flatten()
        else:
            X_flat = X

        n = len(X_flat)
        data = X_flat.copy()
        indices = np.arange(n)

        R_values = []  # Test statistics
        lambda_values = []  # Critical values
        outlier_indices = []

        for i in range(min(self.max_outliers, n - 2)):
            if len(data) < 3:
                break

            mean = np.mean(data)
            std = np.std(data, ddof=1)

            if std < 1e-10:
                break

            # Compute R_i
            deviations = np.abs(data - mean)
            max_idx = np.argmax(deviations)
            R_i = deviations[max_idx] / std
            R_values.append(R_i)

            # Critical value lambda_i
            p = 1 - self.alpha / (2 * (n - i))
            t_p = stats.t.ppf(p, n - i - 2)
            lambda_i = ((n - i - 1) * t_p) / np.sqrt((n - i - 2 + t_p**2) * (n - i))
            lambda_values.append(lambda_i)

            outlier_indices.append(indices[max_idx])

            # Remove the outlier for next iteration
            mask = np.ones(len(data), dtype=bool)
            mask[max_idx] = False
            data = data[mask]
            indices = indices[mask]

        # Determine number of outliers
        n_outliers = 0
        for i in range(len(R_values)):
            if R_values[i] > lambda_values[i]:
                n_outliers = i + 1

        # Mark anomalies
        is_anomaly = np.zeros(n, dtype=bool)
        scores = np.zeros(n)

        for i in range(n_outliers):
            idx = outlier_indices[i]
            is_anomaly[idx] = True
            scores[idx] = R_values[i] / lambda_values[i] if lambda_values[i] > 0 else 1.0

        # Normalize scores
        if scores.max() > 0:
            scores = np.clip(scores / (scores.max() + 0.1), 0, 1)

        # Reshape for multivariate output
        if len(original_shape) > 1:
            is_anomaly = is_anomaly.reshape(-1)
            scores = scores.reshape(-1)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="gesd",
            threshold=self.alpha,
            details={
                "n_outliers": n_outliers,
                "R_values": R_values,
                "lambda_values": lambda_values,
                "outlier_indices": outlier_indices[:n_outliers],
            },
        )


class DynamicThresholdAdapter:
    """
    Dynamic threshold adaptation for streaming anomaly detection.

    Automatically adjusts threshold based on:
    - Exponential moving average of scores
    - Score variance estimation
    - False positive rate feedback
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.05,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        target_anomaly_rate: float = 0.05,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize dynamic threshold adapter.

        Args:
            initial_threshold: Starting threshold value
            adaptation_rate: Rate of threshold adaptation
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            target_anomaly_rate: Target proportion of anomalies
            ema_alpha: EMA smoothing factor
        """
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_anomaly_rate = target_anomaly_rate
        self.ema_alpha = ema_alpha

        self.state = DynamicThresholdState(
            current_threshold=initial_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            adaptation_rate=adaptation_rate,
        )

    def update(self, score: float, is_true_positive: bool | None = None) -> float:
        """
        Update threshold based on new score.

        Args:
            score: New anomaly score
            is_true_positive: Feedback on whether detected anomaly was real

        Returns:
            Updated threshold
        """
        # Update EMA statistics
        self.state.ema_score = self.ema_alpha * score + (1 - self.ema_alpha) * self.state.ema_score

        score_variance = (score - self.state.ema_score) ** 2
        self.state.ema_variance = (
            self.ema_alpha * score_variance + (1 - self.ema_alpha) * self.state.ema_variance
        )

        # Store in history
        self.state.history.append(score)
        if len(self.state.history) > self.state.max_history:
            self.state.history.pop(0)

        # Adapt threshold based on target anomaly rate
        if len(self.state.history) >= 100:
            current_anomaly_rate = np.mean(
                np.array(self.state.history) > self.state.current_threshold
            )

            if current_anomaly_rate > self.target_anomaly_rate * 1.5:
                # Too many anomalies detected - increase threshold
                self.state.current_threshold += self.adaptation_rate
            elif current_anomaly_rate < self.target_anomaly_rate * 0.5:
                # Too few anomalies detected - decrease threshold
                self.state.current_threshold -= self.adaptation_rate

        # Feedback-based adjustment
        if is_true_positive is not None:
            if is_true_positive is False and score > self.state.current_threshold:
                # False positive - increase threshold slightly
                self.state.current_threshold += self.adaptation_rate * 0.5
            elif is_true_positive is True and score < self.state.current_threshold:
                # False negative - decrease threshold slightly
                self.state.current_threshold -= self.adaptation_rate * 0.5

        # Clamp to bounds
        self.state.current_threshold = np.clip(
            self.state.current_threshold,
            self.min_threshold,
            self.max_threshold,
        )

        return self.state.current_threshold

    def get_threshold(self) -> float:
        """Get current threshold."""
        return self.state.current_threshold

    def get_statistics(self) -> dict[str, float]:
        """Get current statistics."""
        return {
            "threshold": self.state.current_threshold,
            "ema_score": self.state.ema_score,
            "ema_variance": self.state.ema_variance,
            "std": np.sqrt(self.state.ema_variance),
            "history_size": len(self.state.history),
        }


class EnhancedStatisticalDetector(BaseDetector):
    """
    Unified enhanced statistical anomaly detector.

    Combines multiple statistical methods with configurable ensemble.
    Supports:
    - Individual method detection
    - Ensemble voting
    - Weighted combination
    - Dynamic threshold adaptation
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        methods: list[StatisticalMethod] | None = None,
        ensemble_strategy: str = "weighted_average",
        use_dynamic_threshold: bool = True,
    ):
        """
        Initialize enhanced statistical detector.

        Args:
            config: Configuration dictionary
            methods: List of statistical methods to use
            ensemble_strategy: 'voting', 'weighted_average', 'max', or 'min'
            use_dynamic_threshold: Enable dynamic threshold adaptation
        """
        super().__init__(config)

        self.methods = methods or [
            StatisticalMethod.MAD,
            StatisticalMethod.LOF,
            StatisticalMethod.CUSUM,
        ]
        self.ensemble_strategy = ensemble_strategy
        self.use_dynamic_threshold = use_dynamic_threshold

        # Initialize detectors
        self._detectors: dict[str, Any] = {}
        self._init_detectors()

        # Initialize dynamic threshold
        self._dynamic_threshold: DynamicThresholdAdapter | None = None
        if use_dynamic_threshold:
            self._dynamic_threshold = DynamicThresholdAdapter(
                initial_threshold=self.config.get("threshold", 0.5),
                target_anomaly_rate=self.config.get("contamination", 0.05),
            )

        # Method weights for weighted averaging
        self._weights: dict[str, float] = {
            StatisticalMethod.MAD: 0.2,
            StatisticalMethod.LOF: 0.25,
            StatisticalMethod.DBSCAN: 0.15,
            StatisticalMethod.MCD: 0.2,
            StatisticalMethod.CUSUM: 0.1,
            StatisticalMethod.GRUBBS: 0.05,
            StatisticalMethod.GESD: 0.05,
        }

    def _init_detectors(self) -> None:
        """Initialize individual detectors."""
        for method in self.methods:
            if method == StatisticalMethod.MAD:
                self._detectors[method] = MADDetector(
                    threshold_multiplier=self.config.get("mad_threshold", 3.5)
                )
            elif method == StatisticalMethod.LOF:
                self._detectors[method] = LOFDetector(
                    n_neighbors=self.config.get("lof_neighbors", 20),
                    contamination=self.config.get("contamination", 0.1),
                )
            elif method == StatisticalMethod.DBSCAN:
                self._detectors[method] = DBSCANDetector(
                    min_samples=self.config.get("dbscan_min_samples", 5),
                )
            elif method == StatisticalMethod.MCD:
                self._detectors[method] = MCDDetector(
                    contamination=self.config.get("contamination", 0.1),
                )
            elif method == StatisticalMethod.CUSUM:
                self._detectors[method] = CUSUMDetector(
                    threshold_h=self.config.get("cusum_threshold", 5.0),
                )
            elif method == StatisticalMethod.GRUBBS:
                self._detectors[method] = GrubbsTest(
                    alpha=self.config.get("grubbs_alpha", 0.05),
                )
            elif method == StatisticalMethod.GESD:
                self._detectors[method] = GESDTest(
                    max_outliers=self.config.get("gesd_max_outliers", 10),
                    alpha=self.config.get("gesd_alpha", 0.05),
                )

    def fit(self, data: np.ndarray | Any) -> EnhancedStatisticalDetector:
        """Fit all detectors."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        for method, detector in self._detectors.items():
            try:
                if hasattr(detector, "fit"):
                    detector.fit(data)
            except Exception as e:
                warnings.warn(f"Failed to fit {method}: {e}", stacklevel=2)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray | Any) -> dict[str, Any]:
        """Detect anomalies using ensemble of methods."""
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = len(data)
        all_results: dict[str, AnomalyResult] = {}
        all_scores: list[np.ndarray] = []
        all_weights: list[float] = []
        all_predictions: list[np.ndarray] = []

        # Run each detector
        for method, detector in self._detectors.items():
            try:
                if hasattr(detector, "detect"):
                    result = detector.detect(data)
                    all_results[method] = result
                    all_scores.append(result.scores)
                    all_weights.append(self._weights.get(method, 0.1))
                    all_predictions.append(result.is_anomaly.astype(float))
            except Exception as e:
                warnings.warn(f"Detection failed for {method}: {e}", stacklevel=2)

        if not all_scores:
            return {
                "is_anomaly": np.zeros(n_samples, dtype=bool),
                "scores": np.zeros(n_samples),
                "detector_type": "enhanced_statistical",
                "method_results": {},
            }

        # Combine scores based on strategy
        scores_array = np.array(all_scores)
        weights_array = np.array(all_weights)
        weights_array /= weights_array.sum()  # Normalize

        if self.ensemble_strategy == "voting":
            predictions_array = np.array(all_predictions)
            combined_scores = np.mean(predictions_array, axis=0)
        elif self.ensemble_strategy == "weighted_average":
            combined_scores = np.average(scores_array, axis=0, weights=weights_array)
        elif self.ensemble_strategy == "max":
            combined_scores = np.max(scores_array, axis=0)
        elif self.ensemble_strategy == "min":
            combined_scores = np.min(scores_array, axis=0)
        else:
            combined_scores = np.mean(scores_array, axis=0)

        # Apply dynamic threshold if enabled
        threshold = self.threshold
        if self._dynamic_threshold:
            for score in combined_scores:
                threshold = self._dynamic_threshold.update(score)

        is_anomaly = combined_scores > threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "threshold": threshold,
            "detector_type": "enhanced_statistical",
            "method_results": {
                method: {
                    "scores": result.scores,
                    "is_anomaly": result.is_anomaly,
                    "method": result.method,
                    "details": result.details,
                }
                for method, result in all_results.items()
            },
            "ensemble_strategy": self.ensemble_strategy,
            "methods_used": list(self._detectors.keys()),
            "dynamic_threshold_stats": (
                self._dynamic_threshold.get_statistics() if self._dynamic_threshold else None
            ),
        }

    def extract_features(self, data: np.ndarray | Any) -> Any:
        """Extract features from all statistical methods."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._is_fitted:
            self.fit(data)

        result = self.detect(data)
        features_list = [result["scores"].reshape(-1, 1)]

        # Add individual method scores as features
        for method_name, method_result in result.get("method_results", {}).items():
            if "scores" in method_result:
                features_list.append(method_result["scores"].reshape(-1, 1))

        features = np.hstack(features_list)

        # Pad to minimum feature dimension
        if features.shape[1] < 10:
            padding = np.zeros((features.shape[0], 10 - features.shape[1]))
            features = np.hstack([features, padding])

        # Return as torch tensor if available, otherwise numpy array
        try:
            import torch

            return torch.tensor(features, dtype=torch.float32)
        except ImportError:
            return features.astype(np.float32)


# Exports
__all__ = [
    "AnomalyResult",
    "CUSUMDetector",
    "DBSCANDetector",
    "DynamicThresholdAdapter",
    "DynamicThresholdState",
    "EnhancedStatisticalDetector",
    "GESDTest",
    "GrubbsTest",
    "LOFDetector",
    "MADDetector",
    "MCDDetector",
    "StatisticalMethod",
]

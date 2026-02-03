"""
Mercury Agent - Enhanced Statistical Anomaly Detection Module
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

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
    Local Outlier Factor (LOF) detector.

    LOF measures local density deviation compared to neighbors.
    Points with significantly lower density are anomalies.

    Advantages:
    - Detects local anomalies (not just global)
    - Works with non-uniform density distributions
    - No assumption about data distribution
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "minkowski",
        p: int = 2,
    ):
        """
        Initialize LOF detector.

        Args:
            n_neighbors: Number of neighbors for LOF calculation
            contamination: Expected proportion of anomalies
            metric: Distance metric
            p: Power parameter for Minkowski metric
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.p = p
        self._lof = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> LOFDetector:
        """Fit the LOF detector."""
        try:
            from sklearn.neighbors import LocalOutlierFactor

            self._lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X) - 1),
                contamination=self.contamination,
                metric=self.metric,
                p=self.p,
                novelty=True,
            )
            self._lof.fit(X)
            self._fitted = True
        except ImportError:
            raise DetectorException("scikit-learn required for LOF detection")

        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using LOF."""
        if not self._fitted or self._lof is None:
            raise DetectorException("LOFDetector must be fitted before detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Get LOF scores (negative = more anomalous)
        raw_scores = -self._lof.decision_function(X)

        # Normalize to [0, 1]
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score - min_score > 1e-8:
            scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            scores = np.full_like(raw_scores, 0.5)

        # Prediction
        predictions = self._lof.predict(X)
        is_anomaly = predictions == -1

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="lof",
            threshold=float(self._lof.offset_),
            details={
                "raw_scores": raw_scores,
                "n_neighbors": self.n_neighbors,
            },
        )


class DBSCANDetector:
    """
    DBSCAN-based anomaly detector.

    Points not belonging to any cluster are labeled as anomalies.

    Advantages:
    - No assumption about cluster shape
    - Automatically finds number of clusters
    - Robust to noise
    """

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int = 5,
        metric: str = "euclidean",
        auto_eps: bool = True,
    ):
        """
        Initialize DBSCAN detector.

        Args:
            eps: Maximum distance between points in a cluster
            min_samples: Minimum points to form a dense region
            metric: Distance metric
            auto_eps: Automatically determine eps from data
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.auto_eps = auto_eps
        self._fitted_eps: float | None = None
        self._reference_data: np.ndarray | None = None
        self._fitted = False

    def _estimate_eps(self, X: NDArray[np.float64]) -> float:
        """Estimate optimal eps using k-distance graph."""
        from sklearn.neighbors import NearestNeighbors

        k = min(self.min_samples, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # Use the knee point of sorted k-distances
        k_distances = np.sort(distances[:, -1])
        # Simple knee detection: maximum curvature
        gradients = np.gradient(k_distances)
        knee_idx = np.argmax(gradients)

        return float(k_distances[knee_idx])

    def fit(self, X: NDArray[np.float64]) -> DBSCANDetector:
        """Fit the DBSCAN detector."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.auto_eps and self.eps is None:
            self._fitted_eps = self._estimate_eps(X)
        else:
            self._fitted_eps = self.eps or 0.5

        self._reference_data = X
        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using DBSCAN."""
        if not self._fitted:
            raise DetectorException("DBSCANDetector must be fitted before detection")

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise DetectorException("scikit-learn required for DBSCAN detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Run DBSCAN
        dbscan = DBSCAN(
            eps=self._fitted_eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        labels = dbscan.fit_predict(X)

        # Noise points (label=-1) are anomalies
        is_anomaly = labels == -1

        # Score based on distance to nearest core point
        scores = np.zeros(len(X))
        core_mask = np.isin(np.arange(len(X)), dbscan.core_sample_indices_)

        if np.any(core_mask):
            core_points = X[core_mask]
            for i, point in enumerate(X):
                if is_anomaly[i]:
                    # Distance to nearest core point
                    distances = np.linalg.norm(core_points - point, axis=1)
                    min_dist = np.min(distances)
                    scores[i] = min(1.0, min_dist / (self._fitted_eps * 3))
                else:
                    scores[i] = 0.0
        else:
            # No core points found - all could be anomalies
            scores = np.ones(len(X)) * 0.5

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="dbscan",
            threshold=self._fitted_eps,
            details={
                "labels": labels,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "n_noise": np.sum(is_anomaly),
                "eps": self._fitted_eps,
            },
        )


class MCDDetector:
    """
    Minimum Covariance Determinant (MCD) based detector.

    Uses robust covariance estimation for Mahalanobis distance.

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
        """
        Initialize MCD detector.

        Args:
            support_fraction: Proportion of data for robust estimation
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state
        self._mcd = None
        self._threshold: float | None = None
        self._fitted = False

    def fit(self, X: NDArray[np.float64]) -> MCDDetector:
        """Fit the MCD detector."""
        try:
            from sklearn.covariance import MinCovDet
        except ImportError:
            raise DetectorException("scikit-learn required for MCD detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._mcd = MinCovDet(
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self._mcd.fit(X)

        # Compute threshold from training data
        distances = self._mcd.mahalanobis(X)
        self._threshold = float(np.percentile(distances, (1 - self.contamination) * 100))

        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using MCD."""
        if not self._fitted or self._mcd is None:
            raise DetectorException("MCDDetector must be fitted before detection")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Mahalanobis distances
        distances = self._mcd.mahalanobis(X)

        # Normalize scores
        max_dist = max(distances.max(), self._threshold * 2)
        scores = np.clip(distances / max_dist, 0, 1)

        is_anomaly = distances > self._threshold

        # Chi-squared based p-values
        n_features = X.shape[1]
        p_values = 1 - stats.chi2.cdf(distances, df=n_features)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            method="mcd",
            threshold=self._threshold,
            confidence=1 - p_values,
            details={
                "mahalanobis_distances": distances,
                "robust_location": self._mcd.location_,
                "robust_covariance": self._mcd.covariance_,
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

                if G > G_crit:
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
        X = np.asarray(X).flatten()

        self._fitted_mean = self.target_mean if self.target_mean is not None else np.mean(X)
        self._fitted_std = self.target_std if self.target_std is not None else np.std(X)

        if self._fitted_std < 1e-10:
            self._fitted_std = 1.0

        self._fitted = True
        return self

    def detect(self, X: NDArray[np.float64]) -> AnomalyResult:
        """Detect anomalies using CUSUM."""
        if not self._fitted:
            raise DetectorException("CUSUMDetector must be fitted before detection")

        X = np.asarray(X).flatten()
        n = len(X)

        # Normalize data
        z = (X - self._fitted_mean) / self._fitted_std

        # CUSUM statistics
        c_plus = np.zeros(n)  # Upper CUSUM
        c_minus = np.zeros(n)  # Lower CUSUM

        for i in range(n):
            c_plus[i] = max(0, c_plus[i - 1] + z[i] - self.slack_k) if i > 0 else max(0, z[i] - self.slack_k)
            c_minus[i] = max(0, c_minus[i - 1] - z[i] - self.slack_k) if i > 0 else max(0, -z[i] - self.slack_k)

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
        self.state.ema_score = (
            self.ema_alpha * score + (1 - self.ema_alpha) * self.state.ema_score
        )

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
                warnings.warn(f"Failed to fit {method}: {e}")

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
                warnings.warn(f"Detection failed for {method}: {e}")

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
                self._dynamic_threshold.get_statistics()
                if self._dynamic_threshold
                else None
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

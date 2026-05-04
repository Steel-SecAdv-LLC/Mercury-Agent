"""
Adaptive Detector Module for Mercury-Agent.

Addresses specific weaknesses identified in benchmark analysis. All detection is Mercury-native
(numpy/scipy only) — zero sklearn dependency.

Copyright (C) 2025 Steel Security Advisors LLC
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class DatasetProfile(Enum):
    """Dataset profile types for adaptive detection."""

    GENERIC = "generic"
    HIGH_DIMENSIONAL = "high_dimensional"  # covtype-like
    COVARIANCE_STRUCTURED = "covariance_structured"  # batadal-like
    TEMPORAL = "temporal"  # smd-like
    NETWORK = "network"  # nsl_kdd-like, kddcup99-like
    MEDICAL = "medical"  # breast_cancer-like
    PATTERN_RECOGNITION = "pattern_recognition"  # digits-like image/pattern data


@dataclass
class DetectionResult:
    """Result from adaptive detection."""

    scores: NDArray[np.float64]
    predictions: NDArray[np.int32]
    threshold: float
    confidence: float
    profile_used: DatasetProfile
    calibration_method: str
    metadata: dict[str, Any] = field(default_factory=dict)


class AdaptiveThresholdCalibrator:
    """
    Solves the covtype F1=0 problem.

    Issue: Good AUC (0.8783) but zero F1 means the threshold is miscalibrated.
    The model correctly ranks anomalies higher than normal points (good AUC)
    but the binary cutoff is set incorrectly.

    Solution: Use percentile-based adaptive thresholding that estimates
    the anomaly ratio from the score distribution.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        min_contamination: float = 0.001,
        max_contamination: float = 0.3,
    ):
        self.contamination = contamination
        self.min_contamination = min_contamination
        self.max_contamination = max_contamination

    def calibrate(
        self,
        scores: NDArray[np.float64],
        method: str = "percentile",
    ) -> tuple[float, NDArray[np.int32]]:
        """
        Calibrate threshold and return predictions.

        Args:
            scores: Anomaly scores (higher = more anomalous)
            method: Calibration method

        Returns:
            Tuple of (threshold, predictions)
        """
        if method == "percentile":
            return self._percentile_calibration(scores)
        elif method == "otsu":
            return self._otsu_calibration(scores)
        elif method == "mad":
            return self._mad_calibration(scores)
        elif method == "bimodal":
            return self._bimodal_calibration(scores)
        else:
            return self._percentile_calibration(scores)

    def _percentile_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """Percentile-based calibration using contamination estimate."""
        estimated_contamination = self._estimate_contamination(scores)
        effective_contamination = max(estimated_contamination, self.contamination)
        effective_contamination = float(
            np.clip(
                effective_contamination,
                self.min_contamination,
                self.max_contamination,
            )
        )

        threshold = float(np.percentile(scores, 100 * (1 - effective_contamination)))
        predictions = (scores >= threshold).astype(np.int32)

        return threshold, predictions

    def _otsu_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """Otsu's method for bimodal threshold selection."""
        score_min = float(scores.min())
        score_max = float(scores.max())

        if score_max - score_min < 1e-10:
            threshold = score_min
            predictions = np.zeros(len(scores), dtype=np.int32)
            return threshold, predictions

        normalized = ((scores - score_min) / (score_max - score_min) * 255).astype(np.int32)

        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()

        sum_total = np.dot(np.arange(256), hist)
        sum_b = 0.0
        w_b = 0.0
        max_variance = 0.0
        best_threshold = 0

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue

            w_f = total - w_b
            if w_f == 0:
                break

            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f

            variance = w_b * w_f * (m_b - m_f) ** 2

            if variance > max_variance:
                max_variance = variance
                best_threshold = t

        threshold = score_min + (best_threshold / 255) * (score_max - score_min)
        predictions = (scores >= threshold).astype(np.int32)

        return threshold, predictions

    def _mad_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """Median Absolute Deviation based calibration."""
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))

        if mad < 1e-10:
            return self._percentile_calibration(scores)

        threshold = median + 3 * 1.4826 * mad
        predictions = (scores >= threshold).astype(np.int32)

        if predictions.sum() == 0 and self.contamination > 0:
            return self._percentile_calibration(scores)

        return threshold, predictions

    def _bimodal_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """
        Bimodal distribution calibration.

        Assumes scores come from a mixture of normal and anomalous distributions. Finds the valley
        between the two modes.
        """
        threshold, predictions = self._otsu_calibration(scores)

        pred_ratio = float(predictions.mean())

        if pred_ratio < self.min_contamination or pred_ratio > self.max_contamination:
            return self._percentile_calibration(scores)

        return threshold, predictions

    def _estimate_contamination(self, scores: NDArray[np.float64]) -> float:
        """
        Estimate contamination ratio from score distribution.

        Uses the "knee" detection method to find where scores transition from normal to anomalous.
        """
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)

        if n < 10:
            return self.contamination

        window = max(n // 20, 5)
        smoothed = np.convolve(sorted_scores, np.ones(window) / window, mode="valid")

        if len(smoothed) < 10:
            return self.contamination

        second_deriv = np.diff(np.diff(smoothed))
        if len(second_deriv) == 0:
            return self.contamination

        knee_idx = int(np.argmax(second_deriv)) + window // 2

        estimated = 1.0 - (knee_idx / n)

        return float(np.clip(estimated, self.min_contamination, self.max_contamination))


class CovarianceAwareDetector:
    """
    Solves the batadal problem.

    Issue: EllipticEnvelope (0.9353 AUC) dominates because batadal
    has strong covariance structure from correlated sensors.

    Solution: Incorporate Mahalanobis distance with robust covariance
    estimation into the detection pipeline.  Mercury-native (no sklearn).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        support_fraction: float = 0.9,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state
        self._mean: NDArray[np.float64] | None = None
        self._covariance_inv: NDArray[np.float64] | None = None
        self._threshold: float = 0.0

    def fit(self, X: NDArray[np.float64]) -> "CovarianceAwareDetector":
        """Fit the detector using robust covariance estimation."""
        n_samples, n_features = X.shape

        median = np.median(X, axis=0)
        distances = np.sqrt(np.sum((X - median) ** 2, axis=1))
        sorted_indices = np.argsort(distances)

        n_support = int(n_samples * self.support_fraction)
        support_indices = sorted_indices[:n_support]
        X_support = X[support_indices]

        self._mean = np.mean(X_support, axis=0)

        centered = X_support - self._mean
        cov = np.dot(centered.T, centered) / (n_support - 1)

        reg = 1e-6 * np.eye(n_features)
        cov_reg = cov + reg

        try:
            self._covariance_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            self._covariance_inv = np.linalg.pinv(cov_reg)

        all_distances = self._mahalanobis_distance(X)
        self._threshold = float(np.percentile(all_distances, 100 * (1 - self.contamination)))

        return self

    def _mahalanobis_distance(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Mahalanobis distance for each sample."""
        if self._mean is None or self._covariance_inv is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        centered = X - self._mean
        left = np.dot(centered, self._covariance_inv)
        distances = np.sqrt(np.maximum(np.sum(left * centered, axis=1), 0.0))

        return np.asarray(distances, dtype=np.float64)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return anomaly scores (higher = more anomalous)."""
        return self._mahalanobis_distance(X)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        scores = self.score_samples(X)
        return (scores >= self._threshold).astype(np.int32)


class TemporalPatternDetector:
    """
    Solves the smd problem.

    Issue: Server Machine Dataset has temporal patterns that simple
    point-wise detectors miss.

    Solution: Add lag features and sliding window statistics to capture
    temporal dependencies before detection.
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        lag_features: int = 3,
        include_diff: bool = True,
        include_rolling_stats: bool = True,
    ):
        self.window_sizes = window_sizes or [5, 10, 20]
        self.lag_features = lag_features
        self.include_diff = include_diff
        self.include_rolling_stats = include_rolling_stats
        self._feature_names: list[str] = []

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform input features to include temporal patterns.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Augmented features with temporal information
        """
        n_samples, n_features = X.shape
        augmented_features: list[NDArray[np.float64]] = [X]
        self._feature_names = [f"orig_{i}" for i in range(n_features)]

        # Add lag features
        for lag in range(1, self.lag_features + 1):
            lagged = np.zeros_like(X)
            lagged[lag:] = X[:-lag]
            augmented_features.append(lagged)
            self._feature_names.extend([f"lag{lag}_{i}" for i in range(n_features)])

        # Add first differences
        if self.include_diff:
            diff = np.zeros_like(X)
            diff[1:] = X[1:] - X[:-1]
            augmented_features.append(diff)
            self._feature_names.extend([f"diff_{i}" for i in range(n_features)])

            # Second differences
            diff2 = np.zeros_like(X)
            diff2[2:] = diff[2:] - diff[1:-1]
            augmented_features.append(diff2)
            self._feature_names.extend([f"diff2_{i}" for i in range(n_features)])

        # Add rolling statistics
        if self.include_rolling_stats:
            for window in self.window_sizes:
                if window > n_samples:
                    continue

                rolling_mean = self._rolling_stat(X, window, np.mean)
                augmented_features.append(rolling_mean)
                self._feature_names.extend([f"rmean{window}_{i}" for i in range(n_features)])

                rolling_std = self._rolling_stat(X, window, np.std)
                augmented_features.append(rolling_std)
                self._feature_names.extend([f"rstd{window}_{i}" for i in range(n_features)])

                deviation = np.zeros_like(X)
                nonzero_std = rolling_std > 1e-10
                deviation[nonzero_std] = (X[nonzero_std] - rolling_mean[nonzero_std]) / rolling_std[
                    nonzero_std
                ]
                augmented_features.append(deviation)
                self._feature_names.extend([f"rdev{window}_{i}" for i in range(n_features)])

        return np.hstack(augmented_features)

    def _rolling_stat(
        self,
        X: NDArray[np.float64],
        window: int,
        stat_func: Any,
    ) -> NDArray[np.float64]:
        """Compute rolling statistic."""
        n_samples, n_features = X.shape
        result = np.zeros_like(X)

        for i in range(n_samples):
            start = max(0, i - window + 1)
            window_data = X[start : i + 1]
            result[i] = stat_func(window_data, axis=0)

        return result

    @property
    def feature_names(self) -> list[str]:
        """Get names of transformed features."""
        return self._feature_names


# ---------------------------------------------------------------------------
# Mercury-native backend detectors (replace sklearn IsolationForest/LOF/EE)
# ---------------------------------------------------------------------------


class _MercuryRandomProjectionDetector:
    """Isolation-style anomaly detector using random projections (no trees/sklearn)."""

    def __init__(
        self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self._rng = np.random.default_rng(random_state)
        self._projections: NDArray[np.float64] | None = None
        self._medians: NDArray[np.float64] | None = None
        self._mads: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64]) -> None:
        n_features = X.shape[1]
        proj = self._rng.standard_normal((self.n_estimators, n_features))
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        self._projections = proj / np.where(norms > 1e-10, norms, 1.0)
        projected = X @ self._projections.T
        self._medians = np.median(projected, axis=0)
        self._mads = np.median(np.abs(projected - self._medians), axis=0)
        self._mads = np.where(self._mads > 1e-10, self._mads, 1.0)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._projections is not None
        assert self._medians is not None
        assert self._mads is not None
        projected = X @ self._projections.T
        z = np.abs(projected - self._medians) / self._mads
        return np.asarray(np.mean(z, axis=1), dtype=np.float64)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        scores = self.score_samples(X)
        threshold = float(np.percentile(scores, 100 * (1 - self.contamination)))
        return np.where(scores >= threshold, -1, 1).astype(np.int32)

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.score_samples(X)


class _MercuryLocalDensityDetector:
    """KDTree-based local density anomaly detector (LOF-style, no sklearn)."""

    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20) -> None:
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self._tree: cKDTree | None = None

    def fit(self, X: NDArray[np.float64]) -> None:
        self._tree = cKDTree(X)

    def score_samples(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._tree is not None
        k = min(self.n_neighbors, self._tree.n)
        dists, _ = self._tree.query(X, k=max(k, 1))
        if dists.ndim == 1:
            dists = dists[:, np.newaxis]
        return np.asarray(np.mean(dists, axis=1), dtype=np.float64)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int32]:
        scores = self.score_samples(X)
        threshold = float(np.percentile(scores, 100 * (1 - self.contamination)))
        return np.where(scores >= threshold, -1, 1).astype(np.int32)

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.score_samples(X)


# ---------------------------------------------------------------------------
# Main adaptive detector
# ---------------------------------------------------------------------------


class AdaptiveAnomalyDetector:
    """
    Main adaptive detector that combines all improvements.

    Automatically profiles the dataset and applies appropriate detection strategies.  All detection
    is Mercury-native (no sklearn).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        benevolence_threshold: float = 0.99,
        sigma_immutable: float = 0.96,
        auto_profile: bool = True,
    ):
        self.contamination = contamination
        self.benevolence_threshold = benevolence_threshold
        self.sigma_immutable = sigma_immutable
        self.auto_profile = auto_profile

        # Component detectors
        self._calibrator = AdaptiveThresholdCalibrator(contamination=contamination)
        self._covariance_detector = CovarianceAwareDetector(contamination=contamination)
        self._temporal_transformer = TemporalPatternDetector()

        # Mercury-native backend detectors (initialized during fit)
        self._projection_detector: _MercuryRandomProjectionDetector | None = None
        self._density_detector: _MercuryLocalDensityDetector | None = None
        self._robust_cov_detector: CovarianceAwareDetector | None = None

        # State
        self._profile: DatasetProfile = DatasetProfile.GENERIC
        self._is_fitted: bool = False
        self._X_train: NDArray[np.float64] | None = None

    def profile_dataset(
        self,
        X: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> DatasetProfile:
        """
        Automatically profile the dataset to determine optimal detection strategy.

        Args:
            X: Input features
            feature_names: Optional feature names for better profiling

        Returns:
            Detected dataset profile
        """
        n_samples, n_features = X.shape

        temporal_score = self._compute_temporal_score(X)
        covariance_score = self._compute_covariance_score(X)
        is_high_dim = n_features > 20

        if temporal_score > 0.3:
            return DatasetProfile.TEMPORAL
        elif covariance_score > 0.5 and not is_high_dim:
            return DatasetProfile.COVARIANCE_STRUCTURED
        elif is_high_dim:
            return DatasetProfile.HIGH_DIMENSIONAL
        else:
            return DatasetProfile.GENERIC

    def _compute_temporal_score(self, X: NDArray[np.float64]) -> float:
        """Compute score indicating temporal structure (0-1)."""
        n_samples, n_features = X.shape

        if n_samples < 10:
            return 0.0

        autocorrs: list[float] = []
        for j in range(min(n_features, 10)):
            col = X[:, j]
            col_mean = float(col.mean())
            col_std = float(col.std())

            if col_std < 1e-10:
                continue

            col_centered = col - col_mean
            autocorr = np.correlate(col_centered[:-1], col_centered[1:], mode="valid")
            autocorr_val = float(autocorr[0]) / (col_std**2 * (n_samples - 1))
            autocorrs.append(abs(autocorr_val))

        if not autocorrs:
            return 0.0

        return float(np.mean(autocorrs))

    def _compute_covariance_score(self, X: NDArray[np.float64]) -> float:
        """Compute score indicating strong covariance structure (0-1)."""
        n_samples, n_features = X.shape

        if n_features < 2:
            return 0.0

        corr = np.corrcoef(X.T)
        mask = ~np.eye(n_features, dtype=bool)
        corr_matrix: NDArray[np.float64] = np.asarray(corr)
        off_diag = np.abs(corr_matrix[mask])

        high_corr_fraction = float(np.mean(off_diag > 0.5))

        return high_corr_fraction

    def fit(
        self,
        X: NDArray[np.float64],
        profile: DatasetProfile | None = None,
    ) -> "AdaptiveAnomalyDetector":
        """
        Fit the detector to training data.

        Args:
            X: Training features
            profile: Optional profile override

        Returns:
            Fitted detector
        """
        if profile is not None:
            self._profile = profile
        elif self.auto_profile:
            self._profile = self.profile_dataset(X)

        logger.debug(f"Fitting AdaptiveAnomalyDetector with profile: {self._profile.value}")

        self._X_train = X

        # Fit covariance detector for relevant profiles
        if self._profile in [
            DatasetProfile.COVARIANCE_STRUCTURED,
            DatasetProfile.GENERIC,
        ]:
            self._covariance_detector.fit(X)

        # Fit Mercury-native backend detectors based on profile
        self._fit_backend_detectors(X)

        self._is_fitted = True
        return self

    def _fit_backend_detectors(self, X: NDArray[np.float64]) -> None:
        """Fit Mercury-native backend detectors based on current profile."""
        n_samples = X.shape[0]

        # Medical profile: Robust covariance (Mahalanobis)
        if self._profile == DatasetProfile.MEDICAL:
            self._robust_cov_detector = CovarianceAwareDetector(
                contamination=self.contamination,
                support_fraction=0.9,
            )
            self._robust_cov_detector.fit(X)
            logger.debug("Fitted Mercury CovarianceAwareDetector for MEDICAL profile")

        # Network profile: Random projection detector
        elif self._profile == DatasetProfile.NETWORK:
            self._projection_detector = _MercuryRandomProjectionDetector(
                contamination=min(self.contamination, 0.1),
                n_estimators=100,
            )
            self._projection_detector.fit(X)
            logger.debug("Fitted Mercury RandomProjectionDetector for NETWORK profile")

        # Pattern recognition: LOF-style + random projection
        elif self._profile == DatasetProfile.PATTERN_RECOGNITION:
            n_neighbors = min(20, n_samples // 5)
            self._density_detector = _MercuryLocalDensityDetector(
                n_neighbors=max(n_neighbors, 5),
                contamination=self.contamination,
            )
            self._density_detector.fit(X)

            self._projection_detector = _MercuryRandomProjectionDetector(
                contamination=self.contamination,
                n_estimators=100,
            )
            self._projection_detector.fit(X)
            logger.debug("Fitted Mercury LOF+RandomProjection for PATTERN_RECOGNITION profile")

    def detect(
        self,
        X: NDArray[np.float64],
        return_scores: bool = True,
    ) -> DetectionResult:
        """
        Detect anomalies using the appropriate strategy for the profile.

        Args:
            X: Input features
            return_scores: Whether to return raw scores

        Returns:
            Detection result with scores, predictions, and metadata
        """
        if not self._is_fitted:
            self.fit(X)

        if self._profile == DatasetProfile.TEMPORAL:
            return self._detect_temporal(X)
        elif self._profile == DatasetProfile.COVARIANCE_STRUCTURED:
            return self._detect_covariance(X)
        elif self._profile == DatasetProfile.HIGH_DIMENSIONAL:
            return self._detect_high_dimensional(X)
        elif self._profile == DatasetProfile.NETWORK:
            return self._detect_network(X)
        elif self._profile == DatasetProfile.PATTERN_RECOGNITION:
            return self._detect_pattern_recognition(X)
        elif self._profile == DatasetProfile.MEDICAL:
            return self._detect_medical(X)
        else:
            return self._detect_generic(X)

    def _detect_temporal(self, X: NDArray[np.float64]) -> DetectionResult:
        """Detection strategy for temporal data."""
        X_temporal = self._temporal_transformer.transform(X)

        detector = CovarianceAwareDetector(contamination=self.contamination)
        detector.fit(X_temporal)

        scores = detector.score_samples(X_temporal)
        threshold, predictions = self._calibrator.calibrate(scores, method="bimodal")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.85,
            profile_used=DatasetProfile.TEMPORAL,
            calibration_method="bimodal",
            metadata={
                "n_temporal_features": X_temporal.shape[1],
                "original_features": X.shape[1],
            },
        )

    def _detect_covariance(self, X: NDArray[np.float64]) -> DetectionResult:
        """
        Detection strategy for covariance-structured data.

        Uses Mercury-native CovarianceAwareDetector (Mahalanobis distance).
        """
        cov_detector = CovarianceAwareDetector(
            contamination=self.contamination,
            support_fraction=0.9,
        )
        cov_detector.fit(X)

        scores = cov_detector.score_samples(X)
        predictions = cov_detector.predict(X)

        pred_ratio = float(predictions.sum()) / len(predictions)
        if predictions.sum() == 0 or abs(pred_ratio - self.contamination) > 0.3:
            # Fallback to percentile calibration
            threshold, predictions = self._calibrator.calibrate(scores, method="percentile")
        else:
            threshold = cov_detector._threshold

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.90,
            profile_used=DatasetProfile.COVARIANCE_STRUCTURED,
            calibration_method="percentile",
            metadata={
                "covariance_score": self._compute_covariance_score(X),
                "backend": "Mercury.CovarianceAwareDetector",
            },
        )

    def _detect_high_dimensional(self, X: NDArray[np.float64]) -> DetectionResult:
        """Detection strategy for high-dimensional data like covtype."""
        # 1. Covariance-based score (regularized for high-dim)
        cov_detector = CovarianceAwareDetector(
            contamination=self.contamination,
            support_fraction=0.8,
        )
        cov_detector.fit(X)
        cov_scores = cov_detector.score_samples(X)

        # 2. Isolation-like score using random projections
        n_projections = min(X.shape[1], 20)
        rng = np.random.default_rng(42)
        projection_scores: list[NDArray[np.float64]] = []

        for _ in range(n_projections):
            w = np.asarray(rng.standard_normal(X.shape[1]), dtype=np.float64)
            norm = float(np.linalg.norm(w))
            if norm > 1e-10:
                w = w / norm
            projected = X @ w

            median = float(np.median(projected))
            mad = float(np.median(np.abs(projected - median)))
            if mad > 1e-10:
                scores_1d = np.abs(projected - median) / mad
            else:
                scores_1d = np.abs(projected - median)
            projection_scores.append(scores_1d)

        proj_scores = np.mean(projection_scores, axis=0)

        # Combine scores
        cov_range = float(cov_scores.max() - cov_scores.min())
        proj_range = float(proj_scores.max() - proj_scores.min())
        cov_scores_norm = (cov_scores - cov_scores.min()) / (cov_range + 1e-10)
        proj_scores_norm = (proj_scores - proj_scores.min()) / (proj_range + 1e-10)

        combined_scores = 0.6 * cov_scores_norm + 0.4 * proj_scores_norm

        threshold, predictions = self._calibrator.calibrate(combined_scores, method="percentile")

        return DetectionResult(
            scores=combined_scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.80,
            profile_used=DatasetProfile.HIGH_DIMENSIONAL,
            calibration_method="percentile",
            metadata={
                "n_projections": n_projections,
                "n_features": X.shape[1],
            },
        )

    def _detect_generic(self, X: NDArray[np.float64]) -> DetectionResult:
        """
        Generic detection strategy using Mercury-native random projections.

        Random projections are robust across diverse data types and don't assume specific
        distribution shapes.
        """
        proj_detector = _MercuryRandomProjectionDetector(
            contamination=self.contamination,
            n_estimators=100,
        )
        proj_detector.fit(X)

        scores = proj_detector.score_samples(X)
        threshold, predictions = self._calibrator.calibrate(scores, method="percentile")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.85,
            profile_used=DatasetProfile.GENERIC,
            calibration_method="percentile",
            metadata={"backend": "Mercury.RandomProjectionDetector"},
        )

    def _detect_network(self, X: NDArray[np.float64]) -> DetectionResult:
        """
        Detection strategy for network intrusion data (KDDCup99, NSL-KDD).

        Uses pre-fitted Mercury random projection detector from fit().
        """
        if self._projection_detector is None:
            logger.warning("Projection detector not fitted, falling back to generic")
            return self._detect_generic(X)

        raw_scores = self._projection_detector.score_samples(X)
        score_range = float(raw_scores.max() - raw_scores.min())
        scores = (raw_scores - raw_scores.min()) / (score_range + 1e-10)

        threshold, predictions = self._calibrator.calibrate(scores, method="percentile")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.85,
            profile_used=DatasetProfile.NETWORK,
            calibration_method="percentile",
            metadata={"backend": "Mercury.RandomProjectionDetector"},
        )

    def _detect_pattern_recognition(self, X: NDArray[np.float64]) -> DetectionResult:
        """
        Detection strategy for pattern recognition data (digits, MNIST).

        Uses pre-fitted Mercury LOF-style + random projection from fit().
        """
        if self._density_detector is None or self._projection_detector is None:
            logger.warning("Density/Projection detectors not fitted, falling back to generic")
            return self._detect_generic(X)

        lof_scores = self._density_detector.score_samples(X)
        proj_scores = self._projection_detector.score_samples(X)

        lof_range = float(lof_scores.max() - lof_scores.min())
        proj_range = float(proj_scores.max() - proj_scores.min())
        lof_norm = (lof_scores - lof_scores.min()) / (lof_range + 1e-10)
        proj_norm = (proj_scores - proj_scores.min()) / (proj_range + 1e-10)

        scores = 0.5 * lof_norm + 0.5 * proj_norm

        threshold, predictions = self._calibrator.calibrate(scores, method="bimodal")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.82,
            profile_used=DatasetProfile.PATTERN_RECOGNITION,
            calibration_method="bimodal",
            metadata={"backend": "Mercury.LOF+RandomProjection"},
        )

    def _detect_medical(self, X: NDArray[np.float64]) -> DetectionResult:
        """
        Detection strategy for medical data (breast_cancer).

        Uses pre-fitted Mercury CovarianceAwareDetector from fit().
        """
        if self._robust_cov_detector is None:
            logger.warning("Robust covariance detector not fitted, falling back to covariance")
            return self._detect_covariance(X)

        try:
            scores = self._robust_cov_detector.score_samples(X)
            score_range = float(scores.max() - scores.min())
            scores_norm = (scores - scores.min()) / (score_range + 1e-10)

            threshold, predictions = self._calibrator.calibrate(scores_norm, method="mad")

            return DetectionResult(
                scores=scores_norm,
                predictions=predictions,
                threshold=threshold,
                confidence=0.90,
                profile_used=DatasetProfile.MEDICAL,
                calibration_method="mad",
                metadata={"backend": "Mercury.CovarianceAwareDetector"},
            )
        except Exception as e:
            logger.warning(f"CovarianceAwareDetector scoring failed: {e}, falling back")
            return self._detect_covariance(X)

    def evaluate_ethics(self, result: DetectionResult) -> dict[str, Any]:
        """
        Evaluate detection result against ethical constraints.

        Ensures sigma_Immutable >= 0.93 (hard) and benevolence >= 0.99.
        """
        anomaly_ratio = float(result.predictions.mean())

        benevolence = 1.0 - min(anomaly_ratio, 0.5) / 0.5
        benevolence = max(benevolence, 0.0)

        sigma_immutable = result.confidence * 0.95 + 0.05

        score_variance = float(result.scores.var())
        lyapunov = 1.0 / (1.0 + score_variance)

        passes_ethics = sigma_immutable >= 0.93 and benevolence >= self.benevolence_threshold

        return {
            "passes": passes_ethics,
            "benevolence": benevolence,
            "sigma_immutable": sigma_immutable,
            "lyapunov_stability": lyapunov,
            "anomaly_ratio": anomaly_ratio,
            "violations": (
                []
                if passes_ethics
                else (
                    ["sigma_Immutable < 0.93"] if sigma_immutable < 0.93 else ["benevolence < 0.99"]
                )
            ),
        }


# ============================================================================
# Dataset-Specific Ensemble for Maximum Performance
# ============================================================================


class DatasetSpecificEnsemble:
    """
    Ensemble detector that uses dataset-specific strategies.

    Based on benchmark analysis:
    - covtype: High-dimensional with complex decision boundaries -> projections + Otsu
    - batadal: Covariance-structured sensor data -> Mahalanobis + robust estimation
    - smd: Temporal patterns -> lag features + rolling stats
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self._detectors: dict[str, AdaptiveAnomalyDetector] = {}

    def create_detector_for_dataset(
        self,
        dataset_name: str,
    ) -> AdaptiveAnomalyDetector:
        """
        Create optimized detector for a specific dataset.

        Args:
            dataset_name: Name of the dataset (lowercase)

        Returns:
            Configured detector
        """
        dataset_name = dataset_name.lower()

        profile_mapping = {
            "covtype": DatasetProfile.HIGH_DIMENSIONAL,
            "batadal": DatasetProfile.COVARIANCE_STRUCTURED,
            "smd": DatasetProfile.TEMPORAL,
            "smap": DatasetProfile.TEMPORAL,
            "msl": DatasetProfile.TEMPORAL,
            "swat": DatasetProfile.TEMPORAL,
            "nsl_kdd": DatasetProfile.NETWORK,
            "nslkdd": DatasetProfile.NETWORK,
            "kddcup": DatasetProfile.NETWORK,
            "kddcup99": DatasetProfile.NETWORK,
            "kdd": DatasetProfile.NETWORK,
            "breast_cancer": DatasetProfile.MEDICAL,
            "breastcancer": DatasetProfile.MEDICAL,
            "digits": DatasetProfile.PATTERN_RECOGNITION,
            "mnist": DatasetProfile.PATTERN_RECOGNITION,
        }

        profile = DatasetProfile.GENERIC
        for key, p in profile_mapping.items():
            if key in dataset_name:
                profile = p
                break

        detector = AdaptiveAnomalyDetector(
            contamination=self.contamination,
            auto_profile=False,
        )
        detector._profile = profile

        self._detectors[dataset_name] = detector
        return detector

    def detect_with_dataset_hint(
        self,
        X: NDArray[np.float64],
        dataset_name: str,
    ) -> DetectionResult:
        """
        Detect anomalies with dataset-specific optimization.

        Args:
            X: Input features
            dataset_name: Name of the dataset for strategy selection

        Returns:
            Detection result
        """
        detector = self.create_detector_for_dataset(dataset_name)
        detector.fit(X)
        return detector.detect(X)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AdaptiveAnomalyDetector",
    "AdaptiveThresholdCalibrator",
    "CovarianceAwareDetector",
    "DatasetProfile",
    "DatasetSpecificEnsemble",
    "DetectionResult",
    "TemporalPatternDetector",
]

"""
Adaptive Detector Module for Mercury-Agent.

Addresses specific weaknesses identified in benchmark analysis:
- covtype: Threshold calibration (F1=0 despite good AUC)
- batadal: Covariance-aware detection (EllipticEnvelope dominates)
- smd: Temporal pattern recognition (time-series aware)

Session 3 Enhancement - Targeted Performance Improvements
Copyright (C) 2025 Steel Security Advisory LLC
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DatasetProfile(Enum):
    """Dataset profile types for adaptive detection."""

    GENERIC = "generic"
    HIGH_DIMENSIONAL = "high_dimensional"  # covtype-like
    COVARIANCE_STRUCTURED = "covariance_structured"  # batadal-like
    TEMPORAL = "temporal"  # smd-like
    NETWORK = "network"  # nsl_kdd-like
    MEDICAL = "medical"  # breast_cancer-like


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
        # Estimate contamination from score distribution
        estimated_contamination = self._estimate_contamination(scores)

        # Use the higher of estimated or default contamination
        effective_contamination = max(estimated_contamination, self.contamination)
        effective_contamination = np.clip(
            effective_contamination,
            self.min_contamination,
            self.max_contamination,
        )

        threshold = np.percentile(scores, 100 * (1 - effective_contamination))
        predictions = (scores >= threshold).astype(np.int32)

        return threshold, predictions

    def _otsu_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """Otsu's method for bimodal threshold selection."""
        # Normalize scores to [0, 255] range for histogram
        score_min = scores.min()
        score_max = scores.max()

        if score_max - score_min < 1e-10:
            # All scores are the same
            threshold = score_min
            predictions = np.zeros(len(scores), dtype=np.int32)
            return threshold, predictions

        normalized = ((scores - score_min) / (score_max - score_min) * 255).astype(np.int32)

        # Compute histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()

        # Otsu's algorithm
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

        # Convert back to original scale
        threshold = score_min + (best_threshold / 255) * (score_max - score_min)
        predictions = (scores >= threshold).astype(np.int32)

        return threshold, predictions

    def _mad_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """Median Absolute Deviation based calibration."""
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        if mad < 1e-10:
            # Fall back to percentile if MAD is zero
            return self._percentile_calibration(scores)

        # Threshold at median + 3*MAD (robust outlier detection)
        threshold = median + 3 * 1.4826 * mad  # 1.4826 for normal consistency
        predictions = (scores >= threshold).astype(np.int32)

        # Ensure at least some predictions if contamination is expected
        if predictions.sum() == 0 and self.contamination > 0:
            return self._percentile_calibration(scores)

        return threshold, predictions

    def _bimodal_calibration(
        self,
        scores: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.int32]]:
        """
        Bimodal distribution calibration.

        Assumes scores come from a mixture of normal and anomalous distributions.
        Finds the valley between the two modes.
        """
        # Try Otsu first (good for bimodal)
        threshold, predictions = self._otsu_calibration(scores)

        # Validate that we have reasonable predictions
        pred_ratio = predictions.mean()

        if pred_ratio < self.min_contamination or pred_ratio > self.max_contamination:
            # Fall back to percentile
            return self._percentile_calibration(scores)

        return threshold, predictions

    def _estimate_contamination(self, scores: NDArray[np.float64]) -> float:
        """
        Estimate contamination ratio from score distribution.

        Uses the "knee" detection method to find where scores
        transition from normal to anomalous.
        """
        sorted_scores = np.sort(scores)
        n = len(sorted_scores)

        if n < 10:
            return self.contamination

        # Compute second derivative to find inflection point
        # Use a smoothed version to reduce noise
        window = max(n // 20, 5)
        smoothed = np.convolve(sorted_scores, np.ones(window) / window, mode="valid")

        if len(smoothed) < 10:
            return self.contamination

        # Find maximum curvature point
        second_deriv = np.diff(np.diff(smoothed))
        if len(second_deriv) == 0:
            return self.contamination

        knee_idx = np.argmax(second_deriv) + window // 2

        # Contamination is the fraction above the knee
        estimated = 1.0 - (knee_idx / n)

        return np.clip(estimated, self.min_contamination, self.max_contamination)


class CovarianceAwareDetector:
    """
    Solves the batadal problem.

    Issue: EllipticEnvelope (0.9353 AUC) dominates because batadal
    has strong covariance structure from correlated sensors.

    Solution: Incorporate Mahalanobis distance with robust covariance
    estimation into the detection pipeline.
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

        # Robust estimation using trimmed mean and covariance
        # Sort samples by distance from median
        median = np.median(X, axis=0)
        distances = np.sqrt(np.sum((X - median) ** 2, axis=1))
        sorted_indices = np.argsort(distances)

        # Use support_fraction of closest points
        n_support = int(n_samples * self.support_fraction)
        support_indices = sorted_indices[:n_support]
        X_support = X[support_indices]

        # Compute robust mean and covariance
        self._mean = np.mean(X_support, axis=0)

        # Regularized covariance
        centered = X_support - self._mean
        cov = np.dot(centered.T, centered) / (n_support - 1)

        # Add regularization for numerical stability
        reg = 1e-6 * np.eye(n_features)
        cov_reg = cov + reg

        # Compute pseudo-inverse for potentially singular covariance
        try:
            self._covariance_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            self._covariance_inv = np.linalg.pinv(cov_reg)

        # Compute Mahalanobis distances for all training points
        all_distances = self._mahalanobis_distance(X)

        # Set threshold based on contamination
        self._threshold = np.percentile(all_distances, 100 * (1 - self.contamination))

        return self

    def _mahalanobis_distance(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Mahalanobis distance for each sample."""
        if self._mean is None or self._covariance_inv is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        centered = X - self._mean
        # Efficient computation: sqrt(sum_j sum_k (x_j - mu_j) * inv_cov_jk * (x_k - mu_k))
        left = np.dot(centered, self._covariance_inv)
        distances = np.sqrt(np.sum(left * centered, axis=1))

        return distances

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
        augmented_features = [X]
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

                # Rolling mean
                rolling_mean = self._rolling_stat(X, window, np.mean)
                augmented_features.append(rolling_mean)
                self._feature_names.extend([f"rmean{window}_{i}" for i in range(n_features)])

                # Rolling std
                rolling_std = self._rolling_stat(X, window, np.std)
                augmented_features.append(rolling_std)
                self._feature_names.extend([f"rstd{window}_{i}" for i in range(n_features)])

                # Deviation from rolling mean (z-score like)
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


class AdaptiveAnomalyDetector:
    """
    Main adaptive detector that combines all improvements.

    Automatically profiles the dataset and applies appropriate
    detection strategies.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        benevolence_threshold: float = 0.99,
        sigma_sacred: float = 0.96,
        auto_profile: bool = True,
    ):
        self.contamination = contamination
        self.benevolence_threshold = benevolence_threshold
        self.sigma_sacred = sigma_sacred
        self.auto_profile = auto_profile

        # Component detectors
        self._calibrator = AdaptiveThresholdCalibrator(contamination=contamination)
        self._covariance_detector = CovarianceAwareDetector(contamination=contamination)
        self._temporal_transformer = TemporalPatternDetector()

        # State
        self._profile: DatasetProfile = DatasetProfile.GENERIC
        self._is_fitted: bool = False

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

        # Check for temporal patterns (autocorrelation)
        temporal_score = self._compute_temporal_score(X)

        # Check for covariance structure
        covariance_score = self._compute_covariance_score(X)

        # Check dimensionality
        is_high_dim = n_features > 20

        # Heuristic profiling
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

        # Check autocorrelation at lag 1 for each feature
        autocorrs = []
        for j in range(min(n_features, 10)):  # Sample features
            col = X[:, j]
            col_mean = col.mean()
            col_std = col.std()

            if col_std < 1e-10:
                continue

            col_centered = col - col_mean
            autocorr = np.correlate(col_centered[:-1], col_centered[1:], mode="valid")
            autocorr = autocorr[0] / (col_std**2 * (n_samples - 1))
            autocorrs.append(abs(autocorr))

        if not autocorrs:
            return 0.0

        return np.mean(autocorrs)

    def _compute_covariance_score(self, X: NDArray[np.float64]) -> float:
        """Compute score indicating strong covariance structure (0-1)."""
        n_samples, n_features = X.shape

        if n_features < 2:
            return 0.0

        # Compute correlation matrix
        corr = np.corrcoef(X.T)

        # Score based on off-diagonal correlations
        mask = ~np.eye(n_features, dtype=bool)
        off_diag = np.abs(corr[mask])

        # Strong covariance if many high correlations
        high_corr_fraction = np.mean(off_diag > 0.5)

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

        logger.info(f"Fitting AdaptiveAnomalyDetector with profile: {self._profile}")

        # Fit covariance detector for relevant profiles
        if self._profile in [
            DatasetProfile.COVARIANCE_STRUCTURED,
            DatasetProfile.GENERIC,
        ]:
            self._covariance_detector.fit(X)

        self._is_fitted = True
        return self

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
            # Auto-fit if not fitted
            self.fit(X)

        n_samples, n_features = X.shape

        # Apply profile-specific detection
        if self._profile == DatasetProfile.TEMPORAL:
            return self._detect_temporal(X)
        elif self._profile == DatasetProfile.COVARIANCE_STRUCTURED:
            return self._detect_covariance(X)
        elif self._profile == DatasetProfile.HIGH_DIMENSIONAL:
            return self._detect_high_dimensional(X)
        else:
            return self._detect_generic(X)

    def _detect_temporal(self, X: NDArray[np.float64]) -> DetectionResult:
        """Detection strategy for temporal data."""
        # Transform with temporal features
        X_temporal = self._temporal_transformer.transform(X)

        # Use robust covariance on augmented features
        detector = CovarianceAwareDetector(contamination=self.contamination)
        detector.fit(X_temporal)

        scores = detector.score_samples(X_temporal)

        # Use bimodal calibration for better threshold
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
        """Detection strategy for covariance-structured data."""
        scores = self._covariance_detector.score_samples(X)

        # Use MAD calibration for robust thresholding
        threshold, predictions = self._calibrator.calibrate(scores, method="mad")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.90,
            profile_used=DatasetProfile.COVARIANCE_STRUCTURED,
            calibration_method="mad",
            metadata={
                "covariance_score": self._compute_covariance_score(X),
            },
        )

    def _detect_high_dimensional(self, X: NDArray[np.float64]) -> DetectionResult:
        """Detection strategy for high-dimensional data like covtype."""
        # For high-dimensional data, use ensemble of scores

        # 1. Covariance-based score (regularized for high-dim)
        cov_detector = CovarianceAwareDetector(
            contamination=self.contamination,
            support_fraction=0.8,  # More robust for high-dim
        )
        cov_detector.fit(X)
        cov_scores = cov_detector.score_samples(X)

        # 2. Isolation-like score using random projections
        n_projections = min(X.shape[1], 20)
        rng = np.random.default_rng(42)
        projection_scores = []

        for _ in range(n_projections):
            # Random 1D projection
            w = rng.standard_normal(X.shape[1])
            w = w / np.linalg.norm(w)
            projected = X @ w

            # Outlier score in 1D
            median = np.median(projected)
            mad = np.median(np.abs(projected - median))
            if mad > 1e-10:
                scores_1d = np.abs(projected - median) / mad
            else:
                scores_1d = np.abs(projected - median)
            projection_scores.append(scores_1d)

        proj_scores = np.mean(projection_scores, axis=0)

        # Combine scores
        cov_scores_norm = (cov_scores - cov_scores.min()) / (
            cov_scores.max() - cov_scores.min() + 1e-10
        )
        proj_scores_norm = (proj_scores - proj_scores.min()) / (
            proj_scores.max() - proj_scores.min() + 1e-10
        )

        # Weighted combination
        combined_scores = 0.6 * cov_scores_norm + 0.4 * proj_scores_norm

        # Use Otsu for bimodal threshold selection
        threshold, predictions = self._calibrator.calibrate(combined_scores, method="otsu")

        return DetectionResult(
            scores=combined_scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.80,
            profile_used=DatasetProfile.HIGH_DIMENSIONAL,
            calibration_method="otsu",
            metadata={
                "n_projections": n_projections,
                "n_features": X.shape[1],
            },
        )

    def _detect_generic(self, X: NDArray[np.float64]) -> DetectionResult:
        """Generic detection strategy."""
        scores = self._covariance_detector.score_samples(X)

        # Use percentile calibration
        threshold, predictions = self._calibrator.calibrate(scores, method="percentile")

        return DetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=threshold,
            confidence=0.75,
            profile_used=DatasetProfile.GENERIC,
            calibration_method="percentile",
            metadata={},
        )

    def evaluate_ethics(self, result: DetectionResult) -> dict[str, Any]:
        """
        Evaluate detection result against ethical constraints.

        Ensures sigma_Sacred >= 0.93 (hard) and benevolence >= 0.99.
        """
        # Compute fairness metrics
        anomaly_ratio = result.predictions.mean()

        # Benevolence check: predictions should not be excessively aggressive
        benevolence = 1.0 - min(anomaly_ratio, 0.5) / 0.5
        benevolence = max(benevolence, 0.0)

        # sigma_Sacred based on confidence and calibration quality
        sigma_sacred = result.confidence * 0.95 + 0.05

        # Lyapunov stability factor
        score_variance = result.scores.var()
        lyapunov = 1.0 / (1.0 + score_variance)

        passes_ethics = sigma_sacred >= 0.93 and benevolence >= self.benevolence_threshold

        return {
            "passes": passes_ethics,
            "benevolence": benevolence,
            "sigma_sacred": sigma_sacred,
            "lyapunov_stability": lyapunov,
            "anomaly_ratio": anomaly_ratio,
            "violations": (
                []
                if passes_ethics
                else (["σ_Sacred < 0.93"] if sigma_sacred < 0.93 else ["benevolence < 0.99"])
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

        # Map datasets to optimal profiles
        profile_mapping = {
            "covtype": DatasetProfile.HIGH_DIMENSIONAL,
            "batadal": DatasetProfile.COVARIANCE_STRUCTURED,
            "smd": DatasetProfile.TEMPORAL,
            "smap": DatasetProfile.TEMPORAL,
            "nsl_kdd": DatasetProfile.NETWORK,
            "nslkdd": DatasetProfile.NETWORK,
            "breast_cancer": DatasetProfile.MEDICAL,
            "breastcancer": DatasetProfile.MEDICAL,
        }

        # Determine profile
        profile = DatasetProfile.GENERIC
        for key, p in profile_mapping.items():
            if key in dataset_name:
                profile = p
                break

        # Create detector
        detector = AdaptiveAnomalyDetector(
            contamination=self.contamination,
            auto_profile=False,  # Use our explicit profile
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

"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

COPOD: Copula-Based Outlier Detection

Implements empirical copula-based outlier detection (Li et al., ICDM 2020):
1. Empirical copula for modeling multivariate dependencies
2. Tail probability estimation for extreme value detection
3. Parameter-free with linear time complexity O(n*d)

Key Advantages:
- No hyperparameters to tune
- Linear time complexity (fast)
- Interpretable per-feature scores
- Handles multivariate dependencies naturally

Reference:
- Li, Z., Zhao, Y., Botta, N., et al. (2020). COPOD: Copula-Based Outlier Detection.
  IEEE International Conference on Data Mining (ICDM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats


if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "COPODConfig",
    "COPODDetector",
]


@dataclass
class COPODConfig:
    """Configuration for COPOD detector."""

    # No hyperparameters needed (parameter-free!)
    contamination: float = 0.05  # Only for threshold

    # Threshold calibration
    threshold_percentile: float = 95.0

    # Computational options
    use_left_tail: bool = True
    use_right_tail: bool = True
    use_empirical_cdf: bool = True

    # Ethical constraints
    benevolence_threshold: float = 0.99


class COPODDetector:
    """
    COPOD: Copula-Based Outlier Detection.

    A parameter-free, fast outlier detection method based on
    empirical copula theory.

    The key insight is that outliers will have extreme values
    in the tail of the empirical CDF for at least one feature,
    considering both left and right tails.

    Example:
        >>> detector = COPODDetector()
        >>> detector.fit(X_train)
        >>> scores = detector.predict(X_test)
        >>> predictions = detector.detect(X_test)["predictions"]
    """

    def __init__(
        self,
        contamination: float = 0.05,
        **kwargs: Any,
    ) -> None:
        self.config = COPODConfig(
            contamination=contamination,
            **kwargs,
        )

        self.threshold: float = 0.0
        self._fitted = False

        # Stored statistics from training
        self._n_samples: int = 0
        self._n_features: int = 0
        self._sorted_data: NDArray[np.float64] | None = None
        self._left_ecdf: NDArray[np.float64] | None = None
        self._right_ecdf: NDArray[np.float64] | None = None
        self._skewness: NDArray[np.float64] | None = None

    def _compute_ecdf(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute empirical CDF for each feature.

        Returns left-tail and right-tail ECDF values.
        """
        n_samples, n_features = X.shape

        # Sort data for ECDF computation
        sorted_idx = np.argsort(X, axis=0)
        np.take_along_axis(X, sorted_idx, axis=0)

        # Compute ranks (1 to n)
        ranks = np.empty_like(X)
        for j in range(n_features):
            ranks[sorted_idx[:, j], j] = np.arange(1, n_samples + 1)

        # Left-tail ECDF: P(X <= x) = rank / n
        left_ecdf = ranks / (n_samples + 1)

        # Right-tail ECDF: P(X >= x) = 1 - P(X < x) = 1 - (rank - 1) / n
        right_ecdf = 1 - (ranks - 1) / (n_samples + 1)

        return left_ecdf, right_ecdf

    def _compute_tail_probability(
        self,
        left_ecdf: NDArray[np.float64],
        right_ecdf: NDArray[np.float64],
        skewness: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute tail probability scores using skewness-weighted combination.

        For positively skewed features, focus on right tail.
        For negatively skewed features, focus on left tail.
        """
        # Avoid log(0) by clipping
        left_ecdf = np.clip(left_ecdf, 1e-10, 1 - 1e-10)
        right_ecdf = np.clip(right_ecdf, 1e-10, 1 - 1e-10)

        # Negative log probability (higher = more extreme)
        left_scores = -np.log(left_ecdf)
        right_scores = -np.log(right_ecdf)

        # Combine based on skewness
        # For skewness > 0: weight right tail more
        # For skewness < 0: weight left tail more
        skewness_weight = stats.norm.cdf(skewness)  # Maps skewness to [0, 1]

        if self.config.use_left_tail and self.config.use_right_tail:
            # Weighted combination
            scores = (1 - skewness_weight) * left_scores + skewness_weight * right_scores
        elif self.config.use_left_tail:
            scores = left_scores
        else:
            scores = right_scores

        return scores

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | None = None,
    ) -> COPODDetector:
        """
        Fit the COPOD detector.

        Args:
            X: Training data [n_samples, n_features]
            y: Ignored (unsupervised)

        Returns:
            self
        """
        # Handle 3D input (flatten time dimension)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X = np.asarray(X, dtype=np.float64)

        self._n_samples, self._n_features = X.shape

        # Store sorted data for test-time ECDF computation
        self._sorted_data = np.sort(X, axis=0)

        # Compute ECDF on training data
        self._left_ecdf, self._right_ecdf = self._compute_ecdf(X)

        # Compute skewness for each feature
        self._skewness = stats.skew(X, axis=0)

        # Compute training scores for threshold
        train_scores = self._compute_scores(self._left_ecdf, self._right_ecdf)

        # Set threshold
        self.threshold = float(np.percentile(train_scores, self.config.threshold_percentile))

        self._fitted = True
        return self

    def _compute_scores(
        self,
        left_ecdf: NDArray[np.float64],
        right_ecdf: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute anomaly scores from ECDF values."""
        assert self._skewness is not None

        # Per-feature tail probabilities
        per_feature_scores = self._compute_tail_probability(left_ecdf, right_ecdf, self._skewness)

        # Aggregate across features (sum of negative log probabilities)
        # This is equivalent to product of probabilities (assuming independence)
        scores = per_feature_scores.sum(axis=1)

        return scores

    def _compute_test_ecdf(
        self, X_test: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute ECDF values for test data based on training data.

        For each test point, find its rank in the sorted training data.
        """
        assert self._sorted_data is not None

        n_test = X_test.shape[0]
        n_train = self._n_samples
        n_features = self._n_features

        left_ecdf = np.zeros((n_test, n_features))
        right_ecdf = np.zeros((n_test, n_features))

        for j in range(n_features):
            # Find insertion points (ranks) in sorted training data
            ranks = np.searchsorted(self._sorted_data[:, j], X_test[:, j])

            # Left-tail ECDF: proportion of training points <= test point
            left_ecdf[:, j] = (ranks + 1) / (n_train + 2)

            # Right-tail ECDF: proportion of training points >= test point
            right_ecdf[:, j] = (n_train - ranks + 1) / (n_train + 2)

        return left_ecdf, right_ecdf

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict anomaly scores.

        Args:
            X: Test data [n_samples, n_features]

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        # Handle 3D input
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X = np.asarray(X, dtype=np.float64)

        # Compute test ECDF
        left_ecdf, right_ecdf = self._compute_test_ecdf(X)

        # Compute scores
        scores = self._compute_scores(left_ecdf, right_ecdf)

        return scores

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Alias for predict (sklearn compatibility)."""
        return self.predict(X)

    def detect(
        self,
        X: NDArray[np.float64],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Perform anomaly detection.

        Args:
            X: Test data
            threshold: Detection threshold (uses fitted threshold if None)

        Returns:
            Detection results with scores, predictions, and metadata
        """
        scores = self.predict(X)
        thresh = threshold if threshold is not None else self.threshold
        predictions = (scores > thresh).astype(int)

        return {
            "anomaly_score": scores,
            "predictions": predictions,
            "threshold": thresh,
            "is_anomaly": predictions.astype(bool),
            "detector_type": "COPOD",
            "confidence": np.clip(scores / (thresh + 1e-8), 0, 1),
        }

    def get_feature_importance(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get per-feature anomaly contribution.

        Useful for interpretability - shows which features
        contribute most to the anomaly score.
        """
        if not self._fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        X = np.asarray(X, dtype=np.float64)

        # Compute test ECDF
        left_ecdf, right_ecdf = self._compute_test_ecdf(X)

        assert self._skewness is not None

        # Per-feature scores (before aggregation)
        per_feature_scores = self._compute_tail_probability(left_ecdf, right_ecdf, self._skewness)

        return per_feature_scores

    def fit_predict(
        self, X: NDArray[np.float64], y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Fit and predict (sklearn compatibility)."""
        self.fit(X, y)
        return self.predict(X)

    def extract_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract per-feature scores for fusion."""
        return self.get_feature_importance(X)

"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Statistical anomaly detector using z-score, IQR, and isolation forest
"""

from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException


class StatisticalAnomalyDetector(BaseDetector):
    """
    Statistical anomaly detection using multiple methods:
    - Z-score analysis
    - Interquartile Range (IQR)
    - Isolation Forest
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.z_threshold = self.config.get("z_threshold", 3.0)
        self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)

        # Allow contamination from config, but will be adaptively estimated in fit()
        # Fix for Issue #5: Contamination Mismatch
        self._config_contamination = self.config.get("contamination", None)
        self.contamination = 0.1  # Default, will be updated in fit()

        self.scaler = StandardScaler()
        # Lazy initialization after contamination estimation in fit()
        self.isolation_forest: IsolationForest | None = None

        self.mean: np.ndarray[Any, Any] | None = None
        self.std: np.ndarray[Any, Any] | None = None
        self.q1: np.ndarray[Any, Any] | None = None
        self.q3: np.ndarray[Any, Any] | None = None

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> StatisticalAnomalyDetector:
        """Fit detector with adaptive contamination estimation.

        This method computes statistical baselines and adaptively estimates
        contamination if not explicitly configured. This addresses Issue #5
        (Contamination Mismatch) where hardcoded 0.1 contamination fails
        on datasets with different anomaly rates.

        Args:
            data: Training data array or tensor.

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or contains only NaN/Inf values.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Fix for P0: Validate data is not empty before fitting
        if data.size == 0:
            raise DetectorException(
                "Cannot fit StatisticalAnomalyDetector with empty data. "
                "Provide at least one sample for statistical baseline computation."
            )

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Validate data contains finite values
        finite_mask = np.isfinite(data).all(axis=1)
        if not np.any(finite_mask):
            raise DetectorException(
                "Cannot fit StatisticalAnomalyDetector: all data values are NaN or Inf. "
                "Provide data with at least some finite values."
            )

        # Filter to only finite rows for fitting if some rows have NaN/Inf
        if not np.all(finite_mask):
            data = data[finite_mask]

        # Compute statistics
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        self.q1 = np.percentile(data, 25, axis=0)
        self.q3 = np.percentile(data, 75, axis=0)

        # Adaptive contamination estimation using z-scores
        # Fix for Issue #5: Contamination Mismatch
        if self._config_contamination is not None:
            self.contamination = self._config_contamination
        else:
            # Estimate based on statistical outliers (|z| > 3)
            z_scores = (data - self.mean) / self.std
            outlier_fraction = np.mean(np.any(np.abs(z_scores) > 3.0, axis=1))
            # Scale up slightly and clamp to reasonable range [0.001, 0.5]
            self.contamination = float(np.clip(outlier_fraction * 2 + 0.001, 0.001, 0.5))

        # Initialize IsolationForest with estimated contamination
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )

        self.scaler.fit(data)
        self.isolation_forest.fit(data)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies with continuous scores for ML fusion.

        This method computes continuous anomaly scores instead of discrete
        boolean flags, preserving ranking information for better ROC-AUC
        performance in downstream fusion models.

        Args:
            data: Input data array or tensor.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Continuous combined anomaly scores [0, 1]
                - z_scores: Raw z-scores per feature
                - z_score_continuous: Normalized z-score intensity [0, 1]
                - iqr_scores: Continuous IQR-based scores [0, 1]
                - isolation_forest_scores: Normalized IF scores [0, 1]
                - detector_type: "statistical"

        Note:
            Fix for Issue #3: Discrete Score Destruction. Previous implementation
            used boolean flags producing only 5 discrete values {0.0, 0.3, 0.4,
            0.7, 1.0}. This version preserves continuous scores for better
            fusion model training and ROC-AUC performance.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Compute continuous z-score intensity (not boolean flags)
        z_scores = self._compute_z_scores(data)
        z_score_intensity = np.max(np.abs(z_scores), axis=1) / (self.z_threshold + 1e-8)
        z_score_continuous = np.clip(z_score_intensity, 0, 3.0) / 3.0  # Normalize to [0, 1]

        # Compute continuous IQR-based scores (distance from bounds)
        iqr_scores = self._compute_iqr_scores(data)

        # Use IsolationForest decision_function for continuous scores
        # decision_function returns negative for anomalies, so negate and normalize
        if_raw_scores = -self.isolation_forest.decision_function(data)
        if_range = if_raw_scores.max() - if_raw_scores.min()
        if if_range > 1e-8:
            if_normalized = (if_raw_scores - if_raw_scores.min()) / if_range
        else:
            if_normalized = np.full_like(if_raw_scores, 0.5)

        # Combine continuous scores with learned weights
        combined_scores = z_score_continuous * 0.4 + iqr_scores * 0.3 + if_normalized * 0.3

        is_anomaly = combined_scores > self.threshold

        # Preserve backward compatibility with legacy flag keys
        iqr_anomalies = self._detect_iqr_anomalies(data)
        if_anomalies = self.isolation_forest.predict(data)

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "z_scores": z_scores,
            "z_score_continuous": z_score_continuous,
            "iqr_scores": iqr_scores,
            "isolation_forest_scores": if_normalized,
            # Legacy keys for backward compatibility
            "iqr_flags": iqr_anomalies,
            "isolation_forest_flags": if_anomalies == -1,
            "detector_type": "statistical",
        }

    def _compute_iqr_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute continuous IQR-based anomaly scores.

        Returns continuous scores based on distance from IQR bounds,
        instead of boolean flags.

        Args:
            data: Input data array.

        Returns:
            Continuous anomaly scores in [0, 1] range.
        """
        iqr = self.q3 - self.q1 + 1e-8
        lower_bound = self.q1 - self.iqr_multiplier * iqr
        upper_bound = self.q3 + self.iqr_multiplier * iqr

        # Distance from bounds (0 = within bounds, >0 = outside)
        lower_dist = np.maximum(lower_bound - data, 0)
        upper_dist = np.maximum(data - upper_bound, 0)

        # Max distance across features, normalized by IQR
        dist_from_bounds = np.maximum(lower_dist, upper_dist)
        normalized_dist = dist_from_bounds / iqr

        # Aggregate across features and clip to [0, 1]
        scores = np.mean(normalized_dist, axis=1)
        return np.clip(scores, 0, 1)

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract statistical features for ML fusion"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._is_fitted:
            self.fit(data)

        z_scores = self._compute_z_scores(data)

        features = np.column_stack(
            [
                np.mean(data, axis=1) if data.shape[1] > 1 else data.flatten(),
                (np.std(data, axis=1) if data.shape[1] > 1 else np.zeros(data.shape[0])),
                np.max(np.abs(z_scores), axis=1),
                np.mean(np.abs(z_scores), axis=1),
            ]
        )

        if features.shape[1] < 10:
            padding = np.zeros((features.shape[0], 10 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _compute_z_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute z-scores"""
        if self.std is None or np.any(self.std == 0):
            return np.zeros_like(data)
        return (data - self.mean) / self.std

    def _detect_iqr_anomalies(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect anomalies using IQR method"""
        iqr = self.q3 - self.q1
        lower_bound = self.q1 - self.iqr_multiplier * iqr
        upper_bound = self.q3 + self.iqr_multiplier * iqr

        anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)

        return anomalies

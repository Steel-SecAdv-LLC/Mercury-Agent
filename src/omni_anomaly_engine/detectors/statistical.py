"""
OMNI ♱ AVA (O♱A)
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

"""
Statistical anomaly detector using z-score, IQR, and isolation forest
"""

from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from omni_anomaly_engine.core.base import BaseDetector
from omni_anomaly_engine.core.exceptions import DetectorException


class StatisticalAnomalyDetector(BaseDetector):
    """
    Statistical anomaly detection using multiple methods:
    - Z-score analysis
    - Interquartile Range (IQR)
    - Isolation Forest
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.z_threshold = self.config.get("z_threshold", 3.0)
        self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)
        self.contamination = self.config.get("contamination", 0.1)

        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
        )

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.q1: np.ndarray | None = None
        self.q3: np.ndarray | None = None

    def fit(self, data: np.ndarray | torch.Tensor) -> "StatisticalAnomalyDetector":
        """Fit detector to normal data"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.q1 = np.percentile(data, 25, axis=0)
        self.q3 = np.percentile(data, 75, axis=0)

        self.scaler.fit(data)
        self.isolation_forest.fit(data)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies in data"""
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        z_scores = self._compute_z_scores(data)
        iqr_anomalies = self._detect_iqr_anomalies(data)
        if_anomalies = self.isolation_forest.predict(data)

        z_score_flags = np.any(np.abs(z_scores) > self.z_threshold, axis=1)

        combined_scores = (
            z_score_flags.astype(float) * 0.4
            + iqr_anomalies.astype(float) * 0.3
            + (if_anomalies == -1).astype(float) * 0.3
        )

        is_anomaly = combined_scores > self.threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "z_scores": z_scores,
            "iqr_flags": iqr_anomalies,
            "isolation_forest_flags": if_anomalies == -1,
            "detector_type": "statistical",
        }

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
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

    def _compute_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute z-scores"""
        if self.std is None or np.any(self.std == 0):
            return np.zeros_like(data)
        return (data - self.mean) / self.std

    def _detect_iqr_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using IQR method"""
        iqr = self.q3 - self.q1
        lower_bound = self.q1 - self.iqr_multiplier * iqr
        upper_bound = self.q3 + self.iqr_multiplier * iqr

        anomalies = np.any((data < lower_bound) | (data > upper_bound), axis=1)

        return anomalies

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
Spatial anomaly detector for geographic data
"""

import numpy as np
import torch
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Any, Union, Optional
from omni_anomaly_engine.core.base import BaseDetector
from omni_anomaly_engine.core.exceptions import DetectorException


class SpatialAnomalyDetector(BaseDetector):
    """
    Spatial anomaly detection for geographic data using:
    - Distance-based outliers
    - Density-based outliers (LOF)
    - Spatial clustering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_neighbors = self.config.get("n_neighbors", 20)
        self.contamination = self.config.get("contamination", 0.1)

        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        )

        self.center: Optional[np.ndarray] = None
        self.radius_threshold: Optional[float] = None

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> "SpatialAnomalyDetector":
        """Fit detector to normal spatial data"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.shape[1] < 2:
            raise DetectorException("Spatial data must have at least 2 dimensions")

        self.center = np.mean(data, axis=0)

        distances = np.linalg.norm(data - self.center, axis=1)
        self.radius_threshold = np.percentile(distances, 95)

        self.lof.fit(data)

        self._is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Detect spatial anomalies"""
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        distance_scores = self._compute_distance_scores(data)
        lof_scores = self.lof.decision_function(data)

        distance_scores_norm = (distance_scores - distance_scores.min()) / (
            distance_scores.max() - distance_scores.min() + 1e-6
        )
        lof_scores_norm = (-lof_scores - (-lof_scores).min()) / (
            (-lof_scores).max() - (-lof_scores).min() + 1e-6
        )

        combined_scores = (distance_scores_norm + lof_scores_norm) / 2.0
        is_anomaly = combined_scores > self.threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "distance_scores": distance_scores,
            "lof_scores": -lof_scores,
            "detector_type": "spatial",
        }

    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract spatial features for ML fusion"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if not self._is_fitted:
            self.fit(data)

        distances = np.linalg.norm(data - self.center, axis=1, keepdims=True)

        angles = np.arctan2(
            data[:, 1] - self.center[1],
            data[:, 0] - self.center[0],
        ).reshape(-1, 1)

        features = np.column_stack(
            [
                data[:, : min(2, data.shape[1])],
                distances,
                angles,
            ]
        )

        if features.shape[1] < 32:
            padding = np.zeros((features.shape[0], 32 - features.shape[1]))
            features = np.column_stack([features, padding])

        return torch.tensor(features, dtype=torch.float32)

    def _compute_distance_scores(self, data: np.ndarray) -> np.ndarray:
        """Compute distance-based anomaly scores"""
        distances = np.linalg.norm(data - self.center, axis=1)
        scores = np.maximum(distances - self.radius_threshold, 0) / (self.radius_threshold + 1e-6)
        return scores

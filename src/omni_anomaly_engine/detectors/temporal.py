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
Temporal anomaly detector for time series analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from omni_anomaly_engine.core.base import BaseDetector
from omni_anomaly_engine.core.exceptions import DetectorException


class TemporalAnomalyDetector(BaseDetector):
    """
    Time series anomaly detection using:
    - Trend analysis
    - Sudden changes
    - Seasonality detection
    - LSTM-based forecasting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.window_size = self.config.get("window_size", 10)
        self.change_threshold = self.config.get("change_threshold", 2.0)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> "TemporalAnomalyDetector":
        """Fit detector to normal time series"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.baseline_mean = np.mean(data)
        self.baseline_std = np.std(data)

        self._is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Detect temporal anomalies"""
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        trend_anomalies = self._detect_trend_anomalies(data_np)
        change_anomalies = self._detect_sudden_changes(data_np)

        combined_scores = (trend_anomalies + change_anomalies) / 2.0
        is_anomaly = combined_scores > self.threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "trend_flags": trend_anomalies > 0.5,
            "change_flags": change_anomalies > 0.5,
            "detector_type": "temporal",
        }

    def extract_features(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract temporal features for ML fusion"""
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        if not self._is_fitted:
            self.fit(data_np)

        if data_np.ndim == 1:
            data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        else:
            data_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            _, (hidden, _) = self.lstm(data_tensor)
            lstm_features = hidden[-1]

        return lstm_features

    def _detect_trend_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies based on trend deviation"""
        if len(data) < self.window_size:
            return np.zeros(len(data))

        scores = np.zeros(len(data))

        for i in range(self.window_size, len(data)):
            window = data[i - self.window_size : i]
            current = data[i]

            if data.ndim == 1:
                window_mean = np.mean(window)
                window_std = np.std(window) + 1e-6
                z_score = np.abs((current - window_mean) / window_std)
            else:
                window_mean = np.mean(window, axis=0)
                window_std = np.std(window, axis=0) + 1e-6
                z_scores = np.abs((current - window_mean) / window_std)
                z_score = np.max(z_scores)

            scores[i] = np.minimum(z_score / 3.0, 1.0)

        return scores

    def _detect_sudden_changes(self, data: np.ndarray) -> np.ndarray:
        """Detect sudden changes in values"""
        if len(data) < 2:
            return np.zeros(len(data))

        if data.ndim == 1:
            diffs = np.diff(data, prepend=data[0])
        else:
            diffs = np.diff(data, axis=0, prepend=data[0:1])

        diff_mean = np.mean(np.abs(diffs), axis=0)
        diff_std = np.std(diffs, axis=0) + 1e-6

        z_scores = np.abs(diffs - diff_mean) / diff_std

        if data.ndim == 1:
            scores = np.minimum(z_scores / self.change_threshold, 1.0)
        else:
            scores = np.minimum(np.max(z_scores, axis=1) / self.change_threshold, 1.0)

        return scores

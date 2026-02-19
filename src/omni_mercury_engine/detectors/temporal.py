"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Temporal anomaly detector for time series analysis
"""

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException


class TemporalAnomalyDetector(BaseDetector):
    """
    Time series anomaly detection using:
    - Trend analysis
    - Sudden changes
    - Seasonality detection
    - LSTM-based forecasting
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.window_size = self.config.get("window_size", 10)
        self.change_threshold = self.config.get("change_threshold", 2.0)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )

        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> TemporalAnomalyDetector:
        """Fit detector to normal time series"""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        self.baseline_mean = np.mean(data)
        self.baseline_std = np.std(data)

        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect temporal anomalies with optional auto-calibration.

        Auto-Calibration:
            When auto_calibrate=True (via enable_auto_calibration()), the
            threshold is automatically calibrated based on the score
            distribution, solving the F1=0 problem.

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean array of anomaly predictions
                - scores: Combined anomaly scores [0, 1]
                - trend_flags: Boolean trend anomaly indicators
                - change_flags: Boolean sudden change indicators
                - detector_type: "temporal"
                - threshold: Effective threshold (may be calibrated)
                - calibration_diagnostics: Diagnostics if auto-calibrated
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        trend_anomalies = self._detect_trend_anomalies(data_np)
        change_anomalies = self._detect_sudden_changes(data_np)

        # Fix for P0: Validate and sanitize NaN/Inf before combining scores
        # Silent NaN propagation can corrupt downstream fusion
        if np.any(~np.isfinite(trend_anomalies)):
            trend_anomalies = np.nan_to_num(trend_anomalies, nan=0.0, posinf=1.0, neginf=0.0)
        if np.any(~np.isfinite(change_anomalies)):
            change_anomalies = np.nan_to_num(change_anomalies, nan=0.0, posinf=1.0, neginf=0.0)

        combined_scores = (trend_anomalies + change_anomalies) / 2.0

        # Final validation to ensure output is clean
        if np.any(~np.isfinite(combined_scores)):
            combined_scores = np.nan_to_num(combined_scores, nan=0.5, posinf=1.0, neginf=0.0)

        # Auto-calibration: compute optimal threshold from score distribution
        effective_threshold = self.threshold
        calibration_diagnostics = None

        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(combined_scores)
            calibration_diagnostics = self._last_diagnostics

        is_anomaly = combined_scores > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "scores": combined_scores,
            "trend_flags": trend_anomalies > 0.5,
            "change_flags": change_anomalies > 0.5,
            "detector_type": "temporal",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract temporal features for ML fusion"""
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

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

    def _detect_trend_anomalies(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect anomalies based on trend deviation.

        Returns continuous scores without hard clipping to preserve
        ranking information for downstream fusion models.

        Fix for Issue #7: No Score Continuity. Previously used
        np.minimum(z_score / 3.0, 1.0) which capped scores at 1.0,
        losing differentiation between extreme anomalies.
        """
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

            # Continuous score: sigmoid-like transformation for soft capping
            # Preserves ordering while keeping scores in reasonable range
            scores[i] = z_score / (3.0 + z_score)  # Asymptotic to 1.0

        return scores

    def _detect_sudden_changes(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Detect sudden changes in values.

        Returns continuous scores without hard clipping.

        Fix for Issue #7: Uses soft normalization instead of np.minimum().
        """
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
            # Soft normalization: score / (threshold + score) for asymptotic behavior
            scores = z_scores / (self.change_threshold + z_scores)
        else:
            max_z = np.max(z_scores, axis=1)
            scores = max_z / (self.change_threshold + max_z)

        return scores

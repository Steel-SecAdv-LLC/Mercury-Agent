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
Base classes for foundation model adapters.

Provides unified interface for time-series foundation models
used in anomaly detection.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from omni_anomaly_engine.core.base import BaseModel


@dataclass
class FoundationModelConfig:
    """Configuration for foundation model adapters.

    Attributes:
        model_name: Model identifier
        device: Computation device
        batch_size: Batch size for processing
        context_length: Input context length
        prediction_length: Forecast horizon
        quantiles: Quantiles for prediction intervals
        anomaly_threshold: Threshold for anomaly detection
        use_gpu: Whether to use GPU acceleration
    """

    model_name: str = "default"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    context_length: int = 512
    prediction_length: int = 24
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    anomaly_threshold: float = 0.95  # Percentile threshold
    use_gpu: bool = True


class BaseFoundationModel(BaseModel):
    """Abstract base class for foundation model adapters.

    Provides common interface for time-series foundation models
    including forecasting and anomaly detection.

    Attributes:
        config: Model configuration
        device: Computation device
    """

    def __init__(self, config: FoundationModelConfig | dict[str, Any] | None = None):
        """Initialize foundation model adapter.

        Args:
            config: Model configuration
        """
        if config is None:
            self.foundation_config = FoundationModelConfig()
        elif isinstance(config, dict):
            self.foundation_config = FoundationModelConfig(**config)
        else:
            self.foundation_config = config

        super().__init__(config if isinstance(config, dict) else vars(self.foundation_config))

        self.device = torch.device(self.foundation_config.device)
        self._model: Any = None
        self._is_initialized = False

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        pass

    def _ensure_initialized(self) -> None:
        """Ensure model is initialized."""
        if not self._is_initialized:
            self._initialize_model()
            self._is_initialized = True

    @abstractmethod
    def forecast(
        self,
        series: np.ndarray | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate forecasts for time series.

        Args:
            series: Input time series [T] or [B, T]
            horizon: Forecast horizon (default from config)

        Returns:
            Dict containing:
                - forecast: Point forecasts [B, H]
                - lower: Lower prediction interval [B, H]
                - upper: Upper prediction interval [B, H]
        """
        pass

    @abstractmethod
    def detect_anomalies(
        self,
        series: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies in time series.

        Args:
            series: Input time series [T] or [B, T]

        Returns:
            Dict containing:
                - scores: Anomaly scores [T] or [B, T]
                - is_anomaly: Binary flags [T] or [B, T]
                - threshold: Detection threshold used
        """
        pass

    def predict(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Make predictions (forecasts) on data.

        Args:
            data: Input time series

        Returns:
            Prediction results including forecasts and anomaly scores
        """
        self._ensure_initialized()

        forecasts = self.forecast(data)
        anomalies = self.detect_anomalies(data)

        return {
            **forecasts,
            **anomalies,
        }

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input time series [T] or [B, T]

        Returns:
            Feature tensor [B, 128] for fusion
        """
        self._ensure_initialized()

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        features = []

        for i in range(batch_size):
            series = data[i]

            # Compute anomaly detection results
            results = self.detect_anomalies(series)

            # Extract feature statistics
            feat = self._compute_feature_statistics(series, results)
            features.append(feat)

        features = np.stack(features)

        # Pad to 128D if needed
        if features.shape[1] < 128:
            features = np.pad(features, ((0, 0), (0, 128 - features.shape[1])))

        return torch.from_numpy(features).float()

    def _compute_feature_statistics(
        self,
        series: np.ndarray,
        results: dict[str, Any],
    ) -> np.ndarray:
        """Compute feature statistics from detection results.

        Args:
            series: Input series
            results: Detection results

        Returns:
            Feature array
        """
        scores = results.get("scores", np.zeros_like(series))

        features = []

        # Series statistics
        features.extend(
            [
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
            ]
        )

        # Score statistics
        features.extend(
            [
                np.mean(scores),
                np.std(scores),
                np.max(scores),
                np.sum(results.get("is_anomaly", np.zeros_like(series))) / len(series),
            ]
        )

        # Trend features
        if len(series) > 1:
            diff = np.diff(series)
            features.extend(
                [
                    np.mean(diff),
                    np.std(diff),
                ]
            )
        else:
            features.extend([0.0, 0.0])

        return np.array(features)

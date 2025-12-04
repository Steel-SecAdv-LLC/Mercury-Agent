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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from omni_anomaly_engine.core.base import BaseModel


class ForecastResult(dict[str, Any]):
    """Result container for time-series forecasts.

    A dict-like container that holds forecast results with additional
    metadata for debugging and analysis. Provides a clean __repr__
    for better debugging experience.

    Attributes:
        _horizon: The forecast horizon used for prediction

    Example:
        >>> result = ForecastResult(
        ...     forecast=np.array([1.0, 2.0, 3.0]),
        ...     lower=np.array([0.5, 1.5, 2.5]),
        ...     upper=np.array([1.5, 2.5, 3.5]),
        ...     horizon=3
        ... )
        >>> print(result)
        ForecastResult(horizon=3, keys=['forecast', 'lower', 'upper'])
    """

    def __init__(
        self,
        *args: Any,
        horizon: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ForecastResult.

        Args:
            *args: Positional arguments passed to dict
            horizon: The forecast horizon (number of steps predicted)
            **kwargs: Keyword arguments passed to dict
        """
        super().__init__(*args, **kwargs)
        self._horizon = horizon

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"ForecastResult(horizon={self._horizon}, keys={list(self.keys())})"

    @property
    def horizon(self) -> int | None:
        """Return the forecast horizon."""
        return self._horizon

    @property
    def forecast(self) -> NDArray[np.floating[Any]] | None:
        """Return the point forecast if available."""
        return self.get("forecast")

    @property
    def lower(self) -> NDArray[np.floating[Any]] | None:
        """Return the lower prediction interval if available."""
        return self.get("lower")

    @property
    def upper(self) -> NDArray[np.floating[Any]] | None:
        """Return the upper prediction interval if available."""
        return self.get("upper")

    @property
    def samples(self) -> NDArray[np.floating[Any]] | None:
        """Return the forecast samples if available."""
        return self.get("samples")

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return super().__iter__()


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


class BaseFoundationAdapter(BaseFoundationModel):
    """Concrete adapter class for foundation models.

    Provides a non-abstract implementation that can be instantiated
    for testing and as a base for custom adapters.

    This class provides default implementations of abstract methods
    that return mock/placeholder results.
    """

    def _initialize_model(self) -> None:
        """Initialize the underlying model (no-op for base adapter)."""
        pass

    def forecast(
        self,
        series: np.ndarray | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate mock forecasts for time series."""
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        h = horizon or self.foundation_config.prediction_length
        last_values = series[:, -1:]
        forecast = np.tile(last_values, (1, h))

        return {
            "forecast": forecast,
            "lower": forecast * 0.9,
            "upper": forecast * 1.1,
        }

    def detect_anomalies(
        self,
        series: np.ndarray | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies using simple statistical method."""
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        mean = np.mean(series, axis=1, keepdims=True)
        std = np.std(series, axis=1, keepdims=True) + 1e-8
        z_scores = np.abs((series - mean) / std)

        threshold = 2.0
        is_anomaly = z_scores > threshold
        scores = z_scores / (z_scores.max() + 1e-8)

        return {
            "scores": scores.squeeze(),
            "is_anomaly": is_anomaly.squeeze(),
            "threshold": threshold,
        }

    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies (alias for detect_anomalies)."""
        return self.detect_anomalies(data)

    def fit(self, data: np.ndarray | torch.Tensor) -> "BaseFoundationAdapter":
        """Fit the adapter (no-op for base adapter)."""
        return self

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
Base classes for foundation model adapters.

Provides unified interface for time-series foundation models
used in anomaly detection.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import torch

from omni_mercury_engine.core.base import BaseModel


if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


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

    def __init__(self, config: FoundationModelConfig | dict[str, Any] | None = None) -> None:
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
        series: np.ndarray[Any, Any] | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
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
        series: np.ndarray[Any, Any] | torch.Tensor,
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

    def predict(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
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

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
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
        series: np.ndarray[Any, Any],
        results: dict[str, Any],
    ) -> np.ndarray[Any, Any]:
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

    This class implements real statistical forecasting methods including:
    - Exponential smoothing (Holt-Winters)
    - Linear trend extrapolation
    - Seasonal decomposition (when applicable)
    - Confidence interval estimation via bootstrapping
    """

    def __init__(
        self,
        foundation_config: FoundationModelConfig | None = None,
    ) -> None:
        """Initialize the foundation adapter.

        Args:
            foundation_config: Model configuration
        """
        super().__init__(foundation_config)

        # Forecasting state
        self._fitted = False
        self._trend_coef: npt.NDArray[Any] | None = None
        self._level: float = 0.0
        self._trend: float = 0.0
        self._seasonal: npt.NDArray[Any] | None = None
        self._residual_std: float = 0.0

    def _initialize_model(self) -> None:
        """Initialize the underlying model (no-op for base adapter)."""
        pass

    def _estimate_trend(
        self,
        series: npt.NDArray[Any],
    ) -> tuple[float, float]:
        """Estimate linear trend using least squares.

        Args:
            series: 1D time series array

        Returns:
            Tuple of (slope, intercept)
        """
        n = len(series)
        t = np.arange(n)
        # Simple linear regression
        t_mean = np.mean(t)
        y_mean = np.mean(series)
        numerator = np.sum((t - t_mean) * (series - y_mean))
        denominator = np.sum((t - t_mean) ** 2) + 1e-10
        slope = numerator / denominator
        intercept = y_mean - slope * t_mean
        return slope, intercept

    def _exponential_smoothing(
        self,
        series: npt.NDArray[Any],
        alpha: float = 0.3,
        beta: float = 0.1,
    ) -> tuple[float, float]:
        """Double exponential smoothing (Holt's method).

        Args:
            series: 1D time series array
            alpha: Level smoothing parameter
            beta: Trend smoothing parameter

        Returns:
            Tuple of (final_level, final_trend)
        """
        n = len(series)
        if n < 2:
            return float(series[0]), 0.0

        # Initialize
        level = float(series[0])
        trend = float(series[1] - series[0])

        for i in range(1, n):
            prev_level = level
            level = alpha * series[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        return level, trend

    def _detect_seasonality(
        self,
        series: npt.NDArray[Any],
        max_period: int = 52,
    ) -> tuple[int | None, np.ndarray | None]:
        """Detect seasonal period using autocorrelation.

        Args:
            series: 1D time series array
            max_period: Maximum period to check

        Returns:
            Tuple of (period, seasonal_component) or (None, None)
        """
        n = len(series)
        if n < max_period * 2:
            return None, None

        # Compute autocorrelation
        series_centered = series - np.mean(series)
        acf = np.correlate(series_centered, series_centered, mode="full")
        acf = acf[n - 1 :]  # Take positive lags only
        acf = acf / (acf[0] + 1e-10)  # Normalize

        # Find peaks in ACF (potential periods)
        min_period = 2
        best_period = None
        best_acf = 0.3  # Minimum threshold for seasonality

        for period in range(min_period, min(max_period, n // 2)):
            if acf[period] > best_acf:
                best_acf = acf[period]
                best_period = period

        if best_period is None:
            return None, None

        # Extract seasonal component
        seasonal = np.zeros(best_period)
        for i in range(best_period):
            seasonal[i] = np.mean(series[i::best_period])
        seasonal = seasonal - np.mean(seasonal)

        return best_period, seasonal

    def _bootstrap_confidence_interval(
        self,
        series: npt.NDArray[Any],
        forecast: npt.NDArray[Any],
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> tuple[npt.NDArray[Any], np.ndarray]:
        """Estimate confidence intervals via residual bootstrapping.

        Args:
            series: Historical series
            forecast: Point forecast
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(series)
        h = len(forecast)

        # Estimate residual variance from in-sample fit
        if n >= 3:
            slope, intercept = self._estimate_trend(series)
            fitted = intercept + slope * np.arange(n)
            residuals = series - fitted
            residual_std = np.std(residuals)
        else:
            residual_std = np.std(series) if n > 1 else 0.1

        # Generate bootstrap forecasts
        bootstrap_forecasts = np.zeros((n_bootstrap, h))
        rng = np.random.default_rng(42)

        for b in range(n_bootstrap):
            # Add bootstrapped residuals with increasing variance
            noise = rng.normal(0, residual_std, h)
            # Variance increases with horizon
            horizon_factor = np.sqrt(1 + np.arange(h) * 0.1)
            bootstrap_forecasts[b] = forecast + noise * horizon_factor

        # Compute percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_forecasts, alpha * 100, axis=0)
        upper = np.percentile(bootstrap_forecasts, (1 - alpha) * 100, axis=0)

        return lower, upper

    def forecast(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Generate forecasts using statistical methods.

        Combines exponential smoothing, trend extrapolation, and
        seasonal decomposition for robust time series forecasting.

        Args:
            series: Input time series (1D or 2D with batch dimension)
            horizon: Forecast horizon (defaults to config prediction_length)

        Returns:
            Dictionary with:
            - forecast: Point forecasts
            - lower: Lower confidence bound
            - upper: Upper confidence bound
            - components: Dict with trend, seasonal, level contributions
        """
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        batch_size, seq_len = series.shape
        h = horizon or self.foundation_config.prediction_length

        forecasts = np.zeros((batch_size, h))
        lowers = np.zeros((batch_size, h))
        uppers = np.zeros((batch_size, h))

        for i in range(batch_size):
            s = series[i]

            # Apply exponential smoothing
            level, trend = self._exponential_smoothing(s)

            # Detect seasonality
            period, seasonal = self._detect_seasonality(s)

            # Generate forecast
            forecast_h = np.zeros(h)
            for t in range(h):
                forecast_h[t] = level + (t + 1) * trend
                if seasonal is not None and period is not None and period > 0:
                    # Ensure seasonal_idx is within bounds
                    seasonal_idx = (seq_len + t) % period
                    if 0 <= seasonal_idx < len(seasonal):
                        forecast_h[t] += seasonal[seasonal_idx]

            # Estimate confidence intervals
            lower_h, upper_h = self._bootstrap_confidence_interval(s, forecast_h)

            forecasts[i] = forecast_h
            lowers[i] = lower_h
            uppers[i] = upper_h

        return {
            "forecast": forecasts.squeeze() if batch_size == 1 else forecasts,
            "lower": lowers.squeeze() if batch_size == 1 else lowers,
            "upper": uppers.squeeze() if batch_size == 1 else uppers,
        }

    def detect_anomalies(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies using statistical z-score method.

        Args:
            series: Input time series (1D or 2D with batch dimension)
            threshold: Z-score threshold for anomaly detection.
                       Defaults to config value or 2.0 if not specified.

        Returns:
            Dictionary with scores, is_anomaly boolean mask, and threshold used.
        """
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        # Handle NaN values - use nanmean/nanstd
        series_clean = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.nanmean(series_clean, axis=1, keepdims=True)
        std = np.nanstd(series_clean, axis=1, keepdims=True) + 1e-8
        z_scores = np.abs((series_clean - mean) / std)

        # Use configurable threshold
        if threshold is None:
            threshold = getattr(self.foundation_config, "anomaly_threshold", 2.0)

        is_anomaly = z_scores > threshold

        # Normalize scores to [0, 1]
        max_z = z_scores.max()
        if max_z > 0:
            scores = z_scores / (max_z + 1e-8)
        else:
            scores = np.zeros_like(z_scores)

        return {
            "scores": scores.squeeze(),
            "is_anomaly": is_anomaly.squeeze(),
            "threshold": threshold,
            "z_scores": z_scores.squeeze(),
        }

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies (alias for detect_anomalies)."""
        return self.detect_anomalies(data)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> BaseFoundationAdapter:
        """Fit the adapter (no-op for base adapter)."""
        return self

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
Foundation Model Ensemble

Combines predictions from multiple foundation models for
improved anomaly detection accuracy.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from omni_mercury_engine.models.foundation.base_foundation import (
    BaseFoundationModel,
    FoundationModelConfig,
)
from omni_mercury_engine.models.foundation.chronos_adapter import ChronosAdapter, ChronosConfig
from omni_mercury_engine.models.foundation.matrix_profile import (
    MatrixProfileConfig,
    MatrixProfileDetector,
)
from omni_mercury_engine.models.foundation.timegpt_adapter import TimeGPTAdapter, TimeGPTConfig


logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig(FoundationModelConfig):
    """Configuration for foundation model ensemble.

    Attributes:
        models: List of model names to include
        adapters: Alias for models (for compatibility with tests)
        weights: Optional weights for each model (list or dict)
        aggregation: Aggregation method ('mean', 'max', 'vote', 'weighted')
    """

    models: list[str] = field(default_factory=lambda: ["matrix_profile"])
    adapters: list[str] | None = None  # Compatibility alias for models
    weights: list[float] | dict[str, float] | None = None
    aggregation: str = "mean"
    model_name: str = "foundation_ensemble"

    def __post_init__(self) -> None:
        """Handle compatibility aliases."""
        # If adapters is provided and models is default, use adapters
        if self.adapters is not None and self.models == ["matrix_profile"]:
            self.models = list(self.adapters)
        # Convert dict weights to list if needed
        if isinstance(self.weights, dict):
            self.weights = [self.weights.get(m, 1.0) for m in self.models]


class FoundationEnsemble(BaseFoundationModel):
    """Ensemble of foundation models for robust anomaly detection.

    Combines predictions from multiple foundation models using
    various aggregation strategies.

    Features:
        - Automatic model initialization
        - Flexible aggregation methods
        - Weighted voting for refined predictions
        - Graceful degradation if models fail

    Example:
        >>> ensemble = FoundationEnsemble(
        ...     models=['timegpt', 'chronos', 'matrix_profile']
        ... )
        >>> results = ensemble.detect_anomalies(time_series)
    """

    AVAILABLE_MODELS = {
        "timegpt": (TimeGPTAdapter, TimeGPTConfig),
        "chronos": (ChronosAdapter, ChronosConfig),
        "matrix_profile": (MatrixProfileDetector, MatrixProfileConfig),
    }

    def __init__(self, config: EnsembleConfig | dict[str, Any] | None = None) -> None:
        """Initialize ensemble.

        Args:
            config: Ensemble configuration
        """
        if config is None:
            config = EnsembleConfig()
        elif isinstance(config, dict):
            config = EnsembleConfig(**config)

        # Set typed config BEFORE calling super().__init__() to avoid AttributeError
        # when the base class accesses self.config property
        self.ensemble_config: EnsembleConfig = config
        super().__init__(config)

        # Initialize weights
        n_models = len(config.models)
        if config.weights is None:
            self._weights = np.ones(n_models) / n_models
        else:
            self._weights = np.array(config.weights)
            self._weights = self._weights / self._weights.sum()

        self._models: dict[str, BaseFoundationModel] = {}

    @property
    def config(self) -> EnsembleConfig:
        """Return the typed Ensemble configuration."""
        return self.ensemble_config

    @config.setter
    def config(self, value: dict[str, Any] | EnsembleConfig) -> None:
        """Set config (required for base class compatibility).

        The base class sets self.config to a dict during __init__.
        We intercept this and store it, but always return the typed config.
        """
        if isinstance(value, EnsembleConfig):
            self.ensemble_config = value
        # If dict, it's from base class init - we already have typed config set

    def _initialize_model(self) -> None:
        """Initialize all ensemble models."""
        for model_name in self.ensemble_config.models:
            if model_name not in self.AVAILABLE_MODELS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue

            model_class, config_class = self.AVAILABLE_MODELS[model_name]

            try:
                model_config = config_class(
                    device=self.ensemble_config.device,
                    context_length=self.ensemble_config.context_length,
                    prediction_length=self.ensemble_config.prediction_length,
                )
                model = model_class(model_config)
                self._models[model_name] = model
                logger.info(f"Initialized ensemble member: {model_name}")

            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")

        if not self._models:
            logger.warning("No models initialized, using matrix profile fallback")
            self._models["matrix_profile"] = MatrixProfileDetector()

    def forecast(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Generate ensemble forecasts.

        Args:
            series: Input time series
            horizon: Forecast horizon

        Returns:
            Aggregated forecast dict
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        horizon = horizon or self.foundation_config.prediction_length

        all_forecasts = []
        all_lowers = []
        all_uppers = []
        valid_weights = []

        for _i, (name, model) in enumerate(self._models.items()):
            try:
                result = model.forecast(series, horizon)
                forecast = result["forecast"]

                # Ensure correct shape
                if forecast.ndim == 1:
                    forecast = forecast.reshape(1, -1)

                all_forecasts.append(forecast)
                all_lowers.append(result.get("lower", forecast * 0.9))
                all_uppers.append(result.get("upper", forecast * 1.1))

                # Get weight for this model
                idx = (
                    self.ensemble_config.models.index(name)
                    if name in self.ensemble_config.models
                    else 0
                )
                valid_weights.append(
                    self._weights[idx] if idx < len(self._weights) else 1.0 / len(self._models)
                )

            except Exception as e:
                logger.warning(f"Forecast from {name} failed: {e}")

        if not all_forecasts:
            # Fallback
            return {
                "forecast": np.full((1, horizon), series.flatten()[-1]),
                "lower": np.full((1, horizon), series.flatten()[-1] * 0.9),
                "upper": np.full((1, horizon), series.flatten()[-1] * 1.1),
            }

        # Normalize weights
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        # Aggregate
        forecast = self._aggregate(all_forecasts, valid_weights)
        lower = self._aggregate(all_lowers, valid_weights)
        upper = self._aggregate(all_uppers, valid_weights)

        return {
            "forecast": forecast,
            "lower": lower,
            "upper": upper,
        }

    def detect_anomalies(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies using ensemble.

        Args:
            series: Input time series

        Returns:
            Aggregated detection results
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        all_scores = []
        all_anomalies = []
        valid_weights = []

        for _i, (name, model) in enumerate(self._models.items()):
            try:
                result = model.detect_anomalies(series)
                scores = result["scores"]
                is_anomaly = result["is_anomaly"]

                # Ensure 1D
                if scores.ndim > 1:
                    scores = scores.flatten()
                if is_anomaly.ndim > 1:
                    is_anomaly = is_anomaly.flatten()

                all_scores.append(scores)
                all_anomalies.append(is_anomaly.astype(float))

                idx = (
                    self.ensemble_config.models.index(name)
                    if name in self.ensemble_config.models
                    else 0
                )
                valid_weights.append(
                    self._weights[idx] if idx < len(self._weights) else 1.0 / len(self._models)
                )

            except Exception as e:
                logger.warning(f"Detection from {name} failed: {e}")

        if not all_scores:
            # Fallback
            length = len(series.flatten())
            return {
                "scores": np.zeros(length),
                "is_anomaly": np.zeros(length, dtype=bool),
                "threshold": 0.5,
            }

        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        # Aggregate scores
        scores = self._aggregate_1d(all_scores, valid_weights)

        # Aggregate anomaly decisions
        if self.ensemble_config.aggregation == "vote":
            # Majority voting
            votes = np.stack(all_anomalies).sum(axis=0)
            is_anomaly = votes > len(all_anomalies) / 2
        else:
            # Weighted average then threshold
            anomaly_scores = self._aggregate_1d(all_anomalies, valid_weights)
            is_anomaly = anomaly_scores > 0.5

        return {
            "scores": scores,
            "is_anomaly": is_anomaly,
            "threshold": self.foundation_config.anomaly_threshold,
            "model_results": {
                name: model.detect_anomalies(series) for name, model in self._models.items()
            },
        }

    def detect(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies using ensemble.

        This is the primary detection interface that provides compatibility
        with the expected test interface.

        Args:
            series: Input time series

        Returns:
            Detection results with adapter_scores for compatibility
        """
        result = self.detect_anomalies(series)
        # Add adapter_scores for test compatibility
        result["adapter_scores"] = result.get("model_results", {})
        return result

    def _aggregate(
        self,
        arrays: list[np.ndarray[Any, Any]],
        weights: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Aggregate arrays using specified method.

        Args:
            arrays: List of arrays to aggregate
            weights: Model weights

        Returns:
            Aggregated array
        """
        method = self.ensemble_config.aggregation

        # Ensure same shape
        shapes = [a.shape for a in arrays]
        if len(set(shapes)) > 1:
            # Find min shape and truncate
            min_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
            arrays = [a[tuple(slice(0, s) for s in min_shape)] for a in arrays]

        stacked = np.stack(arrays)

        if method == "mean":
            return np.average(stacked, axis=0, weights=weights)
        elif method == "max":
            return np.max(stacked, axis=0)
        elif method == "weighted":
            return np.average(stacked, axis=0, weights=weights)
        else:
            return np.mean(stacked, axis=0)

    def _aggregate_1d(
        self,
        arrays: list[np.ndarray[Any, Any]],
        weights: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Aggregate 1D arrays with length matching.

        Args:
            arrays: List of 1D arrays
            weights: Model weights

        Returns:
            Aggregated 1D array
        """
        # Find common length
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]

        return self._aggregate(arrays, weights)

    def add_model(
        self,
        name: str,
        model: BaseFoundationModel,
        weight: float = 1.0,
    ) -> None:
        """Add a model to the ensemble.

        Args:
            name: Model identifier
            model: Foundation model instance
            weight: Model weight
        """
        self._models[name] = model
        self.ensemble_config.models.append(name)

        # Update weights
        current_weights = list(self._weights)
        current_weights.append(weight)
        self._weights = np.array(current_weights)
        self._weights = self._weights / self._weights.sum()

    def get_model_weights(self) -> dict[str, float]:
        """Get current model weights.

        Returns:
            Dict mapping model names to weights
        """
        result = {}
        for i, name in enumerate(self.ensemble_config.models):
            if name in self._models:
                result[name] = float(self._weights[i]) if i < len(self._weights) else 0.0
        return result

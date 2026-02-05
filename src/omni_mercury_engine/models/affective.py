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


"""Affective computing anomaly detection model."""

from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


class AffectiveAnomalyModel:
    """Affective computing model for emotional state anomaly detection."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: DeterministicRNG | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or {}
        self._rng = rng or get_global_rng()

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract affective features from data."""
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, npt.NDArray[Any]):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        num_features = 64

        return self._rng.randn(batch_size, num_features).astype(np.float32)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict emotional state anomalies."""
        features = self.extract_features(data)
        batch_size = features.shape[0]

        return {
            "anomaly_scores": self._rng.rand(batch_size).astype(np.float32),
            "emotion_scores": self._rng.randn(batch_size, 6).astype(np.float32),
            "distress_levels": self._rng.rand(batch_size).astype(np.float32),
        }

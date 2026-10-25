# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Astrophysical anomaly detection model with black hole physics."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.utils.constants import OmniCodes

# Omni-Code: Omni-Indivisible Guardian (unified protection and integrity)
_ETHICAL_ANCHOR = OmniCodes.OMNI_INDIVISIBLE.code


class AstrophysicalAnomalyModel:
    """Astrophysical anomaly detection using black hole physics and cosmic event modeling."""

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Anomaly score is the mean of three black-hole-metaphor "
            "heuristics (inverse event-horizon distance, tanh of the "
            "synthetic gravitational field, horizon-state grade): a "
            "physics-inspired heuristic with no published threshold and no "
            "labelled corpus to calibrate one; verdicts are experimental."
        ),
    )

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self.mass_equivalent = self.config.get("mass_equivalent", 1.0)
        self.speed_of_light = self.config.get("speed_of_light", 1.0)
        self.gravitational_constant = self.config.get("gravitational_constant", 1.0)

        self.schwarzschild_radius = (
            2 * self.gravitational_constant * self.mass_equivalent / (self.speed_of_light**2)
        )

    def _compute_gravitational_field(self, distance: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute gravitational field strength at each distance."""
        with np.errstate(divide="ignore"):
            field = self.gravitational_constant * self.mass_equivalent / (distance**2)
        return np.where(distance < 1e-6, 1e6, field)

    def _compute_time_dilation(self, distance: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute gravitational time dilation factor at each distance."""
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.sqrt(1 - self.schwarzschild_radius / distance)
        return np.where(distance <= self.schwarzschild_radius, 0.0, factor)

    def _compute_hawking_temperature(self) -> float:
        """Compute Hawking temperature."""
        return float(1.0 / (8 * np.pi * self.mass_equivalent))

    def _compute_event_horizon_distance(self, points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute each row's distance from the event horizon (singularity)."""
        return np.linalg.norm(points, axis=1).astype(np.float64)

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract astrophysical features from data."""
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        distance = self._compute_event_horizon_distance(data)
        radius = self.schwarzschild_radius
        horizon_state = np.select(
            [distance > 10 * radius, distance > 2 * radius, distance > radius],
            [0.0, 0.33, 0.66],
            default=1.0,
        )
        info_density = np.std(data, axis=1) / (np.mean(np.abs(data), axis=1) + 1e-8)
        accretion = np.sum(np.abs(data), axis=1) / data.shape[1]

        features = np.zeros((data.shape[0], 24))
        features[:, 0] = distance
        features[:, 1] = self._compute_gravitational_field(distance)
        features[:, 2] = self._compute_time_dilation(distance)
        features[:, 3] = horizon_state
        features[:, 4] = self._compute_hawking_temperature()
        features[:, 5] = info_density
        features[:, 6] = accretion
        features[:, 7] = radius
        return features.astype(np.float32)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict anomalies using astrophysical models."""
        features = self.extract_features(data)

        distances = features[:, 0]
        grav_fields = features[:, 1]
        time_dilations = features[:, 2]
        horizon_states = features[:, 3]

        distance_anomaly = 1.0 / (distances + 1.0)
        grav_anomaly = np.tanh(grav_fields / 10.0)
        horizon_anomaly = horizon_states

        anomaly_scores = (distance_anomaly + grav_anomaly + horizon_anomaly) / 3.0

        return {
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "event_horizons": np.stack([distances, horizon_states], axis=1).astype(np.float32),
            "gravitational_fields": grav_fields.astype(np.float32),
            "time_dilations": time_dilations.astype(np.float32),
        }

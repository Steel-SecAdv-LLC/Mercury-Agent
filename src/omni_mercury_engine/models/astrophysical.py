# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Astrophysical anomaly detection model with black hole physics."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.models.fitted_reference import FittedReferenceScorer
from omni_mercury_engine.utils.constants import OmniCodes

# Omni-Code: Omni-Indivisible Guardian (unified protection and integrity)
_ETHICAL_ANCHOR = OmniCodes.OMNI_INDIVISIBLE.code


class AstrophysicalAnomalyModel(FittedReferenceScorer):
    """Astrophysical feature mapping scored as deviation from a fitted reference.

    The pre-2026-10-27 score was the mean of three fixed-direction
    black-hole-metaphor heuristics whose "distance from the event horizon"
    was simply the row norm - so a high-energy burst read as *far from the
    singularity* and scored as maximally safe (measured AUROC 0.000 against
    clean windows on a planted 12-sigma burst). The black-hole feature
    mapping is kept as the representation; the anomaly score is now the
    calibrated deviation of those features from a reference fitted on
    normal data (:class:`FittedReferenceScorer`), which rises for any
    departure - high- or low-energy - by construction.
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Calibrated robust deviation of black-hole-metaphor features "
            "(row norm, synthetic gravitational field, time dilation, "
            "horizon grade, info density, accretion) from a reference "
            "fitted on normal data; direction-free by construction; no "
            "labelled astrophysical corpus backs a domain tier yet, so "
            "verdicts remain experimental."
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
        """Score rows as calibrated deviation from the fitted reference.

        Raises:
            RuntimeError: Called before :meth:`fit` - the retired
                fixed-direction heuristic is refused, never served.
        """
        features = self.extract_features(data)
        scores = self.score_features(features.astype(np.float64))

        distances = features[:, 0]
        grav_fields = features[:, 1]
        time_dilations = features[:, 2]
        horizon_states = features[:, 3]

        return {
            "anomaly_scores": scores["anomaly_scores"],
            "reference_deviation": scores["reference_deviation"],
            "confidence": scores["confidence"],
            "anomaly_score_basis": self.decision_source.description,
            "event_horizons": np.stack([distances, horizon_states], axis=1).astype(np.float32),
            "gravitational_fields": grav_fields.astype(np.float32),
            "time_dilations": time_dilations.astype(np.float32),
        }

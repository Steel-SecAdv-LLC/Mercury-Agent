# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Affective computing anomaly detection model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng

logger = logging.getLogger(__name__)


class AffectiveAnomalyModel:
    """Affective computing model for emotional state anomaly detection.

    Quarantine status (2026-06-11): no real affective feature extractor is
    implemented. This stub previously emitted *fresh RNG noise per call* as
    its features and anomaly scores, which fed 64 columns of fabricated,
    nondeterministic signal into the fusion feature set at train and serve
    time — repeated ``detect_with_fusion`` calls on identical data moved
    fused probabilities by up to ±0.08, and no checkpoint could reproduce
    the engine that wrote it (ROADMAP row 16). Per the repo's anti-theater
    quarantine pattern (``models/parapsychology.py``,
    ``space/schumann_resonance.py``), it now emits a deterministic *neutral*
    output — zero features, 0.5 (uninformative) scores — with a one-time
    warning, fabricating nothing until a real, measured extractor exists.
    """

    _quarantine_warned = False

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: DeterministicRNG | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self._rng = rng or get_global_rng()

    @classmethod
    def _warn_once(cls) -> None:
        if not cls._quarantine_warned:
            cls._quarantine_warned = True
            logger.warning(
                "AffectiveAnomalyModel has no real feature extractor; emitting "
                "deterministic neutral output (zero features, 0.5 scores) "
                "instead of fabricated noise."
            )

    @staticmethod
    def _batch_size(data: np.ndarray[Any, Any] | dict[str, Any]) -> int:
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return int(data.shape[0])

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract affective features from data (neutral zeros; see class docstring)."""
        self._warn_once()
        return np.zeros((self._batch_size(data), 64), dtype=np.float32)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict emotional state anomalies (neutral prior; see class docstring)."""
        self._warn_once()
        batch_size = self._batch_size(data)
        return {
            "anomaly_scores": np.full(batch_size, 0.5, dtype=np.float32),
            "emotion_scores": np.zeros((batch_size, 6), dtype=np.float32),
            "distress_levels": np.full(batch_size, 0.5, dtype=np.float32),
        }

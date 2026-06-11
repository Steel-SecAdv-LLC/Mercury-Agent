# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Affective computing anomaly detection model.

Honesty note: for generic numeric input this model has no trained affect
network — its features are a *placeholder pseudo-random projection*. They
carry no learned affective signal (the fusion network learns to down-weight
them), and they are documented as such rather than dressed up. What the
projection MUST be is **pure**: a deterministic function of the input bytes
alone, identical across calls, instances, and processes. The previous
implementation drew from a shared, stateful RNG stream, so the same batch
produced different features on every call — one of the root causes of the
fusion checkpoint round-trip drift (ROADMAP v1.7.x deferred item #16).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.utils.rng import DeterministicRNG


def _content_seeded_rng(data: np.ndarray[Any, Any]) -> np.random.Generator:
    """Build a generator seeded from the input's shape, dtype, and bytes.

    Same input -> same generator -> same draws, with no state shared across
    calls, instances, or processes.
    """
    digest = hashlib.sha3_256()
    digest.update(str(data.shape).encode())
    digest.update(str(data.dtype).encode())
    digest.update(np.ascontiguousarray(data).tobytes())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "big"))


class AffectiveAnomalyModel:
    """Affective computing model for emotional state anomaly detection."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: DeterministicRNG | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the instance.

        Args:
            config: Optional configuration mapping.
            rng: Retained for API compatibility. The feature/prediction paths
                no longer consume a shared stateful stream (see module
                docstring); they derive a content-seeded generator per call.
            **kwargs: Ignored extra arguments for constructor compatibility.
        """
        self.config = config or {}
        self._rng = rng

    @staticmethod
    def _as_array(data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Coerce supported inputs to a 2-D array."""
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract affective features from data.

        Returns a deterministic pseudo-random projection keyed on the input
        content (placeholder — no trained affect network; see module
        docstring). Purity contract: same input, same features, everywhere.
        """
        arr = self._as_array(data)
        batch_size = arr.shape[0]
        num_features = 64

        rng = _content_seeded_rng(arr)
        return rng.standard_normal((batch_size, num_features)).astype(np.float32)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict emotional state anomalies.

        Placeholder outputs drawn from the same content-seeded generator
        scheme as :meth:`extract_features` — deterministic per input.
        """
        arr = self._as_array(data)
        features = self.extract_features(arr)
        batch_size = features.shape[0]

        rng = _content_seeded_rng(arr)
        return {
            "anomaly_scores": rng.random(batch_size).astype(np.float32),
            "emotion_scores": rng.standard_normal((batch_size, 6)).astype(np.float32),
            "distress_levels": rng.random(batch_size).astype(np.float32),
        }

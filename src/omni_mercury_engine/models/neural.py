# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neural cognitive anomaly detection model."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class NeuralCognitiveModel:
    """Neural cognitive model for brain activity anomaly detection."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self.memory_capacity = self.config.get("memory_capacity", 100)
        self.memory_buffer = deque[Any](maxlen=self.memory_capacity)

    def _hippocampal_memory(
        self, data: np.ndarray[Any, Any], *, update_memory: bool = True
    ) -> np.ndarray[Any, Any]:
        """Process data through hippocampal memory system.

        Args:
            data: Input batch ``(n, d)`` (1-D input is reshaped).
            update_memory: When True (the ``predict`` memory semantics), the
                batch is committed to ``self.memory_buffer`` so later calls
                see it. When False (the fusion ``extract_features`` contract),
                the computation runs against a snapshot plus the preceding
                rows of *this* batch only and commits nothing — a pure
                function of ``(buffer state, data)``. For a fresh instance the
                two paths produce identical features on the first batch, but
                only the pure path keeps train-time and serve-time fusion
                features in lockstep (ROADMAP v1.7.x deferred item #16).
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        memory_features = np.zeros((batch_size, 16), dtype=np.float32)

        # Bounded working view of the buffer: identical accrual semantics to
        # appending row-by-row, without mutating shared state until (and
        # unless) the batch is committed at the end.
        working: deque[Any] = deque(self.memory_buffer, maxlen=self.memory_capacity)

        for i in range(batch_size):
            pattern = data[i]
            working.append(pattern)

            buffer_array = np.array(list(working))
            similarities = np.dot(buffer_array, pattern) / (
                np.linalg.norm(buffer_array, axis=1) * np.linalg.norm(pattern) + 1e-8
            )
            memory_features[i, :8] = np.histogram(similarities, bins=8)[0].astype(np.float32) / len(
                working
            )
            memory_features[i, 8:] = [
                np.mean(similarities),
                np.std(similarities),
                np.max(similarities),
                np.min(similarities),
                np.median(similarities),
                len(working) / self.memory_capacity,
                np.sum(similarities > 0.7),
                np.sum(similarities < 0.3),
            ]

        if update_memory:
            self.memory_buffer = working

        return memory_features

    def _prefrontal_executive(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process data through prefrontal executive functions."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        executive_features = np.zeros((batch_size, 16), dtype=np.float32)

        for i in range(batch_size):
            pattern = data[i]
            executive_features[i, :8] = [
                np.mean(pattern),
                np.std(pattern),
                np.max(pattern),
                np.min(pattern),
                np.median(pattern),
                np.percentile(pattern, 25),
                np.percentile(pattern, 75),
                np.ptp(pattern),
            ]
            diffs = np.diff(pattern)
            executive_features[i, 8:] = [
                np.mean(diffs),
                np.std(diffs),
                np.max(np.abs(diffs)),
                np.sum(diffs > 0) / len(diffs),
                np.sum(diffs < 0) / len(diffs),
                np.mean(np.abs(diffs)),
                np.sum(np.abs(diffs) > np.std(diffs)),
                len(pattern),
            ]

        return executive_features

    def _amygdala_processing(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process data through amygdala emotional system."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        emotional_features = np.zeros((batch_size, 16), dtype=np.float32)

        for i in range(batch_size):
            pattern = data[i]
            fft_result = np.fft.fft(pattern)
            power_spectrum = np.abs(fft_result) ** 2
            emotional_features[i, :8] = np.histogram(power_spectrum, bins=8)[0].astype(
                np.float32
            ) / len(pattern)
            emotional_features[i, 8:] = [
                np.mean(power_spectrum),
                np.std(power_spectrum),
                np.max(power_spectrum),
                np.sum(power_spectrum > np.mean(power_spectrum)),
                np.mean(np.abs(fft_result)),
                np.std(np.abs(fft_result)),
                np.sum(pattern > 0) / len(pattern),
                np.sum(pattern < 0) / len(pattern),
            ]

        return emotional_features

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract neural cognitive features from data."""
        if isinstance(data, dict):
            data = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Pure path (update_memory=False): fusion features must be a function
        # of the input alone so train-time and serve-time features agree and
        # repeated scoring of the same batch is stable. Memory accrual stays
        # available through predict(), which opts in to committing the batch.
        memory_features = self._hippocampal_memory(data, update_memory=False)
        executive_features = self._prefrontal_executive(data)
        emotional_features = self._amygdala_processing(data)

        return np.concatenate([memory_features, executive_features, emotional_features], axis=1)

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Predict neural cognitive anomalies."""
        if isinstance(data, dict):
            data_array = np.array(next(iter(data.values())))
        elif not isinstance(data, np.ndarray):
            data_array = np.array(data)
        else:
            data_array = data

        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)

        memory_scores = self._hippocampal_memory(data_array)
        executive_scores = self._prefrontal_executive(data_array)
        emotional_scores = self._amygdala_processing(data_array)

        memory_anomaly = np.mean(np.abs(memory_scores - 0.5), axis=1)
        executive_anomaly = np.std(executive_scores, axis=1)
        emotional_anomaly = np.max(np.abs(emotional_scores), axis=1)

        anomaly_scores = (memory_anomaly + executive_anomaly + emotional_anomaly) / 3.0

        return {
            "model_type": "neural",
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "memory_scores": memory_scores.astype(np.float32),
            "executive_scores": executive_scores.astype(np.float32),
            "emotional_scores": emotional_scores.astype(np.float32),
        }

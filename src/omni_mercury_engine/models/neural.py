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

    def reset_state(self) -> None:
        """Clear the transient hippocampal memory buffer.

        The buffer is streaming state: without a reset, identical batches
        score differently call-to-call (each call appends its rows), which
        broke serve-path determinism at the engine's fusion feature
        boundary (defect found 2026-06-11). The engine resets before every
        extraction; direct callers keep the streaming semantics.
        """
        self.memory_buffer.clear()

    def _hippocampal_memory(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process data through hippocampal memory system."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        memory_features = np.zeros((batch_size, 16), dtype=np.float32)

        for i in range(batch_size):
            pattern = data[i]
            self.memory_buffer.append(pattern)

            # Cosine similarity is only defined between equal-length vectors.
            # The streaming buffer can hold patterns of different lengths across
            # calls (variable sequence/batch lengths), so np.array over the whole
            # buffer would build a ragged array and raise on an inhomogeneous
            # shape. Compare only against buffered patterns matching the current
            # pattern's shape; the freshly appended pattern always matches, so
            # this set is non-empty. When every stored pattern shares a length
            # (the common case, and always so on the memory-reset serve path)
            # this is the full buffer and behaviour is unchanged.
            same_shape = [p for p in self.memory_buffer if p.shape == pattern.shape]
            buffer_array = np.array(same_shape)
            similarities = np.dot(buffer_array, pattern) / (
                np.linalg.norm(buffer_array, axis=1) * np.linalg.norm(pattern) + 1e-8
            )
            memory_features[i, :8] = np.histogram(similarities, bins=8)[0].astype(np.float32) / len(
                same_shape
            )
            memory_features[i, 8:] = [
                np.mean(similarities),
                np.std(similarities),
                np.max(similarities),
                np.min(similarities),
                np.median(similarities),
                len(self.memory_buffer) / self.memory_capacity,
                np.sum(similarities > 0.7),
                np.sum(similarities < 0.3),
            ]

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
            if len(pattern) >= 2:
                diffs = np.diff(pattern)
                std_diffs = np.std(diffs)
                executive_features[i, 8:] = [
                    np.mean(diffs),
                    std_diffs,
                    np.max(np.abs(diffs)),
                    np.sum(diffs > 0) / len(diffs),
                    np.sum(diffs < 0) / len(diffs),
                    np.mean(np.abs(diffs)),
                    np.sum(np.abs(diffs) > std_diffs),
                    len(pattern),
                ]
            else:
                # A single-sample pattern (e.g. univariate (N, 1) input) has no
                # finite differences: np.diff is empty, so np.max(|diffs|) would
                # raise on a zero-size reduction and the ratio features would be
                # 0/0 NaN. The derivative-based executive features are undefined
                # here; emit zeros for them while keeping the trailing length
                # feature truthful, so a degenerate pattern yields a finite
                # vector instead of crashing/poisoning the fusion inputs.
                executive_features[i, 8:15] = 0.0
                executive_features[i, 15] = len(pattern)

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

        memory_features = self._hippocampal_memory(data)
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

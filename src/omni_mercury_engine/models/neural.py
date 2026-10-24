# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neural cognitive anomaly detection model."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.provenance import DecisionSource

#: Rows of a batch whose memory windows are scored together. Bounds the
#: ``rows x (rows + window)`` similarity matrix at about 10 MB of float64.
_MEMORY_BLOCK_ROWS = 1024


def _row_histogram(
    values: np.ndarray[Any, Any],
    n_bins: int,
    valid: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Histogram every row of a 2-D array at once.

    Row ``r`` of the result equals ``np.histogram(values[r][valid[r]],
    bins=n_bins)[0]``. This follows numpy's equal-width path step for step
    (outer edges from the row's own minimum and maximum, a constant row
    widened by 0.5 on each side, the last bin closed on the right, and the
    one-ULP index corrections), so no count differs from numpy's.

    One deliberate difference: a row whose entries agree to within rounding
    (the power spectrum of a single spike, which is flat) is too narrow to cut
    into ``n_bins`` bins, and ``np.histogram`` raises ``ValueError`` for it,
    which used to discard the whole batch. Such a row counts every entry in
    the middle bin, the same place a constant row falls.

    Args:
        values: Array of shape ``(rows, n)``.
        n_bins: Number of equal-width bins.
        valid: Optional boolean mask of the entries that belong to each row.

    Returns:
        Integer counts of shape ``(rows, n_bins)``.

    Raises:
        ValueError: When a row's range is not finite, as ``np.histogram``.
    """
    rows = values.shape[0]
    if valid is None:
        low, high = values.min(axis=1), values.max(axis=1)
    else:
        low = np.where(valid, values, np.inf).min(axis=1)
        high = np.where(valid, values, -np.inf).max(axis=1)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("autodetected histogram range is not finite")
    flat = low == high
    low = np.where(flat, low - 0.5, low)
    high = np.where(flat, high + 0.5, high)
    dtype = np.result_type(low, high, values)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.result_type(dtype, np.float64)
    low, high = low.astype(dtype, copy=False), high.astype(dtype, copy=False)

    # numpy.linspace(low, high, n_bins + 1), including its handling of a step
    # that underflows to zero.
    span = high - low
    step = span / n_bins
    ticks = np.arange(n_bins + 1, dtype=dtype)[None, :]
    edges = np.where(
        (step == 0)[:, None],
        ticks / n_bins * span[:, None],
        ticks * step[:, None],
    )
    edges = edges + low[:, None]
    edges[:, -1] = high
    too_narrow = (edges[:, 1:] <= edges[:, :-1]).any(axis=1)
    span = np.where(too_narrow, 1, span)

    safe = values if valid is None else np.where(valid, values, low[:, None])
    safe = np.where(too_narrow[:, None], low[:, None], safe).astype(dtype, copy=False)
    index = (((safe - low[:, None]) / span[:, None]) * n_bins).astype(np.intp)
    index[index == n_bins] -= 1
    index[safe < np.take_along_axis(edges, index, axis=1)] -= 1
    above = (safe >= np.take_along_axis(edges, index + 1, axis=1)) & (index != n_bins - 1)
    index[above] += 1
    index[too_narrow] = n_bins // 2

    index += np.arange(rows)[:, None] * n_bins
    counted = index[valid] if valid is not None else index.ravel()
    return np.bincount(counted, minlength=rows * n_bins).reshape(rows, n_bins)


class NeuralCognitiveModel:
    """Neural cognitive model for brain activity anomaly detection."""

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Anomaly score is the mean of three cognitive-metaphor "
            "heuristics (hippocampal-memory deviation from 0.5, "
            "prefrontal-executive feature std, peak amygdala response): no "
            "published threshold exists for this composite and no labelled "
            "corpus is available to calibrate one; verdicts are "
            "experimental."
        ),
    )

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
        """Process data through hippocampal memory system.

        Each row is compared with the rows in the memory window that ends at
        it: the last ``maxlen`` patterns of the stream, the row itself
        included. Patterns of a different length cannot be compared (cosine
        similarity needs equal lengths); they still occupy window slots but
        are left out of the statistics. The windows of a whole block of rows
        are scored from one similarity matrix.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size, width = data.shape
        memory_features = np.zeros((batch_size, 16), dtype=np.float32)
        if batch_size == 0:
            return memory_features

        window = self.memory_buffer.maxlen or len(self.memory_buffer) + batch_size
        prior = list(self.memory_buffer)
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.dtype(np.float64)
        stream = np.zeros((len(prior) + batch_size, width), dtype=dtype)
        comparable = np.ones(len(stream), dtype=bool)
        for slot, pattern in enumerate(prior):
            if pattern.shape == (width,):
                stream[slot] = pattern
            else:
                comparable[slot] = False
        stream[len(prior) :] = data
        norms = np.linalg.norm(stream, axis=1)
        offsets = np.arange(window)[None, :]

        for start in range(0, batch_size, _MEMORY_BLOCK_ROWS):
            stop = min(start + _MEMORY_BLOCK_ROWS, batch_size)
            first = len(prior) + start  # stream index of the block's first row
            row_index = first + np.arange(stop - start)
            low = max(0, first - (window - 1))
            gram = stream[first : len(prior) + stop] @ stream[low : len(prior) + stop].T

            columns = (row_index - (window - 1) - low)[:, None] + offsets
            valid = (columns >= 0) & comparable[low:][np.maximum(columns, 0)]
            columns = np.maximum(columns, 0)
            similarities = np.take_along_axis(gram, columns, axis=1) / (
                norms[low:][columns] * norms[row_index][:, None] + 1e-8
            )

            seen = valid.sum(axis=1)
            seen_f = seen.astype(similarities.dtype)
            in_window = np.where(valid, similarities, 0.0).astype(np.float64)
            mean = in_window.sum(axis=1) / seen
            spread = np.where(valid, similarities - mean[:, None], 0.0).astype(np.float64)
            ordered = np.sort(np.where(valid, similarities, np.inf), axis=1)
            middle = np.arange(len(seen))
            median = (ordered[middle, (seen - 1) // 2] + ordered[middle, seen // 2]) / 2

            counts = _row_histogram(similarities, 8, valid)
            block = memory_features[start:stop]
            block[:, :8] = counts.astype(np.float32) / seen_f[:, None]
            block[:, 8] = mean
            block[:, 9] = np.sqrt((spread**2).sum(axis=1) / seen)
            block[:, 10] = np.where(valid, similarities, -np.inf).max(axis=1)
            block[:, 11] = np.where(valid, similarities, np.inf).min(axis=1)
            block[:, 12] = median
            block[:, 13] = np.minimum(row_index + 1, window) / self.memory_capacity
            block[:, 14] = (valid & (similarities > 0.7)).sum(axis=1)
            block[:, 15] = (valid & (similarities < 0.3)).sum(axis=1)

        self.memory_buffer.extend(np.array(data[-window:]))
        return memory_features

    def _prefrontal_executive(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process data through prefrontal executive functions."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size, width = data.shape
        executive_features = np.zeros((batch_size, 16), dtype=np.float32)

        executive_features[:, 0] = np.mean(data, axis=1)
        executive_features[:, 1] = np.std(data, axis=1)
        executive_features[:, 2] = np.max(data, axis=1)
        executive_features[:, 3] = np.min(data, axis=1)
        executive_features[:, 4] = np.median(data, axis=1)
        executive_features[:, 5:7] = np.percentile(data, [25, 75], axis=1).T
        executive_features[:, 7] = np.ptp(data, axis=1)
        if width >= 2:
            diffs = np.diff(data, axis=1)
            magnitude = np.abs(diffs)
            std_diffs = np.std(diffs, axis=1)
            executive_features[:, 8] = np.mean(diffs, axis=1)
            executive_features[:, 9] = std_diffs
            executive_features[:, 10] = np.max(magnitude, axis=1)
            executive_features[:, 11] = np.sum(diffs > 0, axis=1) / (width - 1)
            executive_features[:, 12] = np.sum(diffs < 0, axis=1) / (width - 1)
            executive_features[:, 13] = np.mean(magnitude, axis=1)
            executive_features[:, 14] = np.sum(magnitude > std_diffs[:, None], axis=1)
        # A single-sample pattern (e.g. univariate (N, 1) input) has no finite
        # differences: the derivative-based executive features (columns 8-14)
        # are undefined there and stay zero, while the trailing length feature
        # stays truthful, so a degenerate pattern yields a finite vector
        # instead of crashing/poisoning the fusion inputs.
        executive_features[:, 15] = width

        return executive_features

    def _amygdala_processing(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Process data through amygdala emotional system."""
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size, width = data.shape
        emotional_features = np.zeros((batch_size, 16), dtype=np.float32)

        fft_result = np.fft.fft(data, axis=1)
        amplitude = np.abs(fft_result)
        power_spectrum = amplitude**2
        mean_power = np.mean(power_spectrum, axis=1)
        emotional_features[:, :8] = _row_histogram(power_spectrum, 8).astype(np.float32) / width
        emotional_features[:, 8] = mean_power
        emotional_features[:, 9] = np.std(power_spectrum, axis=1)
        emotional_features[:, 10] = np.max(power_spectrum, axis=1)
        emotional_features[:, 11] = np.sum(power_spectrum > mean_power[:, None], axis=1)
        emotional_features[:, 12] = np.mean(amplitude, axis=1)
        emotional_features[:, 13] = np.std(amplitude, axis=1)
        emotional_features[:, 14] = np.sum(data > 0, axis=1) / width
        emotional_features[:, 15] = np.sum(data < 0, axis=1) / width

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

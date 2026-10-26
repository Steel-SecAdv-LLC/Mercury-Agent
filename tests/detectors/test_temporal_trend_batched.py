# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The LEGACY windowed trend scoring matches the point-by-point loop it replaced.

The batch-local windowed z-score is the trend channel's fallback for
detectors restored from legacy checkpoints (no fitted per-feature
references); a fitted detector scores trend against the fitted level (see
tests/detectors/test_base_detector_discrimination.py). These parity tests
drive unfitted detectors, which take the legacy path, and compare raw z:
since the 2026-10 rebuild the channels return raw z and detect() squashes
the combined channels once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors import temporal
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

Array = np.ndarray[Any, Any]


def _reference_trend(data: Array, window_size: int) -> Array:
    """The original per-point implementation."""
    if len(data) < window_size:
        return np.zeros(len(data))
    scores = np.zeros(len(data))
    for i in range(window_size, len(data)):
        window = data[i - window_size : i]
        current = data[i]
        if data.ndim == 1:
            z_score = np.abs((current - np.mean(window)) / (np.std(window) + 1e-6))
        else:
            window_mean = np.mean(window, axis=0)
            window_std = np.std(window, axis=0) + 1e-6
            z_score = np.max(np.abs((current - window_mean) / window_std))
        scores[i] = z_score
    return scores


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape", [(0,), (1,), (9,), (10,), (11,), (300,), (500, 1), (300, 7), (120, 3, 4)]
)
@pytest.mark.parametrize("window_size", [2, 10, 37])
def test_matches_the_point_by_point_scores(
    dtype: Any, shape: tuple[int, ...], window_size: int
) -> None:
    data = np.random.default_rng(sum(shape) + window_size).normal(size=shape).astype(dtype)
    data[len(data) // 2 :] *= 4.0  # a variance shift gives large scores too
    detector = TemporalAnomalyDetector({"window_size": window_size})
    np.testing.assert_allclose(
        detector._detect_trend_anomalies(data),
        _reference_trend(data, window_size),
        rtol=1e-9 if dtype is np.float64 else 1e-4,
        atol=1e-9 if dtype is np.float64 else 1e-5,
    )


def test_constant_and_non_finite_windows() -> None:
    data = np.concatenate([np.full(30, 2.0), [50.0], np.full(5, 2.0), [np.nan], np.ones(30)])
    detector = TemporalAnomalyDetector({"window_size": 8})
    np.testing.assert_allclose(
        detector._detect_trend_anomalies(data),
        _reference_trend(data, 8),
        rtol=1e-9,
        equal_nan=True,
    )


def test_blocking_does_not_change_the_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.random.default_rng(3).normal(size=(400, 5))
    detector = TemporalAnomalyDetector({"window_size": 12})
    whole = detector._detect_trend_anomalies(data)
    monkeypatch.setattr(temporal, "_TREND_BLOCK_ELEMENTS", 12 * 5 * 7)  # 7 points per block
    np.testing.assert_array_equal(detector._detect_trend_anomalies(data), whole)

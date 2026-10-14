# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Moran's I / Geary's C on a sparse kNN graph, and the spatial detector key."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.ml.spatial_autocorrelation import (
    gearys_c,
    knn_weights,
    morans_i,
    spatial_autocorrelation,
)

GRID = np.array([(i, j) for i in range(15) for j in range(15)], dtype=float)


def _dense_reference(values: np.ndarray[Any, Any], w: np.ndarray[Any, Any]) -> tuple[float, float]:
    n = len(values)
    z = values - values.mean()
    i = n / w.sum() * (z @ w @ z) / (z @ z)
    c = (n - 1) / (2 * w.sum()) * np.sum(w * (values[:, None] - values[None, :]) ** 2) / (z @ z)
    return float(i), float(c)


def test_matches_dense_definition() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=len(GRID)) + 0.3 * GRID[:, 0]
    w = knn_weights(GRID, k=6)
    ref_i, ref_c = _dense_reference(values, w.toarray())
    assert morans_i(values, w) == pytest.approx(ref_i, rel=1e-12)
    assert gearys_c(values, w) == pytest.approx(ref_c, rel=1e-12)


def test_weights_are_row_standardised_without_self() -> None:
    w = knn_weights(GRID, k=4)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)
    assert w.diagonal().sum() == 0.0


@pytest.mark.parametrize(
    ("values", "pattern"),
    [
        (((GRID[:, 0] + GRID[:, 1]) % 2).astype(float), "dispersed"),
        (GRID[:, 0].copy(), "clustered"),
    ],
)
def test_patterns_are_recognised(values: np.ndarray[Any, Any], pattern: str) -> None:
    result = spatial_autocorrelation(values, GRID, k=4, n_permutations=99)
    assert result is not None
    assert result.pattern == pattern


def test_random_values_are_not_significant() -> None:
    values = np.random.default_rng(1).normal(size=len(GRID))
    result = spatial_autocorrelation(values, GRID, k=8, n_permutations=199)
    assert result is not None
    assert result.pattern == "random"
    assert abs(result.morans_i - result.expected_i) < 0.1


def test_constant_values_are_undefined() -> None:
    assert spatial_autocorrelation(np.ones(len(GRID)), GRID) is None


def test_too_few_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        knn_weights(np.zeros((2, 2)))


def test_spatial_detector_reports_clustered_anomalies() -> None:
    rng = np.random.default_rng(2)
    train = rng.uniform(0, 10, (400, 2))
    det = SpatialAnomalyDetector().fit(train)
    # A tight remote cluster of anomalies among ordinary points.
    test = np.vstack([rng.uniform(0, 10, (200, 2)), rng.normal(25.0, 0.3, (20, 2))])
    result = det.detect(test)
    auto = result["spatial_autocorrelation"]
    assert auto is not None
    assert auto["pattern"] == "clustered"
    assert auto["n"] == 220


def test_spatial_autocorrelation_can_be_disabled() -> None:
    rng = np.random.default_rng(3)
    det = SpatialAnomalyDetector({"spatial_autocorrelation": False}).fit(rng.normal(size=(50, 2)))
    assert det.detect(rng.normal(size=(30, 2)))["spatial_autocorrelation"] is None

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_vectorised_permutation_test_matches_per_shuffle_loop() -> None:
    """The one-spmm permutation test must reproduce the per-shuffle loop.

    Guards the identity it relies on: with centred ``z``,
    ``sum w_ij (z_i - z_j)^2 = (z^2)'r + (z^2)'c - 2 z'Wz``.
    """
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(300, 2))
    values = rng.normal(size=300) + 0.4 * coords[:, 0]
    result = spatial_autocorrelation(values, coords, k=6, n_permutations=49, seed=11)
    assert result is not None

    w = knn_weights(coords, k=6)
    observed_i, observed_c = morans_i(values, w), gearys_c(values, w)
    shuffle = np.random.default_rng(11)
    z = values - values.mean()
    perm_i = np.empty(49)
    perm_c = np.empty(49)
    for b in range(49):
        shuffled = shuffle.permutation(z)
        perm_i[b] = morans_i(shuffled, w)
        perm_c[b] = gearys_c(shuffled, w)
    assert result.morans_p_clustered == (1 + np.sum(perm_i >= observed_i)) / 50
    assert result.morans_p_dispersed == (1 + np.sum(perm_i <= observed_i)) / 50
    assert result.gearys_p_clustered == (1 + np.sum(perm_c <= observed_c)) / 50


def test_duplicate_points_keep_k_neighbours_without_self() -> None:
    """Zero-distance ties can crowd the self index out of the kNN query."""
    coords = np.vstack([np.zeros((10, 2)), np.random.default_rng(4).normal(size=(50, 2))])
    w = knn_weights(coords, k=4)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)
    assert w.diagonal().sum() == 0.0
    assert (w.getnnz(axis=1) == 4).all()


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


def test_large_batches_estimate_autocorrelation_on_a_bounded_subsample() -> None:
    """Above the limit the kNN graph uses a subsample (its cost was ~O(n^2 d))."""
    rng = np.random.default_rng(4)
    det = SpatialAnomalyDetector().fit(rng.uniform(0, 10, (500, 2)))
    limit = SpatialAnomalyDetector._AUTOCORRELATION_SAMPLE_LIMIT
    n = limit * 2 + 50
    test = np.vstack([rng.uniform(0, 10, (n - 400, 2)), rng.normal(25.0, 0.3, (400, 2))])
    auto = det.detect(test)["spatial_autocorrelation"]
    assert auto is not None
    assert auto["sampled_from"] == n
    assert auto["n"] <= limit
    assert auto["pattern"] == "clustered"  # the remote cluster is still found
    small = det.detect(test[:200])["spatial_autocorrelation"]
    assert small is None or small["sampled_from"] is None

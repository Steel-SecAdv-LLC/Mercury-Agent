# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared leave-one-out kNN distance helper."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from omni_mercury_engine.ml.knn import loo_mean_knn_distance


def test_in_sample_rows_get_leave_one_out_distances() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(200, 3))
    tree = cKDTree(ref)
    in_sample = loo_mean_knn_distance(tree, ref, k=5)
    # Self-match dropped: distances come from real neighbours, never zero.
    assert np.all(in_sample > 0.0)
    # And they equal the naive leave-one-out computation per row.
    all_dists = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    np.fill_diagonal(all_dists, np.inf)
    expected = np.sort(all_dists, axis=1)[:, :5].mean(axis=1)
    np.testing.assert_allclose(in_sample, expected)


def test_k_is_clamped_to_reference_size() -> None:
    ref = np.array([[0.0], [1.0], [2.0]])
    tree = cKDTree(ref)
    scores = loo_mean_knn_distance(tree, np.array([[0.5]]), k=10)
    assert np.all(np.isfinite(scores))  # no inf padding from k > n - 1
    # Clamped to the 2 nearest of the 3 reference rows: 0.0 and 1.0.
    assert scores[0] == np.mean([0.5, 0.5])


def test_matches_naive_reference() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(50, 2))
    tree = cKDTree(ref)
    points = rng.normal(size=(10, 2))
    expected = np.sort(np.linalg.norm(points[:, None, :] - ref[None, :, :], axis=2), axis=1)
    np.testing.assert_allclose(
        loo_mean_knn_distance(tree, points, k=4), expected[:, :4].mean(axis=1)
    )

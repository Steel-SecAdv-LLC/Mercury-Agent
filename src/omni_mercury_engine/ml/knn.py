# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared k-nearest-neighbour scoring primitives (numpy/scipy only)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["loo_mean_knn_distance"]


def loo_mean_knn_distance(tree: Any, points: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
    """Mean distance from each row of *points* to its ``k`` nearest reference rows.

    A row identical to a reference row is not its own neighbour: when the
    nearest distance is exactly zero it is dropped and the next ``k`` are
    used. Scoring the reference set therefore yields leave-one-out
    distances (in-sample scoring stays honest), while rows not in the
    reference are unaffected. ``k`` is clamped to the reference size minus
    one, so a small reference never pads the average with ``inf``.

    Args:
        tree: Fitted :class:`scipy.spatial.cKDTree` over the reference rows.
        points: ``(n, d)`` rows to score, in the tree's coordinate frame.
        k: Neighbours to average over.

    Returns:
        ``(n,)`` array of mean kNN distances (higher = less dense).
    """
    k = max(1, min(int(k), tree.n - 1))
    dist, _ = tree.query(points, k=k + 1)
    dist = np.asarray(dist, dtype=np.float64).reshape(len(points), k + 1)
    self_match = dist[:, 0] <= 0.0
    return np.where(self_match, dist[:, 1:].mean(axis=1), dist[:, :k].mean(axis=1))

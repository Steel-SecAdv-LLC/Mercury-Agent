# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial autocorrelation of values on points: Moran's I and Geary's C.

Answers "are these values spatially clustered?" -- for anomaly scores, a
regional event (neighbouring points anomalous together) versus scattered,
independent glitches. Weights are a sparse, row-standardised k-nearest-
neighbour graph (``O(n k)`` memory, not a dense ``n x n`` matrix), and
significance is a permutation test: the values are shuffled over the fixed
locations ``n_permutations`` times, which assumes nothing about their
distribution.

* Moran's I = ``(n / S0) * z'Wz / z'z`` with ``z`` the centred values and
  ``S0`` the weight total; ``E[I] = -1 / (n - 1)``. Larger = clustered.
* Geary's C = ``(n - 1) / (2 S0) * sum w_ij (x_i - x_j)^2 / z'z``;
  ``E[C] = 1``. Smaller = clustered; it weights local differences, so it
  is the more sensitive of the two to small-scale structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = [
    "SpatialAutocorrelation",
    "gearys_c",
    "knn_weights",
    "morans_i",
    "spatial_autocorrelation",
]


@dataclass(frozen=True)
class SpatialAutocorrelation:
    """Moran's I / Geary's C with one-sided permutation p-values.

    Attributes:
        morans_i: Observed Moran's I.
        expected_i: ``-1 / (n - 1)`` (no autocorrelation).
        morans_p_clustered: P(I_perm >= I) -- evidence of clustering.
        morans_p_dispersed: P(I_perm <= I) -- evidence of dispersion.
        gearys_c: Observed Geary's C (``E[C] = 1``).
        gearys_p_clustered: P(C_perm <= C).
        pattern: ``"clustered"``, ``"dispersed"`` or ``"random"`` from
            Moran's I at ``alpha``.
        n: Number of points.
        k: Neighbours per point.
        n_permutations: Permutations behind the p-values (0 = none).
    """

    morans_i: float
    expected_i: float
    morans_p_clustered: float | None
    morans_p_dispersed: float | None
    gearys_c: float
    gearys_p_clustered: float | None
    pattern: str
    n: int
    k: int
    n_permutations: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "morans_i": self.morans_i,
            "expected_i": self.expected_i,
            "morans_p_clustered": self.morans_p_clustered,
            "morans_p_dispersed": self.morans_p_dispersed,
            "gearys_c": self.gearys_c,
            "gearys_p_clustered": self.gearys_p_clustered,
            "pattern": self.pattern,
            "n": self.n,
            "k": self.k,
            "n_permutations": self.n_permutations,
        }


def knn_weights(coords: np.ndarray[Any, Any], k: int = 8) -> sparse.csr_matrix:
    """Row-standardised k-nearest-neighbour weights (self excluded).

    Args:
        coords: ``(n, d)`` point coordinates.
        k: Neighbours per point (capped at ``n - 1``).

    Returns:
        ``(n, n)`` CSR matrix whose rows each sum to 1.

    Raises:
        ValueError: With fewer than 3 points or non-finite coordinates.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    n = len(pts)
    if n < 3:
        raise ValueError("spatial autocorrelation needs at least 3 points")
    if not np.all(np.isfinite(pts)):
        raise ValueError("coordinates must be finite")
    k = max(1, min(int(k), n - 1))
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    idx = np.asarray(idx).reshape(n, k + 1)
    # Keep each point's first k non-self neighbours. With duplicate points the
    # self index can be crowded out of the query result by other zero-distance
    # ties, leaving k + 1 non-self candidates; the cumulative-count mask then
    # drops the last one, exactly as the former per-row Python loop did.
    non_self = idx != np.arange(n).reshape(-1, 1)
    keep = non_self & (np.cumsum(non_self, axis=1) <= k)
    rows = np.repeat(np.arange(n), keep.sum(axis=1))
    cols = idx[keep]
    data = np.full(len(rows), 1.0 / k)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def morans_i(values: np.ndarray[Any, Any], weights: sparse.spmatrix) -> float:
    """Moran's I of *values* under *weights* (``nan`` for constant values)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    z = x - x.mean()
    denom = float(z @ z)
    s0 = float(weights.sum())
    if denom <= 0.0 or s0 <= 0.0:
        return float("nan")
    return float(len(x) / s0 * (z @ (weights @ z)) / denom)


def gearys_c(values: np.ndarray[Any, Any], weights: sparse.spmatrix) -> float:
    """Geary's C of *values* under *weights* (``nan`` for constant values)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    z = x - x.mean()
    denom = float(z @ z)
    coo = sparse.coo_matrix(weights)
    s0 = float(coo.data.sum())
    if denom <= 0.0 or s0 <= 0.0:
        return float("nan")
    local = float(np.sum(coo.data * (x[coo.row] - x[coo.col]) ** 2))
    return float((len(x) - 1) / (2.0 * s0) * local / denom)


def spatial_autocorrelation(
    values: np.ndarray[Any, Any],
    coords: np.ndarray[Any, Any],
    k: int = 8,
    n_permutations: int = 199,
    alpha: float = 0.05,
    seed: int = 0,
) -> SpatialAutocorrelation | None:
    """Moran's I and Geary's C of *values* at *coords*, with permutation tests.

    Args:
        values: One value per point (e.g. anomaly scores).
        coords: ``(n, d)`` coordinates of the points.
        k: Neighbours per point.
        n_permutations: Shuffles for the p-values (0 skips the test).
        alpha: Significance level for :attr:`SpatialAutocorrelation.pattern`.
        seed: Seed of the permutation generator (deterministic).

    Returns:
        The statistics, or ``None`` when the values are constant (both
        statistics are undefined).
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = knn_weights(coords, k)
    if len(x) != weights.shape[0]:
        raise ValueError(f"{len(x)} values for {weights.shape[0]} points")
    observed_i = morans_i(x, weights)
    observed_c = gearys_c(x, weights)
    if not (np.isfinite(observed_i) and np.isfinite(observed_c)):
        return None
    n = len(x)
    p_high = p_low = p_c = None
    pattern = "random"
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        # One sparse product covers every permutation of both statistics:
        # with z centred, z'Wz gives Moran's I, and the Geary numerator is
        # sum w_ij (z_i - z_j)^2 = (z^2)'r + (z^2)'c - 2 z'Wz with r/c the
        # row/column weight sums. Permutations preserve the mean, so the
        # rows of Z share one centring and one denominator z'z.
        z = x - x.mean()
        # (n, B) with one permutation per column: contiguous for the spmm.
        permuted = np.stack([rng.permutation(z) for _ in range(n_permutations)], axis=1)
        denom = float(z @ z)
        s0 = float(weights.sum())
        row_sums = np.asarray(weights.sum(axis=1)).reshape(-1)
        col_sums = np.asarray(weights.sum(axis=0)).reshape(-1)
        quad = np.einsum("nb,nb->b", permuted, weights @ permuted)
        sum_sq = permuted**2
        geary_num = row_sums @ sum_sq + col_sums @ sum_sq - 2.0 * quad
        perm_i = n / s0 * quad / denom
        perm_c = (n - 1) / (2.0 * s0) * geary_num / denom
        p_high = float((1 + np.sum(perm_i >= observed_i)) / (n_permutations + 1))
        p_low = float((1 + np.sum(perm_i <= observed_i)) / (n_permutations + 1))
        p_c = float((1 + np.sum(perm_c <= observed_c)) / (n_permutations + 1))
        if p_high < alpha:
            pattern = "clustered"
        elif p_low < alpha:
            pattern = "dispersed"
    return SpatialAutocorrelation(
        morans_i=observed_i,
        expected_i=-1.0 / (n - 1),
        morans_p_clustered=p_high,
        morans_p_dispersed=p_low,
        gearys_c=observed_c,
        gearys_p_clustered=p_c,
        pattern=pattern,
        n=n,
        k=int(weights.getnnz(axis=1).max()),
        n_permutations=int(n_permutations),
    )

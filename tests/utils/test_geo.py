# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the canonical geospatial kernel (utils/geo.py).

Several cases reproduce defects measured in an external geospatial
detector during the 2026-06-10 audit (flat-degree distances, fragmenting
cluster growth) and assert the corrected behavior; the rest lock the
chunked implementations to brute-force ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.utils.geo import (
    EARTH_RADIUS_KM,
    dbscan_geo,
    haversine_km,
    haversine_km_to_point,
    neighbor_counts_within_km,
    pairwise_haversine_km,
)


def _brute_force_counts(
    lats: np.ndarray,
    lons: np.ndarray,
    radius_km: float,
    times_s: np.ndarray | None = None,
    window_s: float | None = None,
) -> np.ndarray:
    """Reference O(n^2) scalar neighbor counting."""
    n = len(lats)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if times_s is not None and abs(times_s[i] - times_s[j]) > window_s:
                continue
            if haversine_km(lats[i], lons[i], lats[j], lons[j]) <= radius_km:
                counts[i] += 1.0
    return counts


class TestHaversine:
    """Scalar and vectorized great-circle distances."""

    def test_longitude_distance_at_60n(self) -> None:
        """Audit defect 1: degrees*111 gave 99.9 km for a true 50.0 km span (+100%)."""
        d = haversine_km(60.0, 10.0, 60.0, 10.9)
        assert abs(d - 50.04) < 0.5

    def test_known_distance_paris_to_nyc(self) -> None:
        d = haversine_km(48.8566, 2.3522, 40.7128, -74.0060)
        assert abs(d - 5837.0) < 10.0

    def test_zero_distance(self) -> None:
        assert haversine_km(45.0, 7.0, 45.0, 7.0) == 0.0

    def test_antipodal_is_half_circumference(self) -> None:
        d = haversine_km(0.0, 0.0, 0.0, 180.0)
        assert abs(d - math.pi * EARTH_RADIUS_KM) < 1.0

    def test_pairwise_matches_scalar(self) -> None:
        lats = np.array([60.0, 60.0, 40.0, -33.9])
        lons = np.array([10.0, 10.9, -105.0, 151.2])
        m = pairwise_haversine_km(lats, lons)
        assert m.shape == (4, 4)
        assert np.allclose(np.diag(m), 0.0)
        for i in range(4):
            for j in range(4):
                assert abs(m[i, j] - haversine_km(lats[i], lons[i], lats[j], lons[j])) < 1e-6

    def test_to_point_matches_scalar(self) -> None:
        rng = np.random.default_rng(11)
        lats = rng.uniform(-80, 80, 50)
        lons = rng.uniform(-180, 180, 50)
        d = haversine_km_to_point(lats, lons, 35.5, -97.5)
        for i in range(50):
            assert abs(d[i] - haversine_km(lats[i], lons[i], 35.5, -97.5)) < 1e-6


class TestNeighborCounts:
    """Chunked neighbor counting vs. brute-force ground truth."""

    def test_matches_bruteforce_across_latitudes(self) -> None:
        """Unsorted input spanning two latitude bands exercises the
        argsort + latitude-band pruning path against ground truth."""
        rng = np.random.default_rng(42)
        lats = np.concatenate([62.0 + rng.normal(0, 0.1, 30), 38.0 + rng.normal(0, 0.1, 30)])
        lons = np.concatenate([-150.0 + rng.normal(0, 0.3, 30), -120.0 + rng.normal(0, 0.2, 30)])
        got = neighbor_counts_within_km(lats, lons, 10.0)
        assert np.array_equal(got, _brute_force_counts(lats, lons, 10.0))

    def test_permutation_invariance(self) -> None:
        rng = np.random.default_rng(13)
        lats = 45.0 + rng.normal(0, 0.2, 50)
        lons = 9.0 + rng.normal(0, 0.2, 50)
        base = neighbor_counts_within_km(lats, lons, 15.0)
        perm = rng.permutation(50)
        shuffled = neighbor_counts_within_km(lats[perm], lons[perm], 15.0)
        assert np.array_equal(shuffled, base[perm])

    def test_high_latitude_pair_counted(self) -> None:
        """Two boreal points 9.49 km apart: the old degree pre-filter missed them."""
        lat0 = 62.0
        dlon = 9.5 / (111.32 * math.cos(math.radians(lat0)))
        lats = np.array([lat0, lat0])
        lons = np.array([-150.0, -150.0 + dlon])
        assert haversine_km(lats[0], lons[0], lats[1], lons[1]) < 10.0
        assert neighbor_counts_within_km(lats, lons, 10.0).tolist() == [1.0, 1.0]

    def test_temporal_window(self) -> None:
        rng = np.random.default_rng(7)
        lats = 35.0 + rng.normal(0, 0.3, 40)
        lons = -97.0 + rng.normal(0, 0.3, 40)
        times = rng.uniform(0, 6 * 3600.0, 40)
        got = neighbor_counts_within_km(lats, lons, 50.0, times_s=times, time_window_s=3600.0)
        want = _brute_force_counts(lats, lons, 50.0, times_s=times, window_s=3600.0)
        assert np.array_equal(got, want)

    def test_empty_and_single_point(self) -> None:
        empty = neighbor_counts_within_km(np.array([]), np.array([]), 10.0)
        assert empty.shape == (0,)
        single = neighbor_counts_within_km(np.array([35.0]), np.array([-97.0]), 10.0)
        assert single.tolist() == [0.0]

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            neighbor_counts_within_km(np.array([1.0, 2.0]), np.array([1.0]), 10.0)
        with pytest.raises(ValueError, match="non-negative"):
            neighbor_counts_within_km(np.array([1.0]), np.array([1.0]), -1.0)
        with pytest.raises(ValueError, match="together"):
            neighbor_counts_within_km(
                np.array([1.0]), np.array([1.0]), 10.0, times_s=np.array([0.0])
            )


class TestDbscanGeo:
    """Geographic DBSCAN with kilometre-denominated eps."""

    def test_chain_merges_transitively(self) -> None:
        """Audit defect 2: a 6-point chain (30 km spacing, 40 km radius) was split
        into 3 overlapping fragments with double-assigned points."""
        lats = np.array([40.0 + i * 30.0 / 111.0 for i in range(6)])
        lons = np.full(6, -100.0)
        labels = dbscan_geo(lats, lons, eps_km=40.0, min_samples=2)
        assert labels.tolist() == [0] * 6

    def test_planted_clusters_recovered_no_double_assignment(self) -> None:
        rng = np.random.default_rng(42)

        def blob(center: tuple[float, float], n: int, sigma_km: float) -> list[tuple[float, float]]:
            lat = rng.normal(0.0, sigma_km / 111.0, n)
            lon = rng.normal(0.0, sigma_km / (111.0 * math.cos(math.radians(center[0]))), n)
            return [(center[0] + a, center[1] + b) for a, b in zip(lat, lon)]

        pts: list[tuple[float, float]] = []
        truth: list[int] = []
        for k, c in enumerate([(60.0, 8.0), (61.5, 12.0), (59.0, 14.0)]):
            pts += blob(c, 15, 8.0)
            truth += [k] * 15
        for _ in range(30):
            pts.append((58.5 + rng.uniform(0, 5.4), 6.0 + rng.uniform(0, 10.8)))
            truth.append(-1)

        lats = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        labels = dbscan_geo(lats, lons, eps_km=40.0, min_samples=4)
        assert labels.shape[0] == len(pts)  # exactly one label per point
        found = {int(lb) for lb in labels if lb >= 0}
        assert len(found) == 3
        for k in range(3):  # each planted blob lands in exactly one found cluster
            member = {int(labels[i]) for i in range(len(pts)) if truth[i] == k}
            member.discard(-1)
            assert len(member) == 1

    def test_permutation_invariant_partition(self) -> None:
        """Shuffling the input must yield the same clustering partition
        (cluster ids may be renumbered)."""
        rng = np.random.default_rng(3)
        lats = 45.0 + rng.normal(0, 0.5, 60)
        lons = 9.0 + rng.normal(0, 0.5, 60)
        base = dbscan_geo(lats, lons, eps_km=25.0, min_samples=3)
        perm = rng.permutation(60)
        shuffled = dbscan_geo(lats[perm], lons[perm], eps_km=25.0, min_samples=3)

        def partition(labels: np.ndarray) -> set[frozenset[int]]:
            groups: dict[int, set[int]] = {}
            for idx, lb in enumerate(labels):
                groups.setdefault(int(lb), set()).add(idx)
            noise = groups.pop(-1, set())
            return {frozenset(g) for g in groups.values()} | {frozenset({i}) for i in noise}

        base_perm = base[perm]  # base labels re-indexed into shuffled order
        assert partition(base_perm) == partition(shuffled)

    def test_all_noise_and_empty(self) -> None:
        lats = np.array([0.0, 20.0, 40.0])
        lons = np.array([0.0, 60.0, 120.0])
        assert dbscan_geo(lats, lons, eps_km=10.0, min_samples=2).tolist() == [-1, -1, -1]
        assert dbscan_geo(np.array([]), np.array([]), eps_km=10.0).shape == (0,)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="eps_km"):
            dbscan_geo(np.array([1.0]), np.array([1.0]), eps_km=0.0)
        with pytest.raises(ValueError, match="min_samples"):
            dbscan_geo(np.array([1.0]), np.array([1.0]), eps_km=1.0, min_samples=0)

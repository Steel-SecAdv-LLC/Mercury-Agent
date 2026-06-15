# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity tests for the loader geospatial rewiring.

The tornado and wildfire loaders previously computed neighbor counts,
centroid distances, and spread rates with O(n^2) scalar Python loops and
private haversine copies (Earth radius 6371.0 km).  They now delegate to
the shared kernel in ``omni_mercury_engine.utils.geo`` (IUGG radius
6371.0088 km, vectorized).  These tests pin the rewired methods to
verbatim ports of the original implementations:

- counts must match exactly (integer-valued; the 1.4e-6 relative radius
  difference cannot flip a count except for a pair landing within
  millimetres of the radius boundary),
- continuous outputs must match within the 1.4e-6 relative radius ratio,
- and the one *intended* divergence — the wildfire degree pre-filter
  dropping true neighbours above ~48 deg latitude — is demonstrated
  against brute-force ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.tornado_loader import TornadoLoader
from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader

_RADIUS_RTOL = 3e-6  # IUGG 6371.0088 vs legacy 6371.0


def _legacy_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Verbatim port of the loaders' previous private haversine (R=6371.0)."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _legacy_tornado_clustering(
    timestamps: list[pd.Timestamp],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radius_km: float = 100.0,
) -> np.ndarray:
    """Verbatim port of the original TornadoLoader._compute_temporal_clustering."""
    n = len(timestamps)
    cluster_counts = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return cluster_counts
    epoch_seconds = np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)
    for i in range(n):
        count = 0.0
        for j in range(n):
            if i == j:
                continue
            if abs(epoch_seconds[i] - epoch_seconds[j]) > 3600.0:
                continue
            if (
                _legacy_haversine_km(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                <= radius_km
            ):
                count += 1.0
        cluster_counts[i] = count
    return cluster_counts


def _legacy_wildfire_clustering(
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float = 10.0,
) -> np.ndarray:
    """Verbatim port of the original WildfireLoader._compute_spatial_clustering,
    including the flat degree pre-filter that drops neighbours at high latitude."""
    n = len(lat)
    counts = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return counts
    deg_threshold = radius_km / 111.0 * 1.5
    for i in range(n):
        lat_diff = np.abs(lat - lat[i])
        lon_diff = np.abs(lon - lon[i])
        candidates = np.where((lat_diff <= deg_threshold) & (lon_diff <= deg_threshold))[0]
        count = 0
        for j in candidates:
            if j == i:
                continue
            if _legacy_haversine_km(lat[i], lon[i], lat[j], lon[j]) <= radius_km:
                count += 1
        counts[i] = float(count)
    return counts


def _legacy_wildfire_spread_rate(
    lat: np.ndarray,
    lon: np.ndarray,
    hours: np.ndarray,
) -> np.ndarray:
    """Verbatim port of the original WildfireLoader._compute_spread_rate."""
    n = len(lat)
    rate = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return rate
    for i in range(n - 1):
        min_dist = float("inf")
        min_dt = 0.0
        for j in range(i + 1, min(i + 50, n)):
            dist = _legacy_haversine_km(lat[i], lon[i], lat[j], lon[j])
            if dist < min_dist:
                min_dist = dist
                min_dt = hours[j] - hours[i]
        rate[i] = min_dist / min_dt if min_dt > 0.0 and np.isfinite(min_dist) else 0.0
    return rate


def _bruteforce_counts(lat: np.ndarray, lon: np.ndarray, radius_km: float) -> np.ndarray:
    """Exact neighbor counts with no pre-filter (legacy radius constant)."""
    n = len(lat)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if j != i and _legacy_haversine_km(lat[i], lon[i], lat[j], lon[j]) <= radius_km:
                counts[i] += 1.0
    return counts


class TestTornadoParity:
    """Rewired tornado helpers vs. the original implementations."""

    def test_temporal_clustering_parity(self) -> None:
        rng = np.random.default_rng(19)
        n = 60
        lat = 35.5 + rng.normal(0, 0.8, n)
        lon = -97.5 + rng.normal(0, 0.8, n)
        base = pd.Timestamp("2026-05-20 18:00")
        timestamps = [base + pd.Timedelta(minutes=float(m)) for m in rng.uniform(0, 360, n)]
        got = TornadoLoader._compute_temporal_clustering(timestamps, lat, lon, 100.0)
        want = _legacy_tornado_clustering(timestamps, lat, lon, 100.0)
        assert np.array_equal(got, want)

    def test_geographic_anomaly_parity(self) -> None:
        rng = np.random.default_rng(23)
        lat = 35.5 + rng.normal(0, 3.0, 50)
        lon = -97.5 + rng.normal(0, 5.0, 50)
        got = TornadoLoader._compute_geographic_anomaly(lat, lon)
        want = np.array([_legacy_haversine_km(lat[i], lon[i], 35.5, -97.5) for i in range(50)])
        assert np.allclose(got, want, rtol=_RADIUS_RTOL, atol=1e-9)


class TestWildfireParity:
    """Rewired wildfire helpers vs. the original implementations."""

    def test_spatial_clustering_parity_midlatitude(self) -> None:
        """Below ~48 deg latitude the old pre-filter was harmless: outputs match."""
        rng = np.random.default_rng(31)
        lat = 38.0 + rng.normal(0, 0.05, 80)
        lon = -120.0 + rng.normal(0, 0.05, 80)
        got = WildfireLoader._compute_spatial_clustering(lat, lon, 10.0)
        want = _legacy_wildfire_clustering(lat, lon, 10.0)
        assert np.array_equal(got, want)

    def test_spatial_clustering_boreal_fix(self) -> None:
        """Above ~48 deg latitude the old pre-filter dropped true neighbours;
        the rewired method matches brute-force ground truth instead."""
        rng = np.random.default_rng(3)
        lat = 62.0 + rng.normal(0, 0.05, 80)
        lon = -150.0 + rng.normal(0, 0.2, 80)
        got = WildfireLoader._compute_spatial_clustering(lat, lon, 10.0)
        legacy = _legacy_wildfire_clustering(lat, lon, 10.0)
        truth = _bruteforce_counts(lat, lon, 10.0)
        assert np.array_equal(got, truth)
        assert np.sum(truth - legacy) > 0  # the legacy undercount is real
        assert np.all(got >= legacy)

    def test_spread_rate_parity(self) -> None:
        rng = np.random.default_rng(37)
        n = 120
        lat = 38.0 + np.cumsum(rng.normal(0, 0.01, n))
        lon = -120.0 + np.cumsum(rng.normal(0, 0.01, n))
        hours = np.sort(rng.uniform(0, 48.0, n))
        got = WildfireLoader._compute_spread_rate(lat, lon, hours)
        want = _legacy_wildfire_spread_rate(lat, lon, hours)
        assert np.allclose(got, want, rtol=_RADIUS_RTOL, atol=1e-9)

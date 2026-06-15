# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical geospatial kernel: great-circle distances, neighbor counting, clustering.

This module is the single source of truth for latitude/longitude geometry in
Mercury Agent.  Before it existed, three loaders carried private haversine
copies (tornado, wildfire, hurricane) and counted spatial neighbors with
O(n^2) scalar Python loops; the wildfire loader's degree-based pre-filter
additionally undercounted true neighbors by ~20% at boreal latitudes
(|lat| > ~48 deg), where one degree of longitude spans far less than the
111 km the filter assumed.

Design notes:
    - All public functions take latitude/longitude in decimal degrees and
      return kilometres.
    - Distances use the haversine formula in its ``atan2`` form, which is
      numerically stable for both nearby and antipodal points (the ``asin``
      form loses precision near the antipode and requires clamping).
    - The Earth radius is the IUGG mean radius (6371.0088 km).  The loaders
      previously used 6371.0 km; the relative difference is 1.4e-6 (~14 m
      over 10,000 km), far below the 375 m resolution of a VIIRS pixel.
    - ``neighbor_counts_within_km`` and ``dbscan_geo`` prune candidate pairs
      with an exact latitude band (central angle >= |dlat| on the sphere),
      so they never materialize the full ``n * n`` distance matrix and never
      drop a true neighbor.

Dependencies: numpy only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

EARTH_RADIUS_KM = 6371.0088
"""IUGG mean Earth radius in kilometres."""


def _central_angle_rad(
    lat1: np.ndarray[Any, Any],
    lon1: np.ndarray[Any, Any],
    lat2: np.ndarray[Any, Any],
    lon2: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Haversine central angle in radians for radian inputs (broadcasting).

    Uses the ``atan2`` formulation, which stays well-conditioned at zero
    and antipodal separations.

    Args:
        lat1: Latitudes of the first point set, in radians.
        lon1: Longitudes of the first point set, in radians.
        lat2: Latitudes of the second point set, in radians.
        lon2: Longitudes of the second point set, in radians.

    Returns:
        Central angle(s) in radians, broadcast over the inputs.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # Floating-point round-off can push ``a`` infinitesimally outside [0, 1].
    a = np.clip(a, 0.0, 1.0)
    return np.asarray(2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)), dtype=np.float64)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    angle = _central_angle_rad(
        np.radians(np.float64(lat1)),
        np.radians(np.float64(lon1)),
        np.radians(np.float64(lat2)),
        np.radians(np.float64(lon2)),
    )
    return float(EARTH_RADIUS_KM * angle)


def haversine_km_to_point(
    lats: np.ndarray[Any, Any],
    lons: np.ndarray[Any, Any],
    lat0: float,
    lon0: float,
) -> np.ndarray[Any, Any]:
    """Vectorized great-circle distance from many points to one reference point.

    Args:
        lats: 1-D array of latitudes in decimal degrees.
        lons: 1-D array of longitudes in decimal degrees.
        lat0: Reference latitude in decimal degrees.
        lon0: Reference longitude in decimal degrees.

    Returns:
        1-D float64 array of distances in kilometres, one per input point.
    """
    lats_r = np.radians(np.asarray(lats, dtype=np.float64))
    lons_r = np.radians(np.asarray(lons, dtype=np.float64))
    angle = _central_angle_rad(
        lats_r,
        lons_r,
        np.radians(np.float64(lat0)),
        np.radians(np.float64(lon0)),
    )
    return np.asarray(EARTH_RADIUS_KM * angle, dtype=np.float64)


def pairwise_haversine_km(
    lats: np.ndarray[Any, Any],
    lons: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Symmetric pairwise great-circle distance matrix in kilometres.

    Materializes the full ``(n, n)`` matrix; for neighbor queries over
    large point sets use :func:`neighbor_counts_within_km` or
    :func:`dbscan_geo`, which prune candidates with an exact latitude band
    instead of building the full matrix.

    Args:
        lats: 1-D array of latitudes in decimal degrees.
        lons: 1-D array of longitudes in decimal degrees.

    Returns:
        ``(n, n)`` float64 array of distances in kilometres.
    """
    lats_r = np.radians(np.asarray(lats, dtype=np.float64)).reshape(-1, 1)
    lons_r = np.radians(np.asarray(lons, dtype=np.float64)).reshape(-1, 1)
    angle = _central_angle_rad(lats_r, lons_r, lats_r.T, lons_r.T)
    return np.asarray(EARTH_RADIUS_KM * angle, dtype=np.float64)


def _validate_lat_lon(
    lats: np.ndarray[Any, Any],
    lons: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Coerce and validate paired latitude/longitude arrays.

    Args:
        lats: 1-D array-like of latitudes in decimal degrees.
        lons: 1-D array-like of longitudes in decimal degrees.

    Returns:
        Tuple of float64 arrays ``(lats, lons)``.

    Raises:
        ValueError: If the arrays are not 1-D or differ in length.
    """
    lats_arr = np.asarray(lats, dtype=np.float64)
    lons_arr = np.asarray(lons, dtype=np.float64)
    if lats_arr.ndim != 1 or lons_arr.ndim != 1:
        raise ValueError("lats and lons must be 1-D arrays")
    if lats_arr.shape[0] != lons_arr.shape[0]:
        raise ValueError(
            f"lats and lons must have equal length, got {lats_arr.shape[0]} and {lons_arr.shape[0]}"
        )
    return lats_arr, lons_arr


def _latitude_band_bounds(
    sorted_lats_deg: np.ndarray[Any, Any],
    radius_km: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Per-point candidate index ranges in a latitude-sorted point set.

    On the sphere the central angle between two points is at least their
    latitude separation, so a pair with ``R * |dlat| > radius`` provably
    cannot lie within ``radius``.  Restricting each point's candidates to
    the latitude band ``+- radius / R`` is therefore exact — unlike the
    flat ``radius / 111 * 1.5`` degree box it replaces, which also bounded
    longitude and silently dropped true neighbors above ~48 deg latitude.

    Args:
        sorted_lats_deg: Latitudes in decimal degrees, ascending.
        radius_km: Neighbor radius in kilometres.

    Returns:
        ``(lo, hi)`` int arrays: candidate half-open index ranges per point.
    """
    # 1e-9 deg (~0.1 mm) absorbs round-off so an exactly-at-radius
    # meridional pair stays inside the band.
    dlat_max_deg = math.degrees(radius_km / EARTH_RADIUS_KM) + 1e-9
    lo = np.searchsorted(sorted_lats_deg, sorted_lats_deg - dlat_max_deg, side="left")
    hi = np.searchsorted(sorted_lats_deg, sorted_lats_deg + dlat_max_deg, side="right")
    return lo, hi


def neighbor_counts_within_km(
    lats: np.ndarray[Any, Any],
    lons: np.ndarray[Any, Any],
    radius_km: float,
    times_s: np.ndarray[Any, Any] | None = None,
    time_window_s: float | None = None,
) -> np.ndarray[Any, Any]:
    """Count, per point, the other points within a great-circle radius.

    Optionally restricts neighbors to those within a temporal window, which
    covers the "same hour and within R km" co-occurrence counting used by
    the storm loaders.  Candidates are pruned with an exact latitude band
    (see :func:`_latitude_band_bounds`) and then measured with the exact
    haversine distance, so counts are correct at every latitude while the
    typical cost stays far below the full ``n^2`` distance matrix.

    Args:
        lats: 1-D array of latitudes in decimal degrees.
        lons: 1-D array of longitudes in decimal degrees.
        radius_km: Inclusive neighbor radius in kilometres.
        times_s: Optional 1-D array of timestamps in seconds.  When given,
            ``time_window_s`` must also be given.
        time_window_s: Optional inclusive co-occurrence window in seconds.

    Returns:
        1-D float64 array of neighbor counts (self excluded).

    Raises:
        ValueError: On malformed inputs, a negative radius or time window, or
            ``times_s``/``time_window_s`` given without the other.
    """
    lats_arr, lons_arr = _validate_lat_lon(lats, lons)
    if radius_km < 0.0:
        raise ValueError(f"radius_km must be non-negative, got {radius_km}")
    if (times_s is None) != (time_window_s is None):
        raise ValueError("times_s and time_window_s must be provided together")
    if time_window_s is not None and time_window_s < 0.0:
        # A negative window would exclude even the zero-delta self-match, so the
        # ``- 1.0`` self-subtraction below would drive counts negative (e.g. -1).
        raise ValueError(f"time_window_s must be non-negative, got {time_window_s}")

    n = lats_arr.shape[0]
    counts = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return counts

    times_arr: np.ndarray[Any, Any] | None = None
    if times_s is not None:
        times_arr = np.asarray(times_s, dtype=np.float64)
        if times_arr.ndim != 1 or times_arr.shape[0] != n:
            raise ValueError("times_s must be a 1-D array matching lats/lons in length")

    order = np.argsort(lats_arr, kind="stable")
    slat = lats_arr[order]
    slon = lons_arr[order]
    stimes = times_arr[order] if times_arr is not None else None
    slat_r = np.radians(slat)
    slon_r = np.radians(slon)
    lo, hi = _latitude_band_bounds(slat, radius_km)

    sorted_counts = np.zeros(n, dtype=np.float64)
    for i in range(n):
        band = slice(int(lo[i]), int(hi[i]))
        dist = EARTH_RADIUS_KM * _central_angle_rad(
            slat_r[i], slon_r[i], slat_r[band], slon_r[band]
        )
        within = dist <= radius_km
        if stimes is not None and time_window_s is not None:
            within &= np.abs(stimes[i] - stimes[band]) <= time_window_s
        # Each point trivially matches itself (zero distance, zero time
        # delta), so subtracting one removes exactly the self-match.
        sorted_counts[i] = float(np.count_nonzero(within)) - 1.0

    counts[order] = sorted_counts
    return counts


def dbscan_geo(
    lats: np.ndarray[Any, Any],
    lons: np.ndarray[Any, Any],
    eps_km: float,
    min_samples: int = 3,
) -> np.ndarray[Any, Any]:
    """DBSCAN over geographic points with a kilometre-denominated radius.

    Standard region-growing DBSCAN (Ester et al., 1996): cluster expansion
    is transitive, every point receives exactly one label, and border
    points join the first cluster that reaches them.  Neighborhoods use
    exact great-circle distances (latitude-band pruned, see
    :func:`_latitude_band_bounds`), so a single ``eps_km`` is meaningful at
    every latitude — unlike clustering raw degrees with a euclidean metric,
    where the longitude scale shrinks with ``cos(lat)``.

    The native :class:`omni_mercury_engine.ml.mercury_ml.DBSCAN` delegates
    to ``scipy.cdist`` and therefore cannot use a haversine metric; this
    function is the geographic complement, with numpy as its only
    dependency.

    Args:
        lats: 1-D array of latitudes in decimal degrees.
        lons: 1-D array of longitudes in decimal degrees.
        eps_km: Neighborhood radius in kilometres (inclusive).
        min_samples: Minimum neighborhood size (self included) for a core
            point, matching sklearn's ``min_samples`` convention.

    Returns:
        1-D int64 label array: ``0..k-1`` for clusters, ``-1`` for noise.
        Cluster ids are assigned in ascending-latitude seed order.

    Raises:
        ValueError: On malformed inputs or non-positive ``eps_km``.
    """
    lats_arr, lons_arr = _validate_lat_lon(lats, lons)
    if eps_km <= 0.0:
        raise ValueError(f"eps_km must be positive, got {eps_km}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")

    n = lats_arr.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(lats_arr, kind="stable")
    slat = lats_arr[order]
    slon = lons_arr[order]
    slat_r = np.radians(slat)
    slon_r = np.radians(slon)
    lo, hi = _latitude_band_bounds(slat, eps_km)

    neighbors: list[np.ndarray[Any, Any]] = []
    for i in range(n):
        band_lo = int(lo[i])
        dist = EARTH_RADIUS_KM * _central_angle_rad(
            slat_r[i],
            slon_r[i],
            slat_r[band_lo : int(hi[i])],
            slon_r[band_lo : int(hi[i])],
        )
        neighbors.append(band_lo + np.flatnonzero(dist <= eps_km))

    unvisited = -2  # sentinel distinct from the -1 noise label
    sorted_labels = np.full(n, unvisited, dtype=np.int64)
    cluster_id = 0
    for i in range(n):
        if sorted_labels[i] != unvisited:
            continue
        if neighbors[i].shape[0] < min_samples:
            sorted_labels[i] = -1
            continue
        sorted_labels[i] = cluster_id
        queue: deque[int] = deque(neighbors[i].tolist())
        while queue:
            j = queue.popleft()
            if sorted_labels[j] == -1:
                sorted_labels[j] = cluster_id  # noise becomes a border point
            if sorted_labels[j] != unvisited:
                continue
            sorted_labels[j] = cluster_id
            if neighbors[j].shape[0] >= min_samples:  # core point: expand
                # Only points not yet attached to a cluster can still be claimed;
                # ones already labelled (>= 0) would be skipped on dequeue anyway.
                # Filtering here bounds the queue instead of letting it grow with
                # duplicates in dense clusters, with identical resulting labels.
                queue.extend(int(k) for k in neighbors[j] if sorted_labels[k] in (unvisited, -1))
        cluster_id += 1

    labels = np.empty(n, dtype=np.int64)
    labels[order] = sorted_labels
    return labels


__all__ = [
    "EARTH_RADIUS_KM",
    "dbscan_geo",
    "haversine_km",
    "haversine_km_to_point",
    "neighbor_counts_within_km",
    "pairwise_haversine_km",
]

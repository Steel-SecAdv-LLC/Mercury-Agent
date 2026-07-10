# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic skill-score primitives for hazard-detector evaluation.

Implements the standard forecast-verification measures used by the operational
hazard community, with the exact textbook definitions:

* Contingency-table scores -- POD (hit rate), FAR (false-alarm ratio), CSI
  (critical success index / threat score), frequency bias, and the Heidke
  skill score, as defined in Wilks, *Statistical Methods in the Atmospheric
  Sciences* (3rd ed., 2011, ch. 8) and Jolliffe & Stephenson, *Forecast
  Verification* (2nd ed., 2012). The classic worked example is the Finley
  (1884) tornado dataset reproduced in Wilks table 8.3, which the unit tests
  verify digit-for-digit.
* Warning lead time -- event time minus first-alert time, per NWS warning
  verification practice (positive lead = alert preceded the event).
* Magnitude error (MAE + bias) and great-circle location error via the
  haversine formula (IUGG mean Earth radius 6371.0088 km) for seismic use.
* Ordinal classification accuracy (exact and within-one-level) for
  USGS volcanic alert levels, VEI, and NOAA flare classes, plus the Brier
  (1950) probabilistic score.
* Kp skill -- MAE on the 0-9 planetary Kp index and NOAA G-scale bucket
  accuracy (G1 at Kp 5 ... G5 at Kp 9, per the NOAA Space Weather Scales).

Every function is pure numpy, deterministic, and fail-loud: empty, NaN,
mismatched, or out-of-range inputs raise ``ValueError`` instead of silently
producing a number. Undefined ratios (e.g. POD with no observed events) also
raise -- a skill score computed over nothing is fabrication, not a metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

__all__ = [
    "EARTH_RADIUS_KM",
    "G_SCALE_THRESHOLDS",
    "brier_score",
    "contingency_table",
    "critical_success_index",
    "false_alarm_ratio",
    "frequency_bias",
    "g_bucket_accuracy",
    "g_scale_bucket",
    "heidke_skill_score",
    "kp_mae",
    "lead_times",
    "location_error_km",
    "magnitude_error",
    "ordinal_accuracy",
    "probability_of_detection",
    "vei_accuracy",
]

#: IUGG mean Earth radius, km (Moritz, *Geodetic Reference System 1980*).
EARTH_RADIUS_KM = 6371.0088

#: NOAA Space Weather Scales: G-scale onset thresholds on the planetary Kp
#: index (G1 = Kp 5, G2 = Kp 6, G3 = Kp 7, G4 = Kp 8, G5 = Kp 9).
G_SCALE_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (9.0, "G5"),
    (8.0, "G4"),
    (7.0, "G3"),
    (6.0, "G2"),
    (5.0, "G1"),
)


def _as_finite_1d(name: str, values: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Coerce ``values`` to a finite 1-D float array, failing loud otherwise.

    Args:
        name: Argument name used in error messages.
        values: Array-like input.

    Returns:
        1-D ``float64`` array with at least one element and no NaN/inf.

    Raises:
        ValueError: If the input is empty, not 1-D, or contains non-finite
            values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty: a skill score over nothing is not a measurement")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")
    return arr


def _check_counts(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> None:
    """Validate contingency-table counts (non-negative integers, not all zero).

    Args:
        hits: Event forecast and observed (a).
        misses: Event observed but not forecast (c).
        false_alarms: Event forecast but not observed (b).
        correct_negatives: Neither forecast nor observed (d).

    Raises:
        ValueError: If any count is negative, non-integral, or all are zero.
    """
    for label, value in (
        ("hits", hits),
        ("misses", misses),
        ("false_alarms", false_alarms),
        ("correct_negatives", correct_negatives),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{label} must be an integer count, got {value!r}")
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")
    if hits + misses + false_alarms + correct_negatives == 0:
        raise ValueError("contingency table is empty (all counts zero)")


def contingency_table(y_true: Any, y_pred: Any) -> tuple[int, int, int, int]:
    """Build the 2x2 forecast-verification contingency table.

    Uses the standard cell naming (Wilks 2011, section 8.2): ``a`` = hits,
    ``b`` = false alarms, ``c`` = misses, ``d`` = correct negatives.

    Args:
        y_true: Binary observed outcomes (0/1), 1-D.
        y_pred: Binary forecasts (0/1), 1-D, same length.

    Returns:
        Tuple ``(hits, misses, false_alarms, correct_negatives)``.

    Raises:
        ValueError: On empty/NaN input, length mismatch, or non-binary values.
    """
    t = _as_finite_1d("y_true", y_true)
    p = _as_finite_1d("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    for name, arr in (("y_true", t), ("y_pred", p)):
        if not np.isin(arr, (0.0, 1.0)).all():
            raise ValueError(f"{name} must be binary 0/1")
    hits = int(np.sum((t == 1) & (p == 1)))
    misses = int(np.sum((t == 1) & (p == 0)))
    false_alarms = int(np.sum((t == 0) & (p == 1)))
    correct_negatives = int(np.sum((t == 0) & (p == 0)))
    return hits, misses, false_alarms, correct_negatives


def probability_of_detection(hits: int, misses: int) -> float:
    """Probability of detection (hit rate): ``POD = a / (a + c)``.

    The fraction of observed events that were correctly forecast
    (Wilks 2011, eq. 8.12). Range [0, 1], perfect = 1.

    Args:
        hits: Events both forecast and observed.
        misses: Events observed but not forecast.

    Returns:
        POD in [0, 1].

    Raises:
        ValueError: If no events were observed (``a + c == 0``) -- POD is
            undefined then, and returning a default would fabricate skill.
    """
    _check_counts(hits, misses, 0, 1)
    if hits + misses == 0:
        raise ValueError("POD undefined: no observed events (hits + misses == 0)")
    return hits / (hits + misses)


def false_alarm_ratio(hits: int, false_alarms: int) -> float:
    """False-alarm ratio: ``FAR = b / (a + b)``.

    The fraction of forecast events that failed to materialise (Wilks 2011,
    eq. 8.11). Range [0, 1], perfect = 0. Distinct from the probability of
    false detection (false-alarm *rate*), which normalises by non-events.

    Args:
        hits: Events both forecast and observed.
        false_alarms: Events forecast but not observed.

    Returns:
        FAR in [0, 1].

    Raises:
        ValueError: If nothing was forecast (``a + b == 0``) -- FAR is
            undefined for a forecaster that never alerts.
    """
    _check_counts(hits, 0, false_alarms, 1)
    if hits + false_alarms == 0:
        raise ValueError("FAR undefined: no forecast events (hits + false_alarms == 0)")
    return false_alarms / (hits + false_alarms)


def critical_success_index(hits: int, misses: int, false_alarms: int) -> float:
    """Critical success index (threat score): ``CSI = a / (a + b + c)``.

    Hits divided by all cases where the event was forecast and/or observed
    (Wilks 2011, eq. 8.8); correct negatives are ignored, making CSI the
    operational standard for rare hazards. Range [0, 1], perfect = 1.

    Args:
        hits: Events both forecast and observed.
        misses: Events observed but not forecast.
        false_alarms: Events forecast but not observed.

    Returns:
        CSI in [0, 1].

    Raises:
        ValueError: If ``a + b + c == 0`` (no event forecast or observed).
    """
    _check_counts(hits, misses, false_alarms, 1)
    denom = hits + misses + false_alarms
    if denom == 0:
        raise ValueError("CSI undefined: no events forecast or observed")
    return hits / denom


def frequency_bias(hits: int, misses: int, false_alarms: int) -> float:
    """Frequency bias: ``B = (a + b) / (a + c)``.

    Ratio of forecast to observed event counts (Wilks 2011, eq. 8.10):
    ``B > 1`` over-forecasts, ``B < 1`` under-forecasts, perfect = 1.

    Args:
        hits: Events both forecast and observed.
        misses: Events observed but not forecast.
        false_alarms: Events forecast but not observed.

    Returns:
        Frequency bias (non-negative, unbounded above).

    Raises:
        ValueError: If no events were observed (``a + c == 0``).
    """
    _check_counts(hits, misses, false_alarms, 1)
    if hits + misses == 0:
        raise ValueError("frequency bias undefined: no observed events")
    return (hits + false_alarms) / (hits + misses)


def heidke_skill_score(hits: int, misses: int, false_alarms: int, correct_negatives: int) -> float:
    """Heidke skill score: ``HSS = 2(ad - bc) / [(a+c)(c+d) + (a+b)(b+d)]``.

    Proportion-correct skill relative to random chance (Heidke 1926; Wilks
    2011, eq. 8.15). Range (-1, 1]: 0 = no skill over chance, 1 = perfect.
    For the Finley (1884) tornado forecasts (a=28, b=72, c=23, d=2680) the
    literature value is HSS = 0.355, which the unit tests reproduce.

    Args:
        hits: Events both forecast and observed (a).
        misses: Events observed but not forecast (c).
        false_alarms: Events forecast but not observed (b).
        correct_negatives: Neither forecast nor observed (d).

    Returns:
        HSS in (-1, 1].

    Raises:
        ValueError: If the reference-chance denominator is zero (degenerate
            table with only one observed/forecast category).
    """
    _check_counts(hits, misses, false_alarms, correct_negatives)
    a, b, c, d = hits, false_alarms, misses, correct_negatives
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    if denom == 0:
        raise ValueError("HSS undefined: degenerate contingency table")
    return 2.0 * (a * d - b * c) / denom


def lead_times(event_times: Any, first_alert_times: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Warning lead times: ``event_time - first_alert_time``, elementwise.

    Positive lead means the first alert preceded the event (the NWS warning
    lead-time convention); negative means the alert came after the event
    began. Missed events must be filtered out by the caller *and accounted
    for via POD* -- passing NaN here raises, because averaging a NaN lead
    silently would hide a miss.

    Args:
        event_times: Event onset times, 1-D (any consistent unit).
        first_alert_times: First-alert times, 1-D, same length and unit.

    Returns:
        Array of lead times (same unit as the inputs).

    Raises:
        ValueError: On empty, NaN, or mismatched-length input.
    """
    events = _as_finite_1d("event_times", event_times)
    alerts = _as_finite_1d("first_alert_times", first_alert_times)
    if events.shape != alerts.shape:
        raise ValueError(f"length mismatch: events {events.shape} vs alerts {alerts.shape}")
    return np.asarray(events - alerts, dtype=np.float64)


def magnitude_error(y_true: Any, y_pred: Any) -> tuple[float, float]:
    """Mean absolute error and mean (signed) bias of magnitude estimates.

    ``MAE = mean(|pred - true|)``; ``bias = mean(pred - true)`` (positive =
    overestimation). Standard for seismic magnitude verification.

    Args:
        y_true: Observed magnitudes, 1-D.
        y_pred: Estimated magnitudes, 1-D, same length.

    Returns:
        Tuple ``(mae, bias)``.

    Raises:
        ValueError: On empty, NaN, or mismatched-length input.
    """
    t = _as_finite_1d("y_true", y_true)
    p = _as_finite_1d("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    diff = p - t
    return float(np.mean(np.abs(diff))), float(np.mean(diff))


def location_error_km(
    lat_true: Any, lon_true: Any, lat_pred: Any, lon_pred: Any
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Great-circle location error in km via the haversine formula.

    ``d = 2R * asin(sqrt(sin^2(dphi/2) + cos(phi1)cos(phi2)sin^2(dlambda/2)))``
    with ``R`` the IUGG mean Earth radius (6371.0088 km). One degree of
    longitude on the equator is 111.19508 km, which the unit tests verify.

    Args:
        lat_true: Observed latitudes in decimal degrees, 1-D, in [-90, 90].
        lon_true: Observed longitudes in decimal degrees, in [-180, 360).
        lat_pred: Estimated latitudes, same shape and units.
        lon_pred: Estimated longitudes, same shape and units.

    Returns:
        Array of great-circle distances in km.

    Raises:
        ValueError: On empty/NaN input, length mismatch, or latitude outside
            [-90, 90].
    """
    arrays = {
        "lat_true": _as_finite_1d("lat_true", lat_true),
        "lon_true": _as_finite_1d("lon_true", lon_true),
        "lat_pred": _as_finite_1d("lat_pred", lat_pred),
        "lon_pred": _as_finite_1d("lon_pred", lon_pred),
    }
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"coordinate arrays must share one shape, got {shapes}")
    for name in ("lat_true", "lat_pred"):
        if (np.abs(arrays[name]) > 90.0).any():
            raise ValueError(f"{name} outside [-90, 90] degrees")
    phi1, phi2 = np.radians(arrays["lat_true"]), np.radians(arrays["lat_pred"])
    dphi = phi2 - phi1
    dlam = np.radians(arrays["lon_pred"] - arrays["lon_true"])
    h = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return np.asarray(distances, dtype=np.float64)


def ordinal_accuracy(
    y_true: Sequence[str], y_pred: Sequence[str], levels: Sequence[str]
) -> tuple[float, float]:
    """Exact and within-one-level accuracy for an ordered categorical scale.

    Both labels are mapped onto ``levels`` (lowest to highest severity); the
    within-one score credits predictions off by at most one level, the usual
    tolerance for ordinal alert scales (USGS volcano alert levels, NOAA flare
    classes A/B/C/M/X).

    Args:
        y_true: Observed labels.
        y_pred: Predicted labels, same length.
        levels: The full ordered scale; every label must appear here.

    Returns:
        Tuple ``(exact_accuracy, within_one_accuracy)``, each in [0, 1].

    Raises:
        ValueError: On empty input, length mismatch, duplicate levels, or a
            label not present in ``levels``.
    """
    if len(levels) < 2:
        raise ValueError("levels must define an ordered scale of >= 2 categories")
    if len(set(levels)) != len(levels):
        raise ValueError("levels contains duplicates")
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    index = {level: i for i, level in enumerate(levels)}
    try:
        t = np.array([index[label] for label in y_true])
        p = np.array([index[label] for label in y_pred])
    except KeyError as exc:  # fail loud on unknown labels
        raise ValueError(f"label {exc.args[0]!r} not in levels {list(levels)}") from exc
    delta = np.abs(t - p)
    return float(np.mean(delta == 0)), float(np.mean(delta <= 1))


def vei_accuracy(y_true: Any, y_pred: Any) -> tuple[float, float]:
    """Exact and within-one accuracy for Volcanic Explosivity Index estimates.

    VEI (Newhall & Self 1982) is an ordinal 0-8 scale; within-one credit is
    the accepted tolerance given the scale's logarithmic bin width.

    Args:
        y_true: Observed VEI values, 1-D integers in [0, 8].
        y_pred: Estimated VEI values, 1-D integers in [0, 8], same length.

    Returns:
        Tuple ``(exact_accuracy, within_one_accuracy)``.

    Raises:
        ValueError: On empty/NaN input, length mismatch, non-integral values,
            or values outside [0, 8].
    """
    t = _as_finite_1d("y_true", y_true)
    p = _as_finite_1d("y_pred", y_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true {t.shape} vs y_pred {p.shape}")
    for name, arr in (("y_true", t), ("y_pred", p)):
        if not np.array_equal(arr, np.round(arr)):
            raise ValueError(f"{name} must contain integral VEI values")
        if arr.min() < 0 or arr.max() > 8:
            raise ValueError(f"{name} outside the VEI scale [0, 8]")
    delta = np.abs(t - p)
    return float(np.mean(delta == 0)), float(np.mean(delta <= 1))


def brier_score(probabilities: Any, outcomes: Any) -> float:
    """Brier score for binary probabilistic forecasts.

    ``BS = mean((p_i - o_i)^2)`` (Brier 1950; Wilks 2011, eq. 8.36). Range
    [0, 1], perfect = 0; 0.25 equals an uninformative constant-0.5 forecast.

    Args:
        probabilities: Forecast probabilities in [0, 1], 1-D.
        outcomes: Binary observed outcomes (0/1), 1-D, same length.

    Returns:
        Brier score in [0, 1].

    Raises:
        ValueError: On empty/NaN input, length mismatch, probabilities outside
            [0, 1], or non-binary outcomes.
    """
    p = _as_finite_1d("probabilities", probabilities)
    o = _as_finite_1d("outcomes", outcomes)
    if p.shape != o.shape:
        raise ValueError(f"length mismatch: probabilities {p.shape} vs outcomes {o.shape}")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities outside [0, 1]")
    if not np.isin(o, (0.0, 1.0)).all():
        raise ValueError("outcomes must be binary 0/1")
    return float(np.mean((p - o) ** 2))


def kp_mae(kp_true: Any, kp_pred: Any) -> float:
    """Mean absolute error on the planetary Kp index.

    Kp is bounded to [0, 9] (Bartels 1949; GFZ/NOAA convention); inputs
    outside that range indicate a units bug and raise.

    Args:
        kp_true: Observed Kp values, 1-D, in [0, 9].
        kp_pred: Predicted Kp values, 1-D, in [0, 9], same length.

    Returns:
        MAE in Kp units.

    Raises:
        ValueError: On empty/NaN input, length mismatch, or values outside
            [0, 9].
    """
    t = _as_finite_1d("kp_true", kp_true)
    p = _as_finite_1d("kp_pred", kp_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: kp_true {t.shape} vs kp_pred {p.shape}")
    for name, arr in (("kp_true", t), ("kp_pred", p)):
        if arr.min() < 0.0 or arr.max() > 9.0:
            raise ValueError(f"{name} outside the Kp scale [0, 9]")
    mae, _bias = magnitude_error(t, p)
    return mae


def g_scale_bucket(kp: float) -> str:
    """Map a Kp value to its NOAA geomagnetic-storm G-scale bucket.

    Per the NOAA Space Weather Scales: G1 at Kp 5, G2 at Kp 6, G3 at Kp 7,
    G4 at Kp 8, G5 at Kp 9; below Kp 5 is ``G0`` (no storm).

    Args:
        kp: Kp value in [0, 9].

    Returns:
        One of ``"G0"`` ... ``"G5"``.

    Raises:
        ValueError: If ``kp`` is non-finite or outside [0, 9].
    """
    if not np.isfinite(kp):
        raise ValueError("kp must be finite")
    if kp < 0.0 or kp > 9.0:
        raise ValueError(f"kp outside the Kp scale [0, 9]: {kp}")
    for threshold, bucket in G_SCALE_THRESHOLDS:
        if kp >= threshold:
            return bucket
    return "G0"


def g_bucket_accuracy(kp_true: Any, kp_pred: Any) -> float:
    """Fraction of samples whose predicted G-scale bucket matches the observed.

    Args:
        kp_true: Observed Kp values, 1-D, in [0, 9].
        kp_pred: Predicted Kp values, 1-D, in [0, 9], same length.

    Returns:
        Bucket accuracy in [0, 1].

    Raises:
        ValueError: On empty/NaN input, length mismatch, or out-of-range Kp.
    """
    t = _as_finite_1d("kp_true", kp_true)
    p = _as_finite_1d("kp_pred", kp_pred)
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: kp_true {t.shape} vs kp_pred {p.shape}")
    true_buckets = [g_scale_bucket(float(v)) for v in t]
    pred_buckets = [g_scale_bucket(float(v)) for v in p]
    matches = [a == b for a, b in zip(true_buckets, pred_buckets)]
    return float(np.mean(matches))

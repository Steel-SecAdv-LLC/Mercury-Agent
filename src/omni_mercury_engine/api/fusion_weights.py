# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The scoring kernel the fusion routes share: components and weights.

``POST /api/v1/detect/fusion`` and the batch job's ``fusion`` method score
the same components (statistical and temporal z-scores, with dimensional and
spectral available to the synchronous route) and must reach the same verdict
on the same data. They did not, twice over. The synchronous route normalised
0.35/0.25 for statistical+temporal (0.583/0.417) while the batch path hard
coded 0.6/0.4, so eleven zeros followed by a 10 at sensitivity 0.52 was an
anomaly over HTTP and clean as a batch job. And on multi-feature input the
synchronous route standardised every value against one global mean and
standard deviation and differenced the *flattened* matrix (so a "step" ran
from the last column of one row to the first column of the next), while the
batch path standardised per column and differenced row to row; for
``[[0, 1000]] * 11 + [[10, 1000]]`` the batch job flagged (3.442) and the
synchronous route did not (1.418).

Both routes now take their components from here: :func:`as_matrix` shapes
the input as rows of features (a 1-D series is one feature),
:func:`column_statistics` standardises per feature, :func:`statistical_scores`
and :func:`temporal_scores` give each row its largest per-feature |z| and its
largest per-feature step from the previous row in standard-deviation units,
and :func:`fuse` applies :data:`FUSION_WEIGHTS` normalised over the selected
components. ``tests/api/test_fusion_batch_parity.py`` pins both parities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import (
    chi2,
    norm,
    t as student_t,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "COLLECTIVE_WINDOWS",
    "EPSILON",
    "FUSION_WEIGHTS",
    "MAD_TO_SIGMA",
    "MEAN_AD_TO_SIGMA",
    "SCALE_EFFICIENCY",
    "SEASONAL_MIN_CYCLES",
    "SEASONAL_Z",
    "TEMPORAL_STEP_SCALE",
    "TRIM_FRACTION",
    "TRIM_TO_SIGMA",
    "Z_MAX",
    "as_matrix",
    "calibrated_max_z",
    "chi_norm_z",
    "collective_scores",
    "column_statistics",
    "deseasonalize",
    "extreme_tail",
    "fuse",
    "fuse_calibrated",
    "normalized_weights",
    "null_dof",
    "row_components",
    "series_components",
    "spectral_peak_z",
    "statistical_scores",
    "temporal_scores",
]

#: Added to every standard deviation so a constant feature scores 0, not inf.
EPSILON = 1e-8

#: Relative weight of each component before normalisation over the components
#: a request selects. The order is the documented one.
FUSION_WEIGHTS: Mapping[str, float] = {
    "statistical": 0.30,
    "temporal": 0.20,
    "collective": 0.15,
    "dimensional": 0.15,
    "spectral": 0.20,
}

#: Window lengths (rows) the collective component averages over: a spike
#: spread over its neighbours, a ramp, a level shift and a burst each stand
#: out at one of these scales where no single row does.
COLLECTIVE_WINDOWS: tuple[int, ...] = (3, 5, 9, 17, 33, 65)


def normalized_weights(detectors: Iterable[str]) -> dict[str, float]:
    """Weights of the selected components, scaled to sum to one.

    Unknown names are ignored (the routes validate their own detector
    lists); an empty or wholly unknown selection yields an empty mapping, so
    a fused score over it is ``0.0`` rather than a division by zero.
    """
    selected = {d: FUSION_WEIGHTS[d] for d in detectors if d in FUSION_WEIGHTS}
    total = sum(selected.values())
    if total <= 0.0:
        return {}
    return {d: w / total for d, w in selected.items()}


def fuse(scores: Mapping[str, float], detectors: Iterable[str]) -> float:
    """The weighted sum of ``scores`` over ``detectors`` with the shared weights."""
    weights = normalized_weights(detectors)
    return float(sum(scores.get(d, 0.0) * w for d, w in weights.items()))


def as_matrix(data: Any) -> np.ndarray[Any, Any]:
    """``data`` as rows of features: a 1-D series becomes one feature column."""
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"fusion input must be 1-D or 2-D, got {matrix.ndim}-D")
    return matrix


#: Share of the largest absolute deviations the trimmed scale ignores.
TRIM_FRACTION = 0.10
#: Consistency constant of the trimmed scale: the root mean square of a
#: standard normal restricted to its central ``1 - TRIM_FRACTION``.
_TRIM_Q = float(norm.isf(TRIM_FRACTION / 2.0))
TRIM_TO_SIGMA = float(
    np.sqrt(((1.0 - TRIM_FRACTION) - 2.0 * _TRIM_Q * norm.pdf(_TRIM_Q)) / (1.0 - TRIM_FRACTION))
)
#: Scales a median absolute deviation to the standard deviation of a normal.
MAD_TO_SIGMA = 1.4826
#: Scales a mean absolute deviation (about the median) to that standard deviation.
MEAN_AD_TO_SIGMA = 1.2533
#: The trimmed scale's efficiency against the standard deviation on normal
#: data (measured 0.78 to 0.86 for 12 to 600 rows): the Student-t degrees of
#: freedom the null tails use are this share of the rows' own.
SCALE_EFFICIENCY = 0.8


def column_statistics(
    matrix: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Per-feature robust location and scale over the rows.

    The mean and standard deviation were the standardisation before; both
    move with the very anomaly being scored, so a 100 among nine values near
    1 inflated the standard deviation to 30 and scored a z of 3.0, and eleven
    zeros followed by a 10 scored 3.3 (masking). The location is the median.
    The scale is the root mean square of the absolute deviations after the
    largest :data:`TRIM_FRACTION` of them are dropped, scaled to the normal's
    standard deviation: it ignores up to a tenth of the rows (and always at
    least one) however extreme and keeps about 80% of the standard
    deviation's efficiency, where the MAD
    keeps 38% and its sampling error alone pushed the series false-positive
    rate on 30-row noise to 13% at width 1 and 79% at width 64. Where the
    trimmed deviations are all 0 (more than nine tenths of the values
    identical) the scaled MAD stands in, then the scaled mean absolute
    deviation; a constant feature keeps :data:`EPSILON` and scores 0
    everywhere rather than inf.
    """
    location = np.median(matrix, axis=0)
    deviation = np.abs(matrix - location)
    # At least one deviation is always dropped (two or more rows), so a lone
    # spike among a handful of values cannot sit inside its own scale.
    n_rows = matrix.shape[0]
    dropped = max(1, int(np.floor(TRIM_FRACTION * n_rows))) if n_rows >= 2 else 0
    trimmed = np.sort(deviation, axis=0)[: n_rows - dropped]
    scale = np.sqrt(np.mean(trimmed**2, axis=0)) / TRIM_TO_SIGMA
    scale = np.where(scale > EPSILON, scale, MAD_TO_SIGMA * np.median(deviation, axis=0))
    scale = np.where(scale > EPSILON, scale, MEAN_AD_TO_SIGMA * np.mean(deviation, axis=0))
    return location, np.where(scale > EPSILON, scale, 0.0) + EPSILON


def null_dof(n_rows: int) -> float:
    """Degrees of freedom of the null tails for statistics standardised from ``n_rows``.

    A value standardised by an estimated scale is not normal but Student-t
    with the estimator's effective degrees of freedom; the trimmed scale has
    :data:`SCALE_EFFICIENCY` of the standard deviation's ``n - 1``. Measured:
    normal tails over-flagged 30-row noise series 13% to 79% of the time at a
    4.6% threshold; these tails hold it at or below nominal.
    """
    return max(1.0, SCALE_EFFICIENCY * (n_rows - 1))


def statistical_scores(
    matrix: np.ndarray[Any, Any], mean: np.ndarray[Any, Any], std: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Each row's largest per-feature |z| against ``mean`` and ``std``."""
    z = np.abs((matrix - mean) / std)
    return np.asarray(np.max(z, axis=1), dtype=float)


def temporal_scores(
    matrix: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Each row's largest per-feature step from the row before it, in units of ``std``.

    The first row scores 0 unless ``previous`` (the row before this block,
    for chunked scoring) is given. Steps are taken row to row within each
    feature, never across the flattened matrix.
    """
    if matrix.shape[0] == 0:
        return np.zeros(0)
    if previous is None:
        starts = np.vstack([matrix[:1], matrix[:-1]])
        steps = np.asarray(np.max(np.abs(matrix - starts) / std, axis=1), dtype=float)
        steps[0] = 0.0
        return steps
    starts = np.vstack([np.asarray(previous, dtype=float).reshape(1, -1), matrix[:-1]])
    return np.asarray(np.max(np.abs(matrix - starts) / std, axis=1), dtype=float)


# ---------------------------------------------------------------------------
# Null calibration
# ---------------------------------------------------------------------------
#
# Every component above is a *maximum* of standardised values, and the routes
# compare it with one threshold in z units (3.5 at the default sensitivity).
# A maximum grows with the number of values it ranges over: the largest of
# 600 x 64 |z| values from pure noise is about 4.4, the Euclidean norm of a
# 64-feature standardised row averages 8, so a fixed 3.5 called every row of
# a 64-wide noise matrix anomalous (measured: 0.3% false positives at width
# 1, 14% at 8, 72% at 16, 100% from 32 up) and every fusion verdict on noise
# "anomaly". The functions below turn each statistic into the z whose
# two-sided tail equals the probability of seeing a value at least that large
# under the null, given the comparisons it was taken over. The threshold then
# keeps its stated meaning at every width and length: a per-row false-positive
# rate of 2(1 - Phi(t)) for per-row verdicts, and the same rate for the whole
# series when the verdict is series-wide.

#: The smallest two-sided tail the calibration resolves, and its z: a
#: statistic beyond it reads as this z rather than as infinity.
_P_MIN = 1e-300
Z_MAX = float(norm.isf(_P_MIN / 2.0))
#: A step between two independent values of a feature has standard deviation
#: sqrt(2) times the feature's, so a step in feature-std units is scaled by
#: this before it is read as a z.
TEMPORAL_STEP_SCALE = float(np.sqrt(2.0))


def extreme_tail(p_single: Any, m: Any) -> Any:
    """P(the largest of ``m`` independent values exceeds a level each exceeds with ``p_single``).

    ``1 - (1 - p)^m`` evaluated as ``-expm1(m * log1p(-p))`` so a tail of
    1e-12 over a million comparisons is 1e-6, not 0.
    """
    # A single tail of exactly 1 (a statistic of 0) is held just below it so
    # log1p stays finite; the result is still 1 to double precision.
    p = np.clip(np.asarray(p_single, dtype=float), 0.0, 1.0 - 1e-16)
    count = np.maximum(np.asarray(m, dtype=float), 1.0)
    return -np.expm1(count * np.log1p(-p))


def _z_from_tail(p: Any) -> Any:
    """The z whose two-sided tail is ``p``, capped at :data:`Z_MAX`."""
    tail = np.clip(np.asarray(p, dtype=float), _P_MIN, 1.0)
    return np.minimum(norm.isf(tail / 2.0), Z_MAX)


def calibrated_max_z(x: Any, m: Any, dof: float | None = None) -> Any:
    """The z-equivalent of a maximum of ``m`` standardised values observed at ``x``.

    The single-value tail is normal, or Student-t with ``dof`` degrees of
    freedom when the scale was estimated (:func:`null_dof`). Identity when
    ``m`` is 1 and ``dof`` is None (an unsupported ``x`` below 0 reads as 0).
    """
    level = np.maximum(np.asarray(x, dtype=float), 0.0)
    single = 2.0 * (norm.sf(level) if dof is None else student_t.sf(level, dof))
    return _z_from_tail(extreme_tail(single, m))


def chi_norm_z(r: Any, d: Any, m: Any = 1, dof: float | None = None) -> Any:
    """The z-equivalent of the Euclidean norm ``r`` of ``d`` standardised values.

    Under the null the norm is chi-distributed with ``d`` degrees of freedom;
    ``m`` is how many such norms the maximum ranged over. With an estimated
    scale (``dof``) each squared term carries the estimator's variance, so
    the squared norm is read as a scaled chi-square matched in mean and
    variance (``(1 + 3/dof)`` times a chi-square with ``d / (1 + 3/dof)``
    degrees of freedom). For ``d`` = 1 and no ``dof`` this is |z| exactly, so
    a one-feature matrix scores as the univariate route does.
    """
    radius = np.maximum(np.asarray(r, dtype=float), 0.0)
    width = np.asarray(d, dtype=float)
    inflation = 1.0 if dof is None else 1.0 + 3.0 / dof
    single = chi2.sf(radius**2 / inflation, width / inflation)
    return _z_from_tail(extreme_tail(single, m))


def row_components(
    matrix: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
    previous: np.ndarray[Any, Any] | None = None,
    dof: float | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Each row's calibrated statistical, temporal and dimensional components.

    The statistical and temporal maxima range over the ``d`` features of the
    row; the dimensional norm has ``d`` degrees of freedom; ``dof`` is the
    null tails' degrees of freedom for the rows the scale came from
    (:func:`null_dof`). Each is returned as a z-equivalent, so a per-row
    threshold in z units holds the stated false-positive rate at any width.
    The batch job scores rows with this.
    """
    width = matrix.shape[1] if matrix.ndim == 2 else 1
    statistical = calibrated_max_z(statistical_scores(matrix, mean, std), width, dof)
    steps = temporal_scores(matrix, std, previous=previous) / TEMPORAL_STEP_SCALE
    temporal = calibrated_max_z(steps, width, dof)
    norms = np.linalg.norm((matrix - mean) / std, axis=1)
    dimensional = chi_norm_z(norms, width, 1, dof)
    windows, _, comparisons = collective_scores(matrix, mean, std)
    # Each row's window z is the largest over the windows holding it; it is
    # read as a maximum over the windows and features it could have come from.
    per_row_windows = max(1, comparisons // max(1, matrix.shape[0]))
    collective = calibrated_max_z(windows, per_row_windows, dof)
    return {
        "statistical": np.asarray(statistical, dtype=float),
        "temporal": np.asarray(temporal, dtype=float),
        "dimensional": np.asarray(dimensional, dtype=float),
        "collective": np.asarray(collective, dtype=float),
    }


def series_components(
    matrix: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
    spectral_z: Iterable[float] | None = None,
    dof: float | None = None,
) -> dict[str, float]:
    """The whole-series calibrated components the synchronous fusion route fuses.

    The maxima range over every value of the series: ``n x d`` for the
    statistical component, ``(n - 1) x d`` steps for the temporal one, ``n``
    norms of ``d`` degrees of freedom for the dimensional one, and the ``d``
    per-feature spectral z values (each already a calibrated Fisher test) for
    the spectral one. A series verdict against a z threshold therefore has the
    threshold's false-positive rate however long or wide the input.
    """
    n, width = (matrix.shape[0], matrix.shape[1]) if matrix.ndim == 2 else (matrix.shape[0], 1)
    out: dict[str, float] = {}
    if n == 0:
        return {
            "statistical": 0.0,
            "temporal": 0.0,
            "dimensional": 0.0,
            "collective": 0.0,
            "spectral": 0.0,
        }
    out["statistical"] = float(
        calibrated_max_z(np.max(statistical_scores(matrix, mean, std)), n * width, dof)
    )
    if n > 1:
        step = float(np.max(temporal_scores(matrix, std))) / TEMPORAL_STEP_SCALE
        out["temporal"] = float(calibrated_max_z(step, (n - 1) * width, dof))
    else:
        out["temporal"] = 0.0
    norms = np.linalg.norm((matrix - mean) / std, axis=1)
    out["dimensional"] = float(chi_norm_z(np.max(norms), width, n, dof))
    _, series_max, comparisons = collective_scores(matrix, mean, std)
    out["collective"] = float(calibrated_max_z(series_max, comparisons, dof))
    if spectral_z is not None:
        peaks = np.asarray(list(spectral_z), dtype=float)
        out["spectral"] = float(calibrated_max_z(np.max(peaks), peaks.size)) if peaks.size else 0.0
    return out


# ---------------------------------------------------------------------------
# Structure the point components cannot see
# ---------------------------------------------------------------------------


def spectral_peak_z(values: np.ndarray[Any, Any]) -> tuple[float, int]:
    """Fisher's test for a dominant frequency on a standardised 1-D series.

    Returns the two-sided normal z of the test's p-value and the index of the
    strongest periodogram ordinate (1 = one cycle over the series). The
    ordinates of white noise (DC removed) are independent exponentials, so
    the largest of the ``m`` ordinates has the tail ``1 - (1 - exp(-x))^m``
    in units of their mean. Invariant to offset and scale; ``(0.0, 0)`` for
    fewer than four samples or a constant series.
    """
    n = values.size
    m = (n - 1) // 2
    if m < 1:
        return 0.0, 0
    std = float(np.std(values))
    if std == 0.0:
        return 0.0, 0
    ordinates = np.abs(np.fft.fft((values - np.mean(values)) / std)[1 : m + 1]) ** 2
    peak = int(np.argmax(ordinates)) + 1
    x = float(ordinates[peak - 1] / np.mean(ordinates))
    tail = float(extreme_tail(np.exp(-x), m))
    return float(_z_from_tail(tail)), peak


#: A periodic component is removed before the point components score when
#: Fisher's test finds it at least this surprising (z of its p-value) ...
SEASONAL_Z = 5.0
#: ... and it completes at least this many cycles over the series. One or two
#: "cycles" is what a level shift or a ramp looks like to a periodogram
#: (measured: fitting that as a season turned the pre-shift rows anomalous);
#: a season is something that repeats.
SEASONAL_MIN_CYCLES = 3


def deseasonalize(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], list[int]]:
    """Each feature minus its dominant sinusoid, where Fisher's test finds one.

    A spike riding a strong season is one seasonal amplitude above the crest
    it sits on: against the whole series' spread it is a z of 2 and goes
    unflagged (measured: ranked first, recall 0). The sinusoid at the
    periodogram's peak (one cycle per ``n / k`` rows) is fitted by least
    squares with an intercept and subtracted; the residual carries the spike
    at its true size. Features with no dominant frequency (z below
    :data:`SEASONAL_Z`) or whose peak completes fewer than
    :data:`SEASONAL_MIN_CYCLES` cycles (a shift or a trend, not a season) are
    returned unchanged. Returns the residual matrix
    and, per feature, the removed period in rows (0 when nothing was removed).
    """
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    residual = matrix.copy()
    periods: list[int] = []
    t = np.arange(n, dtype=float)
    for j in range(matrix.shape[1]):
        z, k = spectral_peak_z(matrix[:, j])
        if z < SEASONAL_Z or k < SEASONAL_MIN_CYCLES:
            periods.append(0)
            continue
        omega = 2.0 * np.pi * k / n
        design = np.column_stack([np.ones(n), np.sin(omega * t), np.cos(omega * t)])
        coef, *_ = np.linalg.lstsq(design, matrix[:, j], rcond=None)
        residual[:, j] = matrix[:, j] - design @ coef + coef[0]
        periods.append(round(n / k))
    return residual, periods


def collective_scores(
    matrix: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
    std: np.ndarray[Any, Any],
    windows: tuple[int, ...] = COLLECTIVE_WINDOWS,
) -> tuple[np.ndarray[Any, Any], float, int]:
    """Where a stretch of rows departs together: the standardised window mean.

    For every window length ``w`` that fits the series, the mean of the
    standardised values over each window of ``w`` consecutive rows, times
    ``sqrt(w)``, is a z under the null; a spike spread over five rows (each
    3.6 sigma) is 8 sigma at ``w`` = 5, a 4-sigma level shift is 23 sigma at
    ``w`` = 33, a ramp and a burst stand out at their own scale. Each row's
    score is the largest |window z| of the windows centred on it, one per
    length, taken over the row's features; the series statistic is the
    largest of all windows, with the
    number of (window, position, feature) comparisons it ranged over, for
    Šidák calibration (conservative: the windows overlap).

    Returns ``(per_row_scores, series_max, comparisons)``.
    """
    n, width = matrix.shape
    z = (matrix - mean) / std
    per_row = np.zeros(n)
    series_max = 0.0
    comparisons = 0
    for w in windows:
        if w > n or w < 2:
            continue
        kernel = np.ones(w) / np.sqrt(w)
        for j in range(width):
            window_z = np.abs(np.convolve(z[:, j], kernel, mode="valid"))  # n - w + 1 windows
            comparisons += window_z.size
            series_max = max(series_max, float(window_z.max()))
            # Every row inside a window takes the window's z if it is larger.
            spread = _sliding_max(window_z, w, n)
            per_row = np.maximum(per_row, spread)
    return per_row, series_max, max(comparisons, 1)


def _sliding_max(window_z: np.ndarray[Any, Any], w: int, n: int) -> np.ndarray[Any, Any]:
    """Each row's z from the window of length ``w`` centred on it.

    Attributing a window's z to every row it holds smeared a five-row attack
    over the 64 rows of the longest window around it; the centred window keeps
    the attribution where the departure is. Rows within ``w // 2`` of either
    end take the nearest window that fits.
    """
    centres = np.arange(n) - w // 2
    centres = np.clip(centres, 0, window_z.size - 1)
    return np.asarray(window_z[centres], dtype=float)


def fuse_calibrated(
    scores: Mapping[str, float], detectors: Iterable[str], dof: float | None = None
) -> float:
    """The fused score: the weighted average, or one component decisive on its own.

    A weighted average dilutes a single decisive component: a tone with a
    spectral z of 15 beside three components near 1 averaged to 3.3 and the
    series read "normal" (measured). The largest component is therefore also
    read as the maximum of ``k`` calibrated components (Šidák over the
    selected count), and the fused score is the larger of the two; under the
    null both stay at the threshold's tail, so the false-positive rate holds.
    """
    selected = [d for d in detectors if d in FUSION_WEIGHTS and d in scores]
    weighted = fuse(scores, selected)
    if not selected:
        return weighted
    decisive = float(calibrated_max_z(max(scores[d] for d in selected), len(selected)))
    return float(max(weighted, decisive))

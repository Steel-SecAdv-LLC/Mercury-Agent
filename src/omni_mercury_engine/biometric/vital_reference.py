# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference-sensor rates and agreement statistics for validating RF vital signs.

Contact references give the ground truth a contactless estimate is scored
against: ECG R-peaks or pulse-oximeter (PPG) pulse peaks for heart rate, and
inspiration peaks of a thoracic impedance or respiratory-belt signal for
breathing rate. Rates come from event intervals in the time domain, a
different method from the spectral estimator under test, so the two do not
share a failure mode.

A window's reference rate is ``60 / mean(valid interval)`` over the
intervals that lie wholly inside it, reported only when those intervals
cover at least ``min_coverage`` of the window. An interval is valid when it
is physiologically possible and within ``max_deviation`` of the median of
its neighbours (ectopic beats, missed or doubled detections are dropped).

Agreement follows Bland and Altman (1986): bias, 95 % limits of agreement,
mean absolute error and the share within a tolerance, over the windows that
have both a reference and an estimate; coverage is the share of
reference windows the estimator reported. Confidence intervals resample
subjects (not windows), since windows from one subject are correlated.

References:
    Bland, J. M., Altman, D. G. (1986). Statistical methods for assessing
    agreement between two methods of clinical measurement. Lancet 327(8476),
    307-310.

    Pan, J., Tompkins, W. J. (1985). A real-time QRS detection algorithm.
    IEEE Trans. Biomed. Eng. 32(3), 230-236.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IntervalRule:
    """Admissible event intervals for one vital sign.

    Attributes:
        min_s: Shortest possible interval (s).
        max_s: Longest possible interval (s).
        max_deviation: Largest relative deviation from the median of the
            ``neighbours`` surrounding intervals.
        neighbours: Odd number of intervals in the local median.
    """

    min_s: float
    max_s: float
    max_deviation: float
    neighbours: int = 9


#: 30 to 200 beats/min, at most 25 % from the local median.
HEART_INTERVALS = IntervalRule(min_s=0.3, max_s=2.0, max_deviation=0.25)
#: 3 to 45 breaths/min, at most 50 % from the local median.
BREATH_INTERVALS = IntervalRule(min_s=60.0 / 45.0, max_s=20.0, max_deviation=0.5)


#: Largest fraction of non-finite samples a reference signal may carry.
MAX_MISSING_FRACTION = 0.05


def _signal(x: ArrayLike, fs: float, name: str) -> FloatArray:
    if not (math.isfinite(fs) and fs > 0.0):
        raise ValueError(f"fs must be positive, got {fs}")
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 4 * fs:
        raise ValueError(f"{name} needs at least 4 s of samples")
    missing = ~np.isfinite(arr)
    if missing.any():
        # Dropped reference samples (sensor dropouts) are bridged linearly; a
        # fabricated event there is culled by the interval validity rule.
        fraction = missing.mean()
        if fraction > MAX_MISSING_FRACTION:
            raise ValueError(f"{name} is {fraction:.1%} non-finite; not usable as a reference")
        valid = np.flatnonzero(~missing)
        arr = arr.copy()
        arr[missing] = np.interp(np.flatnonzero(missing), valid, arr[valid])
    return arr


def _bandpass(x: FloatArray, fs: float, low: float, high: float, order: int) -> FloatArray:
    high = min(high, 0.45 * fs)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return np.asarray(signal.sosfiltfilt(sos, x), dtype=np.float64)


def _robust_span(x: FloatArray) -> float:
    lo, hi = np.percentile(x, [5.0, 95.0])
    return float(hi - lo)


def detect_ecg_r_peaks(ecg: ArrayLike, fs: float) -> IntArray:
    """Return R-peak sample indices of an ECG lead.

    QRS energy (8-20 Hz band, squared, 80 ms moving integration, after Pan
    and Tompkins 1985) is peak-picked at least 0.27 s apart above 20 % of
    the median 2 s block maximum; each R-peak is then placed at the largest
    absolute band-passed sample within 100 ms. Polarity does not matter.

    Args:
        ecg: One ECG lead.
        fs: Its sampling rate (Hz), at least 100.

    Returns:
        Increasing sample indices.

    Raises:
        ValueError: On a short, non-finite or under-sampled lead.
    """
    if fs < 100.0:
        raise ValueError("ECG R-peak detection needs fs >= 100 Hz")
    x = _bandpass(_signal(ecg, fs, "ecg"), fs, 8.0, 20.0, order=2)
    width = max(1, round(0.08 * fs))
    energy = np.convolve(x * x, np.ones(width) / width, mode="same")
    block = round(2.0 * fs)
    blocks = energy[: energy.size // block * block].reshape(-1, block).max(axis=1)
    peaks, _ = signal.find_peaks(
        energy, height=0.2 * float(np.median(blocks)), distance=max(1, round(0.27 * fs))
    )
    radius = round(0.1 * fs)
    refined = [
        max(0, p - radius) + int(np.argmax(np.abs(x[max(0, p - radius) : p + radius + 1])))
        for p in peaks
    ]
    return np.unique(np.asarray(refined, dtype=np.int64))


def detect_pulse_peaks(ppg: ArrayLike, fs: float) -> IntArray:
    """Return systolic peak indices of a pulse-oximeter (PPG) waveform.

    Args:
        ppg: PPG waveform (any units; larger means more blood volume).
        fs: Its sampling rate (Hz), at least 20.

    Returns:
        Increasing sample indices.

    Raises:
        ValueError: On a short, non-finite or under-sampled waveform.
    """
    if fs < 20.0:
        raise ValueError("PPG peak detection needs fs >= 20 Hz")
    x = _bandpass(_signal(ppg, fs, "ppg"), fs, 0.5, 5.0, order=2)
    peaks, _ = signal.find_peaks(
        x, distance=max(1, round(0.3 * fs)), prominence=0.2 * _robust_span(x)
    )
    return np.asarray(peaks, dtype=np.int64)


def detect_breaths(resp: ArrayLike, fs: float) -> IntArray:
    """Return inspiration-peak indices of an impedance or belt respiration signal.

    Args:
        resp: Respiration reference (thoracic impedance, belt stretch, ...),
            increasing with inspiration.
        fs: Its sampling rate (Hz), at least 4.

    Returns:
        Increasing sample indices.

    Raises:
        ValueError: On a short, non-finite or under-sampled signal.
    """
    if fs < 4.0:
        raise ValueError("breath detection needs fs >= 4 Hz")
    x = _bandpass(_signal(resp, fs, "resp"), fs, 0.05, 1.0, order=2)
    # Impedance and belt signals carry a cardiac component (46 beats/min is
    # 0.77 Hz, inside the pass band) and often two local maxima per breath;
    # a fixed minimum spacing then double-counts slow breathers. The spacing
    # is set from the signal's own dominant respiration frequency instead:
    # no two breaths closer than 60 % of the dominant period.
    frequencies, power = signal.welch(x, fs, nperseg=min(x.size, round(120.0 * fs)))
    in_band = (frequencies >= 1.0 / BREATH_INTERVALS.max_s) & (
        frequencies <= 1.0 / BREATH_INTERVALS.min_s
    )
    dominant_hz = float(frequencies[in_band][np.argmax(power[in_band])])
    spacing_s = max(BREATH_INTERVALS.min_s, 0.6 / dominant_hz)
    peaks, _ = signal.find_peaks(
        x, distance=max(1, round(spacing_s * fs)), prominence=0.15 * _robust_span(x)
    )
    return np.asarray(peaks, dtype=np.int64)


def valid_intervals(times_s: ArrayLike, rule: IntervalRule) -> tuple[FloatArray, FloatArray]:
    """Return ``(start_s, length_s)`` of the admissible intervals between events."""
    times = np.asarray(times_s, dtype=np.float64).ravel()
    if times.size < 2:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    lengths = np.diff(times)
    half = rule.neighbours // 2
    padded = np.pad(lengths, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, rule.neighbours)
    local = np.median(windows, axis=1)
    keep = (
        (lengths >= rule.min_s)
        & (lengths <= rule.max_s)
        & (np.abs(lengths - local) <= rule.max_deviation * local)
    )
    return times[:-1][keep], lengths[keep]


def windowed_rates(
    times_s: ArrayLike,
    window_start_s: ArrayLike,
    window_end_s: ArrayLike,
    rule: IntervalRule,
    *,
    min_coverage: float,
    min_intervals: int = 2,
) -> FloatArray:
    """Reference rate per window (events/min), ``nan`` where the reference is inadequate.

    Args:
        times_s: Event times (s) on the estimator's clock.
        window_start_s: Window starts.
        window_end_s: Window ends (exclusive).
        rule: Interval admissibility rule.
        min_coverage: Least fraction of a window the valid intervals must
            cover.
        min_intervals: Least number of valid intervals in a window.

    Returns:
        One rate per window.
    """
    starts, lengths = valid_intervals(times_s, rule)
    ends = starts + lengths
    w0 = np.asarray(window_start_s, dtype=np.float64)
    w1 = np.asarray(window_end_s, dtype=np.float64)
    rates = np.full(w0.shape, np.nan)
    for k, (a, b) in enumerate(zip(w0, w1, strict=True)):
        inside = (starts >= a) & (ends <= b)
        n = int(inside.sum())
        if n >= min_intervals and lengths[inside].sum() >= min_coverage * (b - a):
            rates[k] = 60.0 / float(lengths[inside].mean())
    return rates


def agreement(
    estimate: ArrayLike, reference: ArrayLike, *, tolerance: float
) -> dict[str, float | int]:
    """Bland-Altman agreement of ``estimate`` against ``reference`` (``nan`` = absent)."""
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    has_ref = np.isfinite(ref)
    paired = has_ref & np.isfinite(est)
    err = est[paired] - ref[paired]
    n = int(err.size)
    sd = float(err.std(ddof=1)) if n > 1 else math.nan
    bias = float(err.mean()) if n else math.nan
    return {
        "windows": int(est.size),
        "reference_windows": int(has_ref.sum()),
        "reported": n,
        "reported_without_reference": int((~has_ref & np.isfinite(est)).sum()),
        "coverage": n / int(has_ref.sum()) if has_ref.any() else math.nan,
        "mae": float(np.abs(err).mean()) if n else math.nan,
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "within_tolerance": float((np.abs(err) <= tolerance).mean()) if n else math.nan,
        "tolerance": tolerance,
    }


def bootstrap_by_subject(
    rows: list[tuple[str, float, float]],
    *,
    tolerance: float,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pooled agreement with subject-resampled 95 % intervals for MAE and coverage.

    Args:
        rows: ``(subject, estimate, reference)`` per window (``nan`` = absent).
        tolerance: Tolerance for ``within_tolerance``.
        resamples: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        The pooled :func:`agreement` plus ``mae_ci95`` and ``coverage_ci95``.
    """
    subjects = sorted({s for s, _, _ in rows})
    by_subject = {
        s: np.array([(e, r) for t, e, r in rows if t == s], dtype=np.float64).reshape(-1, 2)
        for s in subjects
    }
    pooled = np.concatenate(list(by_subject.values())) if subjects else np.zeros((0, 2))
    result: dict[str, Any] = dict(agreement(pooled[:, 0], pooled[:, 1], tolerance=tolerance))
    result["subjects"] = len(subjects)
    rng = np.random.default_rng(seed)
    maes, coverages = [], []
    for _ in range(resamples if subjects else 0):
        draw = np.concatenate(
            [by_subject[subjects[j]] for j in rng.integers(len(subjects), size=len(subjects))]
        )
        stats = agreement(draw[:, 0], draw[:, 1], tolerance=tolerance)
        maes.append(stats["mae"])
        coverages.append(stats["coverage"])

    def _ci(values: list[float | int]) -> list[float] | None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return [float(v) for v in np.percentile(finite, [2.5, 97.5])] if finite.size else None

    result["mae_ci95"] = _ci(maes)
    result["coverage_ci95"] = _ci(coverages)
    return result

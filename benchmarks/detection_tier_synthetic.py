# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic 1-D synthetic scenario generators for the streaming detector tier.

Each generator returns a ``(series, labels)`` pair matching the convention of
:func:`benchmarks.quick_benchmark_validation.generate_synthetic_anomaly_data`: a
float64 signal and per-point ``0/1`` anomaly labels of identical length. The
scenarios exercise complementary failure modes that streaming anomaly detectors
face in production -- sharp additive bursts, slow mean/variance drift, an abrupt
concept shift, missing-data dropouts, and anomalies camouflaged by correlated
noise -- so a benchmark can measure where each paradigm in the tier excels or
degrades.

All randomness flows through :func:`numpy.random.default_rng` seeded per call, so
a given ``(name, seed, **kwargs)`` triple reproduces byte-identical arrays and no
global RNG state is touched. Every generator keeps the anomaly rate in the
``2-8%`` band and guarantees both classes are present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

__all__ = [
    "SCENARIOS",
    "generate_adversarial_noise",
    "generate_burst",
    "generate_concept_shift",
    "generate_drift",
    "generate_missing_data",
    "generate_scenario",
]


def generate_burst(
    n: int = 2000,
    seed: int = 0,
    baseline_sigma: float = 0.4,
    n_bursts: int = 8,
    burst_len: int = 8,
    spike_amplitude: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Quiet baseline punctuated by a few sharp additive spike bursts.

    Args:
        n: Series length.
        seed: RNG seed for reproducibility.
        baseline_sigma: Standard deviation of the quiet Gaussian baseline.
        n_bursts: Number of spike bursts spread across the series.
        burst_len: Number of consecutive anomalous points per burst.
        spike_amplitude: Mean magnitude of the additive spikes.

    Returns:
        ``(series, labels)`` where ``series`` is float64 and ``labels`` marks
        every burst point with ``1``.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(0.0, baseline_sigma, size=n).astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)

    margin = burst_len + 2
    centers = np.linspace(margin, n - margin, n_bursts).astype(np.int64)
    jitter = rng.integers(-margin // 2, margin // 2 + 1, size=n_bursts)
    starts = np.clip(centers + jitter, 1, n - burst_len - 1)
    for start in starts:
        end = int(start) + burst_len
        signs = rng.choice(np.array([-1.0, 1.0]), size=burst_len)
        magnitudes = spike_amplitude + rng.normal(0.0, 0.5, size=burst_len)
        series[start:end] += signs * magnitudes
        labels[start:end] = 1
    return series, labels


def generate_drift(
    n: int = 2000,
    seed: int = 0,
    drift_slope: float = 3.0,
    var_growth: float = 1.5,
    anomaly_ratio: float = 0.04,
    anomaly_scale: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow mean/variance ramp with point anomalies against the drifting baseline.

    Args:
        n: Series length.
        seed: RNG seed for reproducibility.
        drift_slope: Total mean displacement accumulated across the series.
        var_growth: Fractional growth of the noise scale from start to end.
        anomaly_ratio: Fraction of points that are injected anomalies.
        anomaly_scale: Anomaly magnitude expressed in local-sigma units.

    Returns:
        ``(series, labels)`` with anomalies scaled to the *local* noise level so
        they remain distinguishable from the slowly drifting baseline.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    mean_ramp = drift_slope * t
    sigma_ramp = 0.3 * (1.0 + var_growth * t)
    series = (mean_ramp + rng.normal(0.0, 1.0, size=n) * sigma_ramp).astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)

    n_anom = max(2, round(anomaly_ratio * n))
    idx = rng.choice(n, size=n_anom, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_anom)
    series[idx] += signs * anomaly_scale * sigma_ramp[idx] * (2.0 + rng.random(n_anom))
    labels[idx] = 1
    return series, labels


def generate_concept_shift(
    n: int = 2000,
    seed: int = 0,
    shift_point_frac: float = 0.5,
    mean_shift: float = 3.0,
    var_multiplier: float = 3.0,
    change_region: int = 25,
    anomaly_ratio: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Abrupt regime change (mean + variance) partway with post-shift anomalies.

    The series switches from regime one to a higher-mean, higher-variance regime
    two at ``shift_point_frac``. The transition window is labelled anomalous, and
    scattered point anomalies are injected across *both* regimes so a 50/50
    train/test split sees positives on each side.

    Args:
        n: Series length.
        seed: RNG seed for reproducibility.
        shift_point_frac: Fractional index of the regime change in ``(0, 1)``.
        mean_shift: Mean of regime two (regime one is centred at zero).
        var_multiplier: Multiplier applied to regime two's noise scale.
        change_region: Width of the labelled transition window after the change.
        anomaly_ratio: Fraction of scattered point anomalies.

    Returns:
        ``(series, labels)`` labelling the change region plus scattered anomalies.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(0.0, 0.5, size=n).astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)

    cp = round(shift_point_frac * n)
    series[cp:] = rng.normal(mean_shift, 0.5 * var_multiplier, size=n - cp)
    labels[cp : min(n, cp + change_region)] = 1

    n_anom = max(2, round(anomaly_ratio * n))
    idx = rng.choice(n, size=n_anom, replace=False)
    local_mean = np.where(np.arange(n) >= cp, mean_shift, 0.0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_anom)
    series[idx] = local_mean[idx] + signs * (4.0 + rng.random(n_anom) * 2.0)
    labels[idx] = 1
    return series, labels


def generate_missing_data(
    n: int = 2000,
    seed: int = 0,
    baseline_sigma: float = 0.5,
    dropout_ratio: float = 0.08,
    use_nan: bool = True,
    anomaly_ratio: float = 0.035,
    anomaly_amplitude: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline with NaN/zero dropouts plus labelled point anomalies.

    Dropouts model a data-quality nuisance (sensor gaps) and are *not* labelled
    anomalies; they are placed on indices disjoint from the true anomalies so the
    two never collide. The returned series may therefore contain ``NaN`` while the
    labels stay finite.

    Args:
        n: Series length.
        seed: RNG seed for reproducibility.
        baseline_sigma: Standard deviation of the Gaussian baseline.
        dropout_ratio: Fraction of points blanked out as dropouts.
        use_nan: Blank dropouts with ``NaN`` when ``True`` else ``0.0``.
        anomaly_ratio: Fraction of points that are true (labelled) anomalies.
        anomaly_amplitude: Additive magnitude of the true anomalies.

    Returns:
        ``(series, labels)`` where ``series`` may hold ``NaN`` dropouts and
        ``labels`` marks only the true anomalies.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(0.0, baseline_sigma, size=n).astype(np.float64)
    labels = np.zeros(n, dtype=np.int64)

    n_anom = max(2, round(anomaly_ratio * n))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_anom)
    series[anom_idx] += signs * anomaly_amplitude
    labels[anom_idx] = 1

    available = np.setdiff1d(np.arange(n), anom_idx)
    n_drop = min(available.size, round(dropout_ratio * n))
    drop_idx = rng.choice(available, size=n_drop, replace=False)
    series[drop_idx] = np.nan if use_nan else 0.0
    return series, labels


def generate_adversarial_noise(
    n: int = 2000,
    seed: int = 0,
    noise_sigma: float = 0.6,
    ar_coeff: float = 0.85,
    anomaly_ratio: float = 0.04,
    camouflage: float = 0.6,
    anomaly_amplitude: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlated-noise baseline with anomalies partially camouflaged in the noise.

    The baseline is an ``AR(1)`` process whose excursions superficially resemble
    anomalies. Injected anomalies ride the local noise direction with a modest,
    randomised amplitude (scaled down by ``camouflage``) so they are genuinely
    hard to separate from the correlated background -- the deliberately harder
    scenario in the suite.

    Args:
        n: Series length.
        seed: RNG seed for reproducibility.
        noise_sigma: Standard deviation of the ``AR(1)`` innovations.
        ar_coeff: Autoregressive coefficient in ``[0, 1)``.
        anomaly_ratio: Fraction of points that are anomalies.
        camouflage: Amount ``[0, 1]`` by which anomaly amplitude is suppressed.
        anomaly_amplitude: Nominal additive anomaly magnitude before camouflage.

    Returns:
        ``(series, labels)`` for the camouflaged-anomaly scenario.
    """
    rng = np.random.default_rng(seed)
    innov = rng.normal(0.0, noise_sigma, size=n)
    series = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        series[i] = ar_coeff * series[i - 1] + innov[i]
    labels = np.zeros(n, dtype=np.int64)

    n_anom = max(2, round(anomaly_ratio * n))
    idx = rng.choice(np.arange(1, n), size=n_anom, replace=False)
    local_sign = np.sign(series[idx])
    local_sign[local_sign == 0.0] = 1.0
    amp = anomaly_amplitude * (1.0 - camouflage * rng.random(n_anom))
    series[idx] += local_sign * amp
    labels[idx] = 1
    return series, labels


#: Registry mapping scenario name to its deterministic generator.
SCENARIOS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "burst": generate_burst,
    "drift": generate_drift,
    "concept_shift": generate_concept_shift,
    "missing_data": generate_missing_data,
    "adversarial_noise": generate_adversarial_noise,
}


def generate_scenario(name: str, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a named scenario generator.

    Args:
        name: A key of :data:`SCENARIOS`.
        **kwargs: Forwarded to the selected generator (e.g. ``n``, ``seed``).

    Returns:
        ``(series, labels)`` from the selected generator.

    Raises:
        KeyError: If ``name`` is not a registered scenario.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name](**kwargs)

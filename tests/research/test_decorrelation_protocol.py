# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline guards for the decorrelated-stream fusion protocol.

``research/governed_fusion/measure_decorrelation.py`` executes the one untried
lever logged in ``FINDINGS.md`` -- adding a genuinely *decorrelated* stream and a
learned stacker to test whether fusion can beat best-single on the live suite.
The end-to-end run needs the live-API cache + the native PQC backend, so these
tests pin the *building blocks* offline instead: the two new detectors behave
(flag the injected anomaly), the redundancy/decorrelation statistic is sound, the
stacker scores a separable problem correctly, and the paired bootstrap is
deterministic.  This keeps the protocol's machinery under CI without a network.
"""

from __future__ import annotations

import numpy as np

from research.governed_fusion.measure_decorrelation import (
    _bootstrap_ci,
    _mean_abs_pairwise_spearman,
    _mean_abs_spearman_against,
    _stack_auroc,
    knn_density_score,
    temporal_innovation_score,
)


def test_temporal_innovation_flags_injected_jump() -> None:
    """A single out-of-context spike is the most surprising row to the AR model."""
    rng = np.random.RandomState(0)
    x = np.cumsum(rng.normal(0, 0.05, size=(80, 3)), axis=0)
    spike = 40
    x[spike] += 50.0
    score = temporal_innovation_score(x)
    assert score.shape == (80,)
    assert np.all(np.isfinite(score))
    assert spike in set(np.argsort(score)[-3:])


def test_temporal_innovation_handles_degenerate_short_input() -> None:
    """Fewer rows than lags falls back to the row norm without raising."""
    x = np.ones((2, 4), dtype=np.float64)
    score = temporal_innovation_score(x)
    assert score.shape == (2,)
    assert np.all(np.isfinite(score))


def test_knn_density_flags_isolated_point() -> None:
    """The lone point far from a dense cluster has the largest k-NN distance."""
    rng = np.random.RandomState(1)
    cluster = rng.normal(0, 0.1, size=(60, 2))
    outlier = np.array([[12.0, 12.0]])
    x = np.vstack([cluster, outlier])
    score = knn_density_score(x)
    assert score.shape == (61,)
    assert int(np.argmax(score)) == 60


def test_mean_abs_pairwise_spearman_extremes() -> None:
    """Identical columns -> ~1.0; a constant column is NaN-guarded, not crashed."""
    base = np.linspace(0, 1, 50)
    identical = np.column_stack([base, base, base])
    assert _mean_abs_pairwise_spearman(identical) > 0.999

    with_const = np.column_stack([base, np.ones(50), base[::-1]])
    rho = _mean_abs_pairwise_spearman(with_const)
    assert np.isfinite(rho)
    assert 0.0 <= rho <= 1.0


def test_mean_abs_spearman_against_independent_is_low() -> None:
    """An independent stream is weakly correlated with the existing pool."""
    rng = np.random.RandomState(2)
    base = rng.normal(size=(200, 3))
    new = rng.normal(size=200)
    assert _mean_abs_spearman_against(new, base) < 0.3


def test_stack_auroc_separable_problem() -> None:
    """A logistic stacker on a separable stream scores AUROC 1.0 on eval."""
    y = np.array([0] * 20 + [1] * 20)
    streams = np.column_stack([y.astype(np.float64), np.zeros(40)])
    cal = np.concatenate([np.arange(0, 10), np.arange(20, 30)])
    ev = np.concatenate([np.arange(10, 20), np.arange(30, 40)])
    assert _stack_auroc(streams, y, cal, ev) == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    """Seeded bootstrap is reproducible and the CI contains the sample mean."""
    deltas = [0.02, -0.01, 0.03, 0.00, 0.015, -0.005, 0.01]
    mean_a, low_a, high_a = _bootstrap_ci(deltas, resamples=2000)
    mean_b, low_b, high_b = _bootstrap_ci(deltas, resamples=2000)
    assert (mean_a, low_a, high_a) == (mean_b, low_b, high_b)
    assert low_a <= mean_a <= high_a
    assert abs(mean_a - float(np.mean(deltas))) < 1e-12

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Item 3: bounded-influence reliability pooling primitives (research prototype)."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.robust_pooling import (
    clipped_logodds,
    compute_reliability_weights,
    normalize_weights,
    trimmed_logodds,
)


def test_reliability_weights_downweight_the_diluter() -> None:
    """An at-chance component self-down-weights vs a discriminative one."""
    rng = np.random.default_rng(0)
    n = 400
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    good = np.where(y == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    noise = rng.uniform(0.0, 1.0, n)  # at chance
    comp = np.column_stack([good, noise, good * 0.5 + 0.25])
    w = compute_reliability_weights(comp, y)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] < w[0]  # the at-chance diluter gets less weight than the strong one
    assert w[1] <= 1e-6 + min(w[0], w[2])


def test_clipped_logodds_bounds_single_component_influence() -> None:
    """An extreme outlier opinion cannot dominate the pooled probability."""
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    agree = np.array([[0.5, 0.5, 0.5]])
    # One component screams 0.999999; with c-clipping the pool stays moderate.
    outlier = np.array([[0.5, 0.5, 1.0 - 1e-9]])
    base = float(clipped_logodds(agree, w, c=2.0)[0])
    pulled = float(clipped_logodds(outlier, w, c=2.0)[0])
    assert abs(pulled - base) < 0.25  # bounded influence
    # Without clipping (huge c) the same outlier moves the pool much further.
    pulled_unclipped = float(clipped_logodds(outlier, w, c=100.0)[0])
    assert pulled_unclipped - base > pulled - base


def test_trimmed_logodds_discards_most_deviant() -> None:
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    # Two components agree at 0.8; one dissents at 0.01. Trimming t=1 -> ~0.8.
    probs = np.array([[0.8, 0.8, 0.01]])
    pooled = float(trimmed_logodds(probs, w, t=1)[0])
    assert pooled > 0.7


def test_pooling_is_identity_on_consensus() -> None:
    w = np.array([0.4, 0.3, 0.3])
    probs = np.full((5, 3), 0.73)
    assert np.allclose(clipped_logodds(probs, w, c=2.0), 0.73, atol=1e-6)
    assert np.allclose(trimmed_logodds(probs, w, t=1), 0.73, atol=1e-6)


def test_normalize_weights_handles_degenerate() -> None:
    assert np.allclose(normalize_weights(np.zeros(3)), 1 / 3)
    assert np.allclose(normalize_weights(np.array([2.0, 2.0])), 0.5)

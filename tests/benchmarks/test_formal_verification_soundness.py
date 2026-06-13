# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the formal-verification soundness harness (``benchmarks.formal_verification_soundness``): the revived ``IntervalBoundPropagator`` must produce a *sound* certificate -- its interval must contain the true (densely-sampled) output range of a random ReLU network over an input box -- on every checked case."""

from __future__ import annotations

import numpy as np

from benchmarks.formal_verification_soundness import (
    _forward,
    _ibp_bounds,
    _random_net,
    _true_bounds,
)


def test_forward_matches_manual_relu_net() -> None:
    net = _random_net(np.random.default_rng(0))
    x = np.random.default_rng(1).normal(size=(5, net["W1"].shape[0]))
    expected = np.maximum(x @ net["W1"] + net["b1"], 0.0) @ net["W2"] + net["b2"]
    assert np.allclose(_forward(x, net), expected)


def test_ibp_certificate_is_sound() -> None:
    # On every random case the IBP interval must contain the sampled true range.
    for case in range(25):
        rng = np.random.default_rng(case)
        net = _random_net(rng)
        center = rng.normal(0, 1, net["W1"].shape[0])
        radius = rng.uniform(0.1, 1.0, net["W1"].shape[0])
        lo, hi = center - radius, center + radius
        l_ibp, u_ibp = _ibp_bounds(lo, hi, net)
        l_true, u_true = _true_bounds(lo, hi, net, rng)
        assert np.all(l_ibp <= l_true + 1e-6), (case, l_ibp, l_true)
        assert np.all(u_ibp >= u_true - 1e-6), (case, u_ibp, u_true)


def test_ibp_is_not_vacuous() -> None:
    # A sound-but-infinite certificate would be useless; bounds must be finite
    # and not absurdly wider than the truth.
    rng = np.random.default_rng(3)
    net = _random_net(rng)
    lo, hi = np.full(net["W1"].shape[0], -0.5), np.full(net["W1"].shape[0], 0.5)
    l_ibp, u_ibp = _ibp_bounds(lo, hi, net)
    l_true, u_true = _true_bounds(lo, hi, net, rng)
    assert np.all(np.isfinite(l_ibp)) and np.all(np.isfinite(u_ibp))
    width_ratio = float((u_ibp - l_ibp).mean() / ((u_true - l_true).mean() + 1e-9))
    assert width_ratio < 10.0

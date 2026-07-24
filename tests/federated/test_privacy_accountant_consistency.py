# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test: the DP accountant must account for the noise actually added.

``PrivacyEngine.privatize_gradients`` calibrates its noise from the per-query
epsilon (``mechanism.compute_noise_scale``), but previously handed the accountant
an unrelated ``noise_multiplier * max_grad_norm / batch_size`` scale that the
mechanism never applied -- so the reported (epsilon, delta) was disconnected from
the real noise on the gradients. The accountant now receives the mechanism's own
noise scale.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.federated_learning.privacy import (
    PrivacyAccountant,
    PrivacyEngine,
)


def test_accountant_receives_the_actual_noise_scale() -> None:
    engine = PrivacyEngine(epsilon=8.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=1.0)

    captured: list[float] = []
    original_add_query = engine._accountant.add_query

    def spy(sensitivity: float, noise_scale: float) -> tuple[float, float]:
        captured.append(noise_scale)
        return original_add_query(sensitivity, noise_scale)

    engine._accountant.add_query = spy  # type: ignore[method-assign]

    gradients = np.ones((16, 5), dtype=float)
    private = engine.privatize_gradients(gradients)

    # The scale recorded is the mechanism's own scale for the query epsilon...
    query_epsilon = 8.0 / 1  # first query, n_queries == 0
    expected = engine._mechanism.compute_noise_scale(1.0, query_epsilon)
    assert captured[0] == expected

    # ...and NOT the disconnected ``noise_multiplier * max_grad_norm`` (== 1.0).
    assert abs(captured[0] - 1.0) > 1e-6

    # Noise was actually applied to the aggregated gradient.
    assert not np.allclose(private, gradients.mean(axis=0))


def test_reported_epsilon_is_finite_and_monotonic() -> None:
    engine = PrivacyEngine(epsilon=8.0, delta=1e-5, max_grad_norm=1.0)

    spends = []
    for _ in range(3):
        engine.privatize_gradients(np.ones((8, 4), dtype=float))
        spends.append(engine._accountant.get_current_epsilon())

    assert all(np.isfinite(s) for s in spends)
    assert spends == sorted(spends)  # non-decreasing as queries accumulate


def test_basic_composition_is_linear_not_superlinear() -> None:
    """Basic composition costs add up linearly across identical queries.

    The prior implementation summed every stored *cumulative* value (double-
    counting earlier queries, so epsilon grew super-linearly) and divided delta
    by the query count. Both epsilon and delta must now scale linearly with the
    number of queries.
    """
    accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-5, composition="basic")

    eps_seq = []
    delta_seq = []
    for _ in range(4):
        eps, delta = accountant.add_query(sensitivity=1.0, noise_scale=1.0)
        eps_seq.append(eps)
        delta_seq.append(delta)

    single_eps = eps_seq[0]
    for k, (eps, delta) in enumerate(zip(eps_seq, delta_seq), start=1):
        assert eps == pytest.approx(k * single_eps)
        assert delta == pytest.approx(k * 1e-5)


# =============================================================================
# Property-based invariants for privacy composition (the accounting fixes)
# =============================================================================
_noise = st.floats(min_value=0.25, max_value=10.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=100, deadline=None)
@given(scales=st.lists(_noise, min_size=1, max_size=12))
def test_basic_composition_is_monotonic_and_finite(scales: list[float]) -> None:
    """Cumulative epsilon under basic composition must be finite and never
    decrease as queries accumulate (linear composition adds non-negative cost)."""
    accountant = PrivacyAccountant(total_epsilon=1e9, total_delta=1e-5, composition="basic")
    prev_eps = 0.0
    prev_delta = 0.0
    for noise_scale in scales:
        eps, delta = accountant.add_query(sensitivity=1.0, noise_scale=noise_scale)
        assert np.isfinite(eps) and np.isfinite(delta)
        assert eps >= prev_eps - 1e-9
        assert delta >= prev_delta - 1e-12
        prev_eps, prev_delta = eps, delta

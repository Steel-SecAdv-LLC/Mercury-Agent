# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""R7: the Lyapunov surface is a measured decay-schedule monitor, not a guarantee.

These tests pin the transparency contract:

* ``verify_lyapunov_stability`` reports ``is_stable`` from a *measured*
  contraction of the score trajectory, and returns ``False`` (never a hardcoded
  ``True``) when it cannot measure one (insufficient history or a divergent
  trajectory);
* the learnable-fusion result defaults to ``is_stable=False`` and only flips to
  ``True`` once the observed fusion-score trajectory is measured to contract.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation


def _eq() -> OmniAvaEquation:
    return OmniAvaEquation()


def test_insufficient_history_does_not_claim_stability() -> None:
    eq = _eq()
    eq.convergence_history = [0.5, 0.6]  # fewer than window_size
    eq.time_step = 2
    is_stable, _ = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is False  # no hardcoded True on insufficient data


def test_contracting_trajectory_is_measured_stable() -> None:
    eq = _eq()
    rng = np.random.default_rng(0)
    wide = (rng.normal(0.0, 0.30, 10) + 0.5).tolist()  # high initial variance
    tight = (rng.normal(0.0, 0.01, 10) + 0.5).tolist()  # low recent variance
    eq.convergence_history = wide + tight
    eq.time_step = len(eq.convergence_history)
    is_stable, rate = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is True
    assert rate > 0.0  # positive measured contraction rate


def test_diverging_trajectory_is_not_stable() -> None:
    eq = _eq()
    rng = np.random.default_rng(1)
    tight = (rng.normal(0.0, 0.01, 10) + 0.5).tolist()  # low initial variance
    wide = (rng.normal(0.0, 0.30, 10) + 0.5).tolist()  # high recent variance
    eq.convergence_history = tight + wide
    eq.time_step = len(eq.convergence_history)
    is_stable, rate = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is False  # measured expansion -> not stable
    assert rate < 0.0


def test_learnable_result_defaults_to_not_stable() -> None:
    from omni_mercury_engine.core.three_r.learnable_fusion import Learnable3RResult

    r = Learnable3RResult(
        fusion_score=0.5,
        recursion_score=0.5,
        resonance_score=0.5,
        optimization_score=0.5,
        ethical_gate_output=0.96,
        learned_weights={"w_R": 0.4472, "w_H": 0.2764, "w_O": 0.2764},
        learned_phi=1.618033988749895,
    )
    assert r.is_stable is False  # transparent default: not asserted until measured


def test_learnable_engine_monitor_requires_contraction() -> None:
    from omni_mercury_engine.core.three_r.learnable_fusion import Learnable3REngine

    eng = Learnable3REngine()
    # Empty / short history must not claim contraction.
    assert eng._recent_contraction(window=5) is False
    eng._score_history = [0.9, 0.1, 0.9, 0.1, 0.9] + [0.5, 0.5, 0.5, 0.5, 0.5]
    assert eng._recent_contraction(window=5) is True  # variance collapsed -> measured
    eng._score_history = [0.5, 0.5, 0.5, 0.5, 0.5] + [0.9, 0.1, 0.9, 0.1, 0.9]
    assert eng._recent_contraction(window=5) is False  # variance grew


def test_estimated_rate_matches_known_decay_within_20_percent() -> None:
    """R9 audit cure: the decay-rate estimator must recover a known lambda.

    A fusion-score trajectory ``exp(-0.25 t)`` has window *variance* decaying
    at ``2 * lambda``, and the elapsed time between the initial and recent
    windows is ``len(history) - window``, not the total ``time_step``.  The
    old estimator divided ``-log(var ratio)`` by the total step count with no
    factor 2, overestimating this trajectory's rate by ~33%.
    """
    eq = _eq()
    eq.convergence_history = [float(np.exp(-0.25 * t)) for t in range(30)]
    eq.time_step = 30
    is_stable, rate = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is True
    assert abs(rate - 0.25) <= 0.2 * 0.25, f"estimated lambda {rate} not within 20% of 0.25"


def test_decay_far_below_target_rate_is_not_stable() -> None:
    """A barely-contracting trajectory must not be reported 'stable'.

    ``LYAPUNOV.MIN_CONVERGENCE_RATIO`` (0.5) exists for exactly this: the
    measured rate must reach at least that share of the configured target
    before stability is claimed.  The old check accepted ANY positive rate.
    """
    eq = _eq()
    eq.convergence_history = [
        float(np.exp(-0.001 * t)) * (1.0 + 0.01 * ((-1.0) ** t)) for t in range(30)
    ]
    eq.time_step = 30
    is_stable, rate = eq.verify_lyapunov_stability(window_size=10)
    assert rate < 0.125  # far below MIN_CONVERGENCE_RATIO * 0.25
    assert is_stable is False


def test_flat_trajectory_is_not_stable() -> None:
    eq = _eq()
    eq.convergence_history = [0.5] * 30
    eq.time_step = 30
    is_stable, rate = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is False
    assert rate == 0.0


def test_constant_variance_noise_is_not_stable() -> None:
    eq = _eq()
    rng = np.random.default_rng(7)
    eq.convergence_history = (rng.normal(0.0, 0.1, 30) + 0.5).tolist()
    eq.time_step = 30
    is_stable, _rate = eq.verify_lyapunov_stability(window_size=10)
    assert is_stable is False


def test_complete_contraction_is_stable() -> None:
    """Scores that stop varying altogether have contracted; that is not a failure.

    The estimator divided the recent variance by the initial one and treated a
    zero ratio as "no rate is measurable", so a trajectory that had fully
    converged was reported as not stable.
    """
    equation = OmniAvaEquation()
    rng = np.random.default_rng(0)
    equation.convergence_history = [*(0.5 + 0.1 * rng.normal(size=10)).tolist(), *([0.5] * 10)]
    is_stable, rate = equation.verify_lyapunov_stability()
    assert is_stable is True
    assert np.isfinite(rate) and rate > 1.0  # a lower bound, well past the 0.125 bar

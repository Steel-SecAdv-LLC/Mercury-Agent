# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Cooperative convergence loop for AdaptiveDomainThresholdManager.

Acceptance criteria from the punch list:

(a) convergence within bounded budget (max 4 iterations, ε=1e-3) on synthetic drift
(b) wall-clock cost ≤ 1.3x the one-shot path
(c) no oscillation on stationary input
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from omni_mercury_engine.core.adaptive_domain_thresholding import (
    AdaptiveDomainThresholdManager,
    DomainType,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers — deterministic, no external dependencies.
# ---------------------------------------------------------------------------


def _two_mode_scores(
    n_normal: int,
    n_anom: int,
    mu_n: float,
    mu_a: float,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-Gaussian mixture clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    normal = np.clip(rng.normal(mu_n, sigma, size=n_normal), 0.0, 1.0)
    anom = np.clip(rng.normal(mu_a, sigma, size=n_anom), 0.0, 1.0)
    scores = np.concatenate([normal, anom])
    labels = np.concatenate([np.zeros(n_normal, dtype=np.int32), np.ones(n_anom, dtype=np.int32)])
    # Shuffle so the order doesn't bias any windowing logic downstream.
    perm = rng.permutation(len(scores))
    return scores[perm], labels[perm]


def _drifted_scores(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores after the score distribution has drifted toward the anomaly mode."""
    return _two_mode_scores(n_normal=400, n_anom=100, mu_n=0.30, mu_a=0.65, sigma=0.07, seed=seed)


# ---------------------------------------------------------------------------
# (a) Convergence within bounded budget
# ---------------------------------------------------------------------------


def test_cooperative_loop_converges_within_budget_on_synthetic_drift() -> None:
    """On a clear two-mode mixture, the loop converges in ≤ 4 iterations."""
    scores, labels = _drifted_scores(seed=2026)
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    manager.fit(scores, labels)

    result = manager.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-3)
    assert result["converged"] is True
    assert 1 <= result["iterations"] <= 4
    # The refined threshold sits between the two cluster means by construction.
    refined = result["refined_threshold"]
    assert (
        0.30 < refined < 0.65
    ), f"refined threshold {refined:.4f} did not fall between the two modes"


def test_cooperative_loop_respects_epsilon_strictly() -> None:
    """The final |Δthreshold| is strictly below epsilon when converged=True."""
    scores, labels = _drifted_scores(seed=11)
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    manager.fit(scores, labels)

    result = manager.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-3)
    path = result["threshold_path"]
    assert len(path) >= 2
    if result["converged"]:
        assert abs(path[-1] - path[-2]) < 1e-3


def test_cooperative_loop_caps_at_max_iterations() -> None:
    """When the loop fails to converge within budget, it stops at max_iterations."""
    scores, labels = _drifted_scores(seed=99)
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    manager.fit(scores, labels)

    # Force the loop to exhaust its budget by using an unreachably tight epsilon.
    result = manager.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-15)
    assert result["iterations"] <= 4
    if not result["converged"]:
        assert result["iterations"] == 4


# ---------------------------------------------------------------------------
# (b) Wall-clock cost ≤ 1.3x one-shot path
# ---------------------------------------------------------------------------


def _percentile_runtime(times: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(times, dtype=np.float64), q))


def test_iterative_wall_clock_cost_within_1_3x_one_shot() -> None:
    """Iterative path's median runtime ≤ 1.3x the one-shot calibrate() runtime."""
    scores, labels = _two_mode_scores(
        n_normal=2000, n_anom=400, mu_n=0.25, mu_a=0.70, sigma=0.06, seed=7
    )

    n_repeats = 11
    one_shot_times: list[float] = []
    iterative_times: list[float] = []

    for r in range(n_repeats):
        manager_a = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
        manager_a.fit(scores, labels)
        t0 = time.perf_counter()
        manager_a.calibrate(scores, labels)
        one_shot_times.append(time.perf_counter() - t0)

        manager_b = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
        manager_b.fit(scores, labels)
        t0 = time.perf_counter()
        manager_b.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-3)
        iterative_times.append(time.perf_counter() - t0)

    median_one_shot = _percentile_runtime(one_shot_times, 50)
    median_iterative = _percentile_runtime(iterative_times, 50)

    # The cooperative loop is at most a constant factor on top of one-shot.
    # 1.3x is the punch list's bar; we compare medians to wash out outliers.
    assert median_iterative <= 1.3 * median_one_shot, (
        f"iterative {median_iterative * 1e3:.3f}ms > 1.3x one-shot "
        f"{median_one_shot * 1e3:.3f}ms"
    )


# ---------------------------------------------------------------------------
# (c) No oscillation on stationary input
# ---------------------------------------------------------------------------


def test_cooperative_loop_does_not_oscillate_on_stationary_input() -> None:
    """On a stationary mixture the threshold path is monotone (or very nearly so).

    "No oscillation" is operationalised as: signed differences between
    successive thresholds do not change sign more than once across the
    full path (one direction reversal is permitted because a damped EM
    iteration may overshoot the fixed point on the very first step).
    """
    scores, labels = _two_mode_scores(
        n_normal=1500, n_anom=300, mu_n=0.20, mu_a=0.75, sigma=0.05, seed=2024
    )
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    manager.fit(scores, labels)

    result = manager.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-3)
    path = np.asarray(result["threshold_path"], dtype=np.float64)
    diffs = np.diff(path)
    # Drop near-zero diffs (numerical noise) before counting sign changes.
    significant = diffs[np.abs(diffs) > 1e-9]
    if len(significant) <= 1:
        return  # Trivially monotone.
    sign_changes = int(np.sum(np.sign(significant[1:]) != np.sign(significant[:-1])))
    assert sign_changes <= 1, (
        f"cooperative loop oscillated: path={path.tolist()} " f"(sign_changes={sign_changes})"
    )


def test_cooperative_loop_idempotent_at_fixed_point() -> None:
    """Running the loop a second time from the refined threshold is a no-op."""
    scores, labels = _two_mode_scores(
        n_normal=800, n_anom=160, mu_n=0.25, mu_a=0.70, sigma=0.06, seed=5
    )
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    manager.fit(scores, labels)

    first = manager.calibrate_iterative(scores, labels, max_iterations=4, epsilon=1e-3)
    refined_threshold = first["refined_threshold"]

    # Second pass: the loop should converge in 1 iteration (delta < epsilon
    # immediately) because we are starting from the fixed point.
    second_threshold, second_iters, second_converged, _ = manager._cooperative_refine_threshold(
        np.asarray(first["calibration"].calibrated_scores, dtype=np.float64),
        refined_threshold,
        max_iterations=4,
        epsilon=1e-3,
    )
    assert second_converged is True
    assert abs(second_threshold - refined_threshold) < 1e-3
    assert second_iters <= 2


# ---------------------------------------------------------------------------
# Validation: degenerate inputs
# ---------------------------------------------------------------------------


def test_cooperative_loop_validates_arguments() -> None:
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    scores = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError):
        manager._cooperative_refine_threshold(scores, 0.5, max_iterations=0)
    with pytest.raises(ValueError):
        manager._cooperative_refine_threshold(scores, 0.5, epsilon=0.0)
    with pytest.raises(ValueError):
        manager._cooperative_refine_threshold(scores, 0.5, sigmoid_temperature=0.0)


def test_cooperative_loop_handles_empty_scores() -> None:
    manager = AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)
    refined, iters, converged, path = manager._cooperative_refine_threshold(
        np.array([], dtype=np.float64), threshold_seed=0.5
    )
    assert iters == 0
    assert converged is True
    assert refined == pytest.approx(0.5)
    assert path == [0.5]

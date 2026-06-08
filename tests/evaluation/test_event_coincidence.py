# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for the pre-registered event-coincidence null-test (WS-D harvest).

Validates the harvested GCP machinery as a general tool: it has the correct
false-positive rate under a true null (including for *autocorrelated* streams,
the case the circular-permutation null is designed for), real power against a
planted signal, and correct multiple-comparison control. All offline + seeded.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.evaluation.event_coincidence import (
    PreregisteredCoincidenceTest,
    benjamini_hochberg,
    bonferroni,
    permutation_coincidence_test,
    run_preregistered,
    windows_to_mask,
)


def _mask(n: int, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    m = np.zeros(n, dtype=bool)
    m[rng.choice(n, size=int(n * frac), replace=False)] = True
    return m


def test_null_iid_is_well_calibrated() -> None:
    """IID random scores, fixed events: empirical FPR <= ~alpha (no false psi)."""
    mask = _mask(300, 0.12, seed=1)
    alpha = 0.05
    n_trials = 200
    sig = 0
    for t in range(n_trials):
        scores = np.random.RandomState(1000 + t).randn(300)
        r = permutation_coincidence_test(scores, mask, n_permutations=200, seed=t)
        sig += int(r.p_value <= alpha)
    fpr = sig / n_trials
    assert fpr < 0.15, f"false-positive rate too high: {fpr}"


def test_null_autocorrelated_is_not_systematically_significant() -> None:
    """A strongly autocorrelated random walk unrelated to the events must NOT be
    falsely flagged -- this is the whole point of the circular-shift null."""
    mask = _mask(400, 0.1, seed=2)
    sig = 0
    n_trials = 120
    for t in range(n_trials):
        steps = np.random.RandomState(5000 + t).randn(400)
        walk = np.cumsum(steps)  # heavy autocorrelation
        r = permutation_coincidence_test(walk, mask, n_permutations=200, seed=t)
        sig += int(r.p_value <= 0.05)
    fpr = sig / n_trials
    assert fpr < 0.20, f"autocorrelated null over-fires: {fpr}"


def test_planted_signal_is_detected() -> None:
    """Scores elevated inside the event windows -> significant."""
    mask = _mask(300, 0.15, seed=3)
    scores = np.random.RandomState(7).randn(300)
    scores[mask] += 3.0  # strong real coincidence
    r = permutation_coincidence_test(scores, mask, n_permutations=500, seed=0)
    assert r.observed > 0
    assert r.p_value < 0.05


def test_degenerate_inputs_return_null() -> None:
    scores = np.random.RandomState(0).randn(50)
    # no events
    r1 = permutation_coincidence_test(scores, np.zeros(50, dtype=bool), n_permutations=50)
    assert r1.p_value == 1.0
    # all in-window
    r2 = permutation_coincidence_test(scores, np.ones(50, dtype=bool), n_permutations=50)
    assert r2.p_value == 1.0


def test_windows_to_mask() -> None:
    ts = np.arange(10, dtype=float)
    mask = windows_to_mask(ts, [(2.0, 4.0), (7.0, 7.0)])
    assert mask.tolist() == [False, False, True, True, True, False, False, True, False, False]


def test_bonferroni() -> None:
    assert bonferroni([0.01, 0.02, 0.5], alpha=0.05) == [True, False, False]
    assert bonferroni([], alpha=0.05) == []


def test_benjamini_hochberg_monotone() -> None:
    # All tiny -> all reject; all large -> none.
    assert benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]
    assert benjamini_hochberg([0.6, 0.7, 0.8], alpha=0.05) == [False, False, False]


def test_run_preregistered_applies_correction() -> None:
    test = PreregisteredCoincidenceTest(
        name="unit", n_permutations=300, correction="bonferroni", seed=0
    )
    mask = _mask(300, 0.15, seed=4)
    planted = np.random.RandomState(11).randn(300)
    planted[mask] += 3.0
    null = np.random.RandomState(12).randn(300)
    report = run_preregistered(test, [planted, null], [mask, mask])
    assert report.reject[0] is True  # planted survives correction
    assert report.reject[1] is False  # null does not
    assert report.any_significant is True
    d = report.as_dict()
    assert d["protocol"]["correction"] == "bonferroni"

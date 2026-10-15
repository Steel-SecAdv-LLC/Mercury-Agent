# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``return_samples`` and conformal ``score_function`` paths of the uncertainty quantifier.

Hand-computed, no torch/network/wall-clock: the ensemble sample stack is
attached verbatim and the epistemic term equals the population variance across
members; the conformal set inversion recovers the exact asymmetric interval
boundaries implied by an asymmetric nonconformity score.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.uncertainty import (
    AdaptiveConformalInference,
    UncertaintyQuantifier,
)


def test_return_samples_attaches_measured_ensemble() -> None:
    quantifier = UncertaintyQuantifier(enable_aci=False, seed=0)
    ensemble = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    est = quantifier.estimate_uncertainty(
        np.array([3.0]), ensemble_predictions=ensemble, return_samples=True
    )

    assert est.sample_values is not None
    assert est.sample_values.shape == (5, 1)
    np.testing.assert_allclose(est.sample_values.ravel(), ensemble)
    # epistemic == population variance across members (ddof=0): var([1..5]) == 2.0
    assert abs(est.epistemic - 2.0) < 1e-9
    assert est.epistemic_measured is True
    assert abs(est.prediction - 3.0) < 1e-9
    # to_dict mirrors the attached stack as a nested list.
    assert est.to_dict()["sample_values"] == ensemble.reshape(-1, 1).tolist()


def test_return_samples_off_or_single_member_is_none() -> None:
    quantifier = UncertaintyQuantifier(enable_aci=False, seed=0)

    off = quantifier.estimate_uncertainty(
        np.array([2.0]), ensemble_predictions=np.array([1.0, 2.0, 3.0])
    )
    assert off.sample_values is None  # flag defaults off

    single = quantifier.estimate_uncertainty(
        np.array([2.0]), ensemble_predictions=np.array([2.0]), return_samples=True
    )
    assert single.sample_values is None  # one member -> nothing measured
    assert single.epistemic_measured is False


def _feed(aci: AdaptiveConformalInference, score: float, n: int) -> None:
    for _ in range(n):
        aci.update(score, covered=True)


def test_conformal_inversion_recovers_asymmetric_interval() -> None:
    aci = AdaptiveConformalInference(target_coverage=0.9)
    _feed(aci, 5.0, 20)  # identical scores -> quantile is exactly 5.0 at any level
    assert abs(aci.get_quantile() - 5.0) < 1e-9

    mu = 10.0

    def score(y: float) -> float:
        d = y - mu
        return abs(d) if d < 0 else 0.5 * d  # left slope 1, right slope 0.5

    lo, hi = aci.predict_interval(mu, score_function=score)
    # left boundary: |d| = 5 -> y = 5 ; right boundary: 0.5 * d = 5 -> y = 20
    assert abs(lo - 5.0) < 1e-6
    assert abs(hi - 20.0) < 1e-6


def test_conformal_no_quantile_is_unbounded() -> None:
    aci = AdaptiveConformalInference(target_coverage=0.9)
    lo, hi = aci.predict_interval(0.0, score_function=abs)
    assert lo == float("-inf")
    assert hi == float("inf")


def test_conformal_empty_set_is_degenerate_point() -> None:
    aci = AdaptiveConformalInference(target_coverage=0.9)
    _feed(aci, 0.0, 20)  # quantile is exactly 0.0
    # score at the point itself (1.0) exceeds q -> empty set -> (point, point)
    lo, hi = aci.predict_interval(10.0, score_function=lambda y: 1.0 + abs(y - 10.0))
    assert lo == 10.0
    assert hi == 10.0


def test_conformal_quantile_from_mixed_scores_is_hand_computable() -> None:
    aci = AdaptiveConformalInference(target_coverage=0.9)
    for score in range(1, 11):  # scores 1..10, all covered
        aci.update(float(score), covered=True)
    # alpha only shrinks below its 0.1 start under covered updates, so
    # ceil((n+1)(1-alpha))/n >= ceil(10.89)/10 > 1 -> clamped to the maximum
    # order statistic: q == 10.0 exactly.
    assert aci.get_quantile() == 10.0
    lo, hi = aci.predict_interval(0.0, score_function=abs)
    assert abs(lo - (-10.0)) < 1e-6
    assert abs(hi - 10.0) < 1e-6

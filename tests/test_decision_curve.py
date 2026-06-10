# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3 R2: decision-curve analysis + the single reconciled operating point."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.decision_curve import (
    bayes_threshold,
    decision_curve,
    low_threshold_prior,
    net_benefit,
    reconciled_operating_point,
)


def test_bayes_threshold_is_cost_ratio() -> None:
    assert bayes_threshold(1.0, 1.0) == 0.5
    # Missed detection ~10x costlier (b=10c) -> small threshold.
    assert bayes_threshold(1.0, 10.0) == 1.0 / 11.0
    assert bayes_threshold(0.0, 1.0) == 0.0


def test_net_benefit_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    # All positives caught, no false positives -> NB = prevalence = 0.5.
    assert net_benefit(y, p, 0.5) == 0.5


def test_treat_all_and_none_envelopes() -> None:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=200) < 0.3).astype(int)
    p = rng.uniform(size=200)
    dc = decision_curve(y, p)
    assert dc.treat_none.tolist() == [0.0] * len(dc.thresholds)
    # treat-all at the prevalence-implied break-even is ~0.
    assert abs(dc.prevalence - float(np.mean(y))) < 1e-9
    # A useful model's prior-weighted NB exceeds treat-none (0).
    assert dc.prior_weighted_net_benefit() == dc.prior_weighted_net_benefit()  # finite


def test_low_threshold_prior_upweights_low_t() -> None:
    t = np.array([0.1, 0.5, 0.9])
    w = low_threshold_prior(t)
    assert w[0] > w[1] > w[2]
    assert abs(float(np.sum(w)) - 1.0) < 1e-12


def test_reconciled_operating_point_is_single_pathway() -> None:
    p = np.array([0.05, 0.2, 0.5, 0.95])
    op = reconciled_operating_point(p, cost_fp=1.0, benefit_tp=10.0, conformal_coverage=0.9)
    assert op.bayes_threshold == 1.0 / 11.0
    # Decision is p >= t*; conformal coverage is a *diagnostic*, not a 2nd threshold.
    assert op.decision.tolist() == (p >= op.bayes_threshold).tolist()
    assert op.conformal_recall_floor == 0.9

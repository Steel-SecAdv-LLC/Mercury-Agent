# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Confidence cascade: uncertainty routing + cost/latency instrumentation."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.intel.cascade import (
    CascadeConfig,
    ConfidenceCascadeRouter,
    PathResult,
    RoutePath,
    point_uncertainty,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS


def _deterministic_clock():
    ticks = iter(range(10_000))
    return lambda: next(ticks) * 0.001


def _router(**cfg_kwargs):
    cfg = CascadeConfig(
        low_uncertainty=0.30, high_uncertainty=0.60, cheap_cost=1.0, heavy_cost=20.0, **cfg_kwargs
    )

    def cheap(item):
        answer, prob, disagreement = item
        return PathResult(answer, prob, disagreement)

    def heavy(item):
        return PathResult("HEAVY", 0.99, 0.0)

    return ConfidenceCascadeRouter(cheap, heavy, cfg, clock=_deterministic_clock())


def test_point_uncertainty() -> None:
    assert point_uncertainty(0.5) == pytest.approx(1.0)
    assert point_uncertainty(1.0) == pytest.approx(0.0)
    assert point_uncertainty(0.0) == pytest.approx(0.0)
    assert point_uncertainty(0.75) == pytest.approx(0.5)


def test_low_medium_high_uncertainty_routing() -> None:
    r = _router()
    low = r.route_one(("a", 0.97, 0.0))  # pu ~ 0.06 -> cheap
    borderline = r.route_one(("b", 0.775, 0.0))  # pu ~ 0.45 -> borderline -> heavy
    high = r.route_one(("c", 0.5, 0.0))  # pu = 1.0 -> heavy
    assert low.path is RoutePath.CHEAP and low.tier == "confident_cheap"
    assert borderline.path is RoutePath.HEAVY and borderline.tier == "borderline"
    assert high.path is RoutePath.HEAVY and high.tier == "high_uncertainty"


def test_disagreement_escalates_a_confident_probability() -> None:
    r = _router()
    out = r.route_one(("d", 0.99, 0.9))  # point-confident but split vote
    assert out.path is RoutePath.HEAVY
    assert out.escalated_by == "disagreement"


def test_config_validates_threshold_order() -> None:
    with pytest.raises(ValueError):
        CascadeConfig(low_uncertainty=0.7, high_uncertainty=0.3)


def test_instrumentation_and_compute_savings_meet_value_target() -> None:
    r = _router()
    # 80% easy (confident cheap), 20% hard (routed heavy).
    items = [("easy", 0.99, 0.0)] * 80 + [("hard", 0.5, 0.0)] * 20
    r.route(items)
    report = r.instrumentation.report()
    assert report["n_items"] == 100
    assert report["n_cheap"] == 80
    assert report["n_heavy"] == 20
    # Baseline all-heavy cost = 100 * 20 = 2000; cascade = 100*1 + 20*20 = 500.
    assert report["baseline_heavy_cost"] == pytest.approx(2000.0)
    assert report["total_cost"] == pytest.approx(500.0)
    target = VALUE_METRICS["confidence_cascade"].target
    assert report["compute_saved_fraction"] >= target


def test_calibrator_applied_before_routing() -> None:
    # A calibrator that maps everything to 0.5 forces max uncertainty -> heavy.
    class HalfCalibrator:
        _fitted = True

        def calibrate(self, arr):
            return np.full(np.asarray(arr).shape, 0.5)

    cfg = CascadeConfig(low_uncertainty=0.3, high_uncertainty=0.6)
    r = ConfidenceCascadeRouter(
        lambda i: PathResult("a", 0.99, 0.0),
        lambda i: PathResult("H", 0.99, 0.0),
        cfg,
        calibrator=HalfCalibrator(),
        clock=_deterministic_clock(),
    )
    out = r.route_one("x")
    assert out.path is RoutePath.HEAVY  # calibrated to 0.5 -> uncertainty 1.0

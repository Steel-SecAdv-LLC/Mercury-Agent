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


def test_report_count_fields_are_ints_and_json_serializable() -> None:
    """report() is transparently typed dict[str, float | int]: the n_* fields are exact
    integer counts (not floats masquerading as counts) and the whole mapping is
    JSON-serializable."""
    import json

    r = _router()
    r.route([("easy", 0.99, 0.0)] * 3 + [("hard", 0.5, 0.0)] * 2)
    report = r.instrumentation.report()
    for count_field in ("n_items", "n_cheap", "n_heavy"):
        assert type(report[count_field]) is int  # genuine int, not 3.0
    for float_field in ("cheap_fraction", "compute_saved_fraction", "mean_latency"):
        assert isinstance(report[float_field], float)
    json.dumps(report)  # fully serializable


def test_cascade_accuracy_stays_within_tolerance_of_all_heavy() -> None:
    """The 'at bounded accuracy' half of the metric, not just compute savings.

    Routes a labeled workload that includes a *confident-but-wrong* cheap item
    (served cheap, so the cascade answers it wrong while the accurate heavy
    baseline gets it right). The cascade's accuracy must stay within tolerance of
    all-heavy -- and the delta is genuinely negative here, so the assertion is
    non-vacuous (it would trip if routing trusted more miscalibrated cheap
    answers).
    """
    accuracy_tolerance = 0.02
    cfg = CascadeConfig(
        low_uncertainty=0.30, high_uncertainty=0.60, cheap_cost=1.0, heavy_cost=20.0
    )
    # item = (label, cheap_answer, cheap_prob, cheap_disagreement, heavy_answer)
    items: list[tuple[int, int, float, float, int]] = []
    items += [(1, 1, 0.98, 0.0, 1)] * 40 + [(0, 0, 0.02, 0.0, 0)] * 39  # easy: cheap correct
    items += [(0, 1, 0.98, 0.0, 0)]  # confident-WRONG: served cheap -> cascade wrong
    items += [(1, 0, 0.5, 0.0, 1)] * 10 + [(0, 1, 0.5, 0.0, 0)] * 10  # hard: escalate -> heavy

    def cheap(it: tuple[int, int, float, float, int]) -> PathResult:
        return PathResult(it[1], it[2], it[3])

    def heavy(it: tuple[int, int, float, float, int]) -> PathResult:
        return PathResult(it[4], 0.99, 0.0)

    router = ConfidenceCascadeRouter(cheap, heavy, cfg, clock=_deterministic_clock())
    outcomes = router.route(items)
    cascade_correct = sum(1 for it, o in zip(items, outcomes) if o.result.answer == it[0])
    heavy_correct = sum(1 for it in items if it[4] == it[0])
    delta = cascade_correct / len(items) - heavy_correct / len(items)
    assert delta >= -accuracy_tolerance  # bounded-accuracy guarantee holds
    assert delta < 0.0  # non-vacuous: the cheap path really did cost some accuracy


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

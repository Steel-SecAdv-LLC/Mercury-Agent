# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-stream value board: every stream declares a baseline + target."""

from __future__ import annotations

import math

import pytest

from omni_mercury_engine.intel.value_metrics import (
    VALUE_METRICS,
    Direction,
    ValueMetric,
    get_value_metric,
)

_EXPECTED_STREAMS = {
    "closed_feedback_loop",
    "confidence_cascade",
    "self_consistency",
    "adversarial_co_training",
    "verifier_in_loop",
    "provenance",
}


def test_every_intel_stream_declares_a_value_metric() -> None:
    assert set(VALUE_METRICS) == _EXPECTED_STREAMS


def test_metrics_are_internally_consistent() -> None:
    for stream, metric in VALUE_METRICS.items():
        assert metric.stream == stream
        assert metric.metric and metric.unit and metric.description
        # A target must be an improvement over (or equal to) the baseline.
        if metric.direction is Direction.HIGHER_IS_BETTER:
            assert metric.target >= metric.baseline
        else:
            assert metric.target <= metric.baseline


def test_meets_target_higher_is_better() -> None:
    m = VALUE_METRICS["verifier_in_loop"]  # baseline 0, target 1, higher better
    assert m.meets_target(1.0)
    assert not m.meets_target(0.9)
    assert m.improves_on_baseline(0.0)
    assert m.improves_on_baseline(0.5)


def test_meets_target_lower_is_better() -> None:
    m = VALUE_METRICS["adversarial_co_training"]  # baseline 0.34, target 0.0, lower better
    assert m.meets_target(0.0)
    assert not m.meets_target(0.1)
    assert m.improves_on_baseline(0.34)  # at the floor is not a weakening
    assert m.improves_on_baseline(0.2)
    assert not m.improves_on_baseline(0.5)  # above the floor is a weakening


def test_nan_fails_closed() -> None:
    m = VALUE_METRICS["self_consistency"]
    assert not m.meets_target(math.nan)
    assert not m.improves_on_baseline(math.nan)


def test_summarize_and_as_dict_roundtrip() -> None:
    m = VALUE_METRICS["confidence_cascade"]
    summary = m.summarize(0.6)
    assert summary["measured"] == 0.6
    assert summary["meets_target"] is True
    assert summary["stream"] == "confidence_cascade"


def test_get_value_metric_unknown_raises() -> None:
    assert isinstance(get_value_metric("provenance"), ValueMetric)
    with pytest.raises(KeyError):
        get_value_metric("does_not_exist")

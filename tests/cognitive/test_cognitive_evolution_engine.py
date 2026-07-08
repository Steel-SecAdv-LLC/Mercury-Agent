# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the genuine curiosity/novelty engine.

The decorative self-play / rule-mutation / chain-of-thought / theory-of-mind
components (and their canned outputs) were removed. What remains is a
``CuriosityEngine`` whose novelty score is a *measured* statistical distance
from the observed distribution -- these pin that behaviour: repeated in-family
observations become progressively un-novel, a far-out observation scores high,
and the score is a monotone function of the standardized distance rather than a
hand-tuned constant.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.cognitive_evolution_engine import (
    CuriosityEngine,
    ExplorationResult,
)


def test_to_vector_is_key_order_invariant() -> None:
    """Equivalent dict observations built with different key orders map to the
    same feature vector, so novelty scores do not depend on dict construction
    order (``_to_vector`` reads keys sorted, not by insertion order).
    """
    a = CuriosityEngine._to_vector({"a": 1.0, "b": 2.0, "c": 3.0})
    b = CuriosityEngine._to_vector({"c": 3.0, "a": 1.0, "b": 2.0})
    assert a is not None and b is not None
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.array([1.0, 2.0, 3.0]))


def _feed(engine: CuriosityEngine, vectors: np.ndarray) -> ExplorationResult:
    result = None
    for vec in vectors:
        result = engine.explore("obs", {"a": float(vec[0]), "b": float(vec[1])})
    assert result is not None
    return result


def test_warmup_score_is_neutral_not_a_constant_masquerade() -> None:
    engine = CuriosityEngine()
    first = engine.explore("obs", {"a": 1.0, "b": 2.0})
    # No baseline yet -> honestly undetermined (0.5), not a fabricated novelty.
    assert first.novelty_score == 0.5
    assert first.is_novel is False
    assert first.n_observations == 1
    assert first.standardized_distance == 0.0


def test_in_distribution_observation_is_not_novel() -> None:
    engine = CuriosityEngine()
    rng = np.random.default_rng(0)
    # Build a baseline cluster, then score a point drawn from the same cluster.
    _feed(engine, rng.normal(0.0, 1.0, (60, 2)))
    typical = engine.explore("obs", {"a": 0.1, "b": -0.1})
    assert typical.novelty_score < 0.7
    assert typical.n_observations == 61


def test_out_of_distribution_observation_is_novel() -> None:
    engine = CuriosityEngine()
    rng = np.random.default_rng(1)
    _feed(engine, rng.normal(0.0, 1.0, (60, 2)))
    outlier = engine.explore("obs", {"a": 12.0, "b": -15.0})
    assert outlier.is_novel is True
    assert outlier.novelty_score > 0.7
    # Novelty is a monotone map of the measured standardized distance.
    assert outlier.standardized_distance > 3.0


def test_novelty_is_monotone_in_distance() -> None:
    engine = CuriosityEngine()
    rng = np.random.default_rng(2)
    _feed(engine, rng.normal(0.0, 1.0, (80, 2)))
    near = engine.explore("obs", {"a": 1.0, "b": 1.0})
    far = engine.explore("obs", {"a": 8.0, "b": 8.0})
    assert far.standardized_distance > near.standardized_distance
    assert far.novelty_score > near.novelty_score


def test_constant_feature_does_not_peg_novelty() -> None:
    """A zero-variance feature must not blow the score to 1.0 on every input.

    Dividing a deviation by a ~0 std (a constant feature) would peg novelty at
    1.0 for any observation; the constant dimension is excluded from the
    distance instead, so in-distribution points score low and out-of-distribution
    points in the *varying* feature are still flagged.
    """
    engine = CuriosityEngine()
    rng = np.random.default_rng(0)
    for _ in range(40):
        # feature "a" varies, feature "b" is constant.
        engine.explore("obs", {"a": float(rng.normal()), "b": 5.0})

    typical = engine.explore("obs", {"a": 0.1, "b": 5.0})
    assert np.isfinite(typical.standardized_distance)
    assert typical.novelty_score < 0.7  # not pegged to 1.0 by the constant dim

    outlier = engine.explore("obs", {"a": 20.0, "b": 5.0})
    assert outlier.is_novel is True


def test_non_finite_input_is_rejected_and_does_not_poison_estimate() -> None:
    """NaN/inf observations must not emit NaN novelty or corrupt the estimate.

    A NaN would otherwise flow into the serialized result (invalid JSON) and
    fold into the running mean/variance, permanently killing those dimensions.
    """
    engine = CuriosityEngine()
    rng = np.random.default_rng(0)
    for vec in rng.normal(0.0, 1.0, (30, 2)):
        engine.explore("obs", {"a": float(vec[0]), "b": float(vec[1])})
    healthy_count = engine.observations_seen

    nan_result = engine.explore("obs", {"a": float("nan"), "b": 1.0})
    assert nan_result.novelty_score == 0.0
    assert nan_result.is_novel is False
    # Estimate is untouched (not folded in, not corrupted).
    assert engine.observations_seen == healthy_count
    assert engine._mean is not None and bool(np.all(np.isfinite(engine._mean)))

    inf_result = engine.explore("obs", np.array([np.inf, 1.0]))
    assert inf_result.novelty_score == 0.0

    # A genuine out-of-distribution point still scores after the rejected inputs.
    outlier = engine.explore("obs", {"a": 12.0, "b": -12.0})
    assert outlier.is_novel is True


def test_empty_observation_is_rejected() -> None:
    engine = CuriosityEngine()
    result = engine.explore("obs", np.zeros((0,)))
    assert result.novelty_score == 0.0
    assert engine.observations_seen == 0


def test_non_numeric_input_yields_no_measured_novelty() -> None:
    engine = CuriosityEngine()
    result = engine.explore("obs", data=None)
    assert result.novelty_score == 0.0
    assert result.is_novel is False
    assert engine.observations_seen == 0  # nothing folded in


def test_dimension_change_resets_estimate() -> None:
    engine = CuriosityEngine()
    engine.explore("obs", {"a": 1.0, "b": 2.0})
    engine.explore("obs", {"a": 1.0, "b": 2.0})
    assert engine.observations_seen == 2
    # A 3-D observation is not comparable to the 2-D history -> reset.
    engine.explore("obs", [1.0, 2.0, 3.0])
    assert engine.observations_seen == 1


def test_statistics_report_real_counters() -> None:
    engine = CuriosityEngine(novelty_threshold=0.8)
    engine.explore("obs", {"a": 1.0, "b": 2.0})
    engine.explore("obs", {"a": 3.0, "b": 4.0})
    stats = engine.get_statistics()
    assert stats["explorations_performed"] == 2
    assert stats["observations_seen"] == 2
    assert stats["novelty_threshold"] == 0.8


def test_result_is_serialisable() -> None:
    import json

    engine = CuriosityEngine()
    result = engine.explore("anomaly:cyber", {"score": 0.9, "severity": 0.8})
    payload = result.to_dict()
    json.dumps(payload)
    assert set(payload) == {
        "exploration_id",
        "target",
        "novelty_score",
        "is_novel",
        "standardized_distance",
        "n_observations",
    }

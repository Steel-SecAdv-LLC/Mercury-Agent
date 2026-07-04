# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-consistency: disagreement metric, calibrator integration, value AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.intel.self_consistency import (
    ConsistencyDecision,
    disagreement_error_auroc,
    dispersion,
    normalized_entropy,
    self_consistency,
    self_consistency_decision,
    vote_disagreement,
    widen_uncertainty,
)
from omni_mercury_engine.intel.value_metrics import VALUE_METRICS


def test_as_dict_renders_answer_as_string_and_is_json_serializable() -> None:
    """as_dict() renders the plurality answer as a string (its documented contract),
    so a non-JSON-native sampler answer (a tuple) still serializes cleanly."""
    import json

    result = self_consistency(lambda _rng: (1, 2), n_samples=3, seed=0)
    d = result.as_dict()
    assert isinstance(d["answer"], str)  # rendered as string, not a raw tuple
    assert all(isinstance(k, str) for k in d["distribution"])
    json.dumps(d)  # must not raise


def test_vote_disagreement_bounds() -> None:
    assert vote_disagreement(["a", "a", "a"]) == 0.0
    assert vote_disagreement(["a", "b"]) == 0.5
    assert vote_disagreement(["a", "b", "c", "d"]) == 0.75
    # Empty is maximally uncertain (fail-closed).
    assert vote_disagreement([]) == 1.0


def test_normalized_entropy() -> None:
    assert normalized_entropy(["a", "a"]) == 0.0
    assert normalized_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)
    assert 0.0 < normalized_entropy(["a", "a", "a", "b"]) < 1.0


def test_dispersion_continuous() -> None:
    assert dispersion([0.5, 0.5, 0.5]) == 0.0
    assert dispersion([0.0, 1.0]) == pytest.approx(1.0)
    assert dispersion([]) == 1.0


def test_self_consistency_is_deterministic_with_seed() -> None:
    sampler = lambda rng: "x" if rng.random() < 0.8 else "y"  # noqa: E731
    a = self_consistency(sampler, n_samples=25, seed=7)
    b = self_consistency(sampler, n_samples=25, seed=7)
    assert a == b
    assert a.answer == "x"
    assert 0.0 <= a.disagreement <= 1.0
    assert a.agreement == pytest.approx(1.0 - a.disagreement)
    assert a.support == pytest.approx(a.distribution["x"] / a.n_samples)


def test_self_consistency_rejects_zero_samples() -> None:
    with pytest.raises(ValueError):
        self_consistency(lambda rng: "x", n_samples=0)


def test_widen_uncertainty_pulls_toward_half_never_flips() -> None:
    # No disagreement -> unchanged.
    assert widen_uncertainty(0.95, 0.0) == pytest.approx(0.95)
    # Full disagreement, full strength -> collapses to the boundary (0.5).
    assert widen_uncertainty(0.95, 1.0) == pytest.approx(0.5)
    # Partial -> between, on the same side of 0.5 (never flips the decision).
    w = widen_uncertainty(0.9, 0.5)
    assert 0.5 < w < 0.9


def test_decision_rule_abstains_on_high_disagreement() -> None:
    d = self_consistency_decision(0.95, 0.8, abstain_above=0.6)
    assert isinstance(d, ConsistencyDecision)
    assert d.abstained and d.decision == "abstain"
    # Low disagreement -> commits, widened prob thresholded.
    d2 = self_consistency_decision(0.95, 0.1, abstain_above=0.6)
    assert not d2.abstained and d2.decision == "positive"
    # A confident-looking prob that the paths split on can be pulled below 0.5.
    d3 = self_consistency_decision(0.62, 0.5, decision_threshold=0.5, abstain_above=0.9)
    assert d3.widened_prob < 0.62


def test_disagreement_predicts_error_auroc_meets_value_target() -> None:
    # Build a held-out set where disagreement genuinely tracks error: a latent
    # "hard" set gets both higher disagreement AND more errors.
    rng = np.random.default_rng(0)
    disagreements = []
    errors = []
    for _ in range(400):
        hard = rng.random() < 0.5
        # hard items: high disagreement, ~50% error; easy: low disagreement, ~5% error
        disagreements.append(rng.uniform(0.5, 1.0) if hard else rng.uniform(0.0, 0.4))
        p_err = 0.5 if hard else 0.05
        errors.append(1 if rng.random() < p_err else 0)
    auroc = disagreement_error_auroc(disagreements, errors)
    target = VALUE_METRICS["self_consistency"].target
    assert auroc >= target, f"disagreement AUROC {auroc:.3f} below value target {target}"


def test_disagreement_from_real_self_consistency_predicts_error() -> None:
    """The value metric measured on disagreement the stream ACTUALLY produces.

    Unlike the hand-fabricated-separable test above, this drives real
    :func:`self_consistency` runs: each item's disagreement is computed from
    actual sampled votes, and error is whether the plurality vote is wrong. A
    sampler that near-splits on hard items (which are also more error-prone)
    yields real disagreement that must rank errors above correct items at least as
    well as the declared value target.
    """
    rng = np.random.default_rng(1)
    disagreements: list[float] = []
    errors: list[int] = []
    for i in range(300):
        hard = i % 2 == 0
        # Easy: near-unanimous for the true answer (low disagreement, rarely wrong).
        # Hard: a near coin-flip (high disagreement, often wrong).
        p_true = 0.55 if hard else 0.95

        def sampler(gen: np.random.Generator, p_true: float = p_true) -> str:
            return "A" if gen.random() < p_true else "B"

        result = self_consistency(sampler, n_samples=7, seed=int(rng.integers(0, 2**31)))
        disagreements.append(result.disagreement)  # real, from the votes
        errors.append(0 if result.answer == "A" else 1)
    auroc = disagreement_error_auroc(disagreements, errors)
    target = VALUE_METRICS["self_consistency"].target
    assert auroc >= target, f"real-sampler disagreement AUROC {auroc:.3f} below target {target}"


def test_auroc_single_class_is_uninformative() -> None:
    assert disagreement_error_auroc([0.1, 0.2, 0.3], [0, 0, 0]) == 0.5
    assert disagreement_error_auroc([], []) == 0.5

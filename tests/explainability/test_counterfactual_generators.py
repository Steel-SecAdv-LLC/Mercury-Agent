# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Executing tests for the GENETIC generator, Gower distance, and the gradient-search infeasibility barrier.

Regression context: ``CounterfactualMethod.GENETIC`` and
``DistanceMetric.GOWER`` shipped with zero executing tests, the genetic
fitness silently used L2 while its docstring claimed Gower-when-metadata,
and the DiCE/Wachter NaN infeasibility barrier was unreachable through the
detection seam (whose score wrapper raises instead of returning NaN).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# ``unused-ignore``: the dev venv's editable install may point at a sibling
# worktree that predates the GENETIC generator, ``gower_distance``,
# ``NonFiniteScoreError``, and the ``detection_counterfactuals`` seam; a
# correctly installed tree (CI) stays clean.
from omni_mercury_engine.explainability.counterfactuals import (  # type: ignore[attr-defined,unused-ignore]
    DiCECounterfactual,
    FeatureConstraint,
    GeneticCounterfactual,
    NonFiniteScoreError,
    WachterCounterfactual,
    gower_distance,
)
from omni_mercury_engine.explainability.detection_counterfactuals import (  # type: ignore[import-not-found,unused-ignore]
    explain_detection_counterfactual,
)


def _threshold_model(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Anomalous (1) iff x0 > 0.5 — a crisp, piecewise decision function."""
    X = np.atleast_2d(X)
    return (X[:, 0] > 0.5).astype(float)


class TestGowerDistance:
    def test_numeric_ranges_normalize(self) -> None:
        # |1-3|/range 4 = 0.5 and |10-10| = 0 -> mean 0.25
        d = gower_distance(
            np.array([1.0, 10.0]),
            np.array([3.0, 10.0]),
            feature_types=["numeric", "numeric"],
            feature_ranges=np.array([4.0, 5.0]),
        )
        assert d == pytest.approx(0.25)

    def test_categorical_mismatch_counts_one(self) -> None:
        # categorical equal -> 0, categorical different -> 1; mean over 2 = 0.5
        d = gower_distance(
            np.array([2.0, 7.0]),
            np.array([2.0, 3.0]),
            feature_types=["categorical", "categorical"],
        )
        assert d == pytest.approx(0.5)

    def test_defaults_are_all_numeric_unit_range(self) -> None:
        d = gower_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert d == pytest.approx(0.5)


class TestGeneticCounterfactual:
    def test_flips_threshold_model(self) -> None:
        gen = GeneticCounterfactual(_threshold_model, seed=0)
        result = gen.generate(np.array([1.2, 0.0, 0.0]), target_class=0)
        assert result.counterfactuals, "genetic search returned no candidate"
        best = result.counterfactuals[0]
        assert best.validity is True
        # Verified against the REAL decision function, not the stored field.
        assert float(_threshold_model(best.counterfactual.reshape(1, -1))[0]) == 0.0

    def test_deterministic_for_fixed_seed(self) -> None:
        a = GeneticCounterfactual(_threshold_model, seed=7).generate(
            np.array([1.2, 0.0, 0.0]), target_class=0
        )
        b = GeneticCounterfactual(_threshold_model, seed=7).generate(
            np.array([1.2, 0.0, 0.0]), target_class=0
        )
        assert a.counterfactuals and b.counterfactuals
        np.testing.assert_array_equal(
            a.counterfactuals[0].counterfactual, b.counterfactuals[0].counterfactual
        )

    def test_immutable_feature_is_pinned(self) -> None:
        gen = GeneticCounterfactual(
            _threshold_model,
            feature_constraints=[FeatureConstraint(name="x2", feature_idx=2, is_mutable=False)],
            seed=0,
        )
        original = np.array([1.2, 0.0, 3.5])
        result = gen.generate(original, target_class=0)
        assert result.counterfactuals
        assert result.counterfactuals[0].counterfactual[2] == pytest.approx(3.5)

    def test_gower_metric_engaged_when_metadata_present(self) -> None:
        """The reported distance must BE the Gower distance when metadata
        exists (regression: the fitness and the reported distance silently
        used L2, contradicting the class docstring)."""
        gen = GeneticCounterfactual(
            _threshold_model,
            seed=0,
            feature_types=["numeric", "numeric", "numeric"],
            feature_ranges=np.array([10.0, 10.0, 10.0]),
        )
        original = np.array([1.2, 0.0, 0.0])
        result = gen.generate(original, target_class=0)
        assert result.counterfactuals
        best = result.counterfactuals[0]
        expected = gower_distance(
            original,
            best.counterfactual,
            feature_types=["numeric", "numeric", "numeric"],
            feature_ranges=np.array([10.0, 10.0, 10.0]),
        )
        assert best.distance == pytest.approx(expected)
        # And it is NOT the L2 value (unless they coincide numerically).
        l2 = float(np.sqrt(np.sum((original - best.counterfactual) ** 2)))
        assert best.distance != pytest.approx(l2) or l2 == pytest.approx(expected)


def _nan_region_model(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Scores NaN when x0 < 0 (unscorable region), else a smooth ramp."""
    X = np.atleast_2d(X)
    scores = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.5) * 4.0))
    scores = np.where(X[:, 0] < 0.0, np.nan, scores)
    return scores


class TestInfeasibilityBarrier:
    def test_dice_survives_nan_region(self) -> None:
        """A NaN-scoring region must repel the search, not abort it."""
        gen = DiCECounterfactual(_nan_region_model, seed=0)
        result = gen.generate(np.array([1.5, 0.0]), target_class=0)
        # No exception; any candidate claiming validity must genuinely flip.
        for cf in result.counterfactuals:
            if cf.validity:
                rescored = float(_nan_region_model(cf.counterfactual.reshape(1, -1))[0])
                assert np.isfinite(rescored) and rescored < 0.5

    def test_wachter_survives_nan_region(self) -> None:
        gen = WachterCounterfactual(_nan_region_model, seed=0)
        result = gen.generate(np.array([1.5, 0.0]), target_class=0)
        for cf in result.counterfactuals:
            if cf.validity:
                rescored = float(_nan_region_model(cf.counterfactual.reshape(1, -1))[0])
                assert np.isfinite(rescored) and rescored < 0.5

    def test_barrier_catches_raising_score_wrapper(self) -> None:
        """A fail-loud wrapper raising NonFiniteScoreError == a NaN score."""

        def raising_model(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            X = np.atleast_2d(X)
            if np.any(X[:, 0] < 0.0):
                raise NonFiniteScoreError("unscorable region")
            return 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.5) * 4.0))

        result = DiCECounterfactual(raising_model, seed=0).generate(
            np.array([1.5, 0.0]), target_class=0
        )
        for cf in result.counterfactuals:
            if cf.validity:
                assert float(raising_model(cf.counterfactual.reshape(1, -1))[0]) < 0.5

    def test_search_failure_is_honest_never_a_fabricated_success(self) -> None:
        """A model erroring on every candidate yields a transparent failure.

        Real detectors raise data-dependent errors on extreme candidates
        (e.g. np.histogram on all-NaN intermediates); the search logs the
        failure and, because every candidate is re-scored for validity, an
        aborted search can never be recorded as a successful flip.
        """

        def erroring_model(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            X = np.atleast_2d(X)
            if np.any(np.abs(X[:, 1]) > 0.05):  # everything off the origin errors
                raise ValueError("detector cannot score this candidate")
            return np.ones(X.shape[0])

        result = WachterCounterfactual(erroring_model, seed=0).generate(
            np.array([1.5, 0.0]), target_class=0
        )
        assert all(not cf.validity for cf in result.counterfactuals)
        assert result.coverage_score == 0.0

    def test_detection_seam_dice_survives_nan_candidates(self) -> None:
        """End-to-end through the seam: candidates in the NaN region must not
        abort the DiCE search (the seam's score wrapper raises
        NonFiniteScoreError, which the barrier now absorbs)."""

        def score_fn(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return _nan_region_model(X)

        x = np.array([1.5, 0.0])  # finite score at the original point
        result = explain_detection_counterfactual(score_fn, x, 0.5, method="dice", seed=0)
        # Transparent outcome either way: a genuine flip or an explicit failure.
        if result.flipped:
            rescored = float(score_fn(result.counterfactual_x.reshape(1, -1))[0])
            assert np.isfinite(rescored) and rescored <= 0.5

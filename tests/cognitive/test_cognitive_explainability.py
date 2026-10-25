# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test cognitive explainability."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.cognitive.explainability import (
    ExplainabilityEngine,
    ExplanationType,
    LIMEExplainer,
    SHAPExplainer,
)


def test_shap_explainer_instantiation() -> None:
    explainer = SHAPExplainer(n_samples=10)
    assert explainer is not None
    assert explainer.n_samples == 10


def test_lime_explainer_instantiation() -> None:
    explainer = LIMEExplainer()
    assert explainer is not None


def test_explainability_engine_instantiation() -> None:
    engine = ExplainabilityEngine()
    assert engine is not None


def test_explanation_type_is_enum() -> None:
    assert hasattr(ExplanationType, "SHAP")
    assert hasattr(ExplanationType, "LIME")


Model = Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]
NAMES = ["f0", "f1", "f2"]


class TestLimeSurrogateSlopes:
    """The in-repo LIME surrogate (the path every install takes) fits local slopes.

    Each model has a known local gradient. The replaced surrogate regressed
    without an intercept, so a model's level was spread across every
    feature: 3 + 2*x0 at (1, 1, 1) read as slopes (2.95, 1.06, 0.98), and
    100 + x1 at the origin as (-35.8, 24.3, 11.5).
    """

    @staticmethod
    def _slopes(model: Model, instance: list[float]) -> tuple[list[float], float]:
        weights, r2 = LIMEExplainer(n_samples=2000, seed=0)._approximate_lime(
            model, np.array(instance), NAMES
        )
        return [weights[name] for name in NAMES], r2

    @pytest.mark.parametrize(
        ("model", "instance", "slopes"),
        [
            (lambda x: 3 + 2 * x[:, 0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]),
            (lambda x: 100 + x[:, 1], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            (lambda x: 2 * x[:, 0] - x[:, 1], [5.0, -3.0, 10.0], [2.0, -1.0, 0.0]),
            (lambda x: -4 + 0.5 * x[:, 2], [-2.0, 7.0, 1.0], [0.0, 0.0, 0.5]),
        ],
    )
    def test_affine_models_are_recovered_exactly(
        self, model: Model, instance: list[float], slopes: list[float]
    ) -> None:
        fitted, r2 = self._slopes(model, instance)
        assert fitted == pytest.approx(slopes, abs=1e-9)
        assert r2 == pytest.approx(1.0)

    def test_nonlinear_model_gets_its_local_derivative(self) -> None:
        fitted, r2 = self._slopes(lambda x: x[:, 0] ** 2, [3.0, 0.0, 0.0])
        assert fitted == pytest.approx([6.0, 0.0, 0.0], abs=0.05)
        assert r2 > 0.99

    def test_constant_model_has_no_attribution(self) -> None:
        fitted, r2 = self._slopes(lambda x: np.full(len(x), 7.0), [1.0, 2.0, 3.0])
        assert fitted == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
        assert r2 == 1.0

    def test_explanation_ranks_the_feature_that_moves_the_output(self) -> None:
        explanation = LIMEExplainer(n_samples=2000, seed=0).explain(
            lambda x: 3 - 2 * x[:, 1], np.array([1.0, 1.0, 1.0]), NAMES
        )
        top = explanation.feature_importances[0]
        assert (top.feature_name, top.direction) == ("f1", "negative")
        assert top.importance == pytest.approx(2.0, abs=1e-9)
        assert [fi.importance for fi in explanation.feature_importances[1:]] == pytest.approx(
            [0.0, 0.0], abs=1e-9
        )
        assert explanation.prediction == pytest.approx(1.0)
        assert explanation.local_accuracy == pytest.approx(1.0)


class TestShapFallbackIsShapley:
    """The in-repo SHAP fallback returns Shapley values and a consistent base.

    The replaced fallback averaged random coalitions with equal weight (a
    Banzhaf estimate, which breaks efficiency under interactions) and set
    ``base_value`` to the prediction itself: x0*x1 + x2 at (2, 3, 1) gave
    (4.5, 4.5, 1.0) against Shapley's (3, 3, 1), with local accuracy -0.43.
    """

    @staticmethod
    def _phi(model: Model, instance: list[float], **kwargs: Any) -> list[float]:
        explanation = SHAPExplainer(seed=0, **kwargs).explain(model, np.array(instance), NAMES)
        signed = {
            fi.feature_name: fi.importance * (1.0 if fi.direction == "positive" else -1.0)
            for fi in explanation.feature_importances
        }
        return [signed[name] for name in NAMES]

    @pytest.mark.parametrize(
        ("model", "instance", "phi"),
        [
            (lambda x: 3 + 2 * x[:, 0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]),
            (lambda x: x[:, 0] * x[:, 1] + x[:, 2], [2.0, 3.0, 1.0], [3.0, 3.0, 1.0]),
            (lambda x: x[:, 0] * x[:, 1] * x[:, 2], [1.0, 1.0, 1.0], [1 / 3, 1 / 3, 1 / 3]),
            (lambda x: np.maximum(x[:, 0], x[:, 1]), [4.0, 2.0, 9.0], [3.0, 1.0, 0.0]),
        ],
    )
    def test_exact_shapley_values(
        self, model: Model, instance: list[float], phi: list[float]
    ) -> None:
        assert self._phi(model, instance) == pytest.approx(phi, abs=1e-9)

    def test_efficiency_and_base_value(self) -> None:
        def model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(5 + x[:, 0] * x[:, 1] - x[:, 2], dtype=np.float64)

        explanation = SHAPExplainer(seed=0).explain(model, np.array([2.0, 3.0, 4.0]), NAMES)
        assert explanation.base_value == pytest.approx(5.0)  # f(baseline = 0)
        assert explanation.prediction == pytest.approx(7.0)
        assert explanation.local_accuracy == pytest.approx(1.0)
        assert explanation.metadata["method"] == "exact_shapley"

    def test_background_mean_is_the_baseline_without_the_library(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.cognitive import explainability

        monkeypatch.setattr(explainability, "_optional_module", lambda name: None)
        background = np.array([[1.0, 1.0, 1.0], [3.0, 5.0, 1.0]])  # mean (2, 3, 1)

        def model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(2 * x[:, 0] + x[:, 1], dtype=np.float64)

        explanation = SHAPExplainer(background_data=background, seed=0).explain(
            model, np.array([4.0, 4.0, 4.0]), NAMES
        )
        assert explanation.base_value == pytest.approx(7.0)
        assert explanation.local_accuracy == pytest.approx(1.0)
        assert explanation.feature_importances[0].importance == pytest.approx(4.0)

    def test_background_width_mismatch_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.cognitive import explainability

        monkeypatch.setattr(explainability, "_optional_module", lambda name: None)
        explainer = SHAPExplainer(background_data=np.zeros((4, 2)), seed=0)
        with pytest.raises(ValueError, match="2 features, the instance 3"):
            explainer.explain(lambda x: x[:, 0], np.array([1.0, 1.0, 1.0]), NAMES)

    def test_sampled_orderings_above_the_exact_limit_keep_efficiency(self) -> None:
        from omni_mercury_engine.cognitive.explainability import _EXACT_SHAPLEY_MAX_FEATURES

        n = _EXACT_SHAPLEY_MAX_FEATURES + 2

        def model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            return np.asarray(x[:, 0] * x[:, 1] * x[:, 2] + x[:, 3:].sum(axis=1), dtype=np.float64)

        phi = SHAPExplainer(n_samples=400, seed=0)._approximate_shap(model, np.ones(n))
        assert float(phi.sum()) == pytest.approx(float(n - 3 + 1))
        assert phi[3:] == pytest.approx(np.ones(n - 3))
        assert phi[:3] == pytest.approx([1 / 3] * 3, abs=0.06)

"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Tests for the explainability dependency contract (brief: "SHAP/LIME explainer
revival").  The validated default path — IntegratedGradients + faithfulness
evaluator (PR #265, benchmarks/explanation_fidelity.py) — must work with **no**
third-party explainer installed, and SHAP/LIME must be clean opt-ins behind the
new `[explainability]` extra that degrade gracefully when absent.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.cognitive import explainability as ex

REPO_ROOT = Path(__file__).resolve().parent.parent


def _linear_model(x: np.ndarray) -> np.ndarray:
    """Deterministic differentiable model: score = 3*f0 - 2*f1 + 0.5*f2."""
    x = np.atleast_2d(x)
    w = np.array([3.0, -2.0, 0.5])
    # ``np.asarray`` pins the return to ndarray; the bare ``x @ w`` is typed
    # ``Any`` by the numpy stubs and trips ``warn_return_any`` (no-any-return).
    return np.asarray(x @ w)


def test_availability_flags_are_bool() -> None:
    assert isinstance(ex.SHAP_AVAILABLE, bool)
    assert isinstance(ex.LIME_AVAILABLE, bool)


def test_integrated_gradients_default_works_without_shap_or_lime() -> None:
    """The default explainer must not require SHAP/LIME to be installed."""
    explainer = ex.IntegratedGradientsExplainer(n_steps=32)
    instance = np.array([1.0, 1.0, 1.0])
    result = explainer.explain(_linear_model, instance, ["f0", "f1", "f2"])
    assert result.explanation_type == ex.ExplanationType.INTEGRATED_GRADIENTS
    assert len(result.feature_importances) == 3
    # `importance` is an attribution magnitude; it must track the |weights|
    # 3 (f0) >= 2 (f1) >= 0.5 (f2), proving IG produces sensible attributions
    # with no SHAP/LIME installed.
    mags = {fi.feature_name: abs(fi.importance) for fi in result.feature_importances}
    assert mags["f0"] >= mags["f1"] >= mags["f2"]
    assert mags["f0"] > 0.0


def test_faithfulness_evaluator_is_self_contained() -> None:
    evaluator = ex.FaithfulnessEvaluator()
    assert evaluator is not None  # constructs with no external explainer dep


def test_explainability_extra_declares_shap_and_lime() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "explainability" in extras, "missing [explainability] extra"
    joined = " ".join(extras["explainability"]).lower()
    assert "shap" in joined and "lime" in joined
    # It must be reachable from the umbrella `all` extra.
    assert any("explainability" in dep for dep in extras["all"])


@pytest.mark.skipif(not ex.SHAP_AVAILABLE, reason="shap not installed ([explainability] extra)")
def test_shap_adapter_when_present() -> None:
    rng = np.random.default_rng(0)
    background = rng.normal(size=(20, 3))
    explainer = ex.SHAPExplainer(background_data=background, n_samples=50, seed=0)
    result = explainer.explain(_linear_model, np.array([1.0, 1.0, 1.0]), ["f0", "f1", "f2"])
    assert len(result.feature_importances) == 3


@pytest.mark.skipif(not ex.LIME_AVAILABLE, reason="lime not installed ([explainability] extra)")
def test_lime_adapter_when_present() -> None:
    rng = np.random.default_rng(0)
    training = rng.normal(size=(50, 3))
    explainer = ex.LIMEExplainer(training_data=training, mode="regression", n_samples=500, seed=0)
    result = explainer.explain(_linear_model, np.array([1.0, 1.0, 1.0]), ["f0", "f1", "f2"])
    assert len(result.feature_importances) >= 1

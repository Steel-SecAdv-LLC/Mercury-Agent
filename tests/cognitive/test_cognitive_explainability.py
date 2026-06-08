# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

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

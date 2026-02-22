"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

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

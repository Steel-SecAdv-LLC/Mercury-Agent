# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for BiometricAnomalyModel's single-image visual anomaly path."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.models import biometric as biometric_module
from omni_mercury_engine.models.biometric import BiometricAnomalyModel
from omni_mercury_engine.vision import AnalyzerConfig

from .vision.conftest import inject, woven_surface


@pytest.fixture(scope="module")
def fitted_model() -> BiometricAnomalyModel:
    normals = [woven_surface(s) for s in range(40)]
    return BiometricAnomalyModel().fit_normal(
        normals, AnalyzerConfig(alpha=0.2, type_simulations=4, calibration_fraction=0.5)
    )


def _without_deepface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(biometric_module, "DeepFace", None)


def test_fitted_model_returns_a_calibrated_localised_analysis(
    fitted_model: BiometricAnomalyModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_deepface(monkeypatch)
    image, mask = inject(woven_surface(500), "stain", 1)
    result = fitted_model.predict(image)
    analysis = result["visual_analysis"]
    assert analysis["state"] == "grounded" and analysis["is_anomalous"]
    assert result["anomaly_scores"][0] == pytest.approx(1.0 - analysis["p_value"])
    x0, y0, x1, y1 = analysis["regions"][0]["box"]
    assert mask[y0:y1, x0:x1].any()
    assert analysis["regions"][0]["anomaly_type"] == "color_shift"


def test_defective_images_score_above_normal_ones(
    fitted_model: BiometricAnomalyModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_deepface(monkeypatch)
    normal = [
        float(fitted_model.predict(woven_surface(600 + i))["anomaly_scores"][0]) for i in range(4)
    ]
    defective = [
        float(
            fitted_model.predict(inject(woven_surface(700 + i), "noise", i)[0])["anomaly_scores"][0]
        )
        for i in range(4)
    ]
    assert min(defective) > max(normal)


def test_no_face_attributes_are_invented_without_deepface(
    fitted_model: BiometricAnomalyModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_deepface(monkeypatch)
    result = fitted_model.predict(woven_surface(800))
    assert result["attributes_measured"] is False
    for key in ("age", "gender_confidence", "emotion_confidence"):
        assert key not in result


def test_unfitted_model_screens_but_never_invents_a_content_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_deepface(monkeypatch)
    model = BiometricAnomalyModel()
    clean = model.predict(woven_surface(900))
    assert clean["visual_analysis"]["state"] == "unavailable"
    assert "fit_normal" in clean["visual_analysis"]["reason"]
    assert clean["anomaly_scores"][0] == pytest.approx(0.5)

    # A screen-replay capture: moire peaks are a presentation-attack cue.
    yy, xx = np.indices((96, 96))
    replay = np.clip(
        woven_surface(901) * 0.5 + 0.15 * np.sin(2 * np.pi * (0.31 * xx + 0.07 * yy))[..., None],
        0,
        1,
    )
    cues = model.predict(replay)["visual_analysis"]["presentation_attack_cues"]
    assert any("moire" in cue for cue in cues)


def test_tabular_route_keeps_its_score_and_claims_no_attributes() -> None:
    features = np.random.default_rng(0).random((5, 12))
    result = BiometricAnomalyModel().predict(features)
    assert result["attributes_measured"] is False
    assert "tabular route" in result["anomaly_score_basis"]
    assert "age" not in result


class _StubDeepFace:
    """Stands in for DeepFace with a fixed ``analyze`` result."""

    def __init__(self, analysis: object) -> None:
        self._analysis = analysis

    def analyze(self, *_args: object, **_kwargs: object) -> object:
        return self._analysis


@pytest.mark.parametrize(
    ("analysis", "measured", "present"),
    [
        ({}, False, set()),
        ([{"age": 31}], True, {"age"}),
        ({"gender": {"Woman": 0.9, "Man": 0.1}}, True, {"gender_confidence"}),
    ],
)
def test_attributes_are_reported_only_as_far_as_deepface_measured_them(
    monkeypatch: pytest.MonkeyPatch, analysis: object, measured: bool, present: set[str]
) -> None:
    monkeypatch.setattr(biometric_module, "DeepFace", _StubDeepFace(analysis))
    result = BiometricAnomalyModel().predict(woven_surface(805))
    assert result["attributes_measured"] is measured
    assert {k for k in ("age", "gender_confidence", "emotion_confidence") if k in result} == present

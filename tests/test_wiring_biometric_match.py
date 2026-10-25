# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the real reference-vs-test biometric match path.

Previously ``predict({"reference": ..., "test": ...})`` silently dropped the
test image and returned reference-only analysis. The match is now a genuine
embedding cosine similarity: a self-match is exactly 1.0, structurally
different images score lower, the pipeline anomaly is ``1 - match``, and every
unavailable backend path refuses with a named reason instead of fabricating.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.models.biometric import BiometricAnomalyModel

_RNG = np.random.RandomState(3)
_NOISE_IMG = (_RNG.rand(64, 64, 3) * 255).astype(np.uint8)
# A smooth gradient is spectrally very unlike white noise, so even the
# harmonic-fallback embedding separates them clearly.
_GRADIENT_IMG = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[:, None, None], (1, 64, 3))


def test_self_match_is_exactly_one() -> None:
    model = BiometricAnomalyModel()
    result = model.predict({"reference": _NOISE_IMG, "test": _NOISE_IMG.copy()})
    assert result["match_score"] == pytest.approx(1.0, abs=1e-9)
    assert result["model_type"] == "biometric"
    assert result["embedding_backend"] in ("deepface", "harmonic_fallback")
    assert "verification_note" in result


def test_different_images_score_below_self_match_and_set_anomaly() -> None:
    model = BiometricAnomalyModel()
    self_match = model.predict({"reference": _GRADIENT_IMG, "test": _GRADIENT_IMG.copy()})
    cross = model.predict({"reference": _GRADIENT_IMG, "test": _NOISE_IMG})
    assert cross["match_score"] < self_match["match_score"]
    # The pipeline-facing anomaly is exactly the mismatch.
    assert float(cross["anomaly_scores"][0]) == pytest.approx(1.0 - cross["match_score"], abs=1e-6)


def test_zero_embedding_refuses_with_reason_instead_of_fabricating() -> None:
    # Tabular (non-image) input with harmonic features disabled yields an
    # all-zero embedding on every backend; the match must refuse, not invent.
    model = BiometricAnomalyModel(config={"use_harmonic_features": False})
    tabular = np.ones((4, 16))
    result = model.predict({"reference": tabular, "test": tabular.copy()})
    assert result["match_score"] is None
    assert "error" in result and "zero" in result["error"]


def test_age_progression_refuses_array_reference_with_named_reason() -> None:
    model = BiometricAnomalyModel()
    result = model.predict(
        {"reference": _NOISE_IMG, "test": _NOISE_IMG, "enable_age_progression": True}
    )
    record = result["age_progression"]
    assert record["applied"] is False
    assert "image path" in record["reason"]
    assert "match_score_age_robust" not in result  # never fabricated when not applied


def test_age_progression_on_faceless_image_refuses_honestly(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "noise.png")
    cv2.imwrite(path, _NOISE_IMG[..., ::-1])
    model = BiometricAnomalyModel()
    result = model.predict({"reference": path, "test": _NOISE_IMG, "enable_age_progression": True})
    # Random noise contains no face; the progression must say so, and the
    # plain match still stands on its own.
    record = result["age_progression"]
    assert record["applied"] is False
    assert "failed" in record["reason"].lower() or "face" in record["reason"].lower()
    assert result["match_score"] is not None


def test_single_image_path_contract_unchanged() -> None:
    model = BiometricAnomalyModel()
    result = model.predict({"reference": _NOISE_IMG})
    assert result["model_type"] == "biometric"
    assert "anomaly_scores" in result
    assert "match_score" not in result  # no match is claimed without a test image


def test_harmonic_extractor_refuses_non_numeric_loudly() -> None:
    # Regression: a raw path string used to reach the harmonic extractor as a
    # 0-d numpy string array (only in DeepFace-installed environments) and die
    # with an IndexError deep in reshape. It must refuse with a named reason.
    model = BiometricAnomalyModel()
    with pytest.raises(ValueError, match="numeric array"):
        model._extract_harmonic_features(np.array("some/path.png"))


def test_match_is_symmetric_and_deterministic() -> None:
    model = BiometricAnomalyModel()
    ab = model.predict({"reference": _GRADIENT_IMG, "test": _NOISE_IMG})["match_score"]
    ba = model.predict({"reference": _NOISE_IMG, "test": _GRADIENT_IMG})["match_score"]
    again = model.predict({"reference": _GRADIENT_IMG, "test": _NOISE_IMG})["match_score"]
    assert ab == pytest.approx(ba, abs=1e-9)  # cosine is symmetric
    assert ab == again  # same inputs -> identical score, every call


def test_match_handles_grayscale_and_float_images() -> None:
    model = BiometricAnomalyModel()
    gray = _NOISE_IMG[..., :1]  # (H, W, 1)
    floaty = _NOISE_IMG.astype(np.float64) / 255.0
    for image in (gray, floaty):
        result = model.predict({"reference": image, "test": image.copy()})
        assert result["match_score"] == pytest.approx(1.0, abs=1e-9)


class _StubDeepFace:
    """Stands in for DeepFace and records what reaches it.

    Real ``analyze``/``represent`` download about 1.1 GB of model weights on
    first use (age 539 MB, gender 537 MB, emotion 6 MB) and, under OpenCV 5
    wheels that no longer ship Haar cascades, the detector from GitHub. A unit
    test of the path-loading contract must not depend on that.
    """

    def __init__(self) -> None:
        self.inputs: list[Any] = []

    def analyze(self, img: Any, **_: Any) -> list[dict[str, Any]]:
        self.inputs.append(img)
        return [
            {
                "age": 31,
                "gender": {"Man": 60.0, "Woman": 40.0},
                "emotion": {"neutral": 90.0, "happy": 10.0},
            }
        ]

    def represent(self, img: Any, **_: Any) -> list[dict[str, Any]]:
        self.inputs.append(img)
        return [{"embedding": np.linspace(0.1, 1.0, 128).tolist()}]


def test_single_image_path_input_finally_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The documented "path or array" contract never worked for paths: the old
    # np.array(path) coercion produced a 0-d string array that DeepFace
    # errored on and the harmonic fallback crashed on. A path now loads to
    # pixels and analyses like an array, so what reaches DeepFace is a pixel
    # array, never the path string.
    cv2 = pytest.importorskip("cv2")
    stub = _StubDeepFace()
    monkeypatch.setattr("omni_mercury_engine.models.biometric._deepface", lambda: stub)
    path = str(tmp_path / "ref.png")
    cv2.imwrite(path, _NOISE_IMG[..., ::-1])
    model = BiometricAnomalyModel()

    result = model.predict({"reference": path})
    assert result["model_type"] == "biometric"
    assert "error" not in result
    assert "anomaly_scores" in result
    assert result["age"] == 31 and result["attributes_measured"] is True

    # extract_features honours the same contract (its annotation says str).
    features = np.asarray(model.extract_features(path))
    assert features.shape[-1] == 128
    assert float(np.linalg.norm(features)) > 0.0

    # And a path-loaded image embeds identically to its pixel array.
    direct = model.predict({"reference": path, "test": _NOISE_IMG})
    assert direct["match_score"] == pytest.approx(1.0, abs=1e-9)

    assert stub.inputs, "DeepFace was never reached"
    assert all(isinstance(seen, np.ndarray) for seen in stub.inputs), [type(s) for s in stub.inputs]


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MERCURY_NETWORK_TESTS") != "1",
    reason="runs real DeepFace, which downloads ~1.1 GB of weights; set MERCURY_NETWORK_TESTS=1",
)
def test_single_image_path_input_runs_real_deepface(tmp_path: Path) -> None:
    """End to end against the real library.

    Needs DeepFace installed (``pip install -e ".[face]"``, Python < 3.14) and
    network access. No CI lane installs ``[face]`` today, so this runs locally
    and wherever a lane adds the extra.
    """
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("deepface")
    path = str(tmp_path / "ref.png")
    cv2.imwrite(path, _NOISE_IMG[..., ::-1])
    model = BiometricAnomalyModel()

    result = model.predict({"reference": path})
    assert result["model_type"] == "biometric"
    assert "error" not in result, result.get("error")
    assert "anomaly_scores" in result

    features = np.asarray(model.extract_features(path))
    assert features.shape[-1] == 128
    assert float(np.linalg.norm(features)) > 0.0

    direct = model.predict({"reference": path, "test": _NOISE_IMG})
    assert direct["match_score"] == pytest.approx(1.0, abs=1e-9)


def test_unloadable_path_returns_error_dict_not_exception(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    missing = str(tmp_path / "does_not_exist.png")
    result = BiometricAnomalyModel().predict({"reference": missing})
    assert "error" in result
    assert "does_not_exist.png" in result["error"]

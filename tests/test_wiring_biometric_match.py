# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the real reference-vs-test biometric match path.

Previously ``predict({"reference": ..., "test": ...})`` silently dropped the
test image and returned reference-only analysis. The match is now a genuine
embedding cosine similarity: a self-match is exactly 1.0, structurally
different images score lower, the pipeline anomaly is ``1 - match``, and every
unavailable backend path refuses with a named reason instead of fabricating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

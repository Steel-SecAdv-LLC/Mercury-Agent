# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the hardened heuristic age-progression engine.

The transform itself is real and directional (aging darkens and adds
wrinkle-scale noise; rejuvenation brightens and smooths), seeded and
deterministic. The honesty contract: when no real embedding backend exists,
the embedding is ``None`` and the reported similarity is ``None`` — never a
random stand-in presented as a measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.models.biometric_multimodal import (
    FACENET_AVAILABLE,
    AgeProgressionEngine,
    AgeProgressionResult,
    QuantumAgeVariant,
)

_FACE = (np.random.RandomState(5).rand(160, 160, 3) * 255).astype(np.uint8)


@pytest.mark.skipif(FACENET_AVAILABLE, reason="exercises the backend-absent honesty path")
def test_embedding_without_backend_is_none_not_random() -> None:
    engine = AgeProgressionEngine(seed=0)
    assert engine.extract_facenet_embedding(_FACE) is None


def test_result_similarity_defaults_to_unmeasured() -> None:
    assert AgeProgressionResult(success=False).similarity is None


def test_aging_darkens_and_rejuvenation_smooths() -> None:
    cv2 = pytest.importorskip("cv2")
    engine = AgeProgressionEngine(seed=0)

    aged = engine.apply_polynomial_age_filter(_FACE, 10)
    young = engine.apply_polynomial_age_filter(_FACE, -10)

    # Aging: brightness_adj = 1 - 0.02*delta dominates -> mean drops.
    assert float(aged.mean()) < float(_FACE.mean())

    # Rejuvenation: Gaussian blur removes high-frequency energy.
    def laplacian_var(img: np.ndarray) -> float:
        return float(cv2.Laplacian(img.astype(np.uint8), cv2.CV_64F).var())

    assert laplacian_var(young) < laplacian_var(_FACE)


def test_same_seed_reproduces_identical_progression() -> None:
    a = AgeProgressionEngine(seed=7).apply_polynomial_age_filter(_FACE, 10)
    b = AgeProgressionEngine(seed=7).apply_polynomial_age_filter(_FACE, 10)
    assert np.array_equal(a, b)


def test_different_seeds_produce_different_noise() -> None:
    a = AgeProgressionEngine(seed=7).apply_polynomial_age_filter(_FACE, 10)
    b = AgeProgressionEngine(seed=8).apply_polynomial_age_filter(_FACE, 10)
    assert not np.array_equal(a, b)


def test_age_distribution_is_a_normalized_symmetric_kernel() -> None:
    dist = QuantumAgeVariant.compute_age_probability_distribution(40)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert max(dist, key=lambda k: dist[k]) == 40
    for offset in (1, 3, 7):
        assert dist[40 - offset] == pytest.approx(dist[40 + offset], abs=1e-12)
    # Monotone decay away from the peak on each side.
    assert dist[40] > dist[41] > dist[45] > dist[50]


@pytest.mark.skipif(FACENET_AVAILABLE, reason="exercises the backend-absent honesty path")
def test_multimodal_engine_never_fabricates(tmp_path: Path) -> None:
    import importlib

    multimodal = importlib.import_module("omni_mercury_engine.models.biometric_multimodal")
    if multimodal.DEEPFACE_AVAILABLE or multimodal.FACE_RECOGNITION_AVAILABLE:
        pytest.skip("a real embedding backend is installed; absence paths unreachable")
    cv2 = pytest.importorskip("cv2")

    path = str(tmp_path / "img.png")
    cv2.imwrite(path, _FACE[..., ::-1])
    engine = multimodal.MultimodalBiometricEngine(seed=0)

    # Features: None, never random noise presented as an embedding.
    assert engine.extract_features(path) is None

    # Attributes: an unmeasured record, never a fabricated 30-year-old.
    attributes = engine.analyze_attributes(path)
    assert attributes.get("measured") is False
    assert "age" not in attributes

    # The two-image pipeline fails honestly instead of matching noise to noise.
    result = engine.match_faces(path, path)
    assert result.success is False
    assert "Feature extraction failed" in result.metadata.get("message", "")

    # Symbolic fusion refuses rather than fusing random stand-in features.
    with pytest.raises(ValueError, match="refusing"):
        engine.fuse_with_symbolic(path, {"csdm": 0.9})

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioural tests for craniofacial growth, deterministic texture aging and the progression audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from omni_mercury_engine.models.biometric_multimodal import (
    CRANIOFACIAL_K_RANGE,
    AgeProgressionEngine,
    cardioidal_strain,
    craniofacial_growth_fraction,
    structural_similarity,
)


def _face(size: int = 160) -> np.ndarray[Any, Any]:
    """A synthetic aligned face crop: head ellipse, eyes, mouth, skin texture, sensor noise."""
    yy, xx = np.indices((size, size))
    face = np.full((size, size, 3), 0.15)
    face[((yy - 72) / 62.0) ** 2 + ((xx - 80) / 48.0) ** 2 <= 1] = [0.8, 0.65, 0.55]
    for ex in (62, 98):
        face[(yy - 60) ** 2 + (xx - ex) ** 2 <= 25] = 0.1
    face[(np.abs(yy - 112) <= 2) & (np.abs(xx - 80) <= 14)] = 0.3
    rng = np.random.default_rng(0)
    face += ndimage.gaussian_filter(rng.normal(0, 0.03, face.shape), (1, 1, 0))
    face += rng.normal(0, 0.01, face.shape)
    return np.asarray((np.clip(face, 0, 1) * 255).astype(np.uint8))


def _head_rows(image: np.ndarray[Any, Any]) -> tuple[int, int]:
    rows = np.nonzero((image[..., 0] > 127).any(axis=1))[0]
    return int(rows.min()), int(rows.max())


class TestGrowthModel:
    def test_growth_schedule_is_monotone_and_saturates(self) -> None:
        ages = [0, 2, 5, 10, 15, 18, 30, 60]
        fractions = [craniofacial_growth_fraction(a) for a in ages]
        assert fractions == sorted(fractions)
        assert fractions[0] == 0.0
        assert craniofacial_growth_fraction(18) > 0.94
        assert craniofacial_growth_fraction(60) - craniofacial_growth_fraction(30) < 0.01

    def test_zero_strain_is_the_identity(self) -> None:
        face = _face()
        assert np.array_equal(cardioidal_strain(face, 0.0), face)

    def test_growth_keeps_the_crown_and_elongates_the_lower_face(self) -> None:
        face = _face()
        top0, bottom0 = _head_rows(face)
        top1, bottom1 = _head_rows(cardioidal_strain(face, 0.1))
        assert top1 == top0  # theta = 0: the crown does not move
        assert bottom1 > bottom0 + 8  # theta near pi: the chin moves down

    def test_lower_face_displacement_grows_with_k(self) -> None:
        face = _face()
        bottoms = [_head_rows(cardioidal_strain(face, k))[1] for k in (0.0, 0.05, 0.1, 0.15)]
        assert bottoms == sorted(bottoms) and bottoms[-1] > bottoms[0]

    def test_rejuvenation_is_the_opposite_direction(self) -> None:
        face = _face()
        assert _head_rows(cardioidal_strain(face, -0.08))[1] < _head_rows(face)[1]


class TestProgression:
    def test_juvenile_span_applies_the_growth_model_with_its_provenance(self) -> None:
        result = AgeProgressionEngine(seed=0).progress_face(_face(), 10, source_age=4)
        geometric = result.geometric_model
        assert geometric is not None and geometric["applied"]
        expected_k = CRANIOFACIAL_K_RANGE * (
            craniofacial_growth_fraction(14) - craniofacial_growth_fraction(4)
        )
        assert geometric["k"] == pytest.approx(expected_k, abs=1e-5)
        assert "Todd" in geometric["model"] and "assumption" in geometric["schedule"]

    def test_adult_span_has_no_geometric_change_and_says_why(self) -> None:
        result = AgeProgressionEngine(seed=0).progress_face(_face(), 10, source_age=45)
        assert result.geometric_model is not None and not result.geometric_model["applied"]
        assert "no craniofacial growth" in result.geometric_model["reason"]

    def test_missing_source_age_is_stated_not_guessed(self) -> None:
        result = AgeProgressionEngine(seed=0).progress_face(_face(), 10)
        assert result.geometric_model == {"applied": False, "reason": "source_age not provided"}

    def test_detail_model_is_deterministic_regardless_of_seed(self) -> None:
        a = AgeProgressionEngine(seed=1).progress_face(_face(), 12, source_age=6)
        b = AgeProgressionEngine(seed=2).progress_face(_face(), 12, source_age=6)
        assert a.progressed_face is not None and b.progressed_face is not None
        assert np.array_equal(a.progressed_face, b.progressed_face)
        assert a.texture_model == "detail"

    def test_unknown_texture_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="texture_model"):
            AgeProgressionEngine(seed=0).progress_face(_face(), 5, texture_model="magic")

    def test_similarity_stays_unmeasured_without_an_embedding_backend(self) -> None:
        result = AgeProgressionEngine(seed=0).progress_face(_face(), 10, source_age=8)
        if result.original_embedding is None:
            assert result.similarity is None


class TestProgressionAudit:
    def test_detail_aging_injects_no_noise_and_keeps_structure(self) -> None:
        engine = AgeProgressionEngine(seed=0)
        report = engine.assess_progression(engine.progress_face(_face(), 10, source_age=8))
        assert report["assessed"]
        assert report["noise_ratio"] < 1.5
        assert report["structural_similarity"] > 0.6
        assert "noise_injected" not in report["flags"]

    def test_legacy_noise_filter_is_caught_injecting_noise(self) -> None:
        engine = AgeProgressionEngine(seed=0)
        report = engine.assess_progression(
            engine.progress_face(_face(), 10, texture_model="legacy_noise")
        )
        assert report["noise_ratio"] > report["thresholds"]["noise_ratio"]
        assert "noise_injected" in report["flags"]

    def test_rejuvenation_detail_loss_is_reported(self) -> None:
        engine = AgeProgressionEngine(seed=0)
        report = engine.assess_progression(engine.progress_face(_face(), -10, source_age=18))
        assert "detail_lost" in report["flags"]

    def test_failed_progression_is_not_assessed(self) -> None:
        from omni_mercury_engine.models.biometric_multimodal import AgeProgressionResult

        report = AgeProgressionEngine.assess_progression(
            AgeProgressionResult(success=False, message="Face detection failed")
        )
        assert report == {"assessed": False, "reason": "Face detection failed"}

    def test_ssim_is_one_for_identical_images_and_drops_with_damage(self) -> None:
        face = _face()
        assert structural_similarity(face, face) == pytest.approx(1.0)
        damaged = face.copy()
        damaged[40:120, 40:120] = 0
        assert structural_similarity(face, damaged) < 0.8

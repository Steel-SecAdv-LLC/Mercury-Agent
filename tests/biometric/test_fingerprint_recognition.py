# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the fingerprint recognition module.

Exercises the full public API of
``omni_mercury_engine.biometric.fingerprint_recognition`` -- orientation
field estimation, ridge frequency estimation, Gabor enhancement, minutiae
extraction, minutiae matching, liveness detection and the top-level
``FingerprintRecognizer`` orchestration -- together with the enum/dataclass
value objects.

All inputs are seeded ``numpy`` synthetic fingerprint images built from
sinusoidal ridge patterns (no torch, no network, no wall-clock).  Where the
crossing-number minutiae detector is exercised, the private helper is driven
directly with integer skeletons so its ridge-ending / bifurcation branches are
observable (see ``TestCrossingNumberOverflow`` for the documented uint8 bug).
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric.fingerprint_recognition import (
    FingerprintFeatures,
    FingerprintLivenessDetector,
    FingerprintLivenessResult,
    FingerprintMatcher,
    FingerprintMatchResult,
    FingerprintRecognizer,
    GaborEnhancer,
    Minutia,
    MinutiaeExtractor,
    MinutiaeType,
    OrientationFieldEstimator,
    RidgeFrequencyEstimator,
    Singularity,
    SingularityType,
)

SEED = 20250721


# ---------------------------------------------------------------------------
# Synthetic image / minutiae helpers
# ---------------------------------------------------------------------------
def make_fingerprint(
    seed: int = 0,
    size: int = 64,
    ridge_freq: float = 0.12,
    angle: float = 0.6,
) -> np.ndarray:
    """Build a deterministic sinusoidal ridge pattern in ``[0, 255]``.

    A tilted cosine grating stands in for parallel fingerprint ridges; a small
    amount of seeded noise keeps the gradients well defined so the orientation
    and frequency estimators return finite values.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    img = 0.5 + 0.5 * np.cos(2 * np.pi * ridge_freq * proj)
    img += 0.03 * rng.standard_normal((size, size))
    img = np.clip(img, 0.0, 1.0)
    return np.asarray(img * 255.0, dtype=np.float64)


def line_minutiae(n: int = 12, spacing: float = 6.0) -> list[Minutia]:
    """A deterministic column of well-separated ridge endings."""
    return [
        Minutia(
            x=20.0 + k * spacing,
            y=20.0 + (k % 4) * spacing,
            orientation=(k * 0.2) % (2 * np.pi),
            type=MinutiaeType.RIDGE_ENDING,
            quality=1.0,
        )
        for k in range(n)
    ]


def random_minutiae(n: int, seed: int, span: float = 300.0) -> list[Minutia]:
    """Scattered minutiae that will not align with an unrelated set."""
    rng = np.random.default_rng(seed)
    return [
        Minutia(
            x=float(rng.uniform(0, span)),
            y=float(rng.uniform(0, span)),
            orientation=float(rng.uniform(0, 2 * np.pi)),
            type=MinutiaeType.RIDGE_ENDING,
            quality=1.0,
        )
        for _ in range(n)
    ]


def features_from(minutiae: list[Minutia]) -> FingerprintFeatures:
    """Wrap a minutiae list in a minimal ``FingerprintFeatures``."""
    return FingerprintFeatures(
        minutiae=minutiae,
        singularities=[],
        orientation_field=np.zeros((4, 4)),
        ridge_frequency=np.zeros((4, 4)),
        quality_map=np.ones((4, 4)),
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TestEnums:
    def test_minutiae_type_members(self) -> None:
        names = {m.name for m in MinutiaeType}
        assert names == {
            "RIDGE_ENDING",
            "BIFURCATION",
            "SHORT_RIDGE",
            "ISLAND",
            "SPUR",
            "CROSSOVER",
        }
        # ``auto()`` assigns distinct values.
        assert len({m.value for m in MinutiaeType}) == 6

    def test_singularity_type_members(self) -> None:
        names = {s.name for s in SingularityType}
        assert names == {"CORE_LOOP", "CORE_WHORL", "DELTA"}
        assert len({s.value for s in SingularityType}) == 3


# ---------------------------------------------------------------------------
# Value dataclasses
# ---------------------------------------------------------------------------
class TestDataclasses:
    def test_minutia_defaults(self) -> None:
        m = Minutia(x=1.0, y=2.0, orientation=0.3, type=MinutiaeType.BIFURCATION)
        assert m.quality == 1.0
        assert m.ridge_count == 0

    def test_minutia_distance_to(self) -> None:
        a = Minutia(0.0, 0.0, 0.0, MinutiaeType.RIDGE_ENDING)
        b = Minutia(3.0, 4.0, 0.0, MinutiaeType.RIDGE_ENDING)
        assert a.distance_to(b) == pytest.approx(5.0)
        assert a.distance_to(a) == pytest.approx(0.0)

    def test_minutia_angle_difference_wraps(self) -> None:
        a = Minutia(0.0, 0.0, 0.1, MinutiaeType.RIDGE_ENDING)
        b = Minutia(0.0, 0.0, 6.2, MinutiaeType.RIDGE_ENDING)
        # Raw difference 6.1 wraps to the short way round the circle.
        assert a.angle_difference(b) == pytest.approx(2 * np.pi - 6.1)
        # Symmetric and self-difference is zero.
        assert b.angle_difference(a) == pytest.approx(a.angle_difference(b))
        assert a.angle_difference(a) == pytest.approx(0.0)

    def test_singularity_defaults(self) -> None:
        s = Singularity(x=5.0, y=6.0, type=SingularityType.DELTA)
        assert s.orientation == 0.0

    def test_fingerprint_features_defaults(self) -> None:
        f = FingerprintFeatures(
            minutiae=[],
            singularities=[],
            orientation_field=np.zeros((2, 2)),
            ridge_frequency=np.zeros((2, 2)),
            quality_map=np.zeros((2, 2)),
        )
        assert f.enhanced_image is None
        assert f.overall_quality == 0.0

    def test_match_result_defaults(self) -> None:
        r = FingerprintMatchResult(
            match_score=10.0,
            matched_minutiae=2,
            total_probe_minutiae=5,
            total_gallery_minutiae=6,
            is_match=False,
            confidence=0.1,
        )
        assert r.transformation is None

    def test_liveness_result_default_details(self) -> None:
        r = FingerprintLivenessResult(
            is_live=True,
            confidence=0.8,
            pore_score=0.9,
            perspiration_score=0.7,
            elasticity_score=0.6,
        )
        # ``details`` defaults to an empty dict via default_factory.
        assert r.details == {}


# ---------------------------------------------------------------------------
# OrientationFieldEstimator
# ---------------------------------------------------------------------------
class TestOrientationFieldEstimator:
    def test_estimate_shape_is_block_resolution(self) -> None:
        est = OrientationFieldEstimator(block_size=16)
        field = est.estimate(make_fingerprint(1, 64))
        assert field.shape == (4, 4)
        assert np.all(np.isfinite(field))

    def test_orientation_in_half_pi_band(self) -> None:
        # 0.5 * arctan2(...) is confined to (-pi/2, pi/2].
        est = OrientationFieldEstimator(block_size=16)
        field = est.estimate(make_fingerprint(2, 64))
        assert np.all(field <= np.pi / 2 + 1e-9)
        assert np.all(field >= -np.pi / 2 - 1e-9)

    def test_color_image_is_downmixed(self) -> None:
        est = OrientationFieldEstimator(block_size=16)
        gray = make_fingerprint(3, 64)
        color = np.stack([gray, gray, gray], axis=2)
        assert est.estimate(color).shape == est.estimate(gray).shape

    def test_dominant_orientation_tracks_grating(self) -> None:
        # A near-vertical grating (ridges running along y) yields a consistent
        # dominant orientation across every block.
        est = OrientationFieldEstimator(block_size=16)
        field = est.estimate(make_fingerprint(4, 64, ridge_freq=0.15, angle=0.0))
        assert np.std(field) < 0.2

    def test_convolve2d_matches_reference(self) -> None:
        est = OrientationFieldEstimator()
        img = np.arange(16, dtype=np.float64).reshape(4, 4)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        out = est._convolve2d(img, identity)
        # Convolving with a centred identity kernel is a no-op.
        assert np.allclose(out, img)


# ---------------------------------------------------------------------------
# RidgeFrequencyEstimator
# ---------------------------------------------------------------------------
class TestRidgeFrequencyEstimator:
    def test_estimate_shape_matches_orientation(self) -> None:
        img = make_fingerprint(5, 64)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        assert freq.shape == orient.shape

    def test_estimate_values_clamped_to_valid_band(self) -> None:
        img = make_fingerprint(6, 64)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        assert np.all(freq >= 0.05)
        assert np.all(freq <= 0.5)

    def test_estimate_color_image_downmixed(self) -> None:
        img = make_fingerprint(7, 64)
        color = np.stack([img, img, img], axis=2)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(color, orient)
        assert freq.shape == orient.shape

    def test_flat_projection_returns_fallback(self) -> None:
        est = RidgeFrequencyEstimator()
        # Zero-variance projection -> std < 0.01 fallback.
        assert est._estimate_frequency_from_projection(np.zeros(16)) == pytest.approx(0.1)

    def test_too_few_crossings_returns_fallback(self) -> None:
        est = RidgeFrequencyEstimator()
        proj = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])  # single zero crossing
        assert est._estimate_frequency_from_projection(proj) == pytest.approx(0.1)

    def test_two_crossings_no_period_returns_fallback(self) -> None:
        est = RidgeFrequencyEstimator()
        # Exactly two crossings -> the period loop produces nothing.
        proj = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        assert est._estimate_frequency_from_projection(proj) == pytest.approx(0.1)

    def test_clean_sinusoid_recovers_frequency(self) -> None:
        est = RidgeFrequencyEstimator()
        x = np.arange(64)
        proj = np.cos(2 * np.pi * x / 8.0)  # period 8 -> frequency 1/8
        assert est._estimate_frequency_from_projection(proj) == pytest.approx(0.125)

    def test_high_frequency_clamped_upper(self) -> None:
        est = RidgeFrequencyEstimator()
        proj = np.array([1.0, -1.0] * 16)  # period 2 -> clamped to 0.5
        assert est._estimate_frequency_from_projection(proj) == pytest.approx(0.5)

    def test_low_frequency_clamped_lower(self) -> None:
        est = RidgeFrequencyEstimator()
        x = np.arange(64)
        proj = np.cos(2 * np.pi * x / 40.0)  # long period -> clamped to 0.05
        assert est._estimate_frequency_from_projection(proj) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# GaborEnhancer
# ---------------------------------------------------------------------------
class TestGaborEnhancer:
    def test_enhance_returns_normalized_same_shape(self) -> None:
        img = make_fingerprint(8, 64)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        enhanced = GaborEnhancer(kernel_size=15).enhance(img, orient, freq)
        assert enhanced.shape == img.shape
        assert enhanced.min() >= 0.0
        assert enhanced.max() <= 1.0 + 1e-9
        assert np.all(np.isfinite(enhanced))

    def test_enhance_color_and_quality_mask(self) -> None:
        img = make_fingerprint(9, 64)
        color = np.stack([img, img, img], axis=2)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        mask = np.ones_like(orient)
        enhanced = GaborEnhancer(kernel_size=15).enhance(color, orient, freq, quality_mask=mask)
        assert enhanced.shape == img.shape

    def test_gabor_filter_shape_zero_mean_and_cached(self) -> None:
        enh = GaborEnhancer(kernel_size=15)
        first = enh._get_gabor_filter(0.5, 0.1)
        assert first.shape == (15, 15)
        # Filter is DC-free (mean subtracted) and L1-normalised.
        assert first.mean() == pytest.approx(0.0, abs=1e-9)
        assert np.sum(np.abs(first)) == pytest.approx(1.0, abs=1e-6)
        # Second call with identical rounded key returns the cached object.
        second = enh._get_gabor_filter(0.5, 0.1)
        assert second is first

    def test_apply_filter_preserves_shape(self) -> None:
        enh = GaborEnhancer(kernel_size=5)
        block = make_fingerprint(10, 32) / 255.0
        kernel = enh._get_gabor_filter(0.3, 0.1)
        out = enh._apply_filter(block, kernel)
        assert out.shape == block.shape


# ---------------------------------------------------------------------------
# MinutiaeExtractor
# ---------------------------------------------------------------------------
class TestMinutiaeExtractor:
    def test_extract_returns_list(self) -> None:
        img = make_fingerprint(11, 64)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        enhanced = GaborEnhancer(kernel_size=15).enhance(img, orient, freq)
        extractor = MinutiaeExtractor()
        with np.errstate(over="ignore"):
            minutiae = extractor.extract(enhanced, orient)
        assert isinstance(minutiae, list)
        assert all(isinstance(m, Minutia) for m in minutiae)

    def test_extract_with_quality_map(self) -> None:
        img = make_fingerprint(12, 64)
        orient = OrientationFieldEstimator(block_size=16).estimate(img)
        freq = RidgeFrequencyEstimator(block_size=16, window_size=32).estimate(img, orient)
        enhanced = GaborEnhancer(kernel_size=15).enhance(img, orient, freq)
        quality = np.ones_like(orient)
        with np.errstate(over="ignore"):
            minutiae = MinutiaeExtractor().extract(enhanced, orient, quality)
        assert isinstance(minutiae, list)

    def test_binarize_is_uint8_binary(self) -> None:
        extractor = MinutiaeExtractor()
        img = make_fingerprint(13, 32) / 255.0
        binary = extractor._binarize(img)
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 1})
        # Pixels below the mean are foreground (1).
        assert binary[img < img.mean()].all()

    def test_thin_reduces_or_preserves_foreground(self) -> None:
        extractor = MinutiaeExtractor()
        binary = np.zeros((15, 15), dtype=np.uint8)
        binary[5:10, 3:12] = 1  # a solid bar to be thinned
        skeleton = extractor._thin(binary)
        assert skeleton.shape == binary.shape
        assert skeleton.sum() <= binary.sum()

    def test_find_minutiae_detects_ridge_ending(self) -> None:
        # Driven with an int32 skeleton so the crossing number is exact.
        skeleton = np.zeros((9, 9), dtype=np.int32)
        skeleton[1:5, 4] = 1  # vertical stub -> ridge ending at its top
        orient = np.zeros((2, 2))
        minutiae = MinutiaeExtractor(border_margin=1)._find_minutiae(skeleton, orient)
        types = {m.type for m in minutiae}
        assert MinutiaeType.RIDGE_ENDING in types
        for m in minutiae:
            assert 0.0 <= m.x < 9.0 and 0.0 <= m.y < 9.0

    def test_find_minutiae_detects_bifurcation(self) -> None:
        skeleton = np.zeros((9, 9), dtype=np.int32)
        skeleton[4, 1:5] = 1  # arm from the left
        skeleton[1:4, 4] = 1  # arm upward
        skeleton[5:8, 4] = 1  # arm downward -> 3-way junction at (4, 4)
        orient = np.zeros((2, 2))
        minutiae = MinutiaeExtractor(border_margin=1)._find_minutiae(skeleton, orient)
        bifs = [m for m in minutiae if m.type is MinutiaeType.BIFURCATION]
        assert any(m.x == 4.0 and m.y == 4.0 for m in bifs)

    def test_filter_minutiae_drops_border_points(self) -> None:
        extractor = MinutiaeExtractor(border_margin=10)
        inside = Minutia(50.0, 50.0, 0.0, MinutiaeType.RIDGE_ENDING)
        edge_x = Minutia(2.0, 50.0, 0.0, MinutiaeType.RIDGE_ENDING)  # too near left edge
        edge_y = Minutia(50.0, 96.0, 0.0, MinutiaeType.RIDGE_ENDING)  # too near bottom edge
        kept = extractor._filter_minutiae([inside, edge_x, edge_y], (100, 100), None)
        assert inside in kept
        assert edge_x not in kept
        assert edge_y not in kept

    def test_filter_minutiae_quality_gate(self) -> None:
        extractor = MinutiaeExtractor(quality_threshold=0.5, border_margin=5)
        m = Minutia(50.0, 50.0, 0.0, MinutiaeType.RIDGE_ENDING)
        low_quality = np.zeros((4, 4))  # every block below threshold
        assert extractor._filter_minutiae([m], (100, 100), low_quality) == []
        high_quality = np.ones((4, 4))
        assert extractor._filter_minutiae([m], (100, 100), high_quality) == [m]

    def test_remove_close_minutiae_keeps_highest_quality(self) -> None:
        extractor = MinutiaeExtractor()
        a = Minutia(10.0, 10.0, 0.0, MinutiaeType.RIDGE_ENDING, quality=0.3)
        b = Minutia(12.0, 10.0, 0.0, MinutiaeType.RIDGE_ENDING, quality=0.9)  # within 10px
        far = Minutia(80.0, 80.0, 0.0, MinutiaeType.RIDGE_ENDING, quality=0.5)
        result = extractor._remove_close_minutiae([a, b, far], min_distance=10.0)
        assert far in result
        assert b in result  # higher-quality member of the close cluster
        assert a not in result
        assert len(result) == 2

    def test_remove_close_minutiae_single_passthrough(self) -> None:
        extractor = MinutiaeExtractor()
        only = [Minutia(1.0, 1.0, 0.0, MinutiaeType.RIDGE_ENDING)]
        assert extractor._remove_close_minutiae(only, min_distance=10.0) == only


# ---------------------------------------------------------------------------
# FingerprintMatcher
# ---------------------------------------------------------------------------
class TestFingerprintMatcher:
    def test_identical_features_match_perfectly(self) -> None:
        feats = features_from(line_minutiae(12))
        result = FingerprintMatcher().match(feats, feats)
        assert result.is_match is True
        assert result.match_score == pytest.approx(100.0)
        assert result.matched_minutiae == 12
        assert result.confidence == pytest.approx(1.0)
        assert result.transformation == (0.0, 0.0, 0.0)

    def test_translated_copy_still_matches(self) -> None:
        base = line_minutiae(12)
        shifted = [Minutia(m.x + 25.0, m.y - 13.0, m.orientation, m.type, m.quality) for m in base]
        result = FingerprintMatcher().match(features_from(base), features_from(shifted))
        assert result.is_match is True
        assert result.matched_minutiae == 12
        # The recovered transform undoes the applied offset.
        assert result.transformation is not None
        dx, dy, _ = result.transformation
        assert dx == pytest.approx(25.0)
        assert dy == pytest.approx(-13.0)

    def test_too_few_minutiae_short_circuits(self) -> None:
        probe = features_from(line_minutiae(3))
        gallery = features_from(line_minutiae(12))
        result = FingerprintMatcher(min_matched_minutiae=8).match(probe, gallery)
        assert result.is_match is False
        assert result.match_score == 0.0
        assert result.matched_minutiae == 0
        assert result.total_probe_minutiae == 3
        assert result.total_gallery_minutiae == 12
        assert result.confidence == 0.0
        assert result.transformation is None

    def test_unrelated_prints_do_not_match(self) -> None:
        probe = features_from(random_minutiae(15, 1))
        gallery = features_from(random_minutiae(15, 2))
        result = FingerprintMatcher().match(probe, gallery)
        assert result.is_match is False
        assert result.match_score < 40.0
        # Non-match confidence follows the score/threshold*0.5 formula.
        expected = result.match_score / 40.0 * 0.5
        assert result.confidence == pytest.approx(expected)

    def test_compute_score_zero_counts(self) -> None:
        matcher = FingerprintMatcher()
        assert matcher._compute_score(5, 0, 10) == 0.0
        assert matcher._compute_score(5, 10, 0) == 0.0

    def test_compute_score_formula(self) -> None:
        matcher = FingerprintMatcher()
        # matched^2 / (probe*gallery) * 100.
        assert matcher._compute_score(6, 12, 12) == pytest.approx(25.0)

    def test_count_matched_respects_angle_tolerance(self) -> None:
        matcher = FingerprintMatcher(distance_tolerance=5.0, angle_tolerance=0.1)
        probe = [Minutia(10.0, 10.0, 0.0, MinutiaeType.RIDGE_ENDING)]
        # Same location but orientation far outside tolerance.
        gallery = [Minutia(10.0, 10.0, 1.0, MinutiaeType.RIDGE_ENDING)]
        assert matcher._count_matched_minutiae(probe, gallery, 0.0, 0.0, 0.0) == 0
        # Now aligned within tolerance.
        gallery2 = [Minutia(10.0, 10.0, 0.05, MinutiaeType.RIDGE_ENDING)]
        assert matcher._count_matched_minutiae(probe, gallery2, 0.0, 0.0, 0.0) == 1


# ---------------------------------------------------------------------------
# FingerprintLivenessDetector
# ---------------------------------------------------------------------------
class TestFingerprintLivenessDetector:
    def test_no_images_reports_not_live(self) -> None:
        result = FingerprintLivenessDetector().detect([])
        assert result.is_live is False
        assert result.confidence == 0.0
        assert result.details == {"error": "No images provided"}

    def test_live_sequence_passes(self) -> None:
        rng = np.random.default_rng(SEED)
        images = []
        for scale in (0.6, 1.0, 1.5):
            im = rng.random((40, 40)) * scale * 255.0 + rng.random((40, 40)) * 40.0
            images.append(im)
        result = FingerprintLivenessDetector().detect(images)
        assert result.is_live is True
        assert result.details["pore_live"] is True
        assert result.details["perspiration_live"] is True
        assert 0.0 <= result.confidence <= 1.0

    def test_flat_replica_fails_liveness(self) -> None:
        flat = [np.full((32, 32), 128.0) for _ in range(3)]
        result = FingerprintLivenessDetector().detect(flat)
        assert result.is_live is False
        # Every sub-analysis collapses to its low-signal floor.
        assert result.pore_score == pytest.approx(0.2)
        assert result.perspiration_score == pytest.approx(0.2)
        assert result.elasticity_score == pytest.approx(0.2)

    def test_single_image_uses_neutral_dynamics(self) -> None:
        rng = np.random.default_rng(SEED + 1)
        noisy = rng.random((40, 40)) * 255.0
        result = FingerprintLivenessDetector().detect([noisy])
        # With one frame perspiration/elasticity default to the 0.5 neutral value.
        assert result.perspiration_score == pytest.approx(0.5)
        assert result.elasticity_score == pytest.approx(0.5)

    def test_analyze_sweat_pores_flat_floor(self) -> None:
        det = FingerprintLivenessDetector()
        assert det._analyze_sweat_pores(np.full((16, 16), 100.0)) == pytest.approx(0.2)

    def test_analyze_sweat_pores_color_input(self) -> None:
        det = FingerprintLivenessDetector()
        color = np.stack([np.full((16, 16), 50.0)] * 3, axis=2)
        score = det._analyze_sweat_pores(color)
        assert 0.0 <= score <= 1.0

    def test_analyze_sweat_pores_detects_high_frequency(self) -> None:
        det = FingerprintLivenessDetector()
        rng = np.random.default_rng(SEED + 2)
        textured = rng.random((32, 32)) * 255.0
        # Broadband noise carries plenty of pore-scale high frequency energy.
        assert det._analyze_sweat_pores(textured) > 0.2

    def test_perspiration_single_and_flat(self) -> None:
        det = FingerprintLivenessDetector()
        assert det._analyze_perspiration([np.ones((8, 8))]) == pytest.approx(0.5)
        flat = [np.full((8, 8), 5.0) for _ in range(3)]
        assert det._analyze_perspiration(flat) == pytest.approx(0.2)

    def test_perspiration_color_frames(self) -> None:
        det = FingerprintLivenessDetector()
        frames = [np.stack([np.full((8, 8), v)] * 3, axis=2) for v in (10.0, 20.0, 30.0)]
        score = det._analyze_perspiration(frames)
        assert 0.0 <= score <= 1.0

    def test_elasticity_single_and_flat(self) -> None:
        det = FingerprintLivenessDetector()
        assert det._analyze_elasticity([np.ones((8, 8))]) == pytest.approx(0.5)
        flat = [np.full((8, 8), 5.0) for _ in range(3)]
        assert det._analyze_elasticity(flat) == pytest.approx(0.2)

    def test_elasticity_varying_contrast(self) -> None:
        det = FingerprintLivenessDetector()
        rng = np.random.default_rng(SEED + 3)
        frames = [rng.random((16, 16)) * s for s in (0.2, 1.0, 2.0)]
        score = det._analyze_elasticity(frames)
        assert 0.0 < score <= 1.0

    def test_elasticity_color_frames(self) -> None:
        det = FingerprintLivenessDetector()
        rng = np.random.default_rng(SEED + 4)
        frames = [np.stack([rng.random((12, 12)) * s] * 3, axis=2) for s in (0.3, 1.0, 1.8)]
        score = det._analyze_elasticity(frames)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# FingerprintRecognizer
# ---------------------------------------------------------------------------
class TestFingerprintRecognizer:
    def test_extract_features_structure(self) -> None:
        img = make_fingerprint(21, 48)
        with np.errstate(over="ignore"):
            feats = FingerprintRecognizer(liveness_required=False).extract_features(img)
        assert isinstance(feats, FingerprintFeatures)
        assert feats.orientation_field.shape == feats.ridge_frequency.shape
        assert feats.orientation_field.shape == feats.quality_map.shape
        assert feats.enhanced_image is not None
        assert feats.enhanced_image.shape == img.shape
        assert 0.0 <= feats.overall_quality <= 1.0
        assert isinstance(feats.minutiae, list)
        assert isinstance(feats.singularities, list)

    def test_extract_features_color_image(self) -> None:
        gray = make_fingerprint(22, 48)
        color = np.stack([gray, gray, gray], axis=2)
        with np.errstate(over="ignore"):
            feats = FingerprintRecognizer(liveness_required=False).extract_features(color)
        assert feats.enhanced_image is not None
        assert feats.enhanced_image.shape == gray.shape

    def test_detect_singularities_core_loop(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        size = 10
        yy, xx = np.mgrid[0:size, 0:size]
        theta = np.arctan2(yy - size / 2, xx - size / 2)
        singularities = recognizer._detect_singularities(0.5 * theta)
        assert singularities
        assert all(s.type is SingularityType.CORE_LOOP for s in singularities)

    def test_detect_singularities_delta(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        size = 10
        yy, xx = np.mgrid[0:size, 0:size]
        theta = np.arctan2(yy - size / 2, xx - size / 2)
        # Reversed rotation flips the Poincaré index sign -> delta points.
        singularities = recognizer._detect_singularities(-0.5 * theta)
        assert singularities
        assert all(s.type is SingularityType.DELTA for s in singularities)

    def test_detect_singularities_uniform_field_none(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        assert recognizer._detect_singularities(np.zeros((6, 6))) == []

    def test_compute_coherence_uniform_is_high(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        coh = recognizer._compute_coherence(np.zeros((4, 4)), 1, 1)
        assert coh == pytest.approx(1.0)

    def test_compute_coherence_tiny_field(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        # Fewer than two sampled angles -> the coherence short-circuits to 1.0.
        assert recognizer._compute_coherence(np.zeros((1, 1)), 0, 0) == pytest.approx(1.0)

    def test_compute_quality_map_shape_and_range(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        img = make_fingerprint(23, 48) / 255.0
        orient = recognizer._orientation_estimator.estimate(img)
        quality = recognizer._compute_quality_map(img, orient)
        assert quality.shape == orient.shape
        assert np.all(quality >= 0.0)
        assert np.all(quality <= 1.0)

    def test_compute_overall_quality_blend(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        quality = np.full((4, 4), 0.5)
        # 0.6 * mean_quality + 0.4 * min(1, count/30); count>=30 saturates the term.
        val = recognizer._compute_overall_quality(quality, minutiae_count=30)
        assert val == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)
        val2 = recognizer._compute_overall_quality(quality, minutiae_count=0)
        assert val2 == pytest.approx(0.6 * 0.5)

    def test_verify_without_liveness_returns_none(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=False)
        enrolled = features_from(line_minutiae(12))
        probe = make_fingerprint(24, 48)
        with np.errstate(over="ignore"):
            match_result, liveness = recognizer.verify(probe, enrolled)
        assert liveness is None
        assert isinstance(match_result, FingerprintMatchResult)

    def test_verify_spoof_zeroes_match(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=True)
        img = make_fingerprint(25, 48)
        with np.errstate(over="ignore"):
            enrolled = recognizer.extract_features(img)
        flat = np.full((48, 48), 128.0)  # non-live presentation
        with np.errstate(over="ignore"):
            match_result, liveness = recognizer.verify(flat, enrolled)
        assert liveness is not None
        assert liveness.is_live is False
        # A failed liveness check forces the match result to zero.
        assert match_result.is_match is False
        assert match_result.match_score == 0.0
        assert match_result.matched_minutiae == 0

    def test_verify_live_probe_keeps_match_result(self) -> None:
        recognizer = FingerprintRecognizer(liveness_required=True)
        img = make_fingerprint(26, 48)
        with np.errstate(over="ignore"):
            enrolled = recognizer.extract_features(img)
        rng = np.random.default_rng(SEED + 5)
        probe = rng.random((48, 48)) * 0.6 * 255.0 + rng.random((48, 48)) * 40.0
        live_extra = [
            rng.random((48, 48)) * s * 255.0 + rng.random((48, 48)) * 40.0 for s in (1.0, 1.6)
        ]
        with np.errstate(over="ignore"):
            match_result, liveness = recognizer.verify(probe, enrolled, liveness_images=live_extra)
        assert liveness is not None
        assert liveness.is_live is True
        # Liveness passed, so the match result is preserved (not force-zeroed).
        assert isinstance(match_result, FingerprintMatchResult)
        assert match_result.total_gallery_minutiae == len(enrolled.minutiae)


# ---------------------------------------------------------------------------
# Documented source defect: uint8 crossing-number underflow
# ---------------------------------------------------------------------------
class TestCrossingNumberOverflow:
    def test_uint8_skeleton_crossing_number_has_no_underflow(self) -> None:
        """The crossing-number is dtype-safe on a uint8 skeleton (regression).

        ``_find_minutiae`` now casts each neighbour to a signed ``int`` before
        the ``abs(neighbors[k] - neighbors[k+1])`` difference, so a 0->1
        transition contributes 1 rather than a uint8 wrap-around to 255. A
        genuine ridge ending (crossing number 1) is therefore recognised on the
        raw uint8 skeleton exactly as on an int32 copy -- the two paths must
        agree, and no overflow warning is emitted.

        Before the fix the uint8 path silently missed every ridge ending /
        bifurcation (the underflow drove the crossing number to 128 / 384), so
        the public ``extract()`` returned no minutiae on well-formed ridges.
        """
        skeleton_u8 = np.zeros((9, 9), dtype=np.uint8)
        skeleton_u8[1:5, 4] = 1  # a clear ridge ending at (4, 4)
        orient = np.zeros((2, 2))
        extractor = MinutiaeExtractor(border_margin=1)

        with np.errstate(over="raise"):  # a wrap-around would now raise, not pass
            from_uint8 = extractor._find_minutiae(skeleton_u8, orient)
        from_int = extractor._find_minutiae(skeleton_u8.astype(np.int32), orient)

        # The uint8 path detects the ridge ending, and agrees exactly with int32.
        assert any(m.type is MinutiaeType.RIDGE_ENDING for m in from_uint8)
        assert {(m.type, m.x, m.y) for m in from_uint8} == {(m.type, m.x, m.y) for m in from_int}

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic unit tests for the iris recognition module.

Exercises the full public API of
``omni_mercury_engine.biometric.iris_recognition`` -- the Gabor filter
bank, integro-differential segmentation, Daugman rubber-sheet
normalization, IrisCode encoding, Hamming-distance matching with
rotation compensation, presentation-attack (liveness) detection, and the
top-level ``IrisRecognizer`` orchestration -- together with the
``IrisFeatures`` / ``IrisMatchResult`` / ``LivenessResult`` dataclasses.

All tests are deterministic: synthetic eye images are built from seeded
``numpy`` arrays (no torch, no network, no wall-clock reliance).  The
integro-differential search is O(n^3), so images are kept tiny (default
recognizer paths use 16x16) or a segmenter with small radius ranges is
used on a modest synthetic eye to exercise the real search loops cheaply.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.biometric.iris_recognition import (
    GaborFilter,
    IrisEncoder,
    IrisFeatures,
    IrisLivenessDetector,
    IrisMatcher,
    IrisMatchResult,
    IrisNormalizer,
    IrisRecognizer,
    IrisSegmenter,
    LivenessResult,
)

SEED = 20250721


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _tiny_eye(seed: int = SEED) -> np.ndarray:
    """16x16 image: default segmentation returns fixed defaults quickly."""
    return _rng(seed).integers(0, 256, size=(16, 16)).astype(np.float64)


def make_eye(
    size: int = 40,
    center: tuple[int, int] = (20, 20),
    pupil_radius: int = 5,
    iris_radius: int = 9,
    seed: int = SEED,
) -> np.ndarray:
    """Build a small synthetic eye: dark pupil disk, brighter iris ring.

    Normalized to roughly [0, 1] with a little seeded noise so the
    integro-differential operator has a real (but cheap) boundary to
    localise when driven with small radius ranges.
    """
    rng = _rng(seed)
    cy, cx = center
    img = np.full((size, size), 0.5)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[dist <= iris_radius] = 0.8
    img[dist <= pupil_radius] = 0.05
    img += 0.01 * rng.standard_normal((size, size))
    return img


def _small_segmenter() -> IrisSegmenter:
    return IrisSegmenter(
        pupil_radius_range=(3, 7),
        iris_radius_range=(7, 12),
        search_step=2,
    )


# ---------------------------------------------------------------------------
# Result / feature dataclasses
# ---------------------------------------------------------------------------
class TestDataclasses:
    def test_iris_features_fields_and_optional_default(self) -> None:
        code = np.zeros((2, 4, 4), dtype=np.uint8)
        mask = np.ones((2, 4, 4), dtype=bool)
        f = IrisFeatures(
            iris_code=code,
            mask=mask,
            pupil_center=(8.0, 8.0),
            pupil_radius=50.0,
            iris_center=(8.0, 8.0),
            iris_radius=105.0,
            quality_score=0.7,
        )
        assert f.iris_code.shape == (2, 4, 4)
        assert f.pupil_center == (8.0, 8.0)
        assert f.iris_radius == 105.0
        # normalized_iris is an optional field defaulting to None.
        assert f.normalized_iris is None

    def test_iris_match_result_fields(self) -> None:
        r = IrisMatchResult(
            hamming_distance=0.1,
            match_score=0.9,
            is_match=True,
            bits_compared=4096,
            confidence=0.8,
        )
        assert r.is_match is True
        assert r.bits_compared == 4096
        assert r.match_score == 0.9

    def test_liveness_result_default_details(self) -> None:
        r = LivenessResult(
            is_live=True,
            confidence=0.6,
            pupil_response=0.5,
            specular_reflection_score=0.7,
            texture_authenticity=0.8,
        )
        # details defaults to an empty dict via default_factory.
        assert r.details == {}
        assert r.is_live is True


# ---------------------------------------------------------------------------
# GaborFilter
# ---------------------------------------------------------------------------
class TestGaborFilter:
    def test_default_bank_size_is_scales_times_orientations(self) -> None:
        gf = GaborFilter()  # 4 scales * 8 orientations
        assert len(gf._filters) == 32

    def test_custom_bank_size(self) -> None:
        gf = GaborFilter(kernel_size=15, num_orientations=4, num_scales=2)
        assert len(gf._filters) == 8

    def test_filter_shape_and_dtype(self) -> None:
        gf = GaborFilter(kernel_size=15, num_orientations=2, num_scales=1)
        for filt in gf._filters:
            assert filt.shape == (15, 15)
            assert np.iscomplexobj(filt)

    def test_filters_are_l2_normalized(self) -> None:
        gf = GaborFilter(kernel_size=15, num_orientations=3, num_scales=2)
        for filt in gf._filters:
            energy = np.sqrt(np.sum(np.abs(filt) ** 2))
            assert energy == pytest.approx(1.0)

    def test_apply_returns_one_complex_response_per_filter(self) -> None:
        gf = GaborFilter(kernel_size=15, num_orientations=4, num_scales=2)
        img = _rng(1).standard_normal((20, 24))
        responses = gf.apply(img)
        assert len(responses) == len(gf._filters) == 8
        for resp in responses:
            assert resp.shape == img.shape
            assert np.iscomplexobj(resp)
            assert np.all(np.isfinite(resp))

    def test_convolution_of_zero_image_is_zero(self) -> None:
        gf = GaborFilter(kernel_size=15, num_orientations=2, num_scales=1)
        responses = gf.apply(np.zeros((18, 18)))
        for resp in responses:
            assert np.allclose(resp, 0.0)


# ---------------------------------------------------------------------------
# IrisSegmenter
# ---------------------------------------------------------------------------
class TestIrisSegmenter:
    def test_tiny_image_returns_geometric_defaults(self) -> None:
        # For a 16x16 image the pupil radius range (20, 80) leaves an empty
        # search grid, so the segmenter falls back to image-centre defaults
        # and the mid-point of each radius range.
        seg = IrisSegmenter()
        pc, pr, ic, ir = seg.segment(_tiny_eye())
        assert pc == (8.0, 8.0)
        assert pr == pytest.approx((20 + 80) / 2)  # 50.0
        assert ic == (8.0, 8.0)
        assert ir == pytest.approx((60 + 150) / 2)  # 105.0

    def test_rgb_image_is_downmixed(self) -> None:
        seg = IrisSegmenter()
        gray = _tiny_eye()
        rgb = np.stack([gray, gray, gray], axis=2)
        assert seg.segment(rgb) == seg.segment(gray)

    def test_small_range_search_localizes_pupil_near_center(self) -> None:
        seg = _small_segmenter()
        eye = make_eye()
        pc, pr, ic, ir = seg.segment(eye)
        # The dark pupil is centred at (20, 20) with radius ~5.
        assert pc[0] == pytest.approx(20.0, abs=4.0)
        assert pc[1] == pytest.approx(20.0, abs=4.0)
        assert 3.0 <= pr < 7.0
        # Iris radius stays within its configured search band.
        assert 7.0 <= ir < 12.0

    def test_segment_return_types(self) -> None:
        seg = _small_segmenter()
        pc, pr, ic, ir = seg.segment(make_eye())
        assert isinstance(pc, tuple) and len(pc) == 2
        assert isinstance(pr, float)
        assert isinstance(ic, tuple) and len(ic) == 2
        assert isinstance(ir, float)

    def test_circle_score_dark_vs_bright(self) -> None:
        seg = _small_segmenter()
        # A radial ramp: brightness grows monotonically with the radius, so
        # the outer sample ring is strictly brighter than the inner one.
        yy, xx = np.mgrid[0:30, 0:30]
        img = np.sqrt((xx - 15) ** 2 + (yy - 15) ** 2) / 15.0
        # dark_circle=True rewards dark-inside/bright-outside (outer - inner).
        dark = seg._circle_score(img, (15, 15), 8, dark_circle=True)
        bright = seg._circle_score(img, (15, 15), 8, dark_circle=False)
        assert dark > 0.0
        # dark_circle=False uses the absolute contrast, equal here since the
        # outer ring already dominates.
        assert bright == pytest.approx(abs(dark))

    def test_circle_score_out_of_bounds_returns_neg_inf(self) -> None:
        seg = _small_segmenter()
        img = np.zeros((10, 10))
        # A radius that pushes every sample point outside the image.
        score = seg._circle_score(img, (5, 5), 500, dark_circle=True)
        assert score == float("-inf")


# ---------------------------------------------------------------------------
# IrisNormalizer
# ---------------------------------------------------------------------------
class TestIrisNormalizer:
    def test_output_shapes_and_mask_dtype(self) -> None:
        norm = IrisNormalizer(angular_resolution=32, radial_resolution=16)
        eye = make_eye()
        normalized, mask = norm.normalize(eye, (20.0, 20.0), 5.0, (20.0, 20.0), 9.0)
        assert normalized.shape == (16, 32)
        assert mask.shape == (16, 32)
        assert mask.dtype == np.bool_

    def test_in_bounds_ring_is_fully_valid(self) -> None:
        norm = IrisNormalizer(angular_resolution=32, radial_resolution=16)
        eye = make_eye()
        _, mask = norm.normalize(eye, (20.0, 20.0), 5.0, (20.0, 20.0), 9.0)
        # The whole iris ring lies inside the 40x40 image.
        assert mask.all()

    def test_out_of_bounds_marks_mask_false(self) -> None:
        norm = IrisNormalizer(angular_resolution=32, radial_resolution=16)
        img = np.zeros((30, 30))
        # iris radius 40 pushes the outer ring outside a 30x30 image while
        # the pupil boundary (radius 3) stays inside -> a mixed mask.
        _, mask = norm.normalize(img, (15.0, 15.0), 3.0, (15.0, 15.0), 40.0)
        assert mask.any()
        assert (~mask).any()

    def test_rgb_input_is_downmixed(self) -> None:
        norm = IrisNormalizer(angular_resolution=16, radial_resolution=8)
        eye = make_eye()
        rgb = np.stack([eye, eye, eye], axis=2)
        n_gray, _ = norm.normalize(eye, (20.0, 20.0), 5.0, (20.0, 20.0), 9.0)
        n_rgb, _ = norm.normalize(rgb, (20.0, 20.0), 5.0, (20.0, 20.0), 9.0)
        assert n_rgb.shape == n_gray.shape == (8, 16)
        assert np.allclose(n_rgb, n_gray)


# ---------------------------------------------------------------------------
# IrisEncoder
# ---------------------------------------------------------------------------
class TestIrisEncoder:
    def test_encode_shapes_and_binary_values(self) -> None:
        enc = IrisEncoder(num_filters=4, kernel_size=15)
        normalized = _rng(2).standard_normal((16, 32))
        mask = np.ones((16, 32), dtype=bool)
        code, code_mask = enc.encode(normalized, mask)
        # Each of num_filters produces a real + imaginary bit plane.
        assert code.shape == (8, 16, 32)
        assert code_mask.shape == (8, 16, 32)
        assert code.dtype == np.uint8
        assert set(np.unique(code)).issubset({0, 1})

    def test_encode_replicates_mask_per_bit_plane(self) -> None:
        enc = IrisEncoder(num_filters=2, kernel_size=15)
        normalized = _rng(3).standard_normal((16, 20))
        mask = _rng(4).integers(0, 2, size=(16, 20)).astype(bool)
        _, code_mask = enc.encode(normalized, mask)
        assert code_mask.shape == (4, 16, 20)
        for plane in code_mask:
            assert np.array_equal(plane, mask)

    def test_encode_is_deterministic(self) -> None:
        enc = IrisEncoder(num_filters=3, kernel_size=15)
        normalized = _rng(5).standard_normal((16, 24))
        mask = np.ones((16, 24), dtype=bool)
        c1, _ = enc.encode(normalized, mask)
        c2, _ = enc.encode(normalized, mask)
        assert np.array_equal(c1, c2)


# ---------------------------------------------------------------------------
# IrisMatcher
# ---------------------------------------------------------------------------
class TestIrisMatcher:
    def test_identical_codes_are_a_perfect_match(self) -> None:
        m = IrisMatcher()
        code = _rng(6).integers(0, 2, size=(8, 16, 32)).astype(np.uint8)
        mask = np.ones((8, 16, 32), dtype=bool)
        r = m.match(code, mask, code, mask)
        assert r.hamming_distance == pytest.approx(0.0)
        assert r.match_score == pytest.approx(1.0)
        assert r.is_match is True or bool(r.is_match) is True
        assert r.confidence == pytest.approx(1.0)
        assert r.bits_compared == 8 * 16 * 32

    def test_random_codes_do_not_match(self) -> None:
        m = IrisMatcher()
        rng = _rng(7)
        code1 = rng.integers(0, 2, size=(8, 16, 32)).astype(np.uint8)
        code2 = rng.integers(0, 2, size=(8, 16, 32)).astype(np.uint8)
        mask = np.ones((8, 16, 32), dtype=bool)
        r = m.match(code1, mask, code2, mask)
        # Independent random codes sit near a Hamming distance of 0.5.
        assert 0.35 < r.hamming_distance < 0.65
        assert bool(r.is_match) is False
        assert r.confidence == 0.0
        assert r.match_score == pytest.approx(1.0 - r.hamming_distance)

    def test_insufficient_overlap_returns_default_distance(self) -> None:
        # A tiny code has fewer than the 100-bit minimum overlap, so every
        # rotation shift is skipped and the default distance (1.0) stands.
        m = IrisMatcher()
        code = _rng(8).integers(0, 2, size=(2, 5, 5)).astype(np.uint8)
        mask = np.ones((2, 5, 5), dtype=bool)
        r = m.match(code, mask, code, mask)
        assert r.hamming_distance == 1.0
        assert r.bits_compared == 0
        assert bool(r.is_match) is False
        assert r.match_score == pytest.approx(0.0)
        assert r.confidence == 0.0

    def test_rotation_compensation_aligns_shifted_code(self) -> None:
        m = IrisMatcher(rotation_shifts=3)
        base = _rng(9).integers(0, 2, size=(4, 12, 40)).astype(np.uint8)
        mask = np.ones((4, 12, 40), dtype=bool)
        shifted = np.roll(base, 2, axis=2)
        r = m.match(base, mask, shifted, mask)
        # The matcher searches +-3 shifts and finds the exact -2 alignment.
        assert r.hamming_distance == pytest.approx(0.0)
        assert bool(r.is_match) is True
        assert r.bits_compared == 4 * 12 * 40

    def test_confidence_scales_with_distance_under_threshold(self) -> None:
        m = IrisMatcher(match_threshold=0.5, rotation_shifts=0)
        a = np.zeros((2, 10, 10), dtype=np.uint8)
        b = np.zeros((2, 10, 10), dtype=np.uint8)
        b[0, 0, :] = 1  # 10 differing bits out of 200 -> distance 0.05
        mask = np.ones((2, 10, 10), dtype=bool)
        r = m.match(a, mask, b, mask)
        assert r.hamming_distance == pytest.approx(0.05)
        assert bool(r.is_match) is True
        # confidence = 1 - distance / threshold = 1 - 0.05 / 0.5.
        assert r.confidence == pytest.approx(0.9)
        assert r.match_score == pytest.approx(0.95)
        assert r.bits_compared == 200


# ---------------------------------------------------------------------------
# IrisLivenessDetector
# ---------------------------------------------------------------------------
class TestIrisLivenessDetector:
    def test_fewer_than_two_images_is_not_live(self) -> None:
        ld = IrisLivenessDetector()
        r = ld.detect([_rng(1).standard_normal((20, 20))])
        assert r.is_live is False
        assert r.confidence == 0.0
        assert r.pupil_response == 0.0
        assert "error" in r.details

    def test_pupil_dynamics_high_variation_scores_one(self) -> None:
        ld = IrisLivenessDetector()
        score = ld._analyze_pupil_dynamics([], [10.0, 30.0, 20.0])
        # variation = std/mean is well above the 0.2 saturation point.
        assert score == pytest.approx(1.0)

    def test_pupil_dynamics_constant_radii_scores_zero(self) -> None:
        ld = IrisLivenessDetector()
        score = ld._analyze_pupil_dynamics([], [20.0, 20.0, 20.0])
        assert score == 0.0

    def test_pupil_dynamics_segments_when_radii_missing(self) -> None:
        # No pre-computed radii -> the detector segments each image itself.
        # On 16x16 images segmentation yields a constant default radius, so
        # the variation (and thus the response) is zero.
        ld = IrisLivenessDetector()
        imgs = [_tiny_eye(seed=i) for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            score = ld._analyze_pupil_dynamics(imgs, None)
        assert score == 0.0

    def test_pupil_dynamics_handles_segmentation_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If segmentation raises for every image, fewer than two radii are
        # collected and the detector reports no pupil response.
        ld = IrisLivenessDetector()

        def _boom(self, image):
            raise ValueError("segmentation failed")

        monkeypatch.setattr(IrisSegmenter, "segment", _boom)
        score = ld._analyze_pupil_dynamics([np.zeros((5, 5)), np.zeros((5, 5))], None)
        assert score == 0.0

    def test_specular_flat_image_scores_low(self) -> None:
        ld = IrisLivenessDetector()
        # No pixel exceeds the 99th percentile -> too few spots branch (0.2).
        assert ld._analyze_specular_reflections(np.full((50, 50), 0.5)) == 0.2

    def test_specular_rgb_is_downmixed(self) -> None:
        ld = IrisLivenessDetector()
        assert ld._analyze_specular_reflections(np.full((30, 30, 3), 0.5)) == 0.2

    def test_specular_small_image_hits_too_many_spots_branch(self) -> None:
        ld = IrisLivenessDetector()
        # In a tiny distinct-valued image a single pixel exceeds the 99th
        # percentile, giving a spot ratio (~1/16) well above expectation.
        score = ld._analyze_specular_reflections(_rng(5).random((4, 4)))
        assert score == 0.3

    def test_specular_expected_ratio_scores_one(self) -> None:
        ld = IrisLivenessDetector()
        img = np.zeros((200, 200))
        idx = _rng(11).choice(200 * 200, size=200, replace=False)  # 0.5%
        img.flat[idx] = 1.0
        assert ld._analyze_specular_reflections(img) == pytest.approx(1.0)

    def test_specular_off_expected_ratio_scores_partial(self) -> None:
        ld = IrisLivenessDetector()
        img = np.zeros((200, 200))
        idx = _rng(12).choice(200 * 200, size=300, replace=False)  # 0.75%
        img.flat[idx] = 1.0
        # 1 - |0.0075 - 0.005| / 0.005 = 0.5.
        assert ld._analyze_specular_reflections(img) == pytest.approx(0.5)

    def test_texture_flat_scores_low(self) -> None:
        ld = IrisLivenessDetector()
        # Zero gradient variance -> low-texture branch (0.2).
        assert ld._analyze_texture_authenticity(np.full((40, 40), 0.5)) == 0.2

    def test_texture_low_gradient_ramp_scores_low(self) -> None:
        ld = IrisLivenessDetector()
        ramp = np.tile(np.linspace(0.0, 0.15, 40), (40, 1))
        assert ld._analyze_texture_authenticity(ramp) == 0.2

    def test_texture_step_edge_hits_low_frequency_branch(self) -> None:
        ld = IrisLivenessDetector()
        step = np.zeros((40, 40))
        step[:, :20] = 1.0  # energy concentrated at low frequency
        assert ld._analyze_texture_authenticity(step) == 0.3

    def test_texture_rich_noise_scores_high(self) -> None:
        ld = IrisLivenessDetector()
        score = ld._analyze_texture_authenticity(_rng(13).standard_normal((40, 40)))
        assert score == pytest.approx(1.0)

    def test_texture_rgb_is_downmixed(self) -> None:
        ld = IrisLivenessDetector()
        score = ld._analyze_texture_authenticity(_rng(14).standard_normal((40, 40, 3)))
        assert 0.0 <= score <= 1.0

    def test_detect_combines_all_signals(self) -> None:
        ld = IrisLivenessDetector()
        imgs = [_rng(15).standard_normal((30, 30)) for _ in range(3)]
        r = ld.detect(imgs, pupil_radii=[10.0, 30.0, 20.0])
        assert isinstance(r, LivenessResult)
        assert 0.0 <= r.confidence <= 1.0
        # confidence is the mean of the three component scores.
        expected = (r.pupil_response + r.specular_reflection_score + r.texture_authenticity) / 3.0
        assert r.confidence == pytest.approx(expected)
        assert set(r.details) == {"pupil_live", "reflection_live", "texture_live"}

    def test_detect_all_thresholds_passed_is_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drive the three component analyzers to values above their default
        # thresholds so the combined verdict is a live subject. (Real image
        # statistics rarely push specular reflection high, so the component
        # scores are pinned here; each analyzer is verified on its own above.)
        ld = IrisLivenessDetector()
        monkeypatch.setattr(ld, "_analyze_pupil_dynamics", lambda *a, **k: 0.9)
        monkeypatch.setattr(ld, "_analyze_specular_reflections", lambda *a, **k: 0.8)
        monkeypatch.setattr(ld, "_analyze_texture_authenticity", lambda *a, **k: 0.9)
        imgs = [_rng(16 + i).standard_normal((40, 40)) for i in range(2)]
        r = ld.detect(imgs)
        assert r.is_live is True
        assert all(r.details.values())
        assert r.confidence == pytest.approx((0.9 + 0.8 + 0.9) / 3.0)


# ---------------------------------------------------------------------------
# IrisRecognizer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def enrolled():
    """A recognizer plus one tiny enrolled sample (computed once)."""
    rec = IrisRecognizer(liveness_required=True)
    img = _tiny_eye()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = rec.extract_features(img)
    return rec, img, feats


class TestIrisRecognizer:
    def test_construction_wires_all_components(self) -> None:
        rec = IrisRecognizer(match_threshold=0.25, liveness_required=False)
        assert isinstance(rec._segmenter, IrisSegmenter)
        assert isinstance(rec._normalizer, IrisNormalizer)
        assert isinstance(rec._encoder, IrisEncoder)
        assert isinstance(rec._matcher, IrisMatcher)
        assert isinstance(rec._liveness_detector, IrisLivenessDetector)
        assert rec._liveness_required is False

    def test_extract_features_shapes_and_metadata(self, enrolled: Any) -> None:
        _, _, feats = enrolled
        assert isinstance(feats, IrisFeatures)
        # 8 encoder filters -> 16 bit planes over a 64x256 normalized iris.
        assert feats.iris_code.shape == (16, 64, 256)
        assert feats.mask.shape == (16, 64, 256)
        assert feats.iris_code.dtype == np.uint8
        # Tiny image -> default segmentation geometry.
        assert feats.pupil_center == (8.0, 8.0)
        assert feats.pupil_radius == pytest.approx(50.0)
        assert feats.iris_radius == pytest.approx(105.0)
        assert feats.normalized_iris is not None
        assert feats.normalized_iris.shape == (64, 256)
        assert isinstance(feats.quality_score, float)
        assert 0.0 <= feats.quality_score <= 1.0

    def test_extract_features_is_deterministic(self, enrolled: Any) -> None:
        rec, img, feats = enrolled
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            again = rec.extract_features(img)
        assert np.array_equal(again.iris_code, feats.iris_code)
        assert again.quality_score == feats.quality_score

    def test_verify_blocks_match_when_not_live(self, enrolled: Any) -> None:
        rec, img, feats = enrolled
        # A single probe image cannot satisfy liveness (needs >= 2 frames),
        # so the match is forced to a non-match with zero score.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            match, liveness = rec.verify(img, feats)
        assert liveness is not None
        assert liveness.is_live is False
        assert match.is_match is False
        assert match.match_score == 0.0
        assert match.confidence == 0.0

    def test_verify_runs_liveness_with_extra_images(self, enrolled: Any) -> None:
        rec, img, feats = enrolled
        extra = [_tiny_eye(seed=99), _tiny_eye(seed=100)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            match, liveness = rec.verify(img, feats, liveness_images=extra)
        assert liveness is not None
        assert isinstance(match, IrisMatchResult)
        # Flat tiny frames still fail texture/reflection liveness checks.
        assert liveness.is_live is False

    def test_verify_keeps_match_when_liveness_passes(
        self, enrolled: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When liveness succeeds, the matcher's verdict must pass through
        # unchanged (the not-live override is skipped). Both the liveness
        # detector and the matcher are pinned so a genuine "match kept"
        # outcome is observable on tiny synthetic frames.
        rec, img, feats = enrolled
        live = LivenessResult(
            is_live=True,
            confidence=0.9,
            pupil_response=0.9,
            specular_reflection_score=0.9,
            texture_authenticity=0.9,
        )
        sentinel = IrisMatchResult(
            hamming_distance=0.1,
            match_score=0.9,
            is_match=True,
            bits_compared=4096,
            confidence=0.8,
        )
        monkeypatch.setattr(rec._liveness_detector, "detect", lambda *a, **k: live)
        monkeypatch.setattr(rec._matcher, "match", lambda *a, **k: sentinel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            match, liveness = rec.verify(img, feats)
        assert liveness is live
        assert match is sentinel
        assert match.is_match is True
        assert match.match_score == 0.9

    def test_verify_without_liveness_requirement(self) -> None:
        rec = IrisRecognizer(liveness_required=False)
        img = _tiny_eye(seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = rec.extract_features(img)
            match, liveness = rec.verify(img, feats)
        assert liveness is None
        assert isinstance(match, IrisMatchResult)

    def test_compute_quality_full_mask_high_texture(self) -> None:
        rec = IrisRecognizer(liveness_required=False)
        normalized = _rng(17).standard_normal((16, 32))
        mask = np.ones((16, 32), dtype=bool)
        q = rec._compute_quality(normalized, mask)
        # Full usable area, sharp gradients and high contrast saturate to 1.
        assert q == pytest.approx(1.0)
        assert isinstance(q, float)

    def test_compute_quality_reflects_usable_ratio(self) -> None:
        rec = IrisRecognizer(liveness_required=False)
        flat = np.zeros((16, 32))
        half = np.zeros((16, 32), dtype=bool)
        half[:8, :] = True
        q = rec._compute_quality(flat, half)
        # No sharpness / contrast, so quality is just 0.4 * usable_ratio.
        assert q == pytest.approx(0.4 * 0.5)


# ---------------------------------------------------------------------------
# End-to-end pipeline on a real synthetic eye (cheap small-range segmenter)
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def _encode_eye(self, eye: np.ndarray) -> tuple[Any, Any]:
        seg = _small_segmenter()
        norm = IrisNormalizer(angular_resolution=64, radial_resolution=16)
        enc = IrisEncoder(num_filters=4, kernel_size=15)
        pc, pr, ic, ir = seg.segment(eye)
        normalized, mask = norm.normalize(eye, pc, pr, ic, ir)
        return enc.encode(normalized, mask)

    def test_self_match_is_perfect(self) -> None:
        code, cmask = self._encode_eye(make_eye(seed=1))
        matcher = IrisMatcher()
        r = matcher.match(code, cmask, code, cmask)
        assert r.hamming_distance == pytest.approx(0.0)
        assert bool(r.is_match) is True
        assert r.match_score == pytest.approx(1.0)

    def test_pipeline_produces_binary_iriscode(self) -> None:
        code, cmask = self._encode_eye(make_eye(seed=2))
        # 4 filters -> 8 bit planes over the 16x64 normalized iris.
        assert code.shape == (8, 16, 64)
        assert cmask.shape == (8, 16, 64)
        assert set(np.unique(code)).issubset({0, 1})

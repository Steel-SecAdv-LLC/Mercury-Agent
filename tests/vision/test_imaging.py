# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioural tests for canonical image handling, capture-environment measurement and degradations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import ndimage

from omni_mercury_engine.vision.imaging import (
    DEGRADATIONS,
    assess_environment,
    degrade,
    estimate_noise_sigma,
    local_contrast_normalize,
    to_canonical,
)

from .conftest import woven_surface

if TYPE_CHECKING:
    from pathlib import Path


def _texture(seed: int = 0, size: int = 128) -> np.ndarray:
    base = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(size, size)), 3)
    return np.asarray((base - base.min()) / (base.max() - base.min()) * 0.6 + 0.2)


class TestCanonicalForm:
    def test_uint8_rgb_is_scaled_by_255(self) -> None:
        image = (woven_surface(0) * 255).astype(np.uint8)
        canonical = to_canonical(image)
        assert canonical.provenance.layout == "rgb"
        assert canonical.provenance.scaling == "unsigned 8-bit / 255"
        np.testing.assert_allclose(canonical.pixels, image / 255.0, atol=1e-6)

    def test_uint16_respects_the_true_sensor_bit_depth(self) -> None:
        twelve_bit = (np.random.default_rng(0).random((64, 64)) * 4095).astype(np.uint16)
        default = to_canonical(twelve_bit)
        exact = to_canonical(twelve_bit, bit_depth=12)
        assert "pass bit_depth" in default.provenance.scaling
        assert float(exact.luminance.max()) > 0.99
        assert float(default.luminance.max()) < 0.07

    def test_float_radiometric_input_is_min_max_mapped_and_says_so(self) -> None:
        kelvin = 280.0 + 40.0 * np.random.default_rng(1).random((64, 64))
        canonical = to_canonical(kelvin)
        assert "radiometric scale not preserved" in canonical.provenance.scaling
        assert canonical.luminance.min() == 0.0
        assert canonical.luminance.max() == 1.0

    def test_rgba_separates_alpha_and_reports_coverage(self) -> None:
        rgba = np.ones((64, 64, 4), dtype=np.float32)
        rgba[:16, :, 3] = 0.0
        canonical = to_canonical(rgba)
        assert canonical.pixels.shape[2] == 3
        assert canonical.provenance.alpha_coverage == pytest.approx(0.25)

    def test_channel_first_and_multispectral_layouts(self) -> None:
        channel_first = np.random.default_rng(2).random((3, 64, 80))
        assert to_canonical(channel_first).shape == (64, 80)
        cube = np.random.default_rng(3).random((64, 64, 6))
        canonical = to_canonical(cube)
        assert canonical.provenance.layout == "multichannel"
        np.testing.assert_allclose(canonical.luminance, cube.mean(axis=2), atol=1e-6)

    def test_bgr_order_is_honoured_for_arrays(self) -> None:
        bgr = np.zeros((40, 40, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # pure blue in OpenCV order
        canonical = to_canonical(bgr, channel_order="bgr")
        assert canonical.pixels[0, 0, 2] == pytest.approx(1.0)
        assert canonical.pixels[0, 0, 0] == pytest.approx(0.0)

    def test_nonfinite_pixels_are_masked_and_counted_not_hidden(self) -> None:
        image = np.random.default_rng(4).random((64, 64))
        image[10:20, 10:15] = np.nan
        image[0, 0] = np.inf
        canonical = to_canonical(image)
        assert canonical.provenance.nonfinite_pixels == 51
        assert canonical.invalid_mask.sum() == 51
        assert np.isfinite(canonical.pixels).all()

    def test_path_round_trip_reorders_to_rgb(self, tmp_path: Path) -> None:
        import cv2

        rgb = (woven_surface(5) * 255).astype(np.uint8)
        path = tmp_path / "surface.png"
        cv2.imwrite(str(path), rgb[..., ::-1])
        canonical = to_canonical(str(path))
        assert canonical.provenance.source == "path"
        np.testing.assert_allclose(canonical.pixels, rgb / 255.0, atol=1e-6)

    @pytest.mark.parametrize(
        ("bad", "message"),
        [
            (np.zeros((20, 64)), "at least 32"),
            (np.zeros((64, 64, 40)), "ambiguous"),
            (np.zeros((2, 64, 64, 3)), "one image"),
            (np.array([["a", "b"]]), "numeric"),
        ],
    )
    def test_unusable_inputs_are_refused_with_the_reason(
        self, bad: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            to_canonical(bad)

    def test_unreadable_path_names_the_blocker(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="could not be loaded"):
            to_canonical(str(tmp_path / "missing.png"))


class TestEnvironment:
    @pytest.mark.parametrize("sigma", [0.01, 0.03, 0.06])
    def test_noise_estimate_recovers_the_true_sensor_noise(self, sigma: float) -> None:
        noisy = np.clip(_texture() + np.random.default_rng(7).normal(0, sigma, (128, 128)), 0, 1)
        assert estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.12)

    def test_noise_estimate_ignores_strong_texture(self) -> None:
        yy, xx = np.indices((128, 128))
        stripes = 0.5 + 0.3 * np.sin(2 * np.pi * xx / 6)
        noisy = np.clip(stripes + np.random.default_rng(8).normal(0, 0.02, stripes.shape), 0, 1)
        assert estimate_noise_sigma(noisy) == pytest.approx(0.02, rel=0.35)

    def test_exposure_and_contrast_conditions_are_labelled(self) -> None:
        dark = to_canonical(_texture() * 0.1)
        flat = to_canonical(
            np.full((64, 64), 0.5) + np.random.default_rng(9).normal(0, 0.002, (64, 64))
        )
        assert "underexposed" in assess_environment(dark).conditions
        assert "low_contrast" in assess_environment(flat).conditions

    def test_capture_conditions_are_robust_to_a_localised_defect(self) -> None:
        clean = woven_surface(11)
        defective = clean.copy()
        defective[20:50, 20:50] = np.random.default_rng(12).random((30, 30, 3))
        a, b = assess_environment(to_canonical(clean)), assess_environment(to_canonical(defective))
        # A minority region must not move the capture measurements (per-tile medians, MAD).
        assert b.mean_luminance == pytest.approx(a.mean_luminance, rel=0.1)
        assert b.sharpness == pytest.approx(a.sharpness, rel=0.25)
        assert b.noise_sigma == pytest.approx(a.noise_sigma, rel=0.25)

    def test_color_cast_is_measured_for_colour_only(self) -> None:
        cast = woven_surface(13) * np.array([1.3, 1.0, 0.7])
        assert assess_environment(to_canonical(np.clip(cast, 0, 1))).color_cast is not None
        assert assess_environment(to_canonical(_texture())).color_cast is None

    def test_local_contrast_normalisation_cancels_gain_and_gradient(self) -> None:
        base = _texture(14)
        ramp = np.linspace(0.5, 1.0, base.shape[1])[None, :]
        a = local_contrast_normalize(base)
        b = local_contrast_normalize(base * ramp * 0.8)
        interior = (slice(16, -16), slice(16, -16))
        assert float(np.corrcoef(a[interior].ravel(), b[interior].ravel())[0, 1]) > 0.98


class TestDegradations:
    @pytest.mark.parametrize("kind", DEGRADATIONS)
    def test_every_degradation_preserves_shape_and_is_deterministic(self, kind: str) -> None:
        image = woven_surface(15)
        a = degrade(image, kind, 0.6, rng=np.random.default_rng(1))
        b = degrade(image, kind, 0.6, rng=np.random.default_rng(1))
        assert a.shape == image.shape
        assert np.array_equal(a, b)
        assert a.min() >= 0.0
        assert a.max() <= 1.0
        assert not np.array_equal(a, image)

    def test_severity_orders_the_effect(self) -> None:
        image = woven_surface(16)
        sharp = assess_environment(to_canonical(degrade(image, "gaussian_blur", 0.1))).sharpness
        soft = assess_environment(to_canonical(degrade(image, "gaussian_blur", 0.9))).sharpness
        assert soft < sharp

    def test_invalid_requests_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown degradation"):
            degrade(woven_surface(0), "fog", 0.5)
        with pytest.raises(ValueError, match="severity"):
            degrade(woven_surface(0), "jpeg", 1.5)


class TestPillowFallback:
    """Image files load without OpenCV (Pillow is a core dependency)."""

    @staticmethod
    def _without_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 -> ImportError

    def test_rgb_png_loads_in_rgb_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PIL import Image

        rgb = (woven_surface(5) * 255).astype(np.uint8)
        path = tmp_path / "surface.png"
        Image.fromarray(rgb).save(path)
        self._without_cv2(monkeypatch)
        canonical = to_canonical(str(path))
        np.testing.assert_allclose(canonical.pixels, rgb / 255.0, atol=1e-6)

    def test_sixteen_bit_depth_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PIL import Image

        from omni_mercury_engine.vision.imaging import _load_path

        gray = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 16).astype(np.uint16)
        path = tmp_path / "deep.png"
        Image.fromarray(gray).save(path)
        self._without_cv2(monkeypatch)
        loaded = _load_path(str(path))
        assert loaded.dtype.itemsize >= 2
        np.testing.assert_array_equal(loaded.astype(np.int64), gray.astype(np.int64))

    def test_palette_and_gray_alpha_map_to_rgb_layouts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PIL import Image

        from omni_mercury_engine.vision.imaging import _load_path

        rgb = (woven_surface(3) * 255).astype(np.uint8)
        palette = tmp_path / "palette.png"
        Image.fromarray(rgb).convert("P").save(palette)
        gray_alpha = tmp_path / "la.png"
        Image.fromarray(rgb[..., 0]).convert("LA").save(gray_alpha)
        self._without_cv2(monkeypatch)
        assert _load_path(str(palette)).shape[-1] == 3
        assert _load_path(str(gray_alpha)).shape[-1] == 4

    def test_unreadable_file_is_a_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "not-an-image.png"
        path.write_bytes(b"not an image")
        self._without_cv2(monkeypatch)
        with pytest.raises(ValueError, match="could not be loaded"):
            to_canonical(str(path))

    def test_pillow_and_opencv_decode_the_same_pixels(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        from PIL import Image

        from omni_mercury_engine.vision.imaging import _load_path, _load_path_pillow

        rgb = (woven_surface(7) * 255).astype(np.uint8)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 200, np.uint8)])
        path = tmp_path / "rgba.png"
        Image.fromarray(rgba).save(path)
        np.testing.assert_array_equal(_load_path(str(path)), _load_path_pillow(str(path)))


def test_jpeg_degradation_works_without_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from omni_mercury_engine.vision.imaging import degrade

    image = woven_surface(2)
    with_cv2 = degrade(image, "jpeg", 0.8) if _has_cv2() else None
    monkeypatch.setitem(sys.modules, "cv2", None)
    out = degrade(image, "jpeg", 0.8)
    assert out.shape == image.shape and np.isfinite(out).all()
    assert 0.0 < float(np.abs(out - image).mean()) < 0.1  # lossy, but the same picture
    if with_cv2 is not None:  # two JPEG codecs at one quality: close, not identical
        assert float(np.abs(out - with_cv2).mean()) < 0.02


def _has_cv2() -> bool:
    import importlib.util

    return importlib.util.find_spec("cv2") is not None

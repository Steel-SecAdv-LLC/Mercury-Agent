# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for interpretable patch features and image forensics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from omni_mercury_engine.vision.features import (
    _box_means,
    extract_patch_features,
    lattice_residual,
    lattice_residuals,
    lattice_structure,
    lattice_vectors,
    make_grid,
    uniform_lbp,
)
from omni_mercury_engine.vision.forensics import (
    copy_move_detection,
    jpeg_grid_analysis,
    noise_inconsistency,
    periodic_artifact_analysis,
)
from omni_mercury_engine.vision.imaging import degrade, to_canonical

from .conftest import woven_surface


def _texture(seed: int = 0, size: int = 192) -> np.ndarray:
    base = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(size, size)), 3)
    return np.asarray((base - base.min()) / (base.max() - base.min()) * 0.6 + 0.2)


class TestPatchFeatures:
    def test_box_means_equal_brute_force(self) -> None:
        maps = np.random.default_rng(0).random((3, 50, 61))
        grid = make_grid(50, 61, 16, 8)
        fast = _box_means(maps, grid.rows, grid.cols, 16)
        for i, y in enumerate(grid.rows):
            for j, x in enumerate(grid.cols):
                np.testing.assert_allclose(
                    fast[i, j], maps[:, y : y + 16, x : x + 16].mean(axis=(1, 2))
                )

    def test_grid_always_covers_the_border(self) -> None:
        grid = make_grid(100, 77, 16, 8)
        assert grid.rows[-1] + 16 == 100
        assert grid.cols[-1] + 16 == 77

    def test_lbp_of_a_flat_image_is_the_all_ones_pattern(self) -> None:
        assert np.all(uniform_lbp(np.full((20, 20), 0.5)) == 8)

    def test_lbp_labels_are_in_range_and_non_uniform_is_detected(self) -> None:
        labels = uniform_lbp(np.random.default_rng(1).random((40, 40)))
        assert labels.min() >= 0 and labels.max() <= 9
        assert np.any(labels == 9)

    def test_feature_layout_is_named_grouped_and_finite(self) -> None:
        feats = extract_patch_features(to_canonical(woven_surface(0)))
        assert len(feats.names) == feats.values.shape[-1] == len(feats.groups)
        assert {
            "intensity",
            "color",
            "gradient",
            "orientation",
            "texture",
            "frequency",
            "noise",
            "edges",
            "lattice",
        } <= set(feats.groups)
        assert np.isfinite(feats.values).all()
        assert all(n.startswith(("s1.", "s0.5.")) for n in feats.names)

    def test_histograms_are_distributions(self) -> None:
        feats = extract_patch_features(to_canonical(woven_surface(1)))
        for prefix in ("texture_lbp", "orientation_bin"):
            index = [
                i
                for i, n in enumerate(feats.base_names)
                if n.startswith(prefix) and feats.scales[i] == 1.0
            ]
            np.testing.assert_allclose(feats.values[..., index].sum(axis=-1), 1.0, atol=1e-4)

    def test_grayscale_has_no_colour_dimensions(self) -> None:
        feats = extract_patch_features(to_canonical(_texture(2)))
        assert "color" not in feats.groups

    @pytest.mark.parametrize("gain", [0.6, 0.8, 1.2])
    def test_every_feature_group_is_invariant_to_global_illumination(self, gain: float) -> None:
        image = woven_surface(3)
        reference = extract_patch_features(to_canonical(image))
        relit = extract_patch_features(to_canonical(np.clip(image * gain, 0, 1)))
        # Structure features use contrast-normalised luminance; intensity and
        # noise are median-relative; colour is grey-world balanced. A global
        # gain is a capture condition (measured by the environment check),
        # never content.
        np.testing.assert_allclose(reference.values, relit.values, atol=1e-3)


_N = 128
_YY, _XX = np.indices((_N, _N)).astype(float)


def _lattice(period_y: float = 11.0, period_x: float = 7.3, angle_deg: float = 0.0) -> np.ndarray:
    """Analytic two-period lattice, optionally rotated."""
    theta = np.deg2rad(angle_deg)
    u = _XX * np.cos(theta) + _YY * np.sin(theta)
    w = -_XX * np.sin(theta) + _YY * np.cos(theta)
    return np.asarray(
        0.5 + 0.15 * np.cos(2 * np.pi * u / period_x) + 0.15 * np.cos(2 * np.pi * w / period_y)
    )


def _symmetry_break(lum: np.ndarray) -> np.ndarray:
    structure = lattice_structure(lum)
    return np.asarray(lattice_residuals(structure, lattice_vectors(structure)).max(axis=0))


class TestLatticeSymmetry:
    def test_sub_pixel_periods_are_recovered(self) -> None:
        vectors = lattice_vectors(lattice_structure(_lattice()))
        np.testing.assert_allclose(vectors[0], [0.0, 7.3], atol=0.05)
        np.testing.assert_allclose(vectors[1], [11.0, 0.0], atol=0.05)

    def test_a_rotated_lattice_is_recovered_with_its_angle(self) -> None:
        vectors = lattice_vectors(lattice_structure(_lattice(9.0, 9.0, angle_deg=20.0)))
        lengths = np.hypot(vectors[:, 0], vectors[:, 1])
        np.testing.assert_allclose(lengths, 9.0, atol=0.05)
        # A square lattice at 20 degrees: both basis vectors lie at 20 mod 90, at right angles.
        angles = np.mod(np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])), 90.0)
        np.testing.assert_allclose(angles, 20.0, atol=0.5)
        assert abs(float(vectors[0] @ vectors[1])) / float(lengths.prod()) < 0.01

    def test_single_lag_residual_matches_the_batched_form(self) -> None:
        structure = lattice_structure(_lattice())
        lags = lattice_vectors(structure)
        batched = lattice_residuals(structure, lags)
        for k, lag in enumerate(lags):
            np.testing.assert_array_equal(lattice_residual(structure, lag), batched[k])

    def test_a_clean_lattice_has_no_symmetry_break(self) -> None:
        assert float(np.median(_symmetry_break(_lattice()))) < 0.01

    def test_borders_and_corner_pockets_never_read_as_breaks(self) -> None:
        # A non-square lattice at 35 degrees: both neighbours of a corner pixel
        # leave the image along the diagonal basis vectors. Mirrored samples and
        # the median's border band must give "no evidence", not a break, at the
        # 16-pixel patch level the analyser consumes.
        broken = _symmetry_break(_lattice(23.0, 17.0, angle_deg=35.0))
        pooled = ndimage.uniform_filter(broken, 16)[8:-8:8, 8:-8:8]
        assert float(pooled.max()) < 0.05

    def test_foreign_structure_breaks_symmetry_only_where_it_is(self) -> None:
        mask = np.zeros((_N, _N), dtype=bool)
        mask[50:76, 40:66] = True
        lum = _lattice()
        lum[mask] = _lattice(9.0, 9.0, angle_deg=20.0)[mask]
        broken = _symmetry_break(lum)
        core = ndimage.binary_erosion(mask, iterations=3)
        far = ~ndimage.binary_dilation(mask, iterations=3)
        assert broken[core].mean() > 10 * broken[far].mean()
        assert broken[core].mean() > 0.3

    def test_an_exposure_change_is_not_a_symmetry_break(self) -> None:
        # Log-luminance minus an edge-preserving median background: a darkened
        # region keeps its lattice, so only the exposure step's edge responds.
        mask = np.zeros((_N, _N), dtype=bool)
        mask[50:76, 40:66] = True
        lum = _lattice()
        lum[mask] *= 0.45
        broken = _symmetry_break(lum)
        assert broken[ndimage.binary_erosion(mask, iterations=3)].mean() < 0.03
        assert broken[~ndimage.binary_dilation(mask, iterations=3)].mean() < 0.03

    def test_an_aperiodic_texture_has_a_stable_nonzero_baseline(self) -> None:
        baseline = [float(np.median(_symmetry_break(_texture(seed, 128)))) for seed in range(3)]
        assert all(0.1 < b < 1.0 for b in baseline)
        assert max(baseline) - min(baseline) < 0.15


class TestForensics:
    def test_spliced_noise_level_is_flagged_and_located(self) -> None:
        base = _texture(4)
        spliced = base.copy()
        spliced[64:128, 64:128] += np.random.default_rng(5).normal(0, 0.04, (64, 64))
        result = noise_inconsistency(np.clip(spliced, 0, 1))
        clean = noise_inconsistency(
            np.clip(base + np.random.default_rng(6).normal(0, 0.01, base.shape), 0, 1)
        )
        assert result["max_abs_z"] > 10.0
        assert clean["max_abs_z"] < 4.0
        boxes = [f["box"] for f in result["flagged"]]
        assert boxes
        for x0, y0, x1, y1 in boxes:
            assert x0 >= 64 and y0 >= 64
            assert x1 <= 128 and y1 <= 128

    def test_jpeg_grid_presence_and_phase(self) -> None:
        rgb = np.stack([_texture(7)] * 3, axis=-1)
        compressed = degrade(rgb, "jpeg", 0.8)[..., 0]
        aligned = jpeg_grid_analysis(compressed)
        cropped = jpeg_grid_analysis(compressed[3:, 5:])
        raw = jpeg_grid_analysis(_texture(7))
        assert aligned["grid_present"] and not aligned["grid_misaligned"]
        assert cropped["grid_misaligned"] and cropped["grid_offset"] == (3, 5)
        assert not raw["grid_present"]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_copy_move_recovers_the_exact_shift(self, seed: int) -> None:
        base = _texture(10 + seed)
        rng = np.random.default_rng(seed)
        size = int(rng.integers(32, 48))
        sy, sx = (int(v) for v in rng.integers(0, 40, 2))
        dy, dx = (int(v) for v in rng.integers(90, 192 - 48 - size, 2))
        forged = base.copy()
        forged[sy + dy : sy + dy + size, sx + dx : sx + dx + size] = base[
            sy : sy + size, sx : sx + size
        ]
        forged = np.clip(forged + rng.normal(0, 0.003, forged.shape), 0, 1)
        result = copy_move_detection(forged)
        assert result["detected"]
        assert tuple(result["clusters"][0]["shift"]) == (dx, dy)

    def test_copy_move_has_no_false_positive_on_a_clean_image(self) -> None:
        clean = np.clip(_texture(20) + np.random.default_rng(21).normal(0, 0.003, (192, 192)), 0, 1)
        assert not copy_move_detection(clean)["detected"]

    def test_recapture_moire_is_a_strong_isolated_spectral_peak(self) -> None:
        base = _texture(22)
        yy, xx = np.indices(base.shape)
        moire = np.clip(base + 0.08 * np.sin(2 * np.pi * (0.21 * xx + 0.13 * yy)), 0, 1)
        peak = periodic_artifact_analysis(moire)
        natural = periodic_artifact_analysis(base)
        assert peak["peak_excess"] > 4.0
        assert natural["peak_excess"] < 2.0
        assert peak["frequency_cycles_per_pixel"] == pytest.approx(np.hypot(0.21, 0.13), abs=0.01)

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dense, interpretable multi-scale patch features.

Each patch of an image is described by a fixed set of named measurements, and
every dimension means something a reviewer can check by eye:

* ``intensity``: patch mean and standard deviation of (illumination-relative)
  luminance;
* ``color``: opponent-colour means (red-green, yellow-blue) and saturation,
  after grey-world white balance (colour images only);
* ``gradient``: mean Sobel gradient magnitude of contrast-normalised luminance;
* ``orientation``: an 8-bin, magnitude-weighted histogram of unsigned gradient
  orientation (soft-assigned) — the patch's edge geometry;
* ``texture``: the 10-bin uniform local-binary-pattern histogram (Ojala et al.,
  2002) — illumination-invariant micro-texture;
* ``frequency``: log energy in three difference-of-Gaussian bands (fine, mid,
  coarse) — sharpness, blur and periodic structure;
* ``noise``: log mean absolute Immerkær residual — sensor-noise level;
* ``edges``: fraction of pixels whose normalised gradient exceeds 1;
* ``lattice``: failure of the texture's own translation symmetry. The two
  shortest lattice vectors are found per image from the autocorrelation (to
  sub-pixel precision), and each pixel is compared with its neighbours one
  period away; ``lattice_any_axis`` and ``lattice_both_axes`` are the per-pixel
  max and min over the two vectors. A bend, break or misplaced element in a
  periodic texture (a mesh, weave or tiling) is a local symmetry break that
  the other, purely local, statistics cannot see.

Patch statistics are exact box means over each patch window, computed with
integral images, so a patch's feature does not depend on its neighbours.
Features are computed at several pyramid scales on one shared patch grid (a
coarser scale sees a larger receptive field around the same centre).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from omni_mercury_engine.vision.imaging import (
    CanonicalImage,
    gray_world_balance,
    immerkaer_residual,
    local_contrast_normalize,
)

Array = np.ndarray[Any, Any]

FEATURE_GROUPS: tuple[str, ...] = (
    "intensity",
    "color",
    "gradient",
    "orientation",
    "texture",
    "frequency",
    "noise",
    "edges",
    "lattice",
)

_ORIENTATION_BINS = 8
_LBP_BINS = 10
_LATTICE_STRONG_FRACTION = 0.75
#: Side of the separable median that removes the illumination background.
_LATTICE_MEDIAN = 9
#: Pixels this close to the border carry no lattice evidence: the median
#: background reaches 4 px into reflected pixels and the cubic spline 2 px more.
_LATTICE_BORDER = 6
_EPS = 1e-6


@dataclass(frozen=True)
class PatchGrid:
    """Top-left coordinates of the square patches covering an image.

    Attributes:
        patch_size: Patch side in pixels.
        stride: Step between patches; the last row and column are shifted so
            the grid always covers the image border.
        rows: Top-left y of each patch row.
        cols: Top-left x of each patch column.
        height: Image height.
        width: Image width.
    """

    patch_size: int
    stride: int
    rows: Array
    cols: Array
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(grid rows, grid cols)``."""
        return int(self.rows.size), int(self.cols.size)

    def centers(self) -> tuple[Array, Array]:
        """Return the patch-centre y and x coordinates."""
        half = self.patch_size / 2.0
        return self.rows + half, self.cols + half


def make_grid(height: int, width: int, patch_size: int, stride: int) -> PatchGrid:
    """Build a :class:`PatchGrid` covering an ``height`` x ``width`` image."""
    if patch_size > min(height, width):
        raise ValueError(f"patch size {patch_size} exceeds the image ({height}x{width})")

    def _starts(extent: int) -> Array:
        starts = np.arange(0, extent - patch_size + 1, stride)
        if starts[-1] + patch_size < extent:
            starts = np.append(starts, extent - patch_size)
        return starts.astype(np.int64)

    return PatchGrid(
        patch_size=patch_size,
        stride=stride,
        rows=_starts(height),
        cols=_starts(width),
        height=height,
        width=width,
    )


def _box_means(maps: Array, rows: Array, cols: Array, size: int) -> Array:
    """Exact means over ``size`` x ``size`` windows of each map, via integral images.

    Args:
        maps: ``(K, H, W)`` stack of per-pixel maps.
        rows: Window top-left y coordinates.
        cols: Window top-left x coordinates.
        size: Window side.

    Returns:
        ``(len(rows), len(cols), K)`` window means.
    """
    k, height, width = maps.shape
    integral = np.zeros((k, height + 1, width + 1), dtype=np.float64)
    integral[:, 1:, 1:] = maps.cumsum(axis=1).cumsum(axis=2)
    y0 = rows[:, np.newaxis]
    x0 = cols[np.newaxis, :]
    y1, x1 = y0 + size, x0 + size
    sums = integral[:, y1, x1] - integral[:, y0, x1] - integral[:, y1, x0] + integral[:, y0, x0]
    return np.moveaxis(sums / float(size * size), 0, -1)


def _soft_orientation_bins(gx: Array, gy: Array) -> Array:
    """Return ``(8, H, W)`` magnitude-weighted, linearly soft-assigned orientation maps."""
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    position = angle / np.pi * _ORIENTATION_BINS - 0.5
    lower = np.floor(position).astype(np.int64)
    weight_upper = position - lower
    bins = np.zeros((_ORIENTATION_BINS, *gx.shape), dtype=np.float64)
    for offset, weight in ((0, 1.0 - weight_upper), (1, weight_upper)):
        index = np.mod(lower + offset, _ORIENTATION_BINS)
        for b in range(_ORIENTATION_BINS):
            bins[b] += np.where(index == b, weight * magnitude, 0.0)
    return bins


def uniform_lbp(luminance: Array) -> Array:
    """Return per-pixel uniform LBP labels ``0..9`` (P=8, R=1).

    Labels 0-8 count the neighbours at or above the centre for patterns with at
    most two 0/1 transitions around the circle; label 9 marks every
    non-uniform pattern (Ojala, Pietikäinen & Mäenpää, 2002).
    """
    padded = np.pad(luminance, 1, mode="reflect")
    center = padded[1:-1, 1:-1]
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    height, width = luminance.shape
    bits = np.stack(
        [padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width] >= center for dy, dx in offsets]
    ).astype(np.int64)
    transitions = np.sum(bits != np.roll(bits, -1, axis=0), axis=0)
    ones = bits.sum(axis=0)
    return np.where(transitions <= 2, ones, _LBP_BINS - 1)


def lattice_structure(luminance: Array) -> Array:
    """Return the illumination-free structure image used for lattice analysis.

    Log-luminance turns a multiplicative lighting or exposure change into an
    additive offset. The offset is removed by subtracting an edge-preserving
    background: a separable 9-pixel median (rows, then columns). Unlike a
    Gaussian local mean, a median follows a sharp exposure step instead of
    smearing it (measured: at most 0.005 residual offset beside a 0.8
    log-unit step, against 0.27 for a half-resolution median), so a darkened
    region does not leave a halo of false symmetry breaks around itself.
    """
    lum = np.asarray(luminance, dtype=np.float64)
    # Median-relative first, so a global gain cancels exactly despite the log floor.
    log_lum = np.log(lum / max(float(np.median(lum)), 0.05) + 0.02)
    rows = ndimage.median_filter(log_lum, size=(1, _LATTICE_MEDIAN), mode="reflect")
    background = ndimage.median_filter(rows, size=(_LATTICE_MEDIAN, 1), mode="reflect")
    return np.asarray(log_lum - background)


@functools.lru_cache(maxsize=8)
def _window_autocorrelation(height: int, width: int) -> tuple[Array, Array]:
    """Return the Hann window and its zero-padded, centred autocorrelation (shape-only, cached)."""
    window = np.outer(np.hanning(height), np.hanning(width))
    shape = (2 * height, 2 * width)
    wacf = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(window, shape)) ** 2, shape))
    window.setflags(write=False)
    wacf.setflags(write=False)
    return window, wacf


def lattice_vectors(structure: Array, *, count: int = 2, min_lag: int = 4) -> Array:
    """Return the dominant translation symmetries of a texture as sub-pixel ``(dy, dx)`` lags.

    The normalised autocorrelation is computed with a Hann window and zero
    padding (linear, not circular) and divided by the window's own
    autocorrelation so long lags are not penalised. Local maxima are taken in
    one half-plane, with lag radius at least ``min_lag`` and at most a third of
    each side. Every point of a lattice is an equally strong peak, so among
    the strong ones (at least 75 % of the best) the shortest is the primary
    basis vector and the shortest one not collinear with it (``|sin| > 0.5``)
    the secondary. Each lag is refined to sub-pixel precision by a separable
    parabolic fit, so periods that are not whole pixels still align. Missing
    vectors (an aperiodic or tiny image) fall back to the shortest admissible
    lag along the missing axis, so the output shape is fixed.

    Args:
        structure: ``(H, W)`` map, normally :func:`lattice_structure` output.
        count: Number of vectors to return (1 or 2).
        min_lag: Smallest lag radius considered, in pixels.

    Returns:
        ``(count, 2)`` float64 lags.
    """
    height, width = structure.shape
    # Winsorise so a defect's strong, aperiodic edges cannot swamp the lattice peaks.
    median = float(np.median(structure))
    spread = 1.4826 * float(np.median(np.abs(structure - median))) + _EPS
    clipped = np.clip(structure, median - 3.0 * spread, median + 3.0 * spread)
    window, wacf = _window_autocorrelation(height, width)
    centred = (clipped - clipped.mean()) * window
    shape = (2 * height, 2 * width)
    acf = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(centred, shape)) ** 2, shape))
    # Only lags within a third of each side are admissible; crop to them plus
    # the 2-pixel reach of the peak filter, so the search costs a ninth.
    reach_y, reach_x = height // 3 + 3, width // 3 + 3
    crop = (
        slice(height - reach_y, height + reach_y + 1),
        slice(width - reach_x, width + reach_x + 1),
    )
    ratio = acf[crop] / np.maximum(wacf[crop], wacf.max() * 1e-3)
    ratio /= max(float(ratio[reach_y, reach_x]), _EPS)

    dy, dx = np.mgrid[-reach_y : reach_y + 1, -reach_x : reach_x + 1]
    admissible = (
        (np.hypot(dy, dx) >= min_lag)
        & (np.abs(dy) <= height // 3)
        & (np.abs(dx) <= width // 3)
        & ((dy > 0) | ((dy == 0) & (dx > 0)))
    )
    peaks = admissible & (ratio == ndimage.maximum_filter(ratio, size=5)) & (ratio > 0)
    rows, cols = np.nonzero(peaks)
    centre = ratio[rows, cols]
    # Separable parabolic refinement: sub-pixel offset and interpolated height per axis,
    # so a half-pixel lattice point is not judged weaker than a whole-pixel one.
    offsets, height_gain = [], np.zeros_like(centre)
    for axis_rows, axis_cols in ((1, 0), (0, 1)):
        minus = ratio[rows - axis_rows, cols - axis_cols]
        plus = ratio[rows + axis_rows, cols + axis_cols]
        curvature = minus - 2.0 * centre + plus
        safe = np.where(curvature < 0, curvature, -np.inf)
        offset = np.clip(0.5 * (minus - plus) / safe, -0.5, 0.5)
        offsets.append(offset)
        height_gain += -0.125 * (minus - plus) ** 2 / safe
    strength = centre + np.nan_to_num(height_gain)
    lags = np.stack([rows - reach_y + offsets[0], cols - reach_x + offsets[1]], axis=1)
    length = np.hypot(lags[:, 0], lags[:, 1])
    strong = strength >= _LATTICE_STRONG_FRACTION * (strength.max() if strength.size else 0.0)
    order = np.lexsort((-strength, length, ~strong))

    chosen: list[Array] = []
    for vector in lags[order]:
        if chosen:
            first = chosen[0]
            sine = abs(float(first[0] * vector[1] - first[1] * vector[0])) / (
                float(np.hypot(*first)) * float(np.hypot(*vector)) + _EPS
            )
            if sine <= 0.5:
                continue
        chosen.append(vector)
        if len(chosen) == count:
            break
    for fallback in (np.array([0.0, min_lag]), np.array([float(min_lag), 0.0])):
        if len(chosen) == count:
            break
        chosen.append(fallback)
    return np.asarray(chosen[:count], dtype=np.float64)


def lattice_residuals(structure: Array, lags: Array) -> Array:
    """Per-pixel failure of translation symmetry along each lag (non-negative, energy-normalised).

    For each pixel the squared difference to its neighbour one lattice period
    ahead and one period behind is taken (cubic-spline sampling with mirrored
    borders, so sub-pixel periods align), the smaller is kept (so a defect is
    blamed on itself, not on the pixel a period away), and divided by twice
    the local signal energy. Only comparisons between pixels at least 6 px
    inside the border count; where none exists (the border band, or a corner
    pocket of a diagonal lattice) the image's median residual is reported,
    because there is no evidence of a break there.

    A clean lattice scores near 0. An aperiodic texture has a stable non-zero
    baseline (median about 0.4 for smoothed noise), which the normal model
    learns as that texture's normal level. A bend, break or foreign structure
    in a lattice scores high exactly where it is.

    Args:
        structure: ``(H, W)`` map, normally :func:`lattice_structure` output.
        lags: ``(K, 2)`` ``(dy, dx)`` lags, normally from :func:`lattice_vectors`.

    Returns:
        ``(K, H, W)`` residual maps.
    """
    coefficients = ndimage.spline_filter(structure, order=3, mode="mirror")
    energy = 2.0 * ndimage.gaussian_filter(structure * structure, 2.0, mode="reflect") + 1e-4

    def _sampled(lag: Array) -> Array:
        shift = (-float(lag[0]), -float(lag[1]))
        return np.asarray(
            ndimage.shift(coefficients, shift, order=3, mode="mirror", prefilter=False)
        )

    height, width = structure.shape
    yy, xx = np.indices((height, width), dtype=np.float64)

    band = _LATTICE_BORDER

    def _inside(y: Array, x: Array) -> Array:
        return np.asarray(
            (y >= band) & (y <= height - 1 - band) & (x >= band) & (x <= width - 1 - band)
        )

    centre_inside = _inside(yy, xx)

    out = np.empty((len(lags), height, width), dtype=np.float64)
    for k, lag in enumerate(np.asarray(lags, dtype=np.float64)):
        # Mirrored samples and the border band (where the median background
        # sees reflected pixels) are not the lattice: compare only pixels and
        # neighbours that both lie inside the trusted interior.
        valid_ahead = centre_inside & _inside(yy + lag[0], xx + lag[1])
        valid_behind = centre_inside & _inside(yy - lag[0], xx - lag[1])
        ahead = np.where(valid_ahead, (structure - _sampled(lag)) ** 2, np.inf)
        behind = np.where(valid_behind, (structure - _sampled(-lag)) ** 2, np.inf)
        residual = np.minimum(ahead, behind) / energy
        # Where no valid comparison exists (the border band, or a corner
        # pocket of a diagonal lattice) there is no evidence either way:
        # report the image's typical residual rather than a break.
        known = np.isfinite(residual)
        residual[~known] = float(np.median(residual[known])) if known.any() else 0.0
        out[k] = residual
    return out


def _single_scale_maps(
    image: CanonicalImage, *, illumination_invariant: bool, lattice: bool = True
) -> tuple[Array, list[str], list[str]]:
    """Per-pixel feature maps at the image's native resolution.

    Returns:
        ``(maps, names, groups)`` where ``maps`` is ``(K, H, W)`` and the two
        lists name each map and its feature group. Map ``k`` named
        ``"intensity_std"`` is the per-pixel *second moment*; it is converted
        to a standard deviation after pooling.
    """
    lum = image.luminance.astype(np.float64)
    if illumination_invariant:
        median = max(float(np.median(lum)), 0.05)
        relative = np.clip(lum / median / 2.0, 0.0, 1.0)
        noise_scale = median
    else:
        relative = lum
        noise_scale = 1.0
    lcn = local_contrast_normalize(lum)

    maps: list[Array] = [relative, relative * relative]
    names = ["intensity_mean", "intensity_std"]
    groups = ["intensity", "intensity"]

    if image.is_color:
        rgb = (
            gray_world_balance(image.pixels.astype(np.float64))
            if illumination_invariant
            else (image.pixels.astype(np.float64))
        )
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maps += [
            (r - g) / np.sqrt(2.0),
            (r + g - 2.0 * b) / np.sqrt(6.0),
            (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1e-3),
        ]
        names += ["color_red_green", "color_yellow_blue", "color_saturation"]
        groups += ["color"] * 3

    gx = ndimage.sobel(lcn, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(lcn, axis=0, mode="reflect") / 8.0
    magnitude = np.hypot(gx, gy)
    maps.append(magnitude)
    names.append("gradient_magnitude")
    groups.append("gradient")

    orientation = _soft_orientation_bins(gx, gy)
    maps += list(orientation)
    names += [f"orientation_bin{b}" for b in range(_ORIENTATION_BINS)]
    groups += ["orientation"] * _ORIENTATION_BINS

    labels = uniform_lbp(lcn)
    maps += [(labels == b).astype(np.float64) for b in range(_LBP_BINS)]
    names += [f"texture_lbp{b}" for b in range(_LBP_BINS)]
    groups += ["texture"] * _LBP_BINS

    g1 = ndimage.gaussian_filter(lcn, 1.0, mode="reflect")
    g2 = ndimage.gaussian_filter(lcn, 2.0, mode="reflect")
    g4 = ndimage.gaussian_filter(lcn, 4.0, mode="reflect")
    maps += [(lcn - g1) ** 2, (g1 - g2) ** 2, (g2 - g4) ** 2]
    names += ["frequency_fine", "frequency_mid", "frequency_coarse"]
    groups += ["frequency"] * 3

    # Median-relative like intensity, so a global gain is an environment change.
    maps.append(np.abs(immerkaer_residual(lum)) / noise_scale)
    names.append("noise_residual")
    groups.append("noise")

    maps.append((magnitude > 1.0).astype(np.float64))
    names.append("edges_density")
    groups.append("edges")

    # Per-pixel max/min over the two basis vectors, so the features do not
    # depend on which basis vector an image happens to list first. When the
    # caller has established the domain is aperiodic the group is not
    # computed; constant maps keep the feature layout fixed.
    if lattice:
        structure = lattice_structure(lum)
        residuals = lattice_residuals(structure, lattice_vectors(structure))
        maps += [residuals.max(axis=0), residuals.min(axis=0)]
    else:
        maps += [np.zeros_like(lum), np.zeros_like(lum)]
    names += ["lattice_any_axis", "lattice_both_axes"]
    groups += ["lattice"] * 2

    return np.stack(maps), names, groups


def _pool(maps: Array, names: list[str], grid: PatchGrid) -> Array:
    """Pool per-pixel maps into per-patch features and apply the per-feature transforms."""
    pooled = _box_means(maps, grid.rows, grid.cols, grid.patch_size)
    out = pooled.copy()
    mean_index = names.index("intensity_mean")
    std_index = names.index("intensity_std")
    out[..., std_index] = np.sqrt(
        np.maximum(pooled[..., std_index] - pooled[..., mean_index] ** 2, 0.0)
    )
    orientation = [i for i, n in enumerate(names) if n.startswith("orientation_bin")]
    total = pooled[..., orientation].sum(axis=-1, keepdims=True)
    out[..., orientation] = pooled[..., orientation] / (total + _EPS)
    for i, name in enumerate(names):
        if name.startswith("frequency_") or name == "noise_residual":
            out[..., i] = np.log10(pooled[..., i] + _EPS)
    return out


@dataclass(frozen=True)
class PatchFeatures:
    """Interpretable features for every patch of one image.

    Attributes:
        values: ``(grid rows, grid cols, D)`` float32 features.
        names: Name of each of the ``D`` dimensions (scale-prefixed, e.g. ``"s1.texture_lbp3"``).
        groups: Feature group of each dimension.
        base_names: Scale-free name of each dimension (``"texture_lbp3"``).
        scales: Pyramid scale of each dimension.
        grid: The patch grid at full resolution.
    """

    values: Array
    names: tuple[str, ...]
    groups: tuple[str, ...]
    base_names: tuple[str, ...]
    scales: tuple[float, ...]
    grid: PatchGrid

    @property
    def flat(self) -> Array:
        """Return features as ``(n_patches, D)``."""
        return self.values.reshape(-1, self.values.shape[-1])


def _resize_canonical(image: CanonicalImage, scale: float) -> CanonicalImage:
    height, width = image.shape
    new_h = max(round(height * scale), 8)
    new_w = max(round(width * scale), 8)
    zoom = (new_h / height, new_w / width)
    pixels = np.clip(
        ndimage.zoom(image.pixels.astype(np.float64), (*zoom, 1), order=1, prefilter=False),
        0.0,
        1.0,
    )[:new_h, :new_w]
    luminance = np.clip(
        ndimage.zoom(image.luminance.astype(np.float64), zoom, order=1, prefilter=False), 0.0, 1.0
    )[:new_h, :new_w]
    invalid = ndimage.zoom(image.invalid_mask.astype(np.float64), zoom, order=0) > 0.5
    return CanonicalImage(
        pixels=pixels.astype(np.float32),
        luminance=luminance.astype(np.float32),
        invalid_mask=invalid[:new_h, :new_w],
        provenance=image.provenance,
    )


def extract_patch_features(
    image: CanonicalImage,
    *,
    patch_size: int = 16,
    stride: int = 8,
    scales: tuple[float, ...] = (1.0, 0.5),
    illumination_invariant: bool = True,
    lattice: bool = True,
) -> PatchFeatures:
    """Compute interpretable multi-scale features for every patch of ``image``.

    Args:
        image: The canonical image.
        patch_size: Patch side in pixels at full resolution.
        stride: Step between patch origins at full resolution.
        scales: Pyramid scales; ``1.0`` is native resolution. At a coarser
            scale ``s`` each patch keeps ``patch_size`` pixels, so it covers
            ``patch_size / s`` native pixels around the same centre.
        illumination_invariant: Use illumination-relative intensity and
            grey-world-balanced colour, so a global lighting change is
            reported by the environment check instead of as content.
        lattice: Compute the lattice-symmetry group. ``False`` fills it with
            zeros (same layout) for a domain already known to be aperiodic.

    Returns:
        The patch features on the full-resolution grid.
    """
    if not scales:
        raise ValueError("at least one scale is required")
    height, width = image.shape
    grid = make_grid(height, width, patch_size, stride)
    centers_y, centers_x = grid.centers()

    blocks: list[Array] = []
    names: list[str] = []
    groups: list[str] = []
    base_names: list[str] = []
    scale_of: list[float] = []
    for scale in scales:
        if scale == 1.0:
            scaled = image
        else:
            scaled = _resize_canonical(image, scale)
        maps, map_names, map_groups = _single_scale_maps(
            scaled, illumination_invariant=illumination_invariant, lattice=lattice
        )
        s_height, s_width = scaled.shape
        size = min(patch_size, s_height, s_width)
        rows = np.clip(
            np.round(centers_y * (s_height / height) - size / 2.0), 0, s_height - size
        ).astype(np.int64)
        cols = np.clip(
            np.round(centers_x * (s_width / width) - size / 2.0), 0, s_width - size
        ).astype(np.int64)
        scaled_grid = PatchGrid(size, stride, rows, cols, s_height, s_width)
        blocks.append(_pool(maps, map_names, scaled_grid))
        prefix = "s1" if scale == 1.0 else f"s{scale:g}"
        names += [f"{prefix}.{n}" for n in map_names]
        groups += map_groups
        base_names += map_names
        scale_of += [float(scale)] * len(map_names)

    values = np.concatenate(blocks, axis=-1).astype(np.float32)
    return PatchFeatures(
        values=values,
        names=tuple(names),
        groups=tuple(groups),
        base_names=tuple(base_names),
        scales=tuple(scale_of),
        grid=grid,
    )


__all__ = [
    "FEATURE_GROUPS",
    "PatchFeatures",
    "PatchGrid",
    "extract_patch_features",
    "lattice_residuals",
    "lattice_structure",
    "lattice_vectors",
    "make_grid",
    "uniform_lbp",
]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image forensics: evidence that an image was spliced, duplicated, recompressed or recaptured.

These checks look for manipulation and acquisition traces that content-level
anomaly scores do not see:

* :func:`noise_inconsistency` — block-wise sensor-noise level (texture-robust
  Immerkær estimate). A region pasted from another camera, or retouched,
  usually carries a different noise level from the rest of the image.
* :func:`jpeg_grid_analysis` — the 8x8 JPEG blocking grid (boundary-to-interior
  difference ratio, after Wang, Bovik & Evans, 2000) and its *phase*: a grid
  that is present but not aligned to the image origin indicates cropping or
  recompression after editing.
* :func:`copy_move_detection` — duplicated regions found by matching quantised
  low-frequency DCT coefficients of overlapping blocks and voting on the shift
  between matches (Fridrich, Soukal & Lukáš, 2003).
* :func:`periodic_artifact_analysis` — isolated peaks in the windowed 2-D
  spectrum above the radial background: moiré from photographing a screen, or
  halftone from photographing a print — presentation-attack cues for
  biometric capture.

Each function returns raw, interpretable measurements. Deciding what is
unusual for a given image domain (a woven fabric is periodic by nature) is the
analyser's job, by comparing these values against the normal set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import (
    fft as sp_fft,
    ndimage,
)

from omni_mercury_engine.vision.imaging import immerkaer_residual

Array = np.ndarray[Any, Any]


def noise_level_map(luminance: Array, block: int = 32) -> tuple[Array, Array, Array]:
    """Estimate the sensor-noise level of every ``block`` x ``block`` tile.

    The Immerkær response is read at the lower-gradient 60% of each tile's
    pixels and converted with the Gaussian factor, so edges and texture do not
    masquerade as noise.

    Args:
        luminance: ``(H, W)`` luminance.
        block: Tile side.

    Returns:
        ``(sigma, rows, cols)``: ``(By, Bx)`` noise estimates and the tile
        top-left coordinates.
    """
    lum = luminance.astype(np.float64)
    residual = np.abs(immerkaer_residual(lum))
    gradient = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1))
    height, width = lum.shape
    rows = np.arange(0, height - block + 1, block)
    cols = np.arange(0, width - block + 1, block)
    sigma = np.zeros((rows.size, cols.size))
    for i, y in enumerate(rows):
        for j, x in enumerate(cols):
            tile_res = residual[y : y + block, x : x + block]
            tile_grad = gradient[y : y + block, x : x + block]
            cutoff = np.quantile(tile_grad, 0.6)
            selected = tile_res[tile_grad <= cutoff]
            sigma[i, j] = np.median(selected) / (6.0 * 0.6745)
    return sigma, rows, cols


def noise_inconsistency(luminance: Array, block: int = 32, z_limit: float = 4.0) -> dict[str, Any]:
    """Find tiles whose noise level departs from the image's own robust norm.

    Args:
        luminance: ``(H, W)`` luminance.
        block: Tile side.
        z_limit: Robust z-score (log noise level, median/MAD) above which a
            tile is flagged.

    Returns:
        ``max_abs_z`` (the inconsistency statistic), ``median_sigma``, and
        ``flagged`` tiles as ``{"box": (x0, y0, x1, y1), "sigma", "z"}``.
    """
    sigma, rows, cols = noise_level_map(luminance, block)
    if sigma.size < 4:
        return {"max_abs_z": 0.0, "median_sigma": float(np.median(sigma)), "flagged": []}
    log_sigma = np.log(sigma + 1e-6)
    median = float(np.median(log_sigma))
    mad = float(np.median(np.abs(log_sigma - median))) * 1.4826
    z = (log_sigma - median) / max(mad, 0.05)
    flagged = [
        {
            "box": (int(cols[j]), int(rows[i]), int(cols[j] + block), int(rows[i] + block)),
            "sigma": float(sigma[i, j]),
            "z": float(z[i, j]),
        }
        for i, j in zip(*np.nonzero(np.abs(z) > z_limit), strict=True)
    ]
    return {
        "max_abs_z": float(np.max(np.abs(z))),
        "median_sigma": float(np.exp(median)),
        "flagged": flagged,
    }


def jpeg_grid_analysis(luminance: Array) -> dict[str, Any]:
    """Measure 8x8 JPEG blocking strength and the grid's phase.

    For each of the 8 horizontal and 8 vertical phases, the mean absolute
    luminance step across columns (rows) at that phase is divided by the mean
    step elsewhere. A compressed image shows a ratio above 1 at one phase.

    Returns:
        ``blockiness`` (geometric mean of the two axes' strongest ratios),
        ``axis_ratios``, ``grid_offset`` ``(dx, dy)`` of the strongest phases,
        ``grid_present`` (both axes block at a ratio of at least 1.2 — a
        screening value) and ``grid_misaligned`` (blocking present at a
        non-zero phase: a cropping or post-edit recompression cue). The
        analyser additionally calibrates ``blockiness`` against normal images.
    """
    lum = luminance.astype(np.float64)
    dx = np.abs(np.diff(lum, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(lum, axis=0)).mean(axis=1)

    def _phase_ratios(steps: Array) -> Array:
        index = np.arange(1, steps.size + 1)
        ratios = np.ones(8)
        for phase in range(8):
            on = steps[(index % 8) == phase]
            off = steps[(index % 8) != phase]
            if on.size and off.size:
                ratios[phase] = on.mean() / (off.mean() + 1e-9)
        return ratios

    rx, ry = _phase_ratios(dx), _phase_ratios(dy)
    phase_x, phase_y = int(np.argmax(rx)), int(np.argmax(ry))
    blockiness = float(np.sqrt(rx[phase_x] * ry[phase_y]))
    # JPEG blocks are square: compression blocks both axes. One strongly
    # aligned axis is ordinary scene structure (measured: an uncompressed
    # face with sharp horizontal edges reads 1.15 / 1.25 per axis, moderate
    # JPEG 1.48+ on both), so presence requires the weaker axis to block.
    present = min(rx[phase_x], ry[phase_y]) >= 1.2
    return {
        "blockiness": blockiness,
        "axis_ratios": (float(rx[phase_x]), float(ry[phase_y])),
        "grid_offset": (phase_x, phase_y),
        "grid_present": bool(present),
        "grid_misaligned": bool(present and (phase_x, phase_y) != (0, 0)),
    }


def copy_move_detection(
    luminance: Array,
    *,
    block: int = 16,
    step: int = 1,
    quantization: float = 0.08,
    min_pairs: int = 6,
    min_block_std: float = 0.03,
) -> dict[str, Any]:
    """Detect duplicated regions by DCT block matching with shift-vector voting.

    Overlapping ``block`` x ``block`` windows (every ``step`` pixels) are
    described by their quantised 4x4 low-frequency DCT coefficients. Windows
    with identical descriptors are paired, and pairs are grouped by their
    displacement; a displacement shared by at least ``min_pairs`` pairs is a
    copied region (isolated coincidences do not repeat one shift). Flat windows
    are skipped because uniform areas match everywhere.

    Returns:
        ``detected`` and ``clusters``, each ``{"shift": (dx, dy), "pairs",
        "source_box", "target_box"}`` with boxes as ``(x0, y0, x1, y1)``.
    """
    lum = luminance.astype(np.float64)
    windows = sliding_window_view(lum, (block, block))[::step, ::step]
    n_y, n_x = windows.shape[:2]
    flat_windows = windows.reshape(n_y * n_x, block, block)
    stds = flat_windows.std(axis=(1, 2))
    keep = np.nonzero(stds >= min_block_std)[0]
    if keep.size < 2 * min_pairs:
        return {"detected": False, "clusters": []}
    coefficients = sp_fft.dctn(flat_windows[keep], axes=(1, 2), norm="ortho")[:, :4, :4]
    descriptors = np.round(coefficients.reshape(keep.size, 16) / quantization).astype(np.int64)
    order = np.lexsort(descriptors.T[::-1])
    sorted_desc = descriptors[order]
    positions = np.stack([keep // n_x * step, keep % n_x * step], axis=1)[order]

    sources_parts: list[Array] = []
    shifts_parts: list[Array] = []
    for lag in (1, 2, 3):
        same = np.nonzero(np.all(sorted_desc[lag:] == sorted_desc[:-lag], axis=1))[0]
        if same.size == 0:
            continue
        p, q = positions[same], positions[same + lag]
        shift = q - p
        # Canonical direction: dx > 0, or dx == 0 and dy > 0, so (a->b) and
        # (b->a) vote for the same displacement.
        flip = (shift[:, 1] < 0) | ((shift[:, 1] == 0) & (shift[:, 0] < 0))
        shift = np.where(flip[:, np.newaxis], -shift, shift)
        source = np.where(flip[:, np.newaxis], q, p)
        far = np.max(np.abs(shift), axis=1) >= block
        sources_parts.append(source[far])
        shifts_parts.append(shift[far])
    if not shifts_parts or sum(s.shape[0] for s in shifts_parts) == 0:
        return {"detected": False, "clusters": []}
    sources_all = np.concatenate(sources_parts)
    shifts_all = np.concatenate(shifts_parts)
    unique_shifts, inverse, counts = np.unique(
        shifts_all, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)

    clusters = []
    for index in np.argsort(-counts):
        if counts[index] < min_pairs:
            break
        shift_y, shift_x = int(unique_shifts[index][0]), int(unique_shifts[index][1])
        members = sources_all[inverse == index]
        ys, xs = members[:, 0], members[:, 1]
        source_box = (int(xs.min()), int(ys.min()), int(xs.max() + block), int(ys.max() + block))
        target_box = (
            source_box[0] + shift_x,
            source_box[1] + shift_y,
            source_box[2] + shift_x,
            source_box[3] + shift_y,
        )
        clusters.append(
            {
                "shift": (shift_x, shift_y),
                "pairs": int(counts[index]),
                "source_box": source_box,
                "target_box": target_box,
            }
        )
    return {"detected": bool(clusters), "clusters": clusters}


def periodic_artifact_analysis(
    luminance: Array, min_radius_fraction: float = 0.06
) -> dict[str, Any]:
    """Find isolated spectral peaks (moiré, halftone, screen pixel grids).

    The Hann-windowed, mean-removed luminance spectrum is compared with its own
    radial median at every radius; the excess (in natural-log magnitude units)
    of the strongest off-centre peak is the statistic. Natural scenes have
    smooth power-law spectra with small excess; recaptured screens and prints
    add sharp peaks.

    Args:
        luminance: ``(H, W)`` luminance.
        min_radius_fraction: Ignore frequencies below this fraction of the
            image size (low-frequency scene content).

    Returns:
        ``peak_excess``, the peak's ``frequency_cycles_per_pixel`` and
        ``orientation_degrees``, and ``n_strong_peaks`` (excess above 2.0).
    """
    lum = luminance.astype(np.float64)
    height, width = lum.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.fft.fftshift(np.fft.fft2((lum - lum.mean()) * window))
    log_magnitude = np.log1p(np.abs(spectrum))
    yy, xx = np.indices((height, width))
    cy, cx = height // 2, width // 2
    fy = (yy - cy) / height
    fx = (xx - cx) / width
    radius = np.hypot(fy, fx)
    ring = np.minimum((radius * min(height, width)).astype(np.int64), min(height, width) // 2)
    background = ndimage.median(log_magnitude, labels=ring, index=np.arange(ring.max() + 1))
    excess = log_magnitude - np.asarray(background)[ring]
    valid = (radius >= min_radius_fraction) & (radius <= 0.5) & (fy >= 0)
    if not np.any(valid):
        return {
            "peak_excess": 0.0,
            "frequency_cycles_per_pixel": 0.0,
            "orientation_degrees": 0.0,
            "n_strong_peaks": 0,
        }
    masked = np.where(valid, excess, -np.inf)
    peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return {
        "peak_excess": float(masked[peak]),
        "frequency_cycles_per_pixel": float(radius[peak]),
        "orientation_degrees": float(np.degrees(np.arctan2(fy[peak], fx[peak]))),
        "n_strong_peaks": int(np.sum(masked > 2.0)),
    }


__all__ = [
    "copy_move_detection",
    "jpeg_grid_analysis",
    "noise_inconsistency",
    "noise_level_map",
    "periodic_artifact_analysis",
]

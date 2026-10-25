# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physically defined anomaly synthesis on normal images.

Each anomaly type in the analyser's taxonomy has a generative definition here:
an occluder is a flat fill, a defocus is a local Gaussian blur, sensor
corruption is additive noise, a stain is a chroma gain, foreign matter is a
texture that does not come from the domain, a scratch is a thin stroke, and an
exposure anomaly is a local luminance gain. Shapes (rotated ellipses, strokes),
sizes and strengths are randomised.

The analyser synthesises these on a domain's *own* normal images at fit time
and learns where each type falls in that domain's deviation space, so type
recognition adapts to the domain instead of relying on fixed thresholds (the
self-supervised synthetic-anomaly approach of CutPaste, Li et al., 2021, and
DRAEM, Zavrtanik et al., 2021). The same generator is exported for robustness
testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

Array = np.ndarray[Any, Any]

#: Synthesisable anomaly types, keyed by the analyser's taxonomy name.
SIMULATED_TYPES: tuple[str, ...] = (
    "occlusion_or_missing",
    "blur_or_defocus",
    "noise_or_sensor_corruption",
    "color_shift",
    "foreign_texture_or_object",
    "structural_change",
    "exposure_anomaly",
)


def _ellipse_mask(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    height, width = shape
    side = min(height, width)
    ry = rng.uniform(0.06, 0.16) * side
    rx = rng.uniform(0.06, 0.16) * side
    cy = rng.uniform(ry + 4, height - ry - 4)
    cx = rng.uniform(rx + 4, width - rx - 4)
    angle = rng.uniform(0, np.pi)
    yy, xx = np.indices(shape).astype(float)
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    return np.asarray((u / rx) ** 2 + (v / ry) ** 2 <= 1.0)


def _stroke_mask(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    height, width = shape
    side = min(height, width)
    mask = np.zeros(shape, dtype=bool)
    y, x = rng.uniform(0.2, 0.8) * height, rng.uniform(0.2, 0.8) * width
    angle = rng.uniform(0, np.pi)
    length = rng.uniform(0.25, 0.45) * side
    n_points = int(length * 3)
    curvature = rng.uniform(-1, 1) if rng.random() < 0.6 else 0.0
    for t in np.linspace(-0.5, 0.5, n_points):
        bend = 0.08 * side * np.sin(np.pi * t) * curvature
        py = int(np.clip(y + t * length * np.sin(angle) + bend * np.cos(angle), 0, height - 1))
        px = int(np.clip(x + t * length * np.cos(angle) - bend * np.sin(angle), 0, width - 1))
        mask[py, px] = True
    width_px = int(rng.integers(1, 3))
    return np.asarray(ndimage.binary_dilation(mask, iterations=width_px))


def _foreign_texture(shape: tuple[int, int, int], rng: np.random.Generator) -> Array:
    """A texture with statistics unrelated to the domain: random-scale speckle, random tint."""
    height, width, channels = shape
    scale = rng.uniform(0.6, 2.5)
    base = ndimage.gaussian_filter(rng.normal(size=(height, width)), scale)
    base = (base - base.mean()) / (base.std() + 1e-9)
    binary = rng.random() < 0.5
    texture = (base > 0).astype(float) if binary else 0.5 + 0.25 * base
    tint = rng.uniform(0.3, 1.0, channels)
    level = rng.uniform(0.2, 0.8)
    return np.asarray(
        np.clip(level + (texture - texture.mean())[..., np.newaxis] * 0.5 * tint, 0.0, 1.0)
    )


#: Types whose physical effect has two opposite polarities (darker / brighter);
#: their deviations are bimodal, so they are synthesised and modelled per polarity.
POLAR_TYPES: frozenset[str] = frozenset(
    {"occlusion_or_missing", "structural_change", "exposure_anomaly"}
)


def simulate_anomaly(
    pixels: Array,
    anomaly_type: str,
    rng: np.random.Generator,
    polarity: str | None = None,
) -> tuple[Array, Array]:
    """Synthesise one anomaly of ``anomaly_type`` on a normal image.

    Args:
        pixels: ``(H, W, C)`` float image in ``[0, 1]``.
        anomaly_type: One of :data:`SIMULATED_TYPES`.
        rng: Random generator (shape, size, strength).
        polarity: ``"dark"`` or ``"bright"`` for :data:`POLAR_TYPES`
            (random when omitted); ignored for other types.

    Returns:
        ``(anomalous_pixels, mask)``.

    Raises:
        ValueError: an unknown type or polarity.
    """
    if anomaly_type not in SIMULATED_TYPES:
        raise ValueError(f"unknown anomaly type {anomaly_type!r}; choose from {SIMULATED_TYPES}")
    if polarity not in (None, "dark", "bright"):
        raise ValueError(f"polarity must be 'dark', 'bright' or None; got {polarity!r}")
    if polarity is None:
        polarity = "dark" if rng.random() < 0.5 else "bright"
    darker = polarity == "dark"
    image = np.asarray(pixels, dtype=np.float64)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    shape = image.shape[:2]
    out = image.copy()
    mask = (
        _stroke_mask(shape, rng)
        if anomaly_type == "structural_change"
        else _ellipse_mask(shape, rng)
    )
    region = image[mask]
    local_mean = region.mean(axis=0)

    if anomaly_type == "occlusion_or_missing":
        level = rng.uniform(0.3, 0.7) if darker else rng.uniform(1.4, 1.9)
        fill = np.clip(local_mean * level, 0.0, 1.0)
        out[mask] = fill + rng.normal(0, 0.003, region.shape)
    elif anomaly_type == "blur_or_defocus":
        sigma = rng.uniform(2.0, 4.0)
        out[mask] = ndimage.gaussian_filter(image, (sigma, sigma, 0))[mask]
    elif anomaly_type == "noise_or_sensor_corruption":
        out[mask] = region + rng.normal(0, rng.uniform(0.06, 0.15), region.shape)
    elif anomaly_type == "color_shift":
        if image.shape[2] >= 3:
            direction = rng.normal(size=3)
            direction -= direction.mean()
            direction /= np.linalg.norm(direction) + 1e-9
            gains = 1.0 + rng.uniform(0.25, 0.45) * direction * np.sqrt(3.0)
            out[mask] = region * gains
        else:
            out[mask] = region * rng.uniform(0.7, 1.3)
    elif anomaly_type == "foreign_texture_or_object":
        out[mask] = _foreign_texture(image.shape, rng)[mask]
    elif anomaly_type == "structural_change":
        out[mask] = rng.uniform(0.02, 0.1) if darker else rng.uniform(0.9, 0.98)
    else:  # exposure_anomaly
        gain = rng.uniform(0.3, 0.6) if darker else rng.uniform(1.4, 1.9)
        out[mask] = region * gain

    out = np.clip(out, 0.0, 1.0)
    if np.asarray(pixels).ndim == 2:
        out = out[..., 0]
    return out, mask


__all__ = ["POLAR_TYPES", "SIMULATED_TYPES", "simulate_anomaly"]

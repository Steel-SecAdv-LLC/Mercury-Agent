# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonical image handling, capture-environment assessment, and controlled degradations.

Every analysis in :mod:`omni_mercury_engine.vision` starts from a
:class:`CanonicalImage`: any supported input (an image path, an 8- or 16-bit
array, a float radiometric array such as thermal imagery, grayscale, RGB, RGBA
or a multispectral cube) is mapped to float32 pixels in ``[0, 1]`` with a
recorded :class:`ImageProvenance` stating exactly which value mapping was
applied. Nothing is silently coerced: non-finite pixels are masked and
counted, alpha is separated and its coverage reported, and ambiguous layouts
are refused with the reason.

The capture environment (exposure, contrast, sharpness, sensor noise, colour
cast) is measured separately from image *content* by
:func:`assess_environment`, so an analyser can tell "this scene contains an
anomaly" apart from "this photo was taken in unusual conditions".
:func:`degrade` reproduces those conditions deterministically for robustness
evaluation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

Array = np.ndarray[Any, Any]

#: Smallest side length any analysis accepts; below this, patch statistics are meaningless.
MIN_SIDE = 32

_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Immerkær's Laplacian-difference kernel; applied to pure Gaussian noise of
#: standard deviation sigma its response has standard deviation 6 * sigma.
_IMMERKAER_KERNEL = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])


@dataclass(frozen=True)
class ImageProvenance:
    """How an input became a canonical image: every mapping applied, stated.

    Attributes:
        source: ``"path"`` or ``"array"``.
        source_dtype: The input's numpy dtype name.
        layout: ``"gray"``, ``"rgb"``, ``"rgba"`` or ``"multichannel"``.
        channels_in: Channel count before alpha separation.
        height: Image height in pixels.
        width: Image width in pixels.
        observed_min: Smallest finite input value.
        observed_max: Largest finite input value.
        scaling: The value mapping applied to reach ``[0, 1]``.
        nonfinite_pixels: Pixels whose input value was NaN or infinite.
        alpha_coverage: Fraction of pixels with alpha < 1 (RGBA only).
    """

    source: str
    source_dtype: str
    layout: str
    channels_in: int
    height: int
    width: int
    observed_min: float
    observed_max: float
    scaling: str
    nonfinite_pixels: int
    alpha_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the provenance fields."""
        return asdict(self)


@dataclass(frozen=True)
class CanonicalImage:
    """Float32 pixels in ``[0, 1]`` plus luminance, an invalid-pixel mask and provenance.

    Attributes:
        pixels: ``(H, W, C)`` float32 in ``[0, 1]`` (alpha removed).
        luminance: ``(H, W)`` float32 in ``[0, 1]`` — Rec. 709 luma for RGB,
            the single channel for grayscale, the channel mean otherwise.
        invalid_mask: ``(H, W)`` bool, True where the input was non-finite.
        provenance: How the input was mapped.
    """

    pixels: Array
    luminance: Array
    invalid_mask: Array
    provenance: ImageProvenance

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return int(self.luminance.shape[0]), int(self.luminance.shape[1])

    @property
    def is_color(self) -> bool:
        """Return True for RGB / RGBA inputs (colour features apply)."""
        return self.provenance.layout in ("rgb", "rgba")


def _load_path(path: str) -> Array:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv is a core dependency
        raise ValueError(
            f"image path {path!r} needs OpenCV to load; pass a pixel array instead"
        ) from exc
    loaded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if loaded is None:
        raise ValueError(f"image could not be loaded from path: {path!r}")
    array = np.asarray(loaded)
    if array.ndim == 3 and array.shape[2] == 3:
        array = array[..., ::-1]
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[..., [2, 1, 0, 3]]
    return array


def _to_channel_last(array: Array) -> tuple[Array, str]:
    if array.ndim == 2:
        return array[..., np.newaxis], "gray"
    if array.ndim != 3:
        raise ValueError(
            f"expected one image of shape (H, W) or (H, W, C); got ndim={array.ndim} "
            "(analyse batches one image at a time, e.g. VisualAnomalyAnalyzer.analyze_batch)"
        )
    channels = array.shape[2]
    if channels not in (1, 3, 4) and array.shape[0] in (1, 3, 4) and channels > 4:
        array = np.moveaxis(array, 0, -1)
        channels = array.shape[2]
    if channels == 1:
        return array, "gray"
    if channels == 3:
        return array, "rgb"
    if channels == 4:
        return array, "rgba"
    if 2 <= channels <= 16:
        return array, "multichannel"
    raise ValueError(
        f"ambiguous image layout {tuple(array.shape)}: channel-last with at most 16 "
        "channels, or channel-first with 1/3/4 channels, is required"
    )


def to_canonical(
    image: Any,
    *,
    bit_depth: int | None = None,
    channel_order: str = "rgb",
) -> CanonicalImage:
    """Map any supported image input to a :class:`CanonicalImage`.

    Args:
        image: An image path, a ``(H, W)`` / ``(H, W, C)`` / ``(C, H, W)``
            array of any numeric dtype, or an existing :class:`CanonicalImage`.
        bit_depth: True sensor bit depth for integer inputs (for example 12
            for a 12-bit camera stored in uint16); defaults to the dtype width.
        channel_order: ``"rgb"`` (default) or ``"bgr"`` for arrays that came
            from OpenCV. Paths are always loaded and reordered to RGB.

    Returns:
        The canonical image.

    Raises:
        ValueError: non-numeric data, an ambiguous layout, a side shorter than
            :data:`MIN_SIDE`, or an unloadable path — always naming the reason.
    """
    if isinstance(image, CanonicalImage):
        return image
    source = "array"
    if isinstance(image, (str, os.PathLike)):
        source = "path"
        array = _load_path(os.fspath(image))
    else:
        array = np.asarray(image)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"image must be numeric; got dtype {array.dtype}")
    source_dtype = str(array.dtype)
    array, layout = _to_channel_last(array)
    if channel_order == "bgr" and layout in ("rgb", "rgba") and source == "array":
        order = [2, 1, 0] if layout == "rgb" else [2, 1, 0, 3]
        array = array[..., order]
    elif channel_order not in ("rgb", "bgr"):
        raise ValueError(f"channel_order must be 'rgb' or 'bgr'; got {channel_order!r}")
    height, width, channels_in = (int(s) for s in array.shape)
    if min(height, width) < MIN_SIDE:
        raise ValueError(
            f"image is {height}x{width}; at least {MIN_SIDE} pixels per side are needed "
            "for patch statistics"
        )

    values = array.astype(np.float64)
    finite = np.isfinite(values)
    nonfinite_mask = ~np.all(finite, axis=2)
    finite_values = values[finite]
    observed_min = float(finite_values.min()) if finite_values.size else 0.0
    observed_max = float(finite_values.max()) if finite_values.size else 0.0

    kind = array.dtype.kind
    if kind == "b":
        scaled = values
        scaling = "boolean -> {0, 1}"
    elif kind == "u":
        bits = bit_depth if bit_depth is not None else array.dtype.itemsize * 8
        full_scale = float(2**bits - 1)
        scaled = values / full_scale
        scaling = f"unsigned {bits}-bit / {int(full_scale)}"
        if bit_depth is None and bits == 16 and observed_max < 4096:
            scaling += (
                f" (values occupy {observed_max / full_scale:.1%} of the 16-bit range; "
                "pass bit_depth for sensor-true scaling)"
            )
    elif observed_min >= 0.0 and observed_max <= 1.0:
        scaled = values
        scaling = "values already in [0, 1]"
    elif observed_max > observed_min:
        scaled = (values - observed_min) / (observed_max - observed_min)
        scaling = (
            f"per-image min-max of [{observed_min:.6g}, {observed_max:.6g}] "
            "(radiometric scale not preserved)"
        )
    else:
        scaled = np.zeros_like(values)
        scaling = "constant image -> 0"
    scaled = np.where(finite, scaled, 0.0)
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)

    alpha_coverage: float | None = None
    if layout == "rgba":
        alpha_coverage = float(np.mean(scaled[..., 3] < 1.0))
        pixels = np.ascontiguousarray(scaled[..., :3])
    else:
        pixels = np.ascontiguousarray(scaled)

    if layout in ("rgb", "rgba"):
        luminance = pixels @ _REC709
    elif layout == "gray":
        luminance = pixels[..., 0].copy()
    else:
        luminance = pixels.mean(axis=2)
    luminance = np.clip(luminance, 0.0, 1.0).astype(np.float32)

    provenance = ImageProvenance(
        source=source,
        source_dtype=source_dtype,
        layout=layout,
        channels_in=channels_in,
        height=height,
        width=width,
        observed_min=observed_min,
        observed_max=observed_max,
        scaling=scaling,
        nonfinite_pixels=int(nonfinite_mask.sum()),
        alpha_coverage=alpha_coverage,
    )
    return CanonicalImage(
        pixels=pixels,
        luminance=luminance,
        invalid_mask=nonfinite_mask,
        provenance=provenance,
    )


def immerkaer_residual(luminance: Array) -> Array:
    """Return the Laplacian-difference response used for noise estimation (Immerkær, 1996)."""
    return np.asarray(
        ndimage.convolve(luminance.astype(np.float64), _IMMERKAER_KERNEL, mode="reflect")
    )


def estimate_noise_sigma(luminance: Array, *, gradient_quantile: float = 0.6) -> float:
    """Robust sensor-noise standard deviation in luminance units.

    Immerkær's estimator, made texture-robust: the kernel response is read only
    at pixels whose gradient magnitude is at or below ``gradient_quantile`` of
    the image (edges and strong texture would otherwise read as noise), and the
    median absolute response is converted with the Gaussian factor
    ``sigma = median|r| / (6 * 0.6745)``.

    Args:
        luminance: ``(H, W)`` luminance in ``[0, 1]``.
        gradient_quantile: Fraction of lowest-gradient pixels used.

    Returns:
        The estimated noise standard deviation.
    """
    lum = luminance.astype(np.float64)
    residual = np.abs(immerkaer_residual(lum))[1:-1, 1:-1]
    gradient = np.hypot(ndimage.sobel(lum, axis=0), ndimage.sobel(lum, axis=1))[1:-1, 1:-1]
    cutoff = np.quantile(gradient, gradient_quantile)
    selected = residual[gradient <= cutoff]
    if selected.size == 0:
        selected = residual.ravel()
    return float(np.median(selected) / (6.0 * 0.6745))


#: Numeric environment fields, in the fixed order used for envelope modelling.
ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "mean_luminance",
    "rms_contrast",
    "log10_sharpness",
    "log10_noise_sigma",
    "clipped_dark_fraction",
    "clipped_bright_fraction",
    "color_cast",
)


@dataclass(frozen=True)
class EnvironmentReport:
    """Measured capture conditions of one image, independent of its content.

    Attributes:
        mean_luminance: Mean luminance in ``[0, 1]``.
        luminance_p01: 1st-percentile luminance.
        luminance_p99: 99th-percentile luminance.
        clipped_dark_fraction: Fraction of pixels at or below 1/255.
        clipped_bright_fraction: Fraction of pixels at or above 254/255.
        rms_contrast: Robust luminance spread (1.4826 x median absolute deviation).
        sharpness: Median per-tile variance of the luminance Laplacian (higher is sharper).
        noise_sigma: Texture-robust sensor-noise estimate (luminance units).
        color_cast: Largest relative deviation of a channel mean from the
            grey-world mean; ``None`` for non-colour images.
        conditions: Absolute, uncalibrated screen labels (for example
            ``"underexposed"``). Calibrated envelope checks against the normal
            set are done by the analyser, not here.
    """

    mean_luminance: float
    luminance_p01: float
    luminance_p99: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float
    rms_contrast: float
    sharpness: float
    noise_sigma: float
    color_cast: float | None
    conditions: tuple[str, ...]

    def vector(self) -> Array:
        """Return the numeric fields in :data:`ENVIRONMENT_FIELDS` order (log scales applied)."""
        return np.array(
            [
                self.mean_luminance,
                self.rms_contrast,
                np.log10(self.sharpness + 1e-12),
                np.log10(self.noise_sigma + 1e-6),
                self.clipped_dark_fraction,
                self.clipped_bright_fraction,
                0.0 if self.color_cast is None else self.color_cast,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the measurements."""
        data = asdict(self)
        data["conditions"] = list(self.conditions)
        return data


def _tile_medians(values: Array, tile: int, reducer: Any) -> float:
    """Median over ``tile`` x ``tile`` tiles of ``reducer(tile values)``."""
    height, width = values.shape
    stats = [
        float(reducer(values[y : y + tile, x : x + tile]))
        for y in range(0, height - tile + 1, tile)
        for x in range(0, width - tile + 1, tile)
    ]
    return float(np.median(stats)) if stats else float(reducer(values))


def assess_environment(image: CanonicalImage, tile: int = 32) -> EnvironmentReport:
    """Measure the capture conditions of ``image``, robustly to localised content.

    Conditions are properties of the whole capture, so each is estimated in a
    way a minority region (a defect, an occluder, a sticker) cannot move: mean
    luminance and sharpness are medians over ``tile``-pixel tiles, contrast is
    the scaled median absolute deviation of luminance, and noise uses the
    texture-robust median estimator. Clipping fractions remain whole-image
    fractions (a small clipped region moves them only by its area).

    Args:
        image: The canonical image.
        tile: Tile side for the per-tile medians.

    Returns:
        The environment report.
    """
    lum = image.luminance.astype(np.float64)
    tile = min(tile, *lum.shape)
    mean_lum = _tile_medians(lum, tile, np.mean)
    p01, p99 = (float(v) for v in np.quantile(lum, [0.01, 0.99]))
    dark = float(np.mean(lum <= 1.0 / 255.0))
    bright = float(np.mean(lum >= 254.0 / 255.0))
    contrast = float(np.median(np.abs(lum - np.median(lum))) * 1.4826)
    laplacian = ndimage.laplace(lum)
    sharpness = _tile_medians(laplacian, tile, np.var)
    noise = estimate_noise_sigma(lum)

    cast: float | None = None
    if image.is_color:
        channel_means = image.pixels.reshape(-1, image.pixels.shape[2]).mean(axis=0)
        grey = float(channel_means.mean())
        cast = float(np.max(np.abs(channel_means - grey)) / (grey + 1e-6))

    conditions: list[str] = []
    if mean_lum < 0.15 or dark > 0.3:
        conditions.append("underexposed")
    if mean_lum > 0.85 or bright > 0.3:
        conditions.append("overexposed")
    if contrast < 0.04:
        conditions.append("low_contrast")
    if noise > 0.05:
        conditions.append("noisy")
    if cast is not None and cast > 0.15:
        conditions.append("color_cast")

    return EnvironmentReport(
        mean_luminance=mean_lum,
        luminance_p01=p01,
        luminance_p99=p99,
        clipped_dark_fraction=dark,
        clipped_bright_fraction=bright,
        rms_contrast=contrast,
        sharpness=sharpness,
        noise_sigma=noise,
        color_cast=cast,
        conditions=tuple(conditions),
    )


def local_contrast_normalize(luminance: Array, sigma: float = 6.0, floor: float = 0.03) -> Array:
    """Return ``(I - local mean) / max(local std, floor)``.

    Removes smooth illumination (gain and gradients) so texture, gradient and
    frequency features describe structure rather than lighting. The ``floor``
    keeps genuinely flat regions flat instead of amplifying their noise.

    Args:
        luminance: ``(H, W)`` luminance.
        sigma: Gaussian scale of the local statistics, in pixels.
        floor: Minimum local standard deviation used as the divisor.

    Returns:
        The normalised luminance (roughly zero-mean, unit-variance where textured).
    """
    lum = luminance.astype(np.float64)
    mean = ndimage.gaussian_filter(lum, sigma, mode="reflect")
    var = ndimage.gaussian_filter(lum * lum, sigma, mode="reflect") - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return np.asarray((lum - mean) / np.maximum(std, floor))


def gray_world_balance(pixels: Array) -> Array:
    """Return ``pixels`` with grey-world white balance applied (colour images only)."""
    channel_means = pixels.reshape(-1, pixels.shape[2]).mean(axis=0)
    grey = float(channel_means.mean())
    gains = grey / np.maximum(channel_means, 1e-6)
    return np.asarray(np.clip(pixels * gains, 0.0, 1.0))


#: Supported capture-condition degradations for robustness evaluation.
DEGRADATIONS: tuple[str, ...] = (
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "jpeg",
    "downsample",
    "underexpose",
    "overexpose",
    "color_cast",
    "illumination_gradient",
)


def degrade(
    pixels: Array,
    kind: str,
    severity: float,
    rng: np.random.Generator | None = None,
) -> Array:
    """Apply a deterministic capture-condition degradation.

    Args:
        pixels: ``(H, W)`` or ``(H, W, C)`` float image in ``[0, 1]``.
        kind: One of :data:`DEGRADATIONS`.
        severity: ``0`` (none) to ``1`` (severe).
        rng: Generator for stochastic degradations (noise); seeded default.

    Returns:
        The degraded image, same shape, clipped to ``[0, 1]``.

    Raises:
        ValueError: an unknown kind or a severity outside ``[0, 1]``.
    """
    if kind not in DEGRADATIONS:
        raise ValueError(f"unknown degradation {kind!r}; choose from {DEGRADATIONS}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1]; got {severity}")
    image = np.asarray(pixels, dtype=np.float64)
    squeeze = image.ndim == 2
    if squeeze:
        image = image[..., np.newaxis]
    height, width = image.shape[:2]

    if kind == "gaussian_blur":
        sigma = 0.5 + 3.5 * severity
        out = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    elif kind == "motion_blur":
        length = 1 + round(15 * severity)
        kernel = np.full((1, length, 1), 1.0 / length)
        out = ndimage.convolve(image, kernel, mode="reflect")
    elif kind == "gaussian_noise":
        generator = rng if rng is not None else np.random.default_rng(0)
        out = image + generator.normal(0.0, 0.01 + 0.1 * severity, image.shape)
    elif kind == "jpeg":
        import cv2

        quality = round(95 - 85 * severity)
        as_bytes = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg",
            as_bytes if as_bytes.shape[2] != 1 else as_bytes[..., 0],
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:  # pragma: no cover - encoder failure
            raise ValueError("JPEG encoding failed")
        decoded_raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if decoded_raw is None:  # pragma: no cover - decoder failure
            raise ValueError("JPEG decoding failed")
        decoded = decoded_raw.astype(np.float64) / 255.0
        out = decoded.reshape(image.shape)
    elif kind == "downsample":
        factor = 1.0 + 3.0 * severity
        small = ndimage.zoom(image, (1.0 / factor, 1.0 / factor, 1), order=1)
        out = ndimage.zoom(
            small,
            (height / small.shape[0], width / small.shape[1], 1),
            order=1,
        )[:height, :width]
    elif kind == "underexpose":
        out = image * (1.0 - 0.7 * severity)
    elif kind == "overexpose":
        out = image * (1.0 + 1.5 * severity)
    elif kind == "color_cast":
        gains = np.ones(image.shape[2])
        if image.shape[2] >= 3:
            gains[0], gains[2] = 1.0 + 0.4 * severity, 1.0 - 0.4 * severity
        out = image * gains
    else:  # illumination_gradient
        ramp = np.linspace(1.0 - 0.6 * severity, 1.0, width)[np.newaxis, :, np.newaxis]
        out = image * ramp

    out = np.clip(out, 0.0, 1.0)
    return np.asarray(out[..., 0] if squeeze else out)


__all__ = [
    "DEGRADATIONS",
    "ENVIRONMENT_FIELDS",
    "MIN_SIDE",
    "CanonicalImage",
    "EnvironmentReport",
    "ImageProvenance",
    "assess_environment",
    "degrade",
    "estimate_noise_sigma",
    "gray_world_balance",
    "immerkaer_residual",
    "local_contrast_normalize",
    "to_canonical",
]

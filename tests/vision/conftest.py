# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic synthetic image domains with ground-truth defects for the vision tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from omni_mercury_engine.vision import AnalyzerConfig, VisualAnomalyAnalyzer

SIZE = 96


def woven_surface(seed: int, size: int = SIZE) -> np.ndarray[Any, Any]:
    """A woven-fabric-like RGB surface: periodic weave, fibre noise, gentle shading."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size)).astype(float)
    phase = rng.uniform(0, 2 * np.pi, 2)
    weave = 0.5 + 0.18 * np.sin(2 * np.pi * xx / 9 + phase[0]) * np.sin(
        2 * np.pi * yy / 9 + phase[1]
    )
    fibre = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.2) * 0.08
    shade = 1.0 + 0.05 * np.sin(2 * np.pi * (xx + yy) / (size * 2) + rng.uniform(0, 6))
    lum = np.clip((weave + fibre) * shade, 0, 1)
    tint = np.array([0.55, 0.45, 0.35]) + rng.normal(0, 0.01, 3)
    rgb = np.clip(lum[..., None] * tint[None, None, :] * 1.6, 0, 1)
    return np.asarray(np.clip(rgb + rng.normal(0, 0.004, rgb.shape), 0, 1))


def blob(size: int, seed: int, lo: int = 14, hi: int = 22) -> np.ndarray[Any, Any]:
    """A rectangular ground-truth mask placed away from the border."""
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
    y, x = int(rng.integers(8, size - h - 8)), int(rng.integers(8, size - w - 8))
    mask = np.zeros((size, size), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def inject(
    image: np.ndarray[Any, Any], kind: str, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Inject a ground-truth defect; returns ``(image, mask)``."""
    rng = np.random.default_rng(seed)
    mask = blob(image.shape[0], seed)
    out = image.copy()
    if kind == "occlusion":
        out[mask] = image[mask].mean(axis=0) * 0.5
    elif kind == "blur":
        out[mask] = ndimage.gaussian_filter(image, (3, 3, 0))[mask]
    elif kind == "noise":
        out[mask] = np.clip(image[mask] + rng.normal(0, 0.12, image[mask].shape), 0, 1)
    elif kind == "stain":
        out[mask] = np.clip(image[mask] * np.array([0.7, 1.05, 1.35]), 0, 1)
    elif kind == "dark":
        out[mask] = image[mask] * 0.45
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 1), mask


@pytest.fixture(scope="session")
def fitted_analyzer() -> VisualAnomalyAnalyzer:
    """One analyser fitted once on 40 normal woven surfaces (20 held out for calibration)."""
    normals = [woven_surface(s) for s in range(40)]
    config = AnalyzerConfig(alpha=0.2, type_simulations=8, calibration_fraction=0.5)
    return VisualAnomalyAnalyzer(config).fit(normals)

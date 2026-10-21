# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the wired ``quality_mask`` of the Gabor enhancer.

The per-block blend ``q * filtered + (1 - q) * original`` has two exact
endpoints: ``q == 1`` everywhere reproduces the unmasked enhancement, and
``q == 0`` everywhere reproduces the normalised input. A shape mismatch raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric.fingerprint_recognition import GaborEnhancer

# 16x16 image over a 4x4 orientation/frequency grid -> 4x4 blocks, full coverage.
_IMG = np.random.RandomState(7).rand(16, 16).astype(np.float64)
_ORIENT = np.zeros((4, 4))
_FREQ = np.full((4, 4), 0.1)


def _normalized(img: np.ndarray) -> np.ndarray:
    return np.asarray((img - img.min()) / (img.max() - img.min() + 1e-8))


def test_quality_one_equals_unmasked() -> None:
    enhancer = GaborEnhancer()
    unmasked = enhancer.enhance(_IMG, _ORIENT, _FREQ)
    full_quality = enhancer.enhance(_IMG, _ORIENT, _FREQ, quality_mask=np.ones((4, 4)))
    np.testing.assert_allclose(full_quality, unmasked, atol=1e-9)


def test_quality_zero_equals_normalized_input() -> None:
    enhancer = GaborEnhancer()
    zero_quality = enhancer.enhance(_IMG, _ORIENT, _FREQ, quality_mask=np.zeros((4, 4)))
    np.testing.assert_allclose(zero_quality, _normalized(_IMG), atol=1e-6)


def test_quality_mask_shape_mismatch_raises() -> None:
    enhancer = GaborEnhancer()
    with pytest.raises(ValueError, match="must match"):
        enhancer.enhance(_IMG, _ORIENT, _FREQ, quality_mask=np.ones((3, 3)))

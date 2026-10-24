# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``MatrixProfileConfig.normalize`` reaches STUMPY, and the provenance claims only what is done.

``normalize`` ("whether to z-normalize subsequences") was a declared field that
nothing read, so every profile was z-normalised whatever the caller set.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stumpy")

from omni_mercury_engine.models.foundation.matrix_profile import (
    MatrixProfileConfig,
    MatrixProfileDetector,
)


def _series() -> np.ndarray:
    rng = np.random.RandomState(0)
    values = np.sin(np.arange(300) / 5.0) + rng.normal(0.0, 0.1, 300)
    values[150:160] += 3.0
    return values


def _profile(normalize: bool) -> np.ndarray:
    detector = MatrixProfileDetector(
        MatrixProfileConfig(window_size=30, use_gpu=False, normalize=normalize)
    )
    return detector.compute_matrix_profile(_series())["matrix_profile"]


def test_normalize_false_changes_the_profile() -> None:
    assert not np.allclose(_profile(True), _profile(False))


def test_normalize_true_is_the_released_default_profile() -> None:
    import stumpy

    expected = stumpy.stump(_series(), m=30)[:, 0].astype(float)
    assert np.array_equal(_profile(True), expected)


def test_provenance_does_not_claim_a_z_screen_the_code_does_not_apply() -> None:
    assert "z >= 2.0" not in MatrixProfileDetector.decision_source.description

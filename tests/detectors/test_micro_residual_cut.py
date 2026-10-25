# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The micro-anomaly term is measured against a cut fitted on training rows.

Before this, the cut was the 95th percentile of the *scored* batch's own
residuals, so the rate was about 0.05 for any input and carried no information.
These tests pin the fitted cut, its persistence, and the abstain contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.safeguards.nano_safeguards import NanoSafeguardDetector


def _normal(seed: int, n: int = 200, d: int = 8) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).normal(size=(n, d)), dtype=np.float64)


def _off_manifold(seed: int, n: int = 60, d: int = 8) -> np.ndarray:
    """Rows with a large component outside the normal covariance structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    direction = rng.normal(size=(1, d))
    direction /= np.linalg.norm(direction)
    return np.asarray(
        base + 6.0 * direction * rng.choice([-1.0, 1.0], size=(n, 1)), dtype=np.float64
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_directive_micro_term_rises_off_manifold(seed: int) -> None:
    detector = SigmaDirectiveDetector()
    detector.fit(_normal(seed))
    normal = detector._dimensional_downsampling_detection(_normal(seed + 10, n=60))
    shifted = detector._dimensional_downsampling_detection(_off_manifold(seed + 10))
    assert normal is not None and shifted is not None
    assert shifted > normal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nano_micro_term_rises_off_manifold(seed: int) -> None:
    detector = NanoSafeguardDetector()
    detector.fit(_normal(seed))
    normal = detector._dimensional_downsampling_detection(_normal(seed + 10, n=60))
    shifted = detector._dimensional_downsampling_detection(_off_manifold(seed + 10))
    assert normal is not None and shifted is not None
    assert shifted > normal


def test_directive_unfitted_and_single_column_abstain() -> None:
    """Abstain (None), never 0.0: a 0.0 would be averaged into the blend."""
    detector = SigmaDirectiveDetector()
    assert detector._dimensional_downsampling_detection(_normal(0)) is None
    detector.fit(_normal(0))
    assert detector._dimensional_downsampling_detection(_normal(1, d=1)) is None


def test_nano_unfitted_abstains() -> None:
    assert NanoSafeguardDetector()._dimensional_downsampling_detection(_normal(0)) is None


def test_directive_micro_cut_survives_a_checkpoint_round_trip() -> None:
    fitted = SigmaDirectiveDetector()
    fitted.fit(_normal(3))
    state = fitted.get_fitted_state()
    assert state is not None and "micro_residual_cut" in state

    restored = SigmaDirectiveDetector()
    restored.set_fitted_state(state)
    assert restored._micro_residual_cut == fitted._micro_residual_cut
    probe = _off_manifold(4)
    assert restored._dimensional_downsampling_detection(
        probe
    ) == fitted._dimensional_downsampling_detection(probe)


def test_directive_checkpoint_without_the_cut_abstains_on_the_micro_term() -> None:
    fitted = SigmaDirectiveDetector()
    fitted.fit(_normal(5))
    state = fitted.get_fitted_state()
    assert state is not None
    state.pop("micro_residual_cut")

    legacy = SigmaDirectiveDetector()
    legacy.set_fitted_state(state)
    assert legacy._dimensional_downsampling_detection(_normal(6)) is None

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test for the 3R resonance (H(omega)) score.

``ThreeRMechanism.compute_dominance_score`` derives its resonance component from
the *magnitude* spectrum's energy concentration. ``compute_resonance_spectrum``
returns a ``(frequencies, magnitudes)`` tuple; a prior version passed the whole
tuple to ``np.abs(...).sort()``, which built a ``(2, N)`` array, folded the
frequency axis into the ratio, and made the ``len(spectrum) > 0`` guard a no-op
(``len`` was always 2). The score was therefore systematically wrong and no
longer tracked spectral energy concentration at all.

The concentration score must be near 1.0 for a spectrally concentrated signal (a
pure tone puts nearly all energy in a couple of bins) and markedly lower for a
spectrally flat signal (white noise spreads energy across all bins), which is
what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.three_r_mechanism import ThreeRMechanism


def _resonance_score(mechanism: ThreeRMechanism, data: np.ndarray) -> float:
    """Run the mechanism and return the raw resonance component it recorded."""
    mechanism.compute_dominance_score(data)
    return float(mechanism.last_resonance_score)


class TestResonanceConcentration:
    def test_concentrated_spectrum_scores_higher_than_flat(self) -> None:
        """A pure tone must yield a higher concentration score than white noise."""
        mechanism = ThreeRMechanism()
        t = np.linspace(0.0, 20.0 * np.pi, 256)
        tone = np.sin(t)
        noise = np.random.default_rng(0).standard_normal(256)

        concentrated = _resonance_score(mechanism, tone)
        flat = _resonance_score(mechanism, noise)

        assert concentrated > flat
        # A single-tone spectrum concentrates almost all energy in its top bins.
        assert concentrated > 0.75
        # White noise spreads energy: the top quartile of bins holds well under
        # the whole signal's energy.
        assert flat < 0.65

    def test_resonance_score_is_bounded_and_finite(self) -> None:
        """The resonance component stays a finite probability in [0, 1]."""
        mechanism = ThreeRMechanism()
        for seed in range(3):
            data = np.random.default_rng(seed).standard_normal(200)
            score = _resonance_score(mechanism, data)
            assert np.isfinite(score)
            assert 0.0 <= score <= 1.0

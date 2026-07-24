# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test for the STFT hop guard in the spectral vibration detector.

For a signal longer than ``fft_size`` the power-spectrum helper frames the
signal with ``hop = fft_size * (1 - overlap_ratio)``. An ``overlap_ratio`` of
1.0 (or more) drove ``hop`` to 0 and raised ``ZeroDivisionError`` on the frame
count; the hop is now clamped to at least one sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.spectral_vibration import SpectralVibrationDetector


@pytest.mark.parametrize("overlap_ratio", [1.0, 1.5])
def test_full_overlap_does_not_divide_by_zero(overlap_ratio: float) -> None:
    detector = SpectralVibrationDetector(config={"overlap_ratio": overlap_ratio, "fft_size": 256})
    signal = np.sin(np.linspace(0.0, 50.0 * np.pi, 1024))

    spectrum = detector._compute_power_spectrum(signal)

    assert spectrum.shape[0] == 128  # fft_size // 2 one-sided bins
    assert np.all(np.isfinite(spectrum))

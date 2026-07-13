# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression guard for ``SchumannResonanceDetector._compute_power_spectrum``.

A degenerate ELF window (all-zero, or shorter than two samples) has a spectral
peak of ``0``; an unguarded ``power / np.max(power)`` then emits a
``RuntimeWarning: invalid value encountered in divide`` and propagates
``NaN``/``Inf`` into the downstream fundamental-frequency peak search. The guard
leaves such a spectrum at zero (a truthful "no resonance power" result). These
tests pin both halves of the contract: the degenerate case is finite, and a
real signal is still peak-normalized to 1.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector


def test_power_spectrum_zero_window_is_finite_and_warning_free() -> None:
    """An all-zero ELF window must not produce NaN/Inf (or a divide warning)."""
    det = SchumannResonanceDetector(sampling_rate=100.0)
    zero = np.zeros(256, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        power, xf = det._compute_power_spectrum(zero)
    assert power.shape == xf.shape
    assert np.all(np.isfinite(power)), "degenerate ELF window must not yield NaN/Inf power"
    assert float(np.max(power)) == 0.0


def test_power_spectrum_real_signal_stays_peak_normalized() -> None:
    """A real Schumann-fundamental tone keeps the max-normalized-to-1.0 contract."""
    det = SchumannResonanceDetector(sampling_rate=100.0)
    t = np.arange(1024) / 100.0
    signal = np.sin(2.0 * np.pi * 7.83 * t)
    power, xf = det._compute_power_spectrum(signal)
    assert np.all(np.isfinite(power))
    assert np.isclose(float(np.max(power)), 1.0), "real-signal spectrum must remain peak-normalized"

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the wired ``expected_period`` harmonic fit.

A pure sinusoid of known amplitude/phase/period must be recovered by the
closed-form least-squares fit (``r2 ~= 1``); an off-frequency fit must explain
little variance; and an out-of-range period must raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.ocean.oceanography_patterns import OceanographyPatterns


def test_harmonic_fit_recovers_known_sinusoid() -> None:
    n, period, amplitude, phase = 200, 20.0, 3.0, 0.5
    t = np.arange(n, dtype=float)
    signal = amplitude * np.cos(2.0 * np.pi * t / period - phase)

    out = OceanographyPatterns().tidal_pattern_detection(signal, expected_period=period)

    assert abs(out["harmonic_amplitude"] - amplitude) < 1e-6
    assert abs(out["harmonic_phase_rad"] - phase) < 1e-6
    assert out["harmonic_r2"] > 0.999  # pure constituent explains ~all variance


def test_off_frequency_fit_explains_little_variance() -> None:
    n, period = 200, 20.0
    t = np.arange(n, dtype=float)
    signal = 3.0 * np.cos(2.0 * np.pi * t / period)

    out = OceanographyPatterns().tidal_pattern_detection(signal, expected_period=7.0)
    assert out["harmonic_r2"] < 0.2  # wrong constituent fits poorly


def test_expected_period_out_of_range_raises() -> None:
    signal = np.random.RandomState(0).randn(100)
    patterns = OceanographyPatterns()
    with pytest.raises(ValueError, match="outside"):
        patterns.tidal_pattern_detection(signal, expected_period=1.5)  # <= 2
    with pytest.raises(ValueError, match="outside"):
        patterns.tidal_pattern_detection(signal, expected_period=80.0)  # > n/2


def test_without_expected_period_has_no_harmonic_keys() -> None:
    signal = 3.0 * np.cos(2.0 * np.pi * np.arange(200) / 20.0)
    out = OceanographyPatterns().tidal_pattern_detection(signal)
    assert "harmonic_amplitude" not in out
    assert "detected_period" in out

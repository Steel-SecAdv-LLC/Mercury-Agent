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


@pytest.mark.parametrize("n", [0, 1])
def test_windows_shorter_than_two_samples_are_finite_end_to_end(n: int) -> None:
    """Empty and single-sample windows must not crash or emit NaN anywhere.

    Regression, two distinct failure modes: ``n == 0`` reached
    ``scipy.fft.fft`` (which raises on empty input) before any guard, and
    ``n == 1`` was squeezed to a 0-d scalar and misclassified as off-modality
    input. Both now yield the truthful empty spectrum, and the public
    detection/feature paths take their documented no-data fallbacks.
    """
    det = SchumannResonanceDetector(sampling_rate=100.0)
    window = np.zeros(n, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        power, xf = det._compute_power_spectrum(window)
        result = det.detect_resonance_anomaly(window)
        features = det.extract_features(window)
    assert power.shape == (0,) and xf.shape == (0,)
    assert result.anomaly_detected is False
    assert result.confidence == 0.0
    assert bool(np.isfinite(features.numpy()).all())


def test_power_spectrum_real_signal_stays_peak_normalized() -> None:
    """A real Schumann-fundamental tone keeps the max-normalized-to-1.0 contract."""
    det = SchumannResonanceDetector(sampling_rate=100.0)
    t = np.arange(1024) / 100.0
    signal = np.sin(2.0 * np.pi * 7.83 * t)
    power, xf = det._compute_power_spectrum(signal)
    assert np.all(np.isfinite(power))
    assert np.isclose(float(np.max(power)), 1.0), "real-signal spectrum must remain peak-normalized"


def test_multichannel_input_declines_with_clear_message() -> None:
    """A 2-D tabular window is off-modality: decline loudly, never index-crash.

    Regression: an unvalidated 2-D input sent the FFT along the wrong axis and
    the fundamental/harmonic peak searches then indexed the 1-D frequency array
    with a flattened-2-D ``argmax`` (``index 40 is out of bounds for axis 0
    with size 9``).
    """
    det = SchumannResonanceDetector(sampling_rate=100.0)
    window = np.random.default_rng(42).normal(size=(200, 8))
    with pytest.raises(ValueError, match="1-D ELF time series"):
        det.extract_features(window)
    with pytest.raises(ValueError, match="1-D ELF time series"):
        det.detect_resonance_anomaly(window)


def test_column_vector_input_matches_flat_input() -> None:
    """Trivially-squeezable shapes like ``(n, 1)`` are accepted as one channel."""
    det = SchumannResonanceDetector(sampling_rate=100.0)
    t = np.arange(1024) / 100.0
    signal = np.sin(2.0 * np.pi * 7.83 * t)
    power_flat, xf_flat = det._compute_power_spectrum(signal)
    power_col, xf_col = det._compute_power_spectrum(signal.reshape(-1, 1))
    np.testing.assert_allclose(power_col, power_flat)
    np.testing.assert_allclose(xf_col, xf_flat)


def test_short_temporal_history_windows_are_zero_padded() -> None:
    """Temporal-history spectra shorter than the 103-bin layout must zero-pad."""
    det = SchumannResonanceDetector(sampling_rate=100.0)
    short_history = [np.sin(2.0 * np.pi * 7.83 * np.arange(64) / 100.0) for _ in range(3)]
    tensor = det._process_temporal_history(short_history)
    assert tuple(tensor.shape) == (1, 3, 103)
    assert bool(np.isfinite(tensor.numpy()).all())

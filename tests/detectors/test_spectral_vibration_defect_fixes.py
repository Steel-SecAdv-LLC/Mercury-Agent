# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Peak-selection and utility-function defects in the vibration detector.

* M3: the strongest spectral peak must become the fundamental even when more
  than ``num_harmonics`` weaker peaks sit below it in frequency. The old code
  truncated the frequency-ordered peak list first, dropping strong
  high-frequency peaks before sorting by amplitude.
* STFT crashed on a signal shorter than one window (negative dimension) and
  mislabeled bin/sample indices as frequencies/times.
* The Haar wavelet loop dropped the final sample on odd-length input.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.detectors.spectral_vibration import (
    SpectralVibrationDetector,
    compute_short_time_fourier_transform,
    compute_wavelet_decomposition,
)

FS = 1000.0
FFT = 1024


def test_strongest_peak_is_the_fundamental_even_at_high_frequency() -> None:
    """A strong high-frequency tone must be the fundamental, not dropped.

    Fails today: >num_harmonics weaker low-frequency peaks are kept (lowest
    frequency first) and the strong high-frequency peak is discarded before
    the amplitude sort, so ``dominant_frequencies[0]`` is a weak low tone.
    """
    t = np.arange(FFT) / FS
    signal = 0.02 * np.random.default_rng(0).standard_normal(FFT)
    # Ten weak low-frequency tones (10..100 Hz), all well above 10% of the max.
    for f in range(10, 101, 10):
        signal += 1.0 * np.sin(2.0 * np.pi * f * t)
    # One stronger tone at a high frequency -- the true fundamental.
    signal += 1.6 * np.sin(2.0 * np.pi * 400.0 * t)

    det = SpectralVibrationDetector({"sample_rate": FS, "fft_size": FFT})
    feats = det._extract_spectral_features(signal)
    fundamental_hz = feats.dominant_frequencies[0][0]
    assert (
        abs(fundamental_hz - 400.0) < 3.0
    ), f"fundamental {fundamental_hz:.1f} Hz is not the strongest (400 Hz) peak"


def test_stft_short_signal_does_not_crash() -> None:
    """A signal shorter than one window must not raise a negative-dimension error."""
    freqs, times, spec = compute_short_time_fourier_transform(
        np.random.default_rng(1).standard_normal(10), window_size=64, hop_size=16
    )
    assert len(freqs) == 32  # window_size // 2
    assert spec.shape[0] == 32
    assert spec.shape[1] >= 1
    assert np.isfinite(spec).all()


def test_stft_axes_are_physical_units() -> None:
    """Frequencies must be Hz and times seconds, not bin/sample indices."""
    freqs, times, _ = compute_short_time_fourier_transform(
        np.random.default_rng(2).standard_normal(512),
        window_size=128,
        hop_size=32,
        sample_rate=256.0,
    )
    # Bin spacing is fs/window_size Hz, not 1 (a bin index step).
    assert np.isclose(freqs[1] - freqs[0], 256.0 / 128.0)
    assert freqs[-1] < 128.0 + 1e-9  # strictly below Nyquist in Hz
    # Frame spacing is hop/fs seconds, not hop samples.
    assert np.isclose(times[1] - times[0], 32.0 / 256.0)


def test_wavelet_decomposition_keeps_last_sample_on_odd_length() -> None:
    """Odd-length input must not silently drop its final sample.

    Fails today: ``n_half = n // 2`` ignores ``current[n-1]`` on odd lengths,
    so a 7-sample signal yields 3 approx coeffs instead of 4.
    """
    coeffs = compute_wavelet_decomposition(np.arange(7.0), levels=1)
    assert len(coeffs) == 2  # approx + 1 detail level
    approx = coeffs[0]
    assert len(approx) == 4, f"odd-length final sample dropped: {len(approx)} approx coeffs, not 4"

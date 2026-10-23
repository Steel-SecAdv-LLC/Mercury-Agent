# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The spectral vibration detector's estimator and its random projections.

Two defects, each measured before it was fixed:

* a signal one sample longer than ``fft_size`` used a Hann-windowed Welch
  average while one at or under it used an unwindowed single FFT, so a
  detector fitted on 1,024-sample signals scored every normal 1,025-sample
  signal 1.000 (flagged); and
* the untrained networks drew their weights from the process-global RNG, so
  the same data scored 0.3548, 0.3689, 0.3491 and 0.3400 on four fresh
  detectors, against the module's claim of a reproducible transform.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.fft import fft

from omni_mercury_engine.detectors.spectral_vibration import SpectralVibrationDetector

FS = 1000.0


def _normal(rng: np.random.Generator, n: int) -> np.ndarray:
    """A 100 Hz machine tone in sensor noise."""
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(n)


def _fitted(rng: np.random.Generator) -> SpectralVibrationDetector:
    batch = np.stack([_normal(rng, 1024) for _ in range(16)])
    return SpectralVibrationDetector({"sample_rate": FS}).fit(batch)


@pytest.mark.parametrize("length", [1025, 1100, 2048, 4096])
def test_normal_signals_longer_than_fft_size_are_not_flagged(length: int) -> None:
    """Before: every one of these scored 1.0, as an anomaly."""
    detector = _fitted(np.random.default_rng(0))
    rng = np.random.default_rng(1)
    at_fft_size = [detector.detect(_normal(rng, 1024))["anomaly_score"] for _ in range(10)]
    longer = [detector.detect(_normal(rng, length))["anomaly_score"] for _ in range(10)]

    assert max(longer) < 0.5
    assert abs(np.mean(longer) - np.mean(at_fft_size)) < 0.1


def test_a_full_frame_is_one_hann_windowed_frame() -> None:
    """At ``fft_size`` samples the estimate is exactly the first Welch frame."""
    detector = SpectralVibrationDetector({"sample_rate": FS})
    x = np.random.default_rng(2).standard_normal(1024)
    expected = np.abs(fft(x * np.hanning(1024))[:512]) ** 2
    np.testing.assert_allclose(detector._compute_power_spectrum(x), expected)


@pytest.mark.parametrize("length", [64, 200, 512, 1000, 1025, 2048, 8192])
def test_broadband_level_is_the_same_at_every_length(length: int) -> None:
    """White noise has one level whatever the length (within sampling noise)."""
    detector = SpectralVibrationDetector({"sample_rate": FS})
    rng = np.random.default_rng(3)
    reference = np.mean(
        [detector._compute_power_spectrum(rng.standard_normal(16384)).mean() for _ in range(3)]
    )
    level = np.mean(
        [detector._compute_power_spectrum(rng.standard_normal(length)).mean() for _ in range(40)]
    )
    assert level / reference == pytest.approx(1.0, abs=0.1)


def test_an_empty_signal_has_an_empty_spectrum() -> None:
    detector = SpectralVibrationDetector({"sample_rate": FS})
    assert not detector._compute_power_spectrum(np.zeros(0)).any()


def test_fresh_instances_score_identically() -> None:
    """Before: 0.3548, 0.3689, 0.3491, 0.3400 on the same data."""
    rng = np.random.default_rng(0)
    batch = np.stack([_normal(rng, 1024) for _ in range(16)])
    probe = _normal(rng, 1024)
    scores = [
        SpectralVibrationDetector({"sample_rate": FS}).fit(batch).detect(probe)["anomaly_score"]
        for _ in range(4)
    ]
    assert len(set(scores)) == 1


def test_repeated_detection_is_bit_identical() -> None:
    """Before: ~1e-9 apart, from ARPACK's random start vector in the graph spectrum."""
    rng = np.random.default_rng(0)
    detector = _fitted(rng)
    probe = _normal(rng, 1024)
    scores = [detector.detect(probe)["scores"][0] for _ in range(6)]
    assert len(set(scores)) == 1


def test_the_network_seed_is_configurable_and_matters() -> None:
    a = SpectralVibrationDetector({"network_seed": 1})
    b = SpectralVibrationDetector({"network_seed": 2})
    same = SpectralVibrationDetector({"network_seed": 1})
    first = next(a._cnn.parameters())
    assert torch.equal(first, next(same._cnn.parameters()))
    assert not torch.equal(first, next(b._cnn.parameters()))


def test_building_a_detector_leaves_the_global_rng_untouched() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    SpectralVibrationDetector({"sample_rate": FS})
    assert torch.equal(torch.rand(3), expected)


def _hann_power(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fft(x * np.hanning(len(x)))[: len(x) // 2]) ** 2


class TestDescriptors:
    """Spectral entropy and total harmonic distortion, against known values."""

    def test_total_harmonic_distortion_matches_the_amplitude_ratio(self) -> None:
        from omni_mercury_engine.detectors.spectral_vibration import total_harmonic_distortion

        t = np.arange(4096) / FS
        f0 = 125.0  # bin-aligned: bin 512
        x = (
            np.sin(2 * np.pi * f0 * t)
            + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.05 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        # IEEE 519: sqrt(0.1^2 + 0.05^2) of the fundamental.
        assert total_harmonic_distortion(_hann_power(x)) == pytest.approx(
            np.sqrt(0.1**2 + 0.05**2), rel=0.01
        )
        pure = _hann_power(np.sin(2 * np.pi * f0 * t))
        assert total_harmonic_distortion(pure) == pytest.approx(0.0, abs=1e-3)

    def test_spectral_entropy_orders_tone_below_noise(self) -> None:
        from omni_mercury_engine.detectors.spectral_vibration import spectral_entropy

        t = np.arange(4096) / FS
        noise = np.mean(
            [_hann_power(np.random.default_rng(s).standard_normal(4096)) for s in range(50)], axis=0
        )
        assert spectral_entropy(_hann_power(np.sin(2 * np.pi * 125.0 * t))) < 0.2
        assert spectral_entropy(noise) > 0.99

    @pytest.mark.parametrize("spectrum", [np.zeros(0), np.zeros(10), np.array([1.0])])
    def test_degenerate_spectra_give_zero(self, spectrum: np.ndarray) -> None:
        from omni_mercury_engine.detectors.spectral_vibration import (
            spectral_entropy,
            total_harmonic_distortion,
        )

        assert spectral_entropy(spectrum) == 0.0
        assert total_harmonic_distortion(spectrum) == 0.0

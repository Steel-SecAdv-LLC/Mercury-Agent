# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the wired ``signal_characteristics`` conditioning of contact-protocol exploration.

The fitness weights are a hand-computable function of SNR / bandwidth /
periodicity, and a noisier channel (lower SNR) favours a more
error-correcting top protocol.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from omni_mercury_engine.emergent.emergent_life_detector import (
    MultiverseContactProtocolExplorer,
    _clip01,
    _derive_signal_characteristics,
)

_RNG = np.random.RandomState(0)
_T = np.arange(512, dtype=float)
_SINE = 3.0 * np.sin(2.0 * np.pi * _T / 16.0) + 0.01 * _RNG.randn(512)  # clean, narrow, periodic
_NOISE = np.random.RandomState(1).randn(512)  # broadband, aperiodic


def _must_measure(signal: np.ndarray) -> dict[str, float]:
    measured = _derive_signal_characteristics(signal)
    assert measured is not None  # the signal under test is measurable
    return measured


def test_clip01_hand_values() -> None:
    assert _clip01({}, "snr") == 0.5  # missing -> neutral default
    assert _clip01({"snr": 0.3}, "snr") == 0.3
    assert _clip01({"snr": 5.0}, "snr") == 1.0  # clipped high
    assert _clip01({"snr": -2.0}, "snr") == 0.0  # clipped low


def _explore(chars: dict[str, Any]) -> dict[str, Any]:
    return MultiverseContactProtocolExplorer(num_universes=30).explore_contact_protocols(chars)


def test_weights_track_signal_characteristics() -> None:
    out = _explore({"snr": 0.25, "bandwidth": 0.75, "periodicity": 0.5})
    cond = out["signal_conditioning"]
    assert abs(cond["ecc_weight"] - (0.5 + (1.0 - 0.25))) < 1e-9  # 1.25
    assert abs(cond["info_weight"] - (0.5 + 0.75)) < 1e-9  # 1.25
    assert abs(cond["universality_weight"] - (0.5 + 0.5)) < 1e-9  # 1.0


def test_noisier_channel_prefers_more_error_correction() -> None:
    quiet = _explore({"snr": 1.0, "bandwidth": 0.5, "periodicity": 0.5})
    noisy = _explore({"snr": 0.0, "bandwidth": 0.5, "periodicity": 0.5})
    assert noisy["signal_conditioning"]["ecc_weight"] > quiet["signal_conditioning"]["ecc_weight"]
    quiet_ecc = quiet["optimal_protocols"][0]["characteristics"]["error_resilience"]
    noisy_ecc = noisy["optimal_protocols"][0]["characteristics"]["error_resilience"]
    assert noisy_ecc >= quiet_ecc


def test_wider_band_prefers_more_throughput() -> None:
    narrow = _explore({"snr": 0.5, "bandwidth": 0.0, "periodicity": 0.5})
    wide = _explore({"snr": 0.5, "bandwidth": 1.0, "periodicity": 0.5})
    assert wide["signal_conditioning"]["info_weight"] > narrow["signal_conditioning"]["info_weight"]
    narrow_info = narrow["optimal_protocols"][0]["characteristics"]["information_density"]
    wide_info = wide["optimal_protocols"][0]["characteristics"]["information_density"]
    assert wide_info >= narrow_info


def test_derived_characteristics_separate_tone_from_noise() -> None:
    tone = _must_measure(_SINE)
    noise = _must_measure(_NOISE)
    # A clean tone: strong spectral peak, almost no occupied bandwidth,
    # near-unit autocorrelation peak. White noise: the opposite on all three.
    assert tone["snr"] > 0.9 > noise["snr"]
    assert tone["bandwidth"] < 0.05 < noise["bandwidth"]
    assert tone["periodicity"] > 0.9 > noise["periodicity"]


def test_derived_characteristics_degenerate_inputs_are_unmeasurable() -> None:
    # Too short, constant, and non-finite signals cannot be measured; None
    # (not a placeholder record) lets callers keep what they already know.
    for bad in (np.array([1.0, 2.0]), np.full(64, 7.0), np.array([np.nan] * 32)):
        assert _derive_signal_characteristics(bad) is None


def test_degenerate_signal_preserves_caller_measurements() -> None:
    out = MultiverseContactProtocolExplorer(num_universes=30).explore_contact_protocols(
        {"snr": 0.95, "bandwidth": 0.2}, signal_data=np.full(64, 7.0)
    )
    conditioning = out["signal_conditioning"]
    assert conditioning["derived_from_signal"] is False
    assert conditioning["signal_unmeasurable"] is True
    # The caller's genuine measurements drive the weights, not 0.5 placeholders.
    assert conditioning["characteristics"]["snr"] == 0.95
    assert conditioning["characteristics"]["bandwidth"] == 0.2


def test_signal_data_drives_conditioning_end_to_end() -> None:
    clean = MultiverseContactProtocolExplorer(num_universes=30).explore_contact_protocols(
        signal_data=_SINE
    )
    noisy = MultiverseContactProtocolExplorer(num_universes=30).explore_contact_protocols(
        signal_data=_NOISE
    )
    assert clean["signal_conditioning"]["derived_from_signal"] is True
    assert noisy["signal_conditioning"]["derived_from_signal"] is True
    # The measured SNR difference must reach the fitness weights: the noisy
    # channel weights error correction harder than the clean one.
    assert noisy["signal_conditioning"]["ecc_weight"] > clean["signal_conditioning"]["ecc_weight"]
    # And the conditioning actually used is reported for audit.
    assert clean["signal_conditioning"]["characteristics"]["snr"] > 0.9


def test_measured_signal_overrides_caller_summary() -> None:
    # A caller claiming the channel is noisy is overridden by a measured clean tone.
    out = MultiverseContactProtocolExplorer(num_universes=30).explore_contact_protocols(
        {"snr": 0.0}, signal_data=_SINE
    )
    assert out["signal_conditioning"]["characteristics"]["snr"] > 0.9


def test_measured_snr_decreases_monotonically_with_noise() -> None:
    t = np.arange(512, dtype=float)
    carrier = 3.0 * np.sin(2.0 * np.pi * t / 16.0)
    noise = np.random.RandomState(2).randn(512)
    snrs = [
        _must_measure(carrier + amplitude * noise)["snr"]
        for amplitude in (0.0, 0.3, 1.0, 3.0, 10.0)
    ]
    # More injected noise can never read as a cleaner channel.
    assert all(earlier >= later for earlier, later in itertools.pairwise(snrs))
    assert snrs[0] == 1.0  # noiseless tone saturates the 30 dB scale
    assert snrs[-1] < 0.5  # noise-dominated signal reads as noisy


def test_shortest_valid_signal_is_measured_not_defaulted() -> None:
    # Period-4 tone: the shortest representable strong tone. (An exact-Nyquist
    # alternation would be invisible here: the shared spectrum helper keeps
    # only frequencies >= 0 and the fftfreq convention places the Nyquist bin
    # at a negative frequency — a documented edge of the module convention.)
    eight = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    out = _must_measure(eight)
    assert out["snr"] > 0.5  # a representable pure tone reads as clean


def test_fft_autocorrelation_equals_the_direct_form() -> None:
    # The O(n log n) Wiener-Khinchin autocorrelation must be the same
    # measurement as the O(n^2) direct form it replaced, to floating point.
    x = np.random.RandomState(0).randn(1024)
    centered = x - x.mean()
    energy = float(centered @ centered)
    direct = np.correlate(centered, centered, mode="full")[centered.size - 1 :] / energy
    padded = np.zeros(2048)
    padded[:1024] = centered
    fast = np.fft.irfft(np.abs(np.fft.rfft(padded)) ** 2)[:1024] / energy
    assert float(np.max(np.abs(direct - fast))) < 1e-12


def test_long_capture_is_tractable_and_correctly_characterized() -> None:
    # A one-million-sample receiver capture (the module's stated input class)
    # must be measurable, not a quadratic-time hang.
    t = np.arange(1_000_000, dtype=float)
    signal = np.sin(2.0 * np.pi * t / 500.0) + 0.1 * np.random.RandomState(1).randn(t.size)
    out = _must_measure(signal)
    assert out["snr"] > 0.9
    assert out["bandwidth"] < 0.01
    assert out["periodicity"] > 0.9

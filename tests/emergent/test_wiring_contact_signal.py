# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``signal_characteristics`` conditioning of contact-protocol exploration.

The fitness weights are a hand-computable function of SNR / bandwidth /
periodicity, and a noisier channel (lower SNR) favours a more
error-correcting top protocol.
"""

from __future__ import annotations

from omni_mercury_engine.emergent.emergent_life_detector import (
    MultiverseContactProtocolExplorer,
    _clip01,
)


def test_clip01_hand_values() -> None:
    assert _clip01({}, "snr") == 0.5  # missing -> neutral default
    assert _clip01({"snr": 0.3}, "snr") == 0.3
    assert _clip01({"snr": 5.0}, "snr") == 1.0  # clipped high
    assert _clip01({"snr": -2.0}, "snr") == 0.0  # clipped low


def _explore(chars: dict) -> dict:
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

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioral tests for the wired ``input_length`` / ``num_leads`` contract.

The analyzer resamples any time window to its contracted ``input_length`` (so
the pooled timebase is consistent regardless of the upstream window) while the
lead count is structural and must match ``num_leads`` exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
    ECGRhythmAnalyzer,
)


def test_matching_shape_forwards_and_classifies() -> None:
    net = ECGRhythmAnalyzer(input_length=256, num_leads=12, num_classes=13)
    net.eval()
    with torch.no_grad():
        classification, attention = net(torch.randn(2, 12, 256))
    assert classification.shape == (2, 13)
    assert attention.shape[0] == 2


def test_too_short_input_length_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="< 8"):
        ECGRhythmAnalyzer(input_length=4)


def test_varying_sample_count_is_resampled_to_window() -> None:
    net = ECGRhythmAnalyzer(input_length=256, num_leads=12, num_classes=13)
    net.eval()
    with torch.no_grad():
        for samples in (128, 256, 512):
            classification, _ = net(torch.randn(2, 12, samples))
            assert classification.shape == (2, 13)  # any window -> contracted length


def test_wrong_lead_count_raises_naming_shape() -> None:
    net = ECGRhythmAnalyzer(input_length=256, num_leads=12)
    with pytest.raises(ValueError, match="8 leads"):
        net(torch.randn(1, 8, 256))


def test_same_waveform_at_different_sampling_rates_classifies_consistently() -> None:
    torch.manual_seed(0)
    net = ECGRhythmAnalyzer(input_length=256, num_leads=12, num_classes=13)
    net.eval()

    def waveform(n_samples: int) -> np.ndarray:
        # One continuous 12-lead waveform sampled at two different rates.
        t = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        leads = [
            np.sin(2 * np.pi * (3 + k) * t) + 0.3 * np.sin(2 * np.pi * (7 + k) * t)
            for k in range(12)
        ]
        return np.asarray(np.stack(leads)[None, ...], dtype=np.float32)

    with torch.no_grad():
        at_contract, _ = net(torch.from_numpy(waveform(256)))
        oversampled, _ = net(torch.from_numpy(waveform(512)))

    # The resample-to-contract normalization makes the net see the same
    # physiology either way (measured headroom: actual gap ~5e-5).
    assert float((at_contract - oversampled).abs().max()) < 1e-2
    assert int(at_contract.argmax()) == int(oversampled.argmax())

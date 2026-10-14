# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the wired ``input_length`` / ``num_leads`` contract.

The analyzer resamples any time window to its contracted ``input_length`` (so
the pooled timebase is consistent regardless of the upstream window) while the
lead count is structural and must match ``num_leads`` exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

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

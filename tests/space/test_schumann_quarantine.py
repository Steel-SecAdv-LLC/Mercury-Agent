# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Anti-theater tests for the Schumann resonance detector: the untrained CNN-LSTM must not drive anomaly_type/confidence/risk_score. With random weights the detector falls back to the deterministic FFT-physics assessment, so identical input yields identical output across freshly constructed detectors (it did NOT before the quarantine -- random weights made every instance disagree)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.space.schumann_resonance import SchumannResonanceDetector


def _signal(freq: float, amp: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, 1000)
    return (amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(1000)).astype(float)


class TestUntrainedQuarantine:
    """Untrained network must yield deterministic, physics-grounded output."""

    def test_starts_untrained(self) -> None:
        det = SchumannResonanceDetector()
        assert det._neural_trained is False

    def test_two_instances_agree_on_same_signal(self) -> None:
        # The core bug fix: random-weight inference made instances disagree.
        sig = _signal(7.83)
        r1 = SchumannResonanceDetector().detect_resonance_anomaly(sig)
        r2 = SchumannResonanceDetector().detect_resonance_anomaly(sig)
        assert r1.anomaly_type == r2.anomaly_type
        assert r1.confidence == pytest.approx(r2.confidence)
        assert r1.risk_score == pytest.approx(r2.risk_score)

    def test_confidence_is_bounded(self) -> None:
        for freq, amp in [(7.83, 1.0), (5.0, 5.0), (20.0, 3.0)]:
            r = SchumannResonanceDetector().detect_resonance_anomaly(_signal(freq, amp))
            assert 0.0 <= r.confidence <= 1.0


class TestPhysicsAssessment:
    """The deterministic fallback mapping is correct."""

    def test_combined_when_amplitude_and_frequency(self) -> None:
        kind, conf = SchumannResonanceDetector._physics_assessment(True, True, False, 1.0)
        assert kind == "combined"
        assert conf > 0.0

    def test_frequency_only(self) -> None:
        kind, _ = SchumannResonanceDetector._physics_assessment(False, True, False, 0.6)
        assert kind == "frequency"

    def test_normal_when_no_flags(self) -> None:
        kind, conf = SchumannResonanceDetector._physics_assessment(False, False, False, 0.0)
        assert kind == "normal"
        assert conf == pytest.approx(0.0)

    def test_confidence_increases_with_evidence(self) -> None:
        _, low = SchumannResonanceDetector._physics_assessment(True, False, False, 0.1)
        _, high = SchumannResonanceDetector._physics_assessment(True, True, True, 1.5)
        assert high > low


class TestLoadWeightsEnablesNeural:
    """Loading weights flips the detector onto the learned classifier path."""

    def test_load_weights_sets_trained_flag(self) -> None:
        det = SchumannResonanceDetector()
        # The analyser's own (random) state_dict is a structurally-valid load;
        # this exercises the activation path without claiming the weights are good.
        det.load_neural_weights(det.harmonic_analyzer.state_dict())
        assert det._neural_trained is True
        # Still produces a well-formed result through the neural branch.
        result = det.detect_resonance_anomaly(_signal(7.83))
        assert result.anomaly_type in {"normal", "amplitude", "frequency", "combined"}
        assert 0.0 <= result.confidence <= 1.0

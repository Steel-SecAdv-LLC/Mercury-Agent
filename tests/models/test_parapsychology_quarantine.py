"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Anti-theater tests for the parapsychology detector: the untrained
ConsciousnessFieldAnalyzer must not present random-weight output as a coherence
measurement. With no validated corpus and random weights, field coherence
abstains to the neutral 0.5 prior (deterministically) until trained weights are
explicitly loaded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.models.parapsychology import ParapsychologyDetector


class TestUntrainedQuarantine:
    """Untrained consciousness-field analyser abstains to the neutral prior."""

    def test_starts_untrained(self) -> None:
        det = ParapsychologyDetector()
        assert det._neural_trained is False

    def test_coherence_is_neutral_and_deterministic(self) -> None:
        rng = np.random.RandomState(0)
        seq = rng.random(100)
        a = ParapsychologyDetector()._analyze_field_coherence(seq)
        b = ParapsychologyDetector()._analyze_field_coherence(seq)
        assert a == 0.5
        assert b == 0.5

    def test_short_sequence_returns_neutral(self) -> None:
        det = ParapsychologyDetector()
        assert det._analyze_field_coherence(np.array([0.1, 0.2, 0.3])) == 0.5


class TestLoadWeights:
    """Loading weights enables the neural path; disabled field rejects load."""

    def test_load_weights_enables_network(self) -> None:
        det = ParapsychologyDetector(enable_consciousness_field=True)
        assert det.field_analyzer is not None
        det.load_neural_weights(det.field_analyzer.state_dict())
        assert det._neural_trained is True
        coherence = det._analyze_field_coherence(np.random.RandomState(1).random(100))
        assert 0.0 <= coherence <= 1.0

    def test_load_rejected_when_field_disabled(self) -> None:
        det = ParapsychologyDetector(enable_consciousness_field=False)
        assert det.field_analyzer is None
        with pytest.raises(RuntimeError, match="disabled"):
            det.load_neural_weights({})

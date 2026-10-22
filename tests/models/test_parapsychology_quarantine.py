# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anti-theater tests for the parapsychology detector: the untrained ConsciousnessFieldAnalyzer must not present random-weight output as a coherence measurement. With no validated corpus and random weights, field coherence abstains to the neutral 0.5 prior (deterministically) until trained weights are explicitly loaded."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.models.parapsychology import ParapsychologyDetector


class TestUntrainedQuarantine:
    """Untrained consciousness-field analyser abstains to the neutral prior.

    The default detector serves the ratified ``reg_deviation_gcp`` winner; these
    tests pin the disclosed untrained-abstention fallback, so they construct the
    physics configuration explicitly (``load_shipped_weights=False``).
    """

    def test_default_detector_serves_the_shipped_winner(self) -> None:
        assert ParapsychologyDetector()._neural_trained is True

    def test_physics_configuration_is_untrained(self) -> None:
        assert ParapsychologyDetector(load_shipped_weights=False)._neural_trained is False

    def test_coherence_is_neutral_and_deterministic(self) -> None:
        rng = np.random.RandomState(0)
        seq = rng.random(100)
        a = ParapsychologyDetector(load_shipped_weights=False)._analyze_field_coherence(seq)
        b = ParapsychologyDetector(load_shipped_weights=False)._analyze_field_coherence(seq)
        assert a == 0.5
        assert b == 0.5

    def test_short_sequence_returns_neutral(self) -> None:
        det = ParapsychologyDetector(load_shipped_weights=False)
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


class TestDecisionSourceDeclarations:
    """A7: no published threshold exists for psi or 'consciousness' composites."""

    def test_parapsychology_separates_statistics_from_psi_claims(self) -> None:
        from omni_mercury_engine.models.parapsychology import ParapsychologyDetector

        statistical, fault_model, psi = ParapsychologyDetector.decision_source
        assert statistical.kind == "published"
        assert "p < 0.05" in statistical.description
        assert fault_model.kind == "trained"
        assert "reg_deviation_gcp" in fault_model.operating_point
        assert psi.kind == "experimental"

    def test_consciousness_model_is_marked_experimental(self) -> None:
        from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel

        source = ConsciousnessPreservationModel.decision_source
        assert source.kind == "experimental"
        assert "no labelled corpus" in source.description

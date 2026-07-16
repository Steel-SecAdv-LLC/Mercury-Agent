# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test neural and consciousness models."""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.models.consciousness import ConsciousnessPreservationModel
from omni_mercury_engine.models.neural import NeuralCognitiveModel


def test_neural_model_initialization() -> None:
    """Test neural cognitive model initialization"""
    model = NeuralCognitiveModel()
    assert model is not None
    assert hasattr(model, "memory_buffer")
    assert model.memory_capacity == 100


def test_neural_model_predict(sample_data: Any) -> None:
    """Test neural cognitive model prediction"""
    model = NeuralCognitiveModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "memory_scores" in result
    assert "executive_scores" in result
    assert "emotional_scores" in result
    assert "model_type" in result
    assert result["model_type"] == "neural"


def test_neural_model_features(sample_data: Any) -> None:
    """Test neural feature extraction"""
    model = NeuralCognitiveModel()
    features = model.extract_features(sample_data)

    assert features.shape[0] == len(sample_data)
    assert features.shape[1] >= 48


def test_neural_hippocampal_memory(sample_data: Any) -> None:
    """Test hippocampal memory processing"""
    model = NeuralCognitiveModel()
    memory_scores = model._hippocampal_memory(sample_data)

    assert memory_scores is not None
    assert len(memory_scores) == len(sample_data)


def test_neural_prefrontal_executive(sample_data: Any) -> None:
    """Test prefrontal executive processing"""
    model = NeuralCognitiveModel()
    executive_scores = model._prefrontal_executive(sample_data)

    assert executive_scores is not None
    assert len(executive_scores) == len(sample_data)


def test_neural_amygdala_processing(sample_data: Any) -> None:
    """Test amygdala emotional processing"""
    model = NeuralCognitiveModel()
    emotional_scores = model._amygdala_processing(sample_data)

    assert emotional_scores is not None
    assert len(emotional_scores) == len(sample_data)


def test_consciousness_model_initialization() -> None:
    """Test consciousness preservation model initialization"""
    model = ConsciousnessPreservationModel()
    assert model is not None
    assert model.coherence_threshold == 0.5


def test_consciousness_model_predict(sample_data: Any) -> None:
    """Test consciousness preservation model prediction"""
    model = ConsciousnessPreservationModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "pattern_states" in result
    assert "coherence" in result
    assert "entanglement" in result
    assert "model_type" in result
    assert result["model_type"] == "consciousness"


def test_consciousness_features(sample_data: Any) -> None:
    """Test consciousness feature extraction"""
    model = ConsciousnessPreservationModel()
    features = model.extract_features(sample_data)

    assert features.shape[0] == len(sample_data)
    assert features.shape[1] >= 32


def test_consciousness_pattern_encoding(sample_data: Any) -> None:
    """Test pattern state encoding"""
    model = ConsciousnessPreservationModel()
    pattern_states = model._encode_pattern_states(sample_data)

    assert pattern_states is not None
    assert len(pattern_states) == len(sample_data)


def test_consciousness_coherence_measure(sample_data: Any) -> None:
    """Test pattern coherence measurement"""
    model = ConsciousnessPreservationModel()
    pattern_states = model._encode_pattern_states(sample_data)
    coherence = model._measure_pattern_coherence(pattern_states)

    assert coherence is not None
    assert len(coherence) == len(sample_data)


def test_consciousness_entanglement(sample_data: Any) -> None:
    """Test pattern entanglement computation"""
    model = ConsciousnessPreservationModel()
    pattern_states = model._encode_pattern_states(sample_data)
    entanglement = model._compute_entanglement(pattern_states)

    assert entanglement is not None
    assert len(entanglement) == len(sample_data)


def test_single_sample_pattern_features_are_finite() -> None:
    """A single-sample pattern (e.g. univariate (N, 1) input, or a length-1
    series) has no finite differences; the executive-function features must
    degrade to finite values instead of crashing on np.max of an empty diff.
    """
    import numpy as np

    model = NeuralCognitiveModel()
    for arr in (np.arange(4.0).reshape(4, 1), np.array([5.0])):
        features = np.asarray(model.extract_features(arr))
        assert bool(np.all(np.isfinite(features)))


def test_variable_length_calls_do_not_crash_hippocampal_memory() -> None:
    """The streaming hippocampal buffer can hold patterns of different lengths
    across calls (variable sequence/batch lengths). Cosine similarity is only
    defined between equal-length vectors, so the comparison must not build a
    ragged array and raise an inhomogeneous-shape ValueError.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    model = NeuralCognitiveModel()
    model.extract_features(rng.normal(size=(2, 5)))
    later = np.asarray(model.extract_features(rng.normal(size=(2, 7))))
    assert bool(np.all(np.isfinite(later)))


def test_hippocampal_memory_reset_is_deterministic() -> None:
    """Resetting the streaming buffer before extraction (the serve path) yields
    identical features for identical input across calls.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    model = NeuralCognitiveModel()
    batch = rng.normal(size=(3, 8))
    model.reset_state()
    first = np.asarray(model.extract_features(batch))
    model.reset_state()
    second = np.asarray(model.extract_features(batch))
    assert np.array_equal(first, second)

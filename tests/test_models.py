# Copyright (C) 2025 Steel Security Advisors LLC
"""Test model modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.models.affective import AffectiveAnomalyModel
from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_mercury_engine.models.biometric import BiometricAnomalyModel
from omni_mercury_engine.models.quantum import QuantumAnomalyModel


def test_quantum_model(sample_data: Any) -> None:
    """Test quantum-inspired anomaly model"""
    model = QuantumAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "quantum_states" in result
    assert "coherence" in result
    assert "energy_levels" in result


def test_quantum_features(sample_data: Any) -> None:
    """Test quantum feature extraction"""
    model = QuantumAnomalyModel()
    features = model.extract_features(sample_data)

    assert features.shape[0] == len(sample_data)
    assert features.shape[1] >= 16


def test_astrophysical_model(sample_data: Any) -> None:
    """Test astrophysical anomaly model"""
    model = AstrophysicalAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "event_horizons" in result


def test_affective_model(sample_data: Any) -> None:
    """Test affective computing model"""
    model = AffectiveAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "emotion_scores" in result
    assert "distress_levels" in result


def test_biometric_model() -> None:
    """Test biometric quality model"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = model.predict(image)

    assert "model_type" in result

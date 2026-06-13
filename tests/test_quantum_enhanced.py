# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for enhanced quantum anomaly detection model."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.models.quantum import QuantumAnomalyModel


def test_quantum_extract_features() -> None:
    """Test quantum feature extraction"""
    model = QuantumAnomalyModel()
    data = np.random.randn(10, 20)
    features = model.extract_features(data)

    assert features.shape[0] == 10
    assert features.shape[1] == 16
    assert not np.any(np.isnan(features))


def test_quantum_predict() -> None:
    """Test quantum anomaly prediction"""
    model = QuantumAnomalyModel()
    data = np.random.randn(5, 15)
    result = model.predict(data)

    assert "anomaly_scores" in result
    assert "quantum_states" in result
    assert "coherence" in result
    assert "energy_levels" in result
    assert len(result["anomaly_scores"]) == 5


def test_quantum_entanglement_measurement() -> None:
    """Test quantum entanglement measurement"""
    model = QuantumAnomalyModel(config={"entanglement_strength": 0.5})
    data = np.random.randn(3, 10)

    features = model.extract_features(data)
    entanglement_values = features[:, 12]

    assert np.all(entanglement_values >= 0)


def test_quantum_single_sample() -> None:
    """Test quantum model with single sample"""
    model = QuantumAnomalyModel()
    data = np.random.randn(10)

    result = model.predict(data)
    assert len(result["anomaly_scores"]) == 1

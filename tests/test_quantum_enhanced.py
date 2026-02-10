"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Tests for enhanced quantum anomaly detection model"""

import numpy as np

from omni_mercury_engine.models.quantum import QuantumAnomalyModel


def test_quantum_extract_features():
    """Test quantum feature extraction"""
    model = QuantumAnomalyModel()
    data = np.random.randn(10, 20)
    features = model.extract_features(data)

    assert features.shape[0] == 10
    assert features.shape[1] == 16
    assert not np.any(np.isnan(features))


def test_quantum_predict():
    """Test quantum anomaly prediction"""
    model = QuantumAnomalyModel()
    data = np.random.randn(5, 15)
    result = model.predict(data)

    assert "anomaly_scores" in result
    assert "quantum_states" in result
    assert "coherence" in result
    assert "energy_levels" in result
    assert len(result["anomaly_scores"]) == 5


def test_quantum_entanglement_measurement():
    """Test quantum entanglement measurement"""
    model = QuantumAnomalyModel(config={"entanglement_strength": 0.5})
    data = np.random.randn(3, 10)

    features = model.extract_features(data)
    entanglement_values = features[:, 12]

    assert np.all(entanglement_values >= 0)


def test_quantum_single_sample():
    """Test quantum model with single sample"""
    model = QuantumAnomalyModel()
    data = np.random.randn(10)

    result = model.predict(data)
    assert len(result["anomaly_scores"]) == 1

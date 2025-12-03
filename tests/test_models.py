"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

"""
Test model modules
"""

import numpy as np

from omni_anomaly_engine.models.affective import AffectiveAnomalyModel
from omni_anomaly_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_anomaly_engine.models.biometric import BiometricAnomalyModel
from omni_anomaly_engine.models.quantum import QuantumAnomalyModel


def test_quantum_model(sample_data):
    """Test quantum-inspired anomaly model"""
    model = QuantumAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "quantum_states" in result
    assert "coherence" in result
    assert "energy_levels" in result


def test_quantum_features(sample_data):
    """Test quantum feature extraction"""
    model = QuantumAnomalyModel()
    features = model.extract_features(sample_data)

    assert features.shape[0] == len(sample_data)
    assert features.shape[1] >= 16


def test_astrophysical_model(sample_data):
    """Test astrophysical anomaly model"""
    model = AstrophysicalAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "event_horizons" in result


def test_affective_model(sample_data):
    """Test affective computing model"""
    model = AffectiveAnomalyModel()
    result = model.predict(sample_data)

    assert "anomaly_scores" in result
    assert "emotion_scores" in result
    assert "distress_levels" in result


def test_biometric_model():
    """Test biometric quality model"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = model.predict(image)

    assert "model_type" in result

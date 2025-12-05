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

"""Tests for neurosymbolic engine integration"""

import numpy as np

from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine


def test_neurosymbolic_initialization():
    """Test neurosymbolic engine initialization"""
    engine = NeurosymbolicEngine(input_dim=64)
    assert engine.input_dim == 64
    assert len(engine.knowledge_base) > 0


def test_symbolic_inference():
    """Test symbolic reasoning"""
    engine = NeurosymbolicEngine()
    engine.add_fact("missing_person")
    engine.add_fact("child")

    result = engine.symbolic_inference("priority_high")
    assert result["result"] is True


def test_neural_inference():
    """Test neural inference when PyTorch available"""
    engine = NeurosymbolicEngine(input_dim=32)
    features = np.random.randn(32)

    confidence = engine.neural_inference(features)
    assert 0.0 <= confidence <= 1.0


def test_extract_features():
    """Test feature extraction"""
    engine = NeurosymbolicEngine(input_dim=48)
    data = np.random.randn(5, 20)

    features = engine.extract_features(data)
    assert features.shape[0] == 5


def test_predict():
    """Test anomaly prediction"""
    engine = NeurosymbolicEngine()
    data = np.random.randn(3, 15)

    result = engine.predict(data)
    assert "anomaly_scores" in result
    assert len(result["anomaly_scores"]) == 3

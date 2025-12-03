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

"""Tests for Multiverse Omni Engine"""

import numpy as np

from omni_anomaly_engine.models.multiverse import (
    MultiverseOmniEngine,
    UniverseState,
)


def test_multiverse_initialization():
    """Test multiverse engine initialization"""
    engine = MultiverseOmniEngine(num_universes=10, state_dim=50)
    assert len(engine.universes) == 10
    assert engine.state_dim == 50
    assert engine.timeline == 0


def test_multiverse_convergence():
    """Test multiverse convergence to optimal solution"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=10)

    def fitness_fn(state):
        return -np.sum(state**2)

    converged = engine.converge_multiverse(fitness_fn)
    assert converged.state == UniverseState.CONVERGED
    assert converged.fitness <= 0
    assert engine.best_universe is not None


def test_multiverse_extract_features():
    """Test feature extraction for anomaly detection"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=20)
    data = np.random.randn(3, 15)

    features = engine.extract_features(data)
    assert features.shape[0] == 3
    assert features.shape[1] == 12


def test_multiverse_predict():
    """Test anomaly prediction"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=20)
    data = np.random.randn(4, 10)

    result = engine.predict(data)
    assert "anomaly_scores" in result
    assert len(result["anomaly_scores"]) == 4
    assert "multiverse_features" in result
    assert "best_fitness" in result


def test_multiverse_report():
    """Test multiverse report generation"""
    engine = MultiverseOmniEngine(num_universes=8, state_dim=30)

    def fitness_fn(state):
        return -np.linalg.norm(state)

    engine.converge_multiverse(fitness_fn)
    report = engine.get_multiverse_report()

    assert report["total_universes"] > 0
    assert "system_version" in report
    assert "V20" in report["system_version"]

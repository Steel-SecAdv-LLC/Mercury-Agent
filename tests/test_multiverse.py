# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Multiverse Omni Engine."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.models.multiverse import MultiverseOmniEngine, UniverseState


def test_multiverse_initialization() -> None:
    """Test multiverse engine initialization"""
    engine = MultiverseOmniEngine(num_universes=10, state_dim=50)
    assert len(engine.universes) == 10
    assert engine.state_dim == 50
    assert engine.timeline == 0


def test_multiverse_convergence() -> None:
    """Test multiverse convergence to optimal solution"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=10)

    def fitness_fn(state):
        return -np.sum(state**2)

    converged = engine.converge_multiverse(fitness_fn)
    assert converged.state == UniverseState.CONVERGED
    assert converged.fitness <= 0
    assert engine.best_universe is not None


def test_multiverse_extract_features() -> None:
    """Test feature extraction for anomaly detection"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=20)
    data = np.random.randn(3, 15)

    features = engine.extract_features(data)
    assert features.shape[0] == 3
    assert features.shape[1] == 12


def test_multiverse_extract_features_does_not_mutate_universe_count() -> None:
    """Feature extraction should be per-sample, not universe-growing."""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=20)
    data = np.random.randn(25, 15)

    features = engine.extract_features(data)

    assert features.shape == (25, 12)
    assert len(engine.universes) == 5


def test_multiverse_predict() -> None:
    """Test anomaly prediction"""
    engine = MultiverseOmniEngine(num_universes=5, state_dim=20)
    data = np.random.randn(4, 10)

    result = engine.predict(data)
    assert "anomaly_scores" in result
    assert len(result["anomaly_scores"]) == 4
    assert "multiverse_features" in result
    assert "best_fitness" in result


def test_multiverse_report() -> None:
    """Test multiverse report generation"""
    engine = MultiverseOmniEngine(num_universes=8, state_dim=30)

    def fitness_fn(state):
        return -np.linalg.norm(state)

    engine.converge_multiverse(fitness_fn)
    report = engine.get_multiverse_report()

    assert report["total_universes"] > 0
    assert "system_version" in report
    # Verify system_version contains the OMNI_PERCIPIENT Omni-Code
    # The code contains "20A19" (vigesimal encoding) not "V20"
    assert "20A19" in report["system_version"]

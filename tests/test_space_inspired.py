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

from __future__ import annotations

from typing import Any

"""Tests for Space-Inspired Resilience module"""

import numpy as np

from omni_anomaly_engine.space.space_inspired import SpaceInspiredResilience, SystemState


def test_space_resilience_initialization():
    """Test space-inspired resilience system initialization"""
    system = SpaceInspiredResilience(redundancy_factor=5, degradation_threshold=0.8)
    assert system.redundancy_factor == 5
    assert system.degradation_threshold == 0.8
    assert system.state == SystemState.NOMINAL
    assert isinstance(system.component_health, dict)


def test_graceful_degradation_nominal():
    """Test graceful degradation with no failures"""
    system = SpaceInspiredResilience()

    failures = []
    available = ["comp1", "comp2", "comp3", "comp4"]

    state, priorities = system.graceful_degradation(failures, available)

    assert state == SystemState.NOMINAL
    assert len(priorities) == 4
    assert all(p > 0 for p in priorities.values())


def test_graceful_degradation_partial_failure():
    """Test graceful degradation with partial failures"""
    system = SpaceInspiredResilience(degradation_threshold=0.7, min_operational_components=2)

    failures = ["comp1"]
    available = ["comp1", "comp2", "comp3", "comp4"]

    state, priorities = system.graceful_degradation(failures, available)

    assert state in [SystemState.NOMINAL, SystemState.DEGRADED]
    assert priorities["comp1"] == 0.0
    assert all(priorities[c] > 0 for c in ["comp2", "comp3", "comp4"])


def test_debris_filtering():
    """Test debris filtering for noise reduction"""
    system = SpaceInspiredResilience()

    clean_data = np.random.randn(100, 10) * 0.5
    noisy_data = clean_data + np.random.randn(100, 10) * 5.0

    filtered = system.debris_filtering(noisy_data, noise_threshold=0.2)

    assert filtered.shape == noisy_data.shape
    assert isinstance(filtered, np.ndarray[Any, Any])


def test_trajectory_optimization():
    """Test trajectory optimization for pathfinding"""
    system = SpaceInspiredResilience()

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0, 10.0])

    pathway = system.trajectory_optimization(start, goal, constraints={"max_steps": 20})

    assert len(pathway) > 0
    assert isinstance(pathway[0], np.ndarray[Any, Any])
    assert np.allclose(pathway[0], start, atol=0.1)


def test_reusability_tracking():
    """Test component reusability tracking"""
    system = SpaceInspiredResilience()

    is_reusable, health = system.reusability_tracking("detector_1", usage_count=10, max_reuses=100)

    assert isinstance(is_reusable, bool)
    assert 0.0 <= health <= 1.0
    assert is_reusable is True
    assert health > 0.8


def test_reusability_limit():
    """Test that components become non-reusable at limit"""
    system = SpaceInspiredResilience()

    is_reusable, health = system.reusability_tracking("detector_2", usage_count=95, max_reuses=100)

    assert health < 0.3
    assert is_reusable is False


def test_extreme_environment_adaptation():
    """Test adaptation to extreme environmental stress"""
    system = SpaceInspiredResilience()

    low_stress_performance = system.extreme_environment_adaptation(0.1, adaptation_rate=0.2)
    high_stress_performance = system.extreme_environment_adaptation(0.9, adaptation_rate=0.2)

    assert 0.0 <= low_stress_performance <= 1.0
    assert 0.0 <= high_stress_performance <= 1.0
    assert low_stress_performance > high_stress_performance

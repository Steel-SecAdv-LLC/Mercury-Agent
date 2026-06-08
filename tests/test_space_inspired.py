# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Space-Inspired Resilience module."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.space.space_inspired import SpaceInspiredResilience, SystemState


def test_space_resilience_initialization() -> None:
    """Test space-inspired resilience system initialization"""
    system = SpaceInspiredResilience(redundancy_factor=5, degradation_threshold=0.8)
    assert system.redundancy_factor == 5
    assert system.degradation_threshold == 0.8
    assert system.state == SystemState.NOMINAL
    assert isinstance(system.component_health, dict)


def test_graceful_degradation_nominal() -> None:
    """Test graceful degradation with no failures"""
    system = SpaceInspiredResilience()

    failures: list[str] = []
    available = ["comp1", "comp2", "comp3", "comp4"]

    state, priorities = system.graceful_degradation(failures, available)

    assert state == SystemState.NOMINAL
    assert len(priorities) == 4
    assert all(p > 0 for p in priorities.values())


def test_graceful_degradation_partial_failure() -> None:
    """Test graceful degradation with partial failures"""
    system = SpaceInspiredResilience(degradation_threshold=0.7, min_operational_components=2)

    failures = ["comp1"]
    available = ["comp1", "comp2", "comp3", "comp4"]

    state, priorities = system.graceful_degradation(failures, available)

    assert state in [SystemState.NOMINAL, SystemState.DEGRADED]
    assert priorities["comp1"] == 0.0
    assert all(priorities[c] > 0 for c in ["comp2", "comp3", "comp4"])


def test_debris_filtering() -> None:
    """Test debris filtering for noise reduction"""
    system = SpaceInspiredResilience()

    clean_data = np.random.randn(100, 10) * 0.5
    noisy_data = clean_data + np.random.randn(100, 10) * 5.0

    filtered = system.debris_filtering(noisy_data, noise_threshold=0.2)

    assert filtered.shape == noisy_data.shape
    assert isinstance(filtered, np.ndarray)


def test_trajectory_optimization() -> None:
    """Test trajectory optimization for pathfinding"""
    system = SpaceInspiredResilience()

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0, 10.0])

    pathway = system.trajectory_optimization(start, goal, constraints={"max_steps": 20})

    assert len(pathway) > 0
    assert isinstance(pathway[0], np.ndarray)
    assert np.allclose(pathway[0], start, atol=0.1)


def test_reusability_tracking() -> None:
    """Test component reusability tracking"""
    system = SpaceInspiredResilience()

    is_reusable, health = system.reusability_tracking("detector_1", usage_count=10, max_reuses=100)

    assert isinstance(is_reusable, bool)
    assert 0.0 <= health <= 1.0
    assert is_reusable is True
    assert health > 0.8


def test_reusability_limit() -> None:
    """Test that components become non-reusable at limit"""
    system = SpaceInspiredResilience()

    is_reusable, health = system.reusability_tracking("detector_2", usage_count=95, max_reuses=100)

    assert health < 0.3
    assert is_reusable is False


def test_extreme_environment_adaptation() -> None:
    """Test adaptation to extreme environmental stress"""
    system = SpaceInspiredResilience()

    low_stress_performance = system.extreme_environment_adaptation(0.1, adaptation_rate=0.2)
    high_stress_performance = system.extreme_environment_adaptation(0.9, adaptation_rate=0.2)

    assert 0.0 <= low_stress_performance <= 1.0
    assert 0.0 <= high_stress_performance <= 1.0
    assert low_stress_performance > high_stress_performance

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

"""Tests for Agentic Autonomy module"""

import pytest
import numpy as np
from omni_anomaly_engine.agentic.agentic_autonomy import AgenticAutonomy, AgentState, AgentAction


def test_agentic_initialization():
    """Test agentic autonomy system initialization"""
    system = AgenticAutonomy(autonomy_level=0.9)
    assert system.autonomy_level == 0.9
    assert system.state == AgentState.IDLE
    assert len(system.action_history) == 0
    assert system.decision_threshold == pytest.approx(0.1)


def test_autonomous_detect_with_anomaly():
    """Test autonomous detection when anomaly is present"""
    system = AgenticAutonomy(autonomy_level=0.7)

    anomalous_data = np.random.randn(100) * 10.0 + 50.0

    result = system.autonomous_detect(anomalous_data)

    assert isinstance(result, dict)
    assert "anomaly_detected" in result
    assert "anomaly_score" in result
    assert "action_taken" in result
    assert "autonomous" in result
    assert result["autonomous"] is True


def test_autonomous_detect_without_anomaly():
    """Test autonomous detection when no anomaly is present"""
    system = AgenticAutonomy(autonomy_level=0.8)

    normal_data = np.random.randn(100) * 0.1

    result = system.autonomous_detect(normal_data)

    assert isinstance(result, dict)
    assert "anomaly_score" in result
    assert result["anomaly_score"] >= 0.0


def test_agent_state_transitions():
    """Test that agent transitions through states correctly"""
    system = AgenticAutonomy(autonomy_level=0.5)

    assert system.state == AgentState.IDLE

    data = np.random.randn(100) * 5.0
    system.autonomous_detect(data)

    assert system.state == AgentState.IDLE


def test_action_history_tracking():
    """Test that actions are tracked in history"""
    system = AgenticAutonomy(autonomy_level=0.6)

    initial_count = len(system.action_history)

    anomalous_data = np.random.randn(50) * 20.0 + 100.0
    result = system.autonomous_detect(anomalous_data)

    if result["action_taken"] is not None:
        assert len(system.action_history) > initial_count
        assert isinstance(system.action_history[-1], AgentAction)


def test_autonomy_level_affects_threshold():
    """Test that autonomy level affects decision threshold"""
    high_autonomy = AgenticAutonomy(autonomy_level=0.9)
    low_autonomy = AgenticAutonomy(autonomy_level=0.3)

    assert high_autonomy.decision_threshold < low_autonomy.decision_threshold
    assert high_autonomy.decision_threshold == pytest.approx(0.1)
    assert low_autonomy.decision_threshold == pytest.approx(0.7)


def test_human_oversight_flag():
    """Test that human oversight flag is set correctly"""
    system = AgenticAutonomy(autonomy_level=0.8)

    data = np.random.randn(100)
    result = system.autonomous_detect(data)

    assert "human_oversight_needed" in result
    assert isinstance(result["human_oversight_needed"], bool)

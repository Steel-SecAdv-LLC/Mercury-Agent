"""
Mercury Agent
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

"""Tests for Agentic Autonomy module"""

import numpy as np
import pytest

from omni_mercury_engine.agentic.agentic_autonomy import AgentAction, AgenticAutonomy, AgentState


def test_agentic_initialization() -> None:
    """Test agentic autonomy system initialization"""
    system = AgenticAutonomy(autonomy_level=0.9)
    assert system.autonomy_level == 0.9
    assert system.state == AgentState.IDLE
    assert len(system.action_history) == 0
    assert system.decision_threshold == pytest.approx(0.1)


def test_autonomous_detect_with_anomaly() -> None:
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


def test_autonomous_detect_without_anomaly() -> None:
    """Test autonomous detection when no anomaly is present"""
    system = AgenticAutonomy(autonomy_level=0.8)

    normal_data = np.random.randn(100) * 0.1

    result = system.autonomous_detect(normal_data)

    assert isinstance(result, dict)
    assert "anomaly_score" in result
    assert result["anomaly_score"] >= 0.0


def test_agent_state_transitions() -> None:
    """Test that agent transitions through states correctly"""
    system = AgenticAutonomy(autonomy_level=0.5)

    assert system.state == AgentState.IDLE

    data = np.random.randn(100) * 5.0
    system.autonomous_detect(data)

    assert system.state == AgentState.IDLE


def test_action_history_tracking() -> None:
    """Test that actions are tracked in history"""
    system = AgenticAutonomy(autonomy_level=0.6)

    initial_count = len(system.action_history)

    anomalous_data = np.random.randn(50) * 20.0 + 100.0
    result = system.autonomous_detect(anomalous_data)

    if result["action_taken"] is not None:
        assert len(system.action_history) > initial_count
        assert isinstance(system.action_history[-1], AgentAction)


def test_autonomy_level_affects_threshold() -> None:
    """Test that autonomy level affects decision threshold"""
    high_autonomy = AgenticAutonomy(autonomy_level=0.9)
    low_autonomy = AgenticAutonomy(autonomy_level=0.3)

    assert high_autonomy.decision_threshold < low_autonomy.decision_threshold
    assert high_autonomy.decision_threshold == pytest.approx(0.1)
    assert low_autonomy.decision_threshold == pytest.approx(0.7)


def test_human_oversight_flag() -> None:
    """Test that human oversight flag is set correctly"""
    system = AgenticAutonomy(autonomy_level=0.8)

    data = np.random.randn(100)
    result = system.autonomous_detect(data)

    assert "human_oversight_needed" in result
    assert isinstance(result["human_oversight_needed"], bool)


# ---------------------------------------------------------------------------
# Workflow execution — branching contract.
#
# ``execute_workflow`` previously had ``decision_point`` steps with
# ``on_true`` / ``on_false`` branches scaffolded but the actual branch
# transition was a no-op (two ``pass`` statements at lines 547-549 of
# agentic_autonomy.py).  The workflow advanced linearly regardless of
# the decision, ignoring the branch keys entirely.  These tests pin the
# real branching contract that replaces the stub.


def test_execute_workflow_linear_completes() -> None:
    """A workflow with no decision_points completes linearly."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "linear",
        "steps": [
            {"id": "norm", "type": "data_transformation", "transformation": "normalize"},
            {"id": "act", "type": "action", "action": "log"},
        ],
    }
    result = system.execute_workflow(workflow, np.array([1.0, 2.0, 3.0]))
    assert result["status"] == "completed"
    assert result["total_steps"] == 2
    assert result["completed_steps"] == 2
    executed_ids = [r["step_id"] for r in result["steps_executed"]]
    assert executed_ids == ["norm", "act"]


def test_execute_workflow_branches_on_true() -> None:
    """A decision_point with on_true must jump to the named target step,
    skipping any intermediate steps."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "branch-true",
        "steps": [
            {
                "id": "decide",
                "type": "decision_point",
                # mean of [10, 20, 30] = 20 > 0.5 → True
                "condition": {"metric": "mean", "operator": ">", "threshold": 0.5},
                "on_true": "true_path",
                "on_false": "false_path",
            },
            # This step MUST be skipped — it sits between the decision
            # and the true target.  Its presence in steps_executed
            # would mean branching did nothing.
            {"id": "false_path", "type": "action", "action": "log"},
            {"id": "true_path", "type": "action", "action": "alert"},
        ],
    }
    result = system.execute_workflow(workflow, np.array([10.0, 20.0, 30.0]))
    assert result["status"] == "completed"
    executed_ids = [r["step_id"] for r in result["steps_executed"]]
    # The action recorded must be the one on the true branch.
    assert "true_path" in executed_ids
    assert "false_path" not in executed_ids, (
        f"branch on_true did not skip the false_path step; executed={executed_ids!r}"
    )
    # The decision itself is recorded in autonomous_decisions.
    assert result["autonomous_decisions"][0]["decision"] is True


def test_execute_workflow_branches_on_false() -> None:
    """A decision_point with on_false must jump when condition is False."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "branch-false",
        "steps": [
            {
                "id": "decide",
                "type": "decision_point",
                # mean of [-1, -2, -3] = -2 < 0.5 → False
                "condition": {"metric": "mean", "operator": ">", "threshold": 0.5},
                "on_true": "true_path",
                "on_false": "false_path",
            },
            {"id": "true_path", "type": "action", "action": "alert"},
            {"id": "false_path", "type": "action", "action": "log"},
        ],
    }
    result = system.execute_workflow(workflow, np.array([-1.0, -2.0, -3.0]))
    executed_ids = [r["step_id"] for r in result["steps_executed"]]
    assert "false_path" in executed_ids
    assert "true_path" not in executed_ids, (
        f"branch on_false did not skip the true_path step; executed={executed_ids!r}"
    )
    assert result["autonomous_decisions"][0]["decision"] is False


def test_execute_workflow_unknown_branch_target_recorded() -> None:
    """An ``on_true`` pointing at a missing step id must record a
    ``branching_errors`` entry and fall through to linear advance —
    silently dropping the branch would mask operator authoring bugs."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "bad-target",
        "steps": [
            {
                "id": "decide",
                "type": "decision_point",
                "condition": {"metric": "mean", "operator": ">", "threshold": 0.5},
                "on_true": "does_not_exist",
            },
            {"id": "next", "type": "action", "action": "log"},
        ],
    }
    result = system.execute_workflow(workflow, np.array([10.0, 20.0, 30.0]))
    assert "branching_errors" in result
    assert result["branching_errors"][0]["target"] == "does_not_exist"
    assert result["branching_errors"][0]["reason"] == "unknown step id"
    # Linear fallthrough: the ``next`` action still runs after the
    # decision point because the unresolved branch did not jump.
    executed_ids = [r["step_id"] for r in result["steps_executed"]]
    assert "next" in executed_ids


def test_execute_workflow_branch_cycle_detected() -> None:
    """A self-referencing branch must NOT hang — the executor caps
    the total number of jumps and fails closed with an explicit
    ``branching_cycle_detected`` status."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "cycle",
        "steps": [
            {
                "id": "decide",
                "type": "decision_point",
                "condition": {"metric": "mean", "operator": ">", "threshold": 0.5},
                # Self-loop: every iteration the same decision fires
                # and jumps back to itself.
                "on_true": "decide",
            },
        ],
    }
    result = system.execute_workflow(workflow, np.array([10.0, 20.0, 30.0]))
    assert result["status"] == "branching_cycle_detected", (
        f"expected branching_cycle_detected, got {result['status']!r}"
    )
    assert "branching_errors" in result
    cycle_entry = next(e for e in result["branching_errors"] if "max_jumps" in e.get("reason", ""))
    assert cycle_entry["max_jumps"] >= 64


def test_execute_workflow_no_branch_keys_advances_linearly() -> None:
    """A decision_point without on_true/on_false advances to the next
    step in order (no branching) — preserves the historical contract
    for operator workflows that only USE decision_point for logging."""
    system = AgenticAutonomy(autonomy_level=0.5)
    workflow = {
        "id": "decision-only",
        "steps": [
            {
                "id": "decide",
                "type": "decision_point",
                "condition": {"metric": "mean", "operator": ">", "threshold": 0.5},
                # No on_true / on_false — pure decision recording.
            },
            {"id": "after", "type": "action", "action": "log"},
        ],
    }
    result = system.execute_workflow(workflow, np.array([10.0, 20.0, 30.0]))
    assert result["status"] == "completed"
    assert "branching_errors" not in result
    executed_ids = [r["step_id"] for r in result["steps_executed"]]
    assert "after" in executed_ids
    # The decision itself is still recorded.
    assert len(result["autonomous_decisions"]) == 1

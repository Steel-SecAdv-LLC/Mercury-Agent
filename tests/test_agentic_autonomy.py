# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Agentic Autonomy module."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.agentic.agentic_autonomy import (
    AgentAction,
    AgenticAutonomy,
    AgentState,
    LearningConfig,
)


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
    assert (
        "false_path" not in executed_ids
    ), f"branch on_true did not skip the false_path step; executed={executed_ids!r}"
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
    assert (
        "true_path" not in executed_ids
    ), f"branch on_false did not skip the true_path step; executed={executed_ids!r}"
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
    assert (
        result["status"] == "branching_cycle_detected"
    ), f"expected branching_cycle_detected, got {result['status']!r}"
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


# ---------------------------------------------------------------------------
# Reinforcement-learning policy is wired into autonomous_detect.
#
# Before this suite the Q-table / experience-replay / epsilon-greedy
# machinery existed but ``autonomous_detect`` hardcoded ``flag_anomaly`` and
# never consulted the policy.  These tests assert the RL loop is real:
# Q-table writes, reward shaping, epsilon-greedy explore/exploit, and replay
# all drive observable behaviour.
# ---------------------------------------------------------------------------


def _anomalous(seed: int) -> np.ndarray:
    """High-mean / high-std batch that clears the decision threshold."""
    return np.random.default_rng(seed).standard_normal(100) * 10.0 + 50.0


class TestReinforcementLearningPolicy:
    def test_policy_selects_action_in_autonomous_detect(self) -> None:
        """The action type comes from the policy, not a hardcoded constant."""
        system = AgenticAutonomy(
            autonomy_level=0.9,
            learning_config=LearningConfig(exploration_rate=0.0, min_exploration_rate=0.0),
            seed=3,
        )
        result = system.autonomous_detect(_anomalous(0))
        assert result["anomaly_detected"] is True
        # The returned action type is surfaced and is a real policy action.
        assert result["action_type"] in AgenticAutonomy.ACTION_TYPES
        assert result["action_taken"].action_type == result["action_type"]

    def test_q_table_updates_toward_reward(self) -> None:
        """A detection must write the selected (state, action) Q-value.

        Terminal TD update with an empty table: new_q == lr * reward.
        """
        cfg = LearningConfig(
            exploration_rate=0.0,
            min_exploration_rate=0.0,
            learning_rate=0.1,
            reward_scale=1.0,
        )
        system = AgenticAutonomy(autonomy_level=0.9, learning_config=cfg, seed=5)
        assert len(system._q_table) == 0

        result = system.autonomous_detect(_anomalous(1))
        action = result["action_taken"]
        assert len(system._q_table) == 1

        # The written key must be the exact state the policy selected on.
        bucket = system._discretize_state(action.state_features)
        key = (bucket, action.action_type)
        assert key in system._q_table
        expected_q = cfg.learning_rate * action.outcome
        assert system._q_table[key] == pytest.approx(expected_q, abs=1e-9)

    def test_q_key_consistent_between_selection_and_learning(self) -> None:
        """The policy reads and the TD update writes the SAME Q-key.

        ``action.state_features`` is captured at selection and reused at
        learning, so ``action_history`` mutating in between cannot drift the
        bucket.
        """
        system = AgenticAutonomy(
            autonomy_level=0.9,
            learning_config=LearningConfig(exploration_rate=0.0, min_exploration_rate=0.0),
            seed=9,
        )
        # Prime action_history so len() is non-trivial at selection time.
        for i in range(15):
            system.autonomous_detect(_anomalous(i))
        # Every Q-table entry must correspond to a real discretized state the
        # learner wrote — i.e. no orphan keys from a selection/learn mismatch.
        assert all(
            isinstance(b, int) and a in AgenticAutonomy.ACTION_TYPES for (b, a) in system._q_table
        )

    def test_reward_shaping_matches_contract(self) -> None:
        """``_compute_action_reward`` implements the documented reward table."""
        system = AgenticAutonomy(autonomy_level=0.8, seed=0)

        def reward(action_type: str, confidence: float, severity: str = "medium") -> float:
            return system._compute_action_reward(
                AgentAction(
                    action_type=action_type,
                    parameters={"severity": severity},
                    confidence=confidence,
                    rationale="t",
                )
            )

        assert reward("flag_anomaly", 0.9) == pytest.approx(1.0)
        assert reward("flag_anomaly", 0.6) == pytest.approx(0.6)
        assert reward("flag_anomaly", 0.2) == pytest.approx(-0.5)
        assert reward("escalate", 0.9, "high") == pytest.approx(0.8)
        assert reward("escalate", 0.9, "critical") == pytest.approx(1.0)
        assert reward("suppress", 0.2) == pytest.approx(0.5)
        assert reward("suppress", 0.9) == pytest.approx(-0.3)
        assert reward("investigate", 0.5) == pytest.approx(0.2)
        assert reward("log", 0.5) == pytest.approx(0.1)

    def test_epsilon_greedy_exploits_best_q_when_exploration_off(self) -> None:
        """With exploration disabled, the policy picks the max-Q action."""
        system = AgenticAutonomy(
            autonomy_level=0.9,
            learning_config=LearningConfig(exploration_rate=0.0, min_exploration_rate=0.0),
            seed=2,
        )
        state = (0.9, 1.0, 0.0, 0.9, 0.0)
        bucket = system._discretize_state(state)
        # Make "investigate" the unambiguous best action for this state.
        for a in AgenticAutonomy.ACTION_TYPES:
            system._q_table[(bucket, a)] = 0.1
        system._q_table[(bucket, "investigate")] = 5.0
        assert system.select_action_with_policy(state) == "investigate"

    def test_epsilon_greedy_explores_all_actions_when_exploration_on(self) -> None:
        """With exploration forced on, selection ranges over all actions."""
        system = AgenticAutonomy(
            autonomy_level=0.9,
            learning_config=LearningConfig(
                exploration_rate=1.0, min_exploration_rate=1.0, exploration_decay=1.0
            ),
            seed=1,
        )
        chosen = {system.select_action_with_policy((0.9, 1.0, 0.0, 0.9, 0.0)) for _ in range(200)}
        assert chosen == set(AgenticAutonomy.ACTION_TYPES)

    def test_experience_replay_runs_and_records_convergence(self) -> None:
        """Once the buffer reaches batch_size, replay runs and logs TD error."""
        cfg = LearningConfig(
            exploration_rate=0.3,
            min_exploration_rate=0.05,
            batch_size=4,
            memory_size=100,
        )
        system = AgenticAutonomy(autonomy_level=0.95, learning_config=cfg, seed=11)
        for i in range(10):
            system.autonomous_detect(_anomalous(i))
        # Buffer filled past batch_size, so replay must have run at least once.
        assert len(system.experience_buffer) >= cfg.batch_size
        assert len(system.policy_metrics.convergence_history) >= 1
        assert all(td >= 0.0 for td in system.policy_metrics.convergence_history)

    def test_exploration_rate_decays_over_episodes(self) -> None:
        """Epsilon decays toward the floor as the agent learns."""
        cfg = LearningConfig(exploration_rate=0.5, min_exploration_rate=0.01, exploration_decay=0.9)
        system = AgenticAutonomy(autonomy_level=0.95, learning_config=cfg, seed=4)
        start = system.exploration_rate
        for i in range(20):
            system.autonomous_detect(_anomalous(i))
        assert system.exploration_rate < start
        assert system.exploration_rate >= cfg.min_exploration_rate

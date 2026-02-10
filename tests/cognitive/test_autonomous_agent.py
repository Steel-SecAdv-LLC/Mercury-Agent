"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Tests for Autonomous Agent - OODA Loop, User Sync, Self-Maintenance
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.cognitive.autonomous_agent import (
    ActionResult,
    ActionRisk,
    AgentState,
    ApprovalStatus,
    Decision,
    DiagnosticResult,
    Observation,
    OODAAgent,
    Orientation,
    Reflection,
    SelfMaintenance,
    UserSyncInterface,
)


class TestUserSyncInterface:
    """Tests for UserSyncInterface."""

    def test_init(self):
        """Test interface initialization."""
        interface = UserSyncInterface(approval_timeout=60.0)
        assert interface.approval_timeout == 60.0
        assert len(interface.pending_approvals) == 0

    def test_request_approval(self):
        """Test requesting approval."""
        interface = UserSyncInterface()
        decision = Decision(
            decision_id="dec_001",
            action="test_action",
            risk_level=ActionRisk.HIGH,
            ethical_score=0.95,
            confidence=0.9,
            reasoning="Test reasoning",
            requires_approval=True,
        )

        request = interface.request_approval(decision, {"context": "test"})

        assert request.request_id.startswith("approval_")
        assert request.status == ApprovalStatus.PENDING
        assert request.decision == decision

    def test_provide_approval_approved(self):
        """Test providing approval."""
        interface = UserSyncInterface()
        decision = Decision(
            decision_id="dec_001",
            action="test_action",
            risk_level=ActionRisk.HIGH,
            ethical_score=0.95,
            confidence=0.9,
            reasoning="Test",
            requires_approval=True,
        )

        request = interface.request_approval(decision, {})
        result = interface.provide_approval(request.request_id, approved=True)

        assert result is True
        assert interface.pending_approvals[request.request_id].status == ApprovalStatus.APPROVED

    def test_provide_approval_rejected(self):
        """Test rejecting approval."""
        interface = UserSyncInterface()
        decision = Decision(
            decision_id="dec_001",
            action="test_action",
            risk_level=ActionRisk.HIGH,
            ethical_score=0.95,
            confidence=0.9,
            reasoning="Test",
            requires_approval=True,
        )

        request = interface.request_approval(decision, {})
        interface.provide_approval(request.request_id, approved=False, response="Not safe")

        assert interface.pending_approvals[request.request_id].status == ApprovalStatus.REJECTED

    def test_check_approval_status(self):
        """Test checking approval status."""
        interface = UserSyncInterface()
        decision = Decision(
            decision_id="dec_001",
            action="test_action",
            risk_level=ActionRisk.HIGH,
            ethical_score=0.95,
            confidence=0.9,
            reasoning="Test",
            requires_approval=True,
        )

        request = interface.request_approval(decision, {})
        status = interface.check_approval_status(request.request_id)

        assert status == ApprovalStatus.PENDING

    def test_add_user_input(self):
        """Test adding user input."""
        interface = UserSyncInterface()
        interface.add_user_input({"feedback": "test feedback"})

        assert len(interface.user_inputs) == 1
        assert interface.user_inputs[0]["data"]["feedback"] == "test feedback"

    def test_get_pending_inputs(self):
        """Test getting pending inputs."""
        interface = UserSyncInterface()
        interface.add_user_input({"input": "1"})
        interface.add_user_input({"input": "2"})

        pending = interface.get_pending_inputs()
        assert len(pending) == 2

        interface.mark_input_processed(0)
        pending = interface.get_pending_inputs()
        assert len(pending) == 1

    def test_preferences(self):
        """Test user preferences."""
        interface = UserSyncInterface()
        interface.set_preference("risk_tolerance", "low")

        assert interface.get_preference("risk_tolerance") == "low"
        assert interface.get_preference("unknown", "default") == "default"

    def test_get_statistics(self):
        """Test statistics retrieval."""
        interface = UserSyncInterface()
        stats = interface.get_statistics()

        assert "pending_approvals" in stats
        assert "user_inputs" in stats


class TestSelfMaintenance:
    """Tests for SelfMaintenance."""

    def test_init(self):
        """Test maintenance initialization."""
        maintenance = SelfMaintenance(
            confidence_threshold=0.85,
            memory_limit=5000,
        )
        assert maintenance.confidence_threshold == 0.85
        assert maintenance.memory_limit == 5000

    def test_run_diagnostics(self):
        """Test running diagnostics."""
        maintenance = SelfMaintenance()

        class MockComponent:
            def get_statistics(self):
                return {"error_count": 5, "confidence": 0.95}

        results = maintenance.run_diagnostics({"test_component": MockComponent()})

        assert len(results) == 1
        assert results[0].component == "test_component"
        assert results[0].status == "healthy"

    def test_run_diagnostics_degraded(self):
        """Test diagnostics detecting degraded state."""
        maintenance = SelfMaintenance(confidence_threshold=0.9)

        class MockComponent:
            def get_statistics(self):
                return {"error_count": 15, "confidence": 0.8}

        results = maintenance.run_diagnostics({"test": MockComponent()})

        assert results[0].status == "degraded"
        assert len(results[0].issues) >= 1

    def test_prune_memories(self):
        """Test memory pruning."""
        maintenance = SelfMaintenance(memory_limit=5)
        memories = [{"id": f"m{i}", "importance": i * 0.1} for i in range(10)]

        pruned, removed = maintenance.prune_memories(memories)

        assert len(pruned) == 5
        assert removed == 5
        assert pruned[0]["importance"] == 0.9

    def test_prune_memories_under_limit(self):
        """Test pruning when under limit."""
        maintenance = SelfMaintenance(memory_limit=100)
        memories = [{"id": f"m{i}"} for i in range(10)]

        pruned, removed = maintenance.prune_memories(memories)

        assert len(pruned) == 10
        assert removed == 0

    def test_detect_redundant_memories(self):
        """Test redundant memory detection."""
        maintenance = SelfMaintenance(redundancy_threshold=0.9)
        memories = [
            {"content": {"a": 1, "b": 2}},
            {"content": {"a": 1, "b": 2}},
            {"content": {"c": 3, "d": 4}},
        ]

        redundant = maintenance.detect_redundant_memories(memories)

        assert len(redundant) >= 1
        assert redundant[0][0] == 0
        assert redundant[0][1] == 1

    def test_repair_rule_inconsistencies(self):
        """Test rule repair."""
        maintenance = SelfMaintenance()
        rules = [
            {"id": "r1"},
            {"id": "r2", "confidence": 0.05},
            {"id": "r3", "confidence": 0.8, "enabled": True},
        ]

        repaired, log = maintenance.repair_rule_inconsistencies(rules)

        assert len(repaired) == 3
        assert repaired[0]["confidence"] == 0.5
        assert repaired[1]["enabled"] is False
        assert len(log) >= 2

    def test_should_trigger_reflection(self):
        """Test reflection trigger check."""
        maintenance = SelfMaintenance(confidence_threshold=0.9)

        assert maintenance.should_trigger_reflection(0.85) is True
        assert maintenance.should_trigger_reflection(0.95) is False

    def test_generate_maintenance_task(self):
        """Test task generation."""
        maintenance = SelfMaintenance()
        task = maintenance.generate_maintenance_task("Low confidence detected", "high")

        assert task["issue"] == "Low confidence detected"
        assert task["priority"] == "high"
        assert task["status"] == "pending"

    def test_get_statistics(self):
        """Test statistics retrieval."""
        maintenance = SelfMaintenance()
        stats = maintenance.get_statistics()

        assert "diagnostics_run" in stats
        assert "maintenance_actions" in stats


class TestOODAAgent:
    """Tests for OODAAgent main interface."""

    def test_init(self):
        """Test agent initialization."""
        agent = OODAAgent(
            risk_threshold=ActionRisk.LOW,
            ethical_threshold=0.95,
        )
        assert agent.risk_threshold == ActionRisk.LOW
        assert agent.ethical_threshold == 0.95
        assert agent.state == AgentState.IDLE

    def test_observe(self):
        """Test observe phase."""
        agent = OODAAgent()
        observation = agent.observe(
            data={"event": "test_event", "value": 42},
            source="test_source",
        )

        assert observation.observation_id.startswith("obs_")
        assert observation.source == "test_source"
        assert agent.state == AgentState.OBSERVING

    def test_orient(self):
        """Test orient phase."""
        agent = OODAAgent()
        observation = agent.observe({"event": "test"})

        def mock_analyzer(data):
            return {
                "patterns": [{"type": "trend"}],
                "predictions": [],
                "threats": [],
                "opportunities": [{"type": "insight"}],
            }

        orientation = agent.orient(observation, analyzer=mock_analyzer)

        assert orientation.orientation_id.startswith("orient_")
        assert len(orientation.patterns) == 1
        assert len(orientation.opportunities) == 1

    def test_decide(self):
        """Test decide phase."""
        agent = OODAAgent()
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)

        decision = agent.decide(orientation)

        assert decision.decision_id.startswith("dec_")
        assert decision.action is not None
        assert 0 <= decision.ethical_score <= 1

    def test_decide_with_ethical_scorer(self):
        """Test decide with custom ethical scorer."""
        agent = OODAAgent()
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)

        def ethical_scorer(action, context):
            return 0.99

        decision = agent.decide(orientation, ethical_scorer=ethical_scorer)

        assert decision.ethical_score == 0.99

    def test_act_low_risk(self):
        """Test act phase with low risk action."""
        agent = OODAAgent(risk_threshold=ActionRisk.HIGH)
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)
        decision = agent.decide(orientation)

        decision.risk_level = ActionRisk.LOW
        decision.requires_approval = False
        decision.ethical_score = 0.99

        result = agent.act(decision)

        assert result is not None
        assert result.result_id.startswith("result_")

    def test_act_blocked_ethical(self):
        """Test act blocked due to low ethical score."""
        agent = OODAAgent(ethical_threshold=0.99)
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)
        decision = agent.decide(orientation)

        decision.ethical_score = 0.5

        result = agent.act(decision)

        assert result is None

    def test_reflect(self):
        """Test reflect phase."""
        agent = OODAAgent()
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)
        decision = agent.decide(orientation)

        decision.requires_approval = False
        decision.ethical_score = 0.99
        result = agent.act(decision)

        reflection = agent.reflect(decision, result)

        assert reflection.reflection_id.startswith("reflect_")
        assert len(reflection.lessons_learned) >= 1
        assert agent.state == AgentState.IDLE

    def test_reflect_blocked_action(self):
        """Test reflect on blocked action."""
        agent = OODAAgent()
        observation = agent.observe({"event": "test"})
        orientation = agent.orient(observation)
        decision = agent.decide(orientation)

        reflection = agent.reflect(decision, None)

        assert "blocked" in reflection.outcome_assessment.lower()
        assert reflection.confidence_adjustment < 0

    def test_run_cycle(self):
        """Test complete OODA cycle."""
        agent = OODAAgent(
            risk_threshold=ActionRisk.CRITICAL,
            ethical_threshold=0.5,
            confidence_threshold=0.3,
        )

        def high_ethical_scorer(action, context):
            return 0.99

        results = agent.run_cycle(
            data={"event": "test_event"},
            source="test",
            ethical_scorer=high_ethical_scorer,
        )

        assert "observation" in results
        assert "orientation" in results
        assert "decision" in results
        assert "reflection" in results

    def test_kill_switch(self):
        """Test kill switch functionality."""
        agent = OODAAgent()
        agent.activate_kill_switch()

        assert agent._kill_switch is True
        assert agent.state == AgentState.ERROR

        with pytest.raises(RuntimeError):
            agent.observe({"event": "test"})

        agent.deactivate_kill_switch()
        assert agent._kill_switch is False

    def test_pause_resume(self):
        """Test pause and resume."""
        agent = OODAAgent()
        agent.pause()

        assert agent._paused is True
        assert agent.state == AgentState.PAUSED

        agent.resume()
        assert agent._paused is False
        assert agent.state == AgentState.IDLE

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = OODAAgent()
        agent.observe({"event": "test"})

        stats = agent.get_statistics()

        assert stats["observations"] == 1
        assert stats["state"] == AgentState.OBSERVING.value
        assert "user_sync" in stats
        assert "maintenance" in stats

    def test_get_audit_log(self):
        """Test audit log retrieval."""
        agent = OODAAgent()
        agent.observe({"event": "test"})

        log = agent.get_audit_log(limit=10)

        assert len(log) >= 1
        assert log[-1]["action"] == "observe"


class TestAgentStates:
    """Tests for agent state enums."""

    def test_agent_states(self):
        """Test all agent states exist."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.OBSERVING.value == "observing"
        assert AgentState.ORIENTING.value == "orienting"
        assert AgentState.DECIDING.value == "deciding"
        assert AgentState.ACTING.value == "acting"
        assert AgentState.REFLECTING.value == "reflecting"
        assert AgentState.PAUSED.value == "paused"
        assert AgentState.AWAITING_APPROVAL.value == "awaiting_approval"
        assert AgentState.ERROR.value == "error"


class TestActionRisk:
    """Tests for action risk enums."""

    def test_action_risk_levels(self):
        """Test all risk levels exist."""
        assert ActionRisk.LOW.value == "low"
        assert ActionRisk.MEDIUM.value == "medium"
        assert ActionRisk.HIGH.value == "high"
        assert ActionRisk.CRITICAL.value == "critical"


class TestApprovalStatus:
    """Tests for approval status enums."""

    def test_approval_statuses(self):
        """Test all approval statuses exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.TIMEOUT.value == "timeout"


class TestDataclasses:
    """Tests for dataclasses."""

    def test_observation(self):
        """Test Observation dataclass."""
        obs = Observation(
            observation_id="obs_001",
            source="test",
            data={"event": "test"},
        )
        assert obs.observation_id == "obs_001"
        assert obs.confidence == 0.8

    def test_orientation(self):
        """Test Orientation dataclass."""
        orient = Orientation(
            orientation_id="orient_001",
            patterns=[],
            predictions=[],
            threats=[],
            opportunities=[],
        )
        assert orient.orientation_id == "orient_001"
        assert orient.confidence == 0.8

    def test_decision(self):
        """Test Decision dataclass."""
        dec = Decision(
            decision_id="dec_001",
            action="test_action",
            risk_level=ActionRisk.LOW,
            ethical_score=0.95,
            confidence=0.9,
            reasoning="Test",
            requires_approval=False,
        )
        assert dec.decision_id == "dec_001"
        assert dec.risk_level == ActionRisk.LOW

    def test_action_result(self):
        """Test ActionResult dataclass."""
        result = ActionResult(
            result_id="result_001",
            action="test_action",
            success=True,
            outcome={"status": "completed"},
            side_effects=[],
        )
        assert result.result_id == "result_001"
        assert result.success is True

    def test_reflection(self):
        """Test Reflection dataclass."""
        ref = Reflection(
            reflection_id="reflect_001",
            action="test_action",
            outcome_assessment="Success",
            lessons_learned=["Lesson 1"],
            rule_updates=[],
            memory_updates=[],
            confidence_adjustment=0.01,
        )
        assert ref.reflection_id == "reflect_001"
        assert len(ref.lessons_learned) == 1

    def test_diagnostic_result(self):
        """Test DiagnosticResult dataclass."""
        diag = DiagnosticResult(
            diagnostic_id="diag_001",
            component="test_component",
            status="healthy",
            issues=[],
            recommendations=[],
        )
        assert diag.diagnostic_id == "diag_001"
        assert diag.status == "healthy"


class TestIntegration:
    """Integration tests for autonomous agent."""

    def test_full_autonomous_cycle(self):
        """Test complete autonomous operation cycle."""
        agent = OODAAgent(
            risk_threshold=ActionRisk.CRITICAL,
            ethical_threshold=0.5,
            confidence_threshold=0.3,
        )

        def analyzer(data):
            return {
                "patterns": [{"type": "normal_operation"}],
                "predictions": [{"type": "stable"}],
                "threats": [],
                "opportunities": [{"type": "optimization"}],
            }

        def ethical_scorer(action, context):
            return 0.98

        def executor(action, context):
            return {"success": True, "result": "completed"}

        results = agent.run_cycle(
            data={"event": "system_check", "status": "normal"},
            source="monitoring",
            analyzer=analyzer,
            ethical_scorer=ethical_scorer,
            executor=executor,
        )

        assert results["observation"] is not None
        assert results["orientation"] is not None
        assert results["decision"] is not None
        assert results["reflection"] is not None

        stats = agent.get_statistics()
        assert stats["observations"] == 1
        assert stats["decisions"] == 1

    def test_user_sync_integration(self):
        """Test user synchronization integration."""
        agent = OODAAgent()

        agent.user_sync.add_user_input({"preference": "conservative"})
        agent.user_sync.set_preference("risk_tolerance", "low")

        observation = agent.observe({"event": "test"})

        assert "user_input" in observation.data
        assert agent.user_sync.get_preference("risk_tolerance") == "low"

    def test_maintenance_integration(self):
        """Test self-maintenance integration."""
        agent = OODAAgent(confidence_threshold=0.95)

        class MockComponent:
            def get_statistics(self):
                return {"confidence": 0.8, "error_count": 0}

        diagnostics = agent.maintenance.run_diagnostics({"agent": MockComponent()})

        assert len(diagnostics) == 1
        assert agent.maintenance.should_trigger_reflection(0.8) is True

    def test_ethical_blocking_integration(self):
        """Test that unethical actions are blocked."""
        agent = OODAAgent(ethical_threshold=0.99)

        observation = agent.observe({"event": "risky_operation"})
        orientation = agent.orient(observation)

        def low_ethical_scorer(action, context):
            return 0.5

        decision = agent.decide(orientation, ethical_scorer=low_ethical_scorer)
        result = agent.act(decision)

        assert result is None

        audit = agent.get_audit_log()
        blocked_entries = [e for e in audit if e["action"] == "act_blocked"]
        assert len(blocked_entries) >= 1

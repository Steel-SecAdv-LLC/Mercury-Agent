"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Behavioural tests for ``agentic/mercury_a_agent.py``.

Before this suite ``MercuryAgent._execute_task`` was a no-op that always
reported ``completed`` (success_rate forced to 1.0) and carried zero tests.
These tests pin the truthed-up contract: real tool dispatch with genuine
success / failure, honest ``skipped`` for unbound tasks, and a fail-closed
ethical gate that cannot be swallowed by the execution ``try/except``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.agentic.mercury_a_agent import (
    DomainType,
    MercuryAgent,
    Task,
    TaskPriority,
    create_mercury_agent,
)
from omni_mercury_engine.ethical import EthicalViolation


def _task(
    description: str,
    *,
    tool: str | None = None,
    tool_args: dict[str, Any] | None = None,
    deps: list[str] | None = None,
    task_id: str = "t1",
    domain: DomainType = DomainType.GENERAL,
) -> Task:
    meta: dict[str, Any] = {}
    if tool is not None:
        meta["tool"] = tool
    if tool_args is not None:
        meta["tool_args"] = tool_args
    return Task(
        task_id=task_id,
        description=description,
        domain=domain,
        priority=TaskPriority.MEDIUM,
        dependencies=deps or [],
        metadata=meta,
    )


class TestRealToolDispatch:
    def test_bound_tool_executes_for_real(self) -> None:
        agent = MercuryAgent(name="t")
        seen = {}

        def detect(data=None):
            seen["data"] = data
            return {"anomalies": int(np.sum(np.asarray(data) > 2.0))}

        agent.register_tool("detect", detect)
        task = _task("run detection", tool="detect")
        result = agent._execute_task(task, {"data": np.array([0.0, 3.0, 4.0, 1.0])})

        assert result["status"] == "completed"
        assert result["output"] == {"anomalies": 2}
        assert task.status == "completed"
        assert task.completed_at is not None
        # The analysed batch was injected into the tool's ``data`` parameter.
        assert seen["data"] is not None

    def test_tool_args_are_forwarded(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("add", lambda a, b: a + b)
        result = agent._execute_task(_task("add", tool="add", tool_args={"a": 2, "b": 5}), {})
        assert result["status"] == "completed"
        assert result["output"] == 7

    def test_raising_tool_yields_genuine_failure(self) -> None:
        agent = MercuryAgent(name="t")

        def boom():
            raise RuntimeError("kaboom")

        agent.register_tool("boom", boom)
        result = agent._execute_task(_task("explode", tool="boom"), {})
        assert result["status"] == "failed"
        assert "kaboom" in result["error"]

    def test_unregistered_tool_fails(self) -> None:
        agent = MercuryAgent(name="t")
        result = agent._execute_task(_task("x", tool="ghost"), {})
        assert result["status"] == "failed"
        assert "not registered" in result["error"]

    def test_unbound_task_is_skipped_not_fake_completed(self) -> None:
        agent = MercuryAgent(name="t")
        result = agent._execute_task(_task("just think"), {})
        assert result["status"] == "skipped"
        assert "no tool bound" in result["reason"]


class TestExecutionEthicalGate:
    def test_harmful_task_is_blocked_fail_closed(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("detect", lambda data=None: {"ok": True})
        harmful = _task(
            "destroy all data harm victims maliciously exploit attack damage",
            tool="detect",
        )
        with pytest.raises(EthicalViolation) as exc:
            agent._execute_task(harmful, {})
        assert exc.value.check == "benevolence"

    def test_gate_runs_before_tool_side_effect(self) -> None:
        """The ethical gate must fire before the tool runs and must NOT be
        swallowed by the execution ``try/except``."""
        agent = MercuryAgent(name="t")
        ran = {"tool": False}

        def detect(data=None):
            ran["tool"] = True
            return {}

        agent.register_tool("detect", detect)
        harmful = _task(
            "harm destroy attack exploit malicious damage victims weaponize",
            tool="detect",
        )
        with pytest.raises(EthicalViolation):
            agent._execute_task(harmful, {})
        assert ran["tool"] is False  # tool never executed

    def test_legitimate_task_passes_gate(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("detect", lambda data=None: {"ok": True})
        result = agent._execute_task(
            _task("analyze network traffic for anomalies", tool="detect"), {}
        )
        assert result["status"] == "completed"


class TestPlanExecutionAccounting:
    def test_success_rate_measured_over_executed_not_inflated_by_skips(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("ok", lambda: "done")
        agent.register_tool("bad", lambda: (_ for _ in ()).throw(ValueError("x")))

        from omni_mercury_engine.agentic.mercury_a_agent import PlanResult

        plan = PlanResult(
            plan_id="p",
            tasks=[
                _task("ok", tool="ok", task_id="a"),
                _task("bad", tool="bad", task_id="b"),
                _task("noexec", task_id="c"),  # skipped
            ],
            estimated_duration=0.0,
            confidence=0.5,
            domain=DomainType.GENERAL,
        )
        out = agent._execute_plan(plan, {})
        assert out["tasks_completed"] == 1
        assert out["tasks_failed"] == 1
        assert out["tasks_skipped"] == 1
        # 1 completed / (1 completed + 1 failed) == 0.5; skip does not inflate.
        assert out["success_rate"] == pytest.approx(0.5)

    def test_dependency_gating_skips_unsatisfied(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("ok", lambda: "done")
        from omni_mercury_engine.agentic.mercury_a_agent import PlanResult

        plan = PlanResult(
            plan_id="p",
            tasks=[
                _task("first", tool="ok", task_id="a", deps=["missing"]),
            ],
            estimated_duration=0.0,
            confidence=0.5,
            domain=DomainType.GENERAL,
        )
        out = agent._execute_plan(plan, {})
        # Unsatisfied dependency => task never executed (not in results).
        assert out["task_results"] == []


class TestAnalyzeFlow:
    def test_analyze_returns_structured_result(self) -> None:
        agent = create_mercury_agent(name="t")
        result = agent.analyze(
            np.random.default_rng(0).standard_normal((10, 4)),
            domain=DomainType.SECURITY,
        )
        assert result["agent"] == "t"
        assert "execution" in result
        assert "reasoning" in result
        # A pure-reasoning plan (no tools bound) reports honest skips, not a
        # fabricated success_rate of 1.0.
        assert result["execution"]["success_rate"] == 0.0
        assert result["execution"]["tasks_skipped"] >= 1

    def test_registered_tools_visible_in_state(self) -> None:
        agent = MercuryAgent(name="t")
        agent.register_tool("detect", lambda: None)
        assert "detect" in agent.get_state()["registered_tools"]

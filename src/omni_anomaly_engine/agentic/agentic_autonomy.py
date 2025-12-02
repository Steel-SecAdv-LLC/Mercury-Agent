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

"""
Agentic AI Autonomy Module

Inspired by Bain 2025 report on agentic AI transformation:
"At full potential, agents will run complete processes and workflows."

Implements autonomous agent framework for anomaly detection
that can operate with minimal human oversight.

Research source: Bain & Company Technology Report 2025

MIT License compatible - original implementation
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class AgentState(Enum):
    """Agent operational states."""

    IDLE = 1
    OBSERVING = 2
    ANALYZING = 3
    ACTING = 4
    LEARNING = 5


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""

    action_type: str
    parameters: Dict
    confidence: float
    rationale: str


class AgenticAutonomy:
    """
    Autonomous agent framework for anomaly detection.

    Agents can operate complete processes and workflows with
    minimal human oversight, inspired by Bain's agentic AI vision.
    """

    def __init__(self, autonomy_level: float = 0.8):
        """
        Initialize agentic autonomy system.

        Args:
            autonomy_level: Level of autonomy (0-1), higher = more autonomous
        """
        self.autonomy_level = autonomy_level
        self.state = AgentState.IDLE
        self.action_history: List[AgentAction] = []
        self.decision_threshold = 1.0 - autonomy_level

    def autonomous_detect(self, data: np.ndarray, context: Optional[Dict] = None) -> Dict:
        """
        Autonomously detect anomalies with minimal human oversight.

        Args:
            data: Input data to analyze
            context: Optional context information

        Returns:
            Detection results with actions taken
        """
        self.state = AgentState.OBSERVING

        observations = self._observe_patterns(data)

        self.state = AgentState.ANALYZING

        anomaly_score = self._analyze_anomalies(observations)

        if anomaly_score > self.decision_threshold:
            self.state = AgentState.ACTING
            action = self._decide_action(anomaly_score, observations)
            self.action_history.append(action)

            self.state = AgentState.LEARNING
            self._learn_from_action(action)
        else:
            action = None

        self.state = AgentState.IDLE

        return {
            "anomaly_detected": bool(anomaly_score > self.decision_threshold),
            "anomaly_score": float(anomaly_score),
            "action_taken": action,
            "autonomous": True,
            "human_oversight_needed": bool(anomaly_score < self.decision_threshold),
        }

    def _observe_patterns(self, data: np.ndarray) -> Dict:
        """Observe patterns in data."""
        return {"mean": np.mean(data), "std": np.std(data), "trend": self._detect_trend(data)}

    def _analyze_anomalies(self, observations: Dict) -> float:
        """Analyze observations for anomalies."""
        score = abs(observations["mean"]) / (observations["std"] + 1e-8)
        return min(score / 10.0, 1.0)

    def _decide_action(self, anomaly_score: float, observations: Dict) -> AgentAction:
        """Decide what action to take."""
        return AgentAction(
            action_type="flag_anomaly",
            parameters={"severity": "high" if anomaly_score > 0.8 else "medium"},
            confidence=anomaly_score,
            rationale=f"Autonomous detection with score {anomaly_score:.3f}",
        )

    def _learn_from_action(self, action: AgentAction):
        """Learn from action outcomes (placeholder for reinforcement learning)."""
        pass

    def _detect_trend(self, data: np.ndarray) -> str:
        """Detect trend in data."""
        flat_data = data.flatten()
        if len(flat_data) < 2:
            return "stable"
        diff = np.diff(flat_data)
        if np.mean(diff) > 0:
            return "increasing"
        elif np.mean(diff) < 0:
            return "decreasing"
        return "stable"

    def execute_workflow(self, workflow_definition: Dict, input_data: np.ndarray) -> Dict:
        """
        Execute complete workflow autonomously.

        Bain 2025: "At full potential, agents will run complete processes and workflows."
        Implements end-to-end workflow execution with minimal human oversight.

        Args:
            workflow_definition: Dict defining workflow steps and conditions
            input_data: Input data for workflow

        Returns:
            Workflow execution results with outcomes and actions
        """
        workflow_id = workflow_definition.get("id", "workflow_001")
        steps = workflow_definition.get("steps", [])

        self.state = AgentState.OBSERVING

        workflow_results = {
            "workflow_id": workflow_id,
            "steps_executed": [],
            "outputs": {},
            "autonomous_decisions": [],
            "human_oversight_required": False,
        }

        current_data = input_data

        for step_idx, step in enumerate(steps):
            step_id = step.get("id", f"step_{step_idx}")
            step_type = step.get("type", "unknown")

            self.state = AgentState.ANALYZING

            if step_type == "anomaly_detection":
                result = self.autonomous_detect(current_data)
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "result": result,
                    }
                )

                if result["anomaly_detected"] and step.get("escalate_on_anomaly", False):
                    workflow_results["human_oversight_required"] = True
                    workflow_results["escalation_reason"] = f"Anomaly detected in {step_id}"
                    break

            elif step_type == "data_transformation":
                transformation = step.get("transformation", "normalize")
                transformed_data = self._apply_transformation(current_data, transformation)
                current_data = transformed_data
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "transformation": transformation,
                    }
                )

            elif step_type == "decision_point":
                condition = step.get("condition", {})
                decision = self._evaluate_condition(current_data, condition)
                workflow_results["autonomous_decisions"].append(
                    {
                        "step_id": step_id,
                        "decision": decision,
                        "rationale": f"Condition {condition} evaluated to {decision}",
                    }
                )

                if decision and step.get("on_true"):
                    next_step_id = step["on_true"]
                elif not decision and step.get("on_false"):
                    next_step_id = step["on_false"]
                else:
                    continue

            elif step_type == "action":
                self.state = AgentState.ACTING
                action_type = step.get("action", "log")
                action_result = self._execute_action(action_type, current_data)
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "action": action_type,
                        "result": action_result,
                    }
                )

        self.state = AgentState.LEARNING
        self._learn_from_workflow(workflow_results)

        self.state = AgentState.IDLE

        workflow_results["status"] = (
            "completed" if not workflow_results["human_oversight_required"] else "escalated"
        )
        workflow_results["total_steps"] = len(steps)
        workflow_results["completed_steps"] = len(workflow_results["steps_executed"])

        return workflow_results

    def _apply_transformation(self, data: np.ndarray, transformation: str) -> np.ndarray:
        """Apply data transformation."""
        if transformation == "normalize":
            return (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif transformation == "scale":
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        else:
            return data

    def _evaluate_condition(self, data: np.ndarray, condition: Dict) -> bool:
        """Evaluate decision condition."""
        metric = condition.get("metric", "mean")
        operator = condition.get("operator", ">")
        threshold = condition.get("threshold", 0.5)

        if metric == "mean":
            value = np.mean(data)
        elif metric == "max":
            value = np.max(data)
        elif metric == "std":
            value = np.std(data)
        else:
            value = 0.0

        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6
        else:
            return False

    def _execute_action(self, action_type: str, data: np.ndarray) -> Dict:
        """Execute workflow action."""
        if action_type == "log":
            return {"logged": True, "data_summary": f"mean={np.mean(data):.3f}"}
        elif action_type == "alert":
            return {"alert_sent": True, "severity": "medium"}
        elif action_type == "store":
            return {"stored": True, "timestamp": "now"}
        else:
            return {"action": action_type, "status": "unknown"}

    def _learn_from_workflow(self, workflow_results: Dict):
        """Learn from workflow execution outcomes."""
        pass

    def get_autonomy_metrics(self) -> Dict:
        """
        Get metrics on autonomous operation.

        Returns:
            Metrics showing autonomy level, decision count, intervention rate
        """
        total_actions = len(self.action_history)

        return {
            "autonomy_level": self.autonomy_level,
            "total_autonomous_actions": total_actions,
            "current_state": self.state.name,
            "decision_threshold": self.decision_threshold,
            "actions_without_intervention": total_actions,
            "bain_vision_alignment": "Agents running complete processes with minimal oversight",
        }

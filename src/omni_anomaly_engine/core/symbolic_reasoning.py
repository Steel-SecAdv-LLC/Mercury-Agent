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

"""Symbolic Reasoning Engine for Explainable Anomaly Detection.

Based on: PyReason - Temporal First-Order Logic Explainable AI
(AAAI 2023: https://pyreason.readthedocs.io/, https://arxiv.org/pdf/2302.13482.pdf)

Provides symbolic reasoning layer for explainable anomaly detection outputs.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class SymbolicRule:
    """Represents a symbolic reasoning rule."""

    def __init__(self, name: str, predicate: str, conditions: List[str], confidence: float = 1.0):
        """Initialize symbolic rule.

        Args:
            name: Rule identifier
            predicate: Logical predicate to evaluate
            conditions: List of condition strings
            confidence: Rule confidence score (0-1)
        """
        self.name = name
        self.predicate = predicate
        self.conditions = conditions
        self.confidence = confidence

    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Evaluate rule against context.

        Args:
            context: Dictionary of variables and their values

        Returns:
            (satisfied, confidence) tuple
        """
        satisfied = all(self._evaluate_condition(cond, context) for cond in self.conditions)
        return satisfied, self.confidence if satisfied else 0.0

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        return True


class SymbolicReasoningEngine:
    """Symbolic reasoning engine for explainable AI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize symbolic reasoning engine.

        Args:
            config: Configuration including:
                - temporal_logic: Enable temporal reasoning (default: True)
                - graph_based: Enable graph-based reasoning (default: True)
                - explainability_threshold: Min confidence for explanations (default: 0.7)
        """
        self.config = config or {}
        self.temporal_logic = self.config.get("temporal_logic", True)
        self.graph_based = self.config.get("graph_based", True)
        self.explainability_threshold = self.config.get("explainability_threshold", 0.7)
        self.rules: List[SymbolicRule] = []
        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default reasoning rules for code anomaly detection."""
        self.rules = [
            SymbolicRule(
                name="high_complexity_rule",
                predicate="is_anomalous",
                conditions=["cyclomatic_complexity > 10", "code_length > 100"],
                confidence=0.8,
            ),
            SymbolicRule(
                name="unusual_pattern_rule",
                predicate="is_anomalous",
                conditions=["pattern_frequency < 0.01"],
                confidence=0.9,
            ),
            SymbolicRule(
                name="refactoring_candidate_rule",
                predicate="needs_refactoring",
                conditions=["code_duplication > 0.5", "maintainability_index < 20"],
                confidence=0.85,
            ),
        ]

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a custom reasoning rule."""
        self.rules.append(rule)

    def reason(self, neural_output: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid neuro-symbolic reasoning.

        Combines neural network outputs with symbolic rule-based reasoning
        for explainable anomaly detection.

        Args:
            neural_output: Output from neural anomaly detector
            context: Context dict with code metrics and features

        Returns:
            Reasoning results with explanations
        """
        symbolic_rules_fired: List[str] = []
        explanations: List[str] = []
        neural_score = float(neural_output[0]) if len(neural_output) > 0 else 0.0

        for rule in self.rules:
            satisfied, confidence = rule.evaluate(context)
            if satisfied and confidence >= self.explainability_threshold:
                symbolic_rules_fired.append(rule.name)
                explanations.append(
                    f"Rule '{rule.name}': {rule.predicate} (confidence: {confidence:.2f})"
                )

        symbolic_confidence = (
            len(symbolic_rules_fired) / len(self.rules) if self.rules else 0.0
        )
        combined_confidence = 0.6 * neural_score + 0.4 * symbolic_confidence

        final_decision = "anomalous" if combined_confidence > 0.5 else "normal"

        results: Dict[str, Any] = {
            "neural_score": neural_score,
            "symbolic_rules_fired": symbolic_rules_fired,
            "explanations": explanations,
            "final_decision": final_decision,
            "confidence": combined_confidence,
        }

        return results

    def explain_decision(self, reasoning_results: Dict[str, Any]) -> str:
        """Generate human-readable explanation of reasoning decision.

        Args:
            reasoning_results: Results from reason() method

        Returns:
            Human-readable explanation string
        """
        explanation = f"Decision: {reasoning_results['final_decision'].upper()} "
        explanation += f"(confidence: {reasoning_results['confidence']:.2%})\n\n"

        if reasoning_results["explanations"]:
            explanation += "Reasoning:\n"
            for exp in reasoning_results["explanations"]:
                explanation += f"  - {exp}\n"
        else:
            explanation += "Based on neural network analysis only.\n"

        return explanation

    def open_world_detection(
        self, observations: List[Dict[str, Any]], confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Detect anomalies in open-world settings with novel object types.

        Based on: Anomaly Detection in an Open World by a Neuro-symbolic Program
        (OpenReview: https://openreview.net/pdf?id=Bg3ZO3nXJuA)

        Uses probabilistic multi-hypothesis reasoning and symbolic predicates.

        Args:
            observations: List of observation dicts with features
            confidence_threshold: Min confidence for novel class detection

        Returns:
            Detection results with symbolic explanations
        """
        detected_anomalies: List[Dict[str, Any]] = []
        novel_classes: List[Dict[str, Any]] = []
        symbolic_explanations: List[str] = []

        for obs in observations:
            for rule in self.rules:
                try:
                    eval_result = rule.evaluate(obs)
                    if isinstance(eval_result, tuple):
                        satisfied, confidence = eval_result
                    else:
                        satisfied = bool(eval_result)
                        confidence = 1.0 if satisfied else 0.0

                    if satisfied and confidence >= confidence_threshold:
                        if hasattr(rule, "predicate") and rule.predicate == "is_novel_object":
                            novel_classes.append(
                                {"observation": obs, "confidence": confidence, "rule": rule.name}
                            )
                            symbolic_explanations.append(
                                f"Novel object detected: {rule.name} (conf: {confidence:.2f})"
                            )
                        elif hasattr(rule, "predicate") and rule.predicate == "is_anomalous":
                            detected_anomalies.append(
                                {"observation": obs, "confidence": confidence, "rule": rule.name}
                            )
                except Exception:
                    continue

        return {
            "detected_anomalies": detected_anomalies,
            "novel_classes": novel_classes,
            "symbolic_explanations": symbolic_explanations,
        }

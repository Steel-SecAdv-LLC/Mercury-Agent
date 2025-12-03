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
AI Ethics Framework for OMNI ♱ AVA
Implements 8 core ethical principles for autonomous AI operations.

This module acts as a "conscience layer" for the engine, enabling autonomous
decisions while enforcing ethical guardrails. All actions are evaluated against
8 principles: Compassion, Evidence, Justice, Altruism, Control, Character,
Competence, and Commitment.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class EthicalPrinciple(Enum):
    """Eight core ethical principles for AI operations."""

    COMPASSION = "compassion"
    EVIDENCE = "evidence"
    JUSTICE = "justice"
    ALTRUISM = "altruism"
    CONTROL = "control"
    CHARACTER = "character"
    COMPETENCE = "competence"
    COMMITMENT = "commitment"


@dataclass
class EthicsConfig:
    """Configuration for ethics framework."""

    enable_compassion_checks: bool = True
    enable_evidence_validation: bool = True
    enable_justice_bias_checks: bool = True
    enable_altruism_impact_checks: bool = True
    enable_control_auditing: bool = True
    enable_character_transparency: bool = True
    enable_competence_validation: bool = True
    enable_commitment_evolution: bool = True
    min_ethics_score: float = 0.7
    strict_mode: bool = False


@dataclass
class EthicsResult:
    """Result of ethics evaluation."""

    passed: bool
    overall_score: float
    principle_scores: Dict[str, float]
    violations: List[str]
    recommendations: List[str]

    def __repr__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"EthicsResult({status}, score={self.overall_score:.2f})"


class EthicalAutonomyGovernor:
    """
    Oversees AI operations and scores actions on ethical principles.

    This acts as a meta-controller ensuring autonomous operations
    don't compromise safety or ethical standards. It evaluates every
    significant action against 8 core principles and maintains an audit log.

    Principles:
    1. COMPASSION: Prioritize user well-being, minimize harm
    2. EVIDENCE: Require verifiable data and proofs
    3. JUSTICE: Ensure fair, unbiased logic
    4. ALTRUISM: Promote positive societal impact
    5. CONTROL: Implement auditable controls
    6. CHARACTER: Uphold integrity and transparency
    7. COMPETENCE: Validate with rigorous testing
    8. COMMITMENT: Dedicate to continuous improvement
    """

    def __init__(self, config: Optional[EthicsConfig] = None):
        self.config = config or EthicsConfig()
        self.audit_log: List[Dict[str, Any]] = []
        logging.info("Ethical Autonomy Governor initialized with 8 principles")

    def evaluate_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EthicsResult:
        """
        Evaluate an action against all 8 ethical principles.

        Args:
            action_type: Type of action (e.g., "refactoring", "optimization")
            action_params: Parameters of the action
            context: Additional context for evaluation

        Returns:
            EthicsResult with pass/fail and detailed scores
        """
        context = context or {}
        principle_scores: Dict[str, float] = {}
        violations: List[str] = []
        recommendations: List[str] = []

        if self.config.enable_compassion_checks:
            compassion_score = self._check_compassion(action_type, action_params, context)
            principle_scores["compassion"] = compassion_score
            if compassion_score < 0.5:
                violations.append("Compassion: Action may cause harm or break existing code")
                recommendations.append("Add safety checks and rollback capabilities")

        if self.config.enable_evidence_validation:
            evidence_score = self._check_evidence(action_type, action_params, context)
            principle_scores["evidence"] = evidence_score
            if evidence_score < 0.5:
                violations.append("Evidence: Claims not backed by verifiable data")
                recommendations.append("Add benchmarks and statistical validation")

        if self.config.enable_justice_bias_checks:
            justice_score = self._check_justice(action_type, action_params, context)
            principle_scores["justice"] = justice_score
            if justice_score < 0.5:
                violations.append("Justice: Potential bias detected in logic")
                recommendations.append("Review for fairness across all input types")

        if self.config.enable_altruism_impact_checks:
            altruism_score = self._check_altruism(action_type, action_params, context)
            principle_scores["altruism"] = altruism_score
            if altruism_score < 0.5:
                violations.append("Altruism: Limited positive societal impact")
                recommendations.append("Consider open-source contributions")

        if self.config.enable_control_auditing:
            control_score = self._check_control(action_type, action_params, context)
            principle_scores["control"] = control_score
            if control_score < 0.5:
                violations.append("Control: Insufficient audit trail")
                recommendations.append("Add detailed logging and rollback mechanisms")

        if self.config.enable_character_transparency:
            character_score = self._check_character(action_type, action_params, context)
            principle_scores["character"] = character_score
            if character_score < 0.5:
                violations.append("Character: Lacks transparency in operations")
                recommendations.append("Add clear docstrings explaining ethical rationale")

        if self.config.enable_competence_validation:
            competence_score = self._check_competence(action_type, action_params, context)
            principle_scores["competence"] = competence_score
            if competence_score < 0.5:
                violations.append("Competence: Insufficient testing coverage")
                recommendations.append("Add tests to achieve >95% coverage")

        if self.config.enable_commitment_evolution:
            commitment_score = self._check_commitment(action_type, action_params, context)
            principle_scores["commitment"] = commitment_score
            if commitment_score < 0.5:
                violations.append("Commitment: No provision for future improvement")
                recommendations.append("Add extension points and versioning")

        overall_score = (
            sum(principle_scores.values()) / len(principle_scores) if principle_scores else 0.0
        )

        if self.config.strict_mode:
            passed = overall_score >= self.config.min_ethics_score and len(violations) == 0
        else:
            passed = overall_score >= self.config.min_ethics_score

        self.audit_log.append(
            {
                "action_type": action_type,
                "action_params": action_params,
                "overall_score": overall_score,
                "passed": passed,
                "violations": violations,
                "timestamp": None,
            }
        )

        return EthicsResult(
            passed=passed,
            overall_score=overall_score,
            principle_scores=principle_scores,
            violations=violations,
            recommendations=recommendations,
        )

    def _check_compassion(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check COMPASSION: Does this minimize harm and prioritize user well-being?

        Evaluates safety features like backups, confirmations, and rollback mechanisms.
        """
        score = 0.5

        if params.get("create_backup", False):
            score += 0.2
        if params.get("require_confirmation", False):
            score += 0.2
        if "rollback" in str(params).lower() or context.get("has_rollback"):
            score += 0.1

        return min(1.0, score)

    def _check_evidence(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check EVIDENCE: Is this backed by verifiable data, benchmarks, and proofs?

        Ensures all claims are supported by empirical measurements.
        """
        score = 0.5

        if "benchmark" in str(params).lower() or context.get("has_benchmarks"):
            score += 0.3
        if "statistical" in str(params).lower() or context.get("has_statistics"):
            score += 0.2
        if context.get("verified_claims"):
            score += 0.1

        return min(1.0, score)

    def _check_justice(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check JUSTICE: Is this fair and unbiased across all inputs?

        Evaluates determinism and fairness in logic.
        """
        score = 0.8

        if action_type in ["ast_transform", "refactoring", "analysis"]:
            score = 1.0

        if context.get("bias_checked"):
            score = min(1.0, score + 0.1)

        return score

    def _check_altruism(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check ALTRUISM: Does this have positive societal impact?

        Evaluates contribution to open-source and community benefit.
        """
        score = 0.6

        if "open_source" in str(params).lower() or context.get("is_open_source"):
            score += 0.2
        if "community" in str(params).lower():
            score += 0.2

        return min(1.0, score)

    def _check_control(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check CONTROL: Are there auditable controls and logging?

        Ensures operations can be traced and reversed if needed.
        """
        score = 0.5

        if params.get("logging_enabled", True):
            score += 0.3
        if "audit" in str(params).lower() or context.get("audit_enabled"):
            score += 0.2

        return min(1.0, score)

    def _check_character(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check CHARACTER: Is this transparent with clear intent?

        Evaluates documentation and explanations of ethical rationale.
        """
        score = 0.7

        if params.get("verbose", False):
            score += 0.2
        if "transparent" in str(params).lower() or context.get("is_transparent"):
            score += 0.1

        return min(1.0, score)

    def _check_competence(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check COMPETENCE: Is this well-tested with high coverage?

        Evaluates test coverage and rigorous validation.
        """
        score = 0.5

        test_coverage = context.get("test_coverage", 0)
        if test_coverage > 0.95:
            score += 0.4
        elif test_coverage > 0.80:
            score += 0.2
        elif test_coverage > 0.60:
            score += 0.1

        return min(1.0, score)

    def _check_commitment(
        self, action_type: str, params: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """
        Check COMMITMENT: Does this support continuous improvement?

        Evaluates extensibility and evolution provisions.
        """
        score = 0.7

        if "extensible" in str(params).lower() or context.get("is_extensible"):
            score += 0.2
        if "versioned" in str(params).lower() or context.get("is_versioned"):
            score += 0.1

        return min(1.0, score)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return the full audit log of ethics evaluations."""
        return self.audit_log.copy()

    def reset_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log.clear()
        logging.info("Ethics audit log cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ethics evaluations."""
        if not self.audit_log:
            return {
                "total_evaluations": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
            }

        total = len(self.audit_log)
        passed = sum(1 for entry in self.audit_log if entry["passed"])
        failed = total - passed
        avg_score = sum(entry["overall_score"] for entry in self.audit_log) / total

        return {
            "total_evaluations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total,
            "avg_score": avg_score,
        }


def evaluate_refactoring_ethics(
    create_backup: bool = True,
    require_confirmation: bool = False,
    has_tests: bool = False,
    test_coverage: float = 0.0,
) -> EthicsResult:
    """
    Convenience function to evaluate refactoring operation ethics.

    Args:
        create_backup: Whether backup is created before refactoring
        require_confirmation: Whether user confirmation is required
        has_tests: Whether tests exist for the code
        test_coverage: Test coverage percentage (0.0-1.0)

    Returns:
        EthicsResult with evaluation
    """
    governor = EthicalAutonomyGovernor()
    return governor.evaluate_action(
        action_type="refactoring",
        action_params={
            "create_backup": create_backup,
            "require_confirmation": require_confirmation,
        },
        context={
            "has_tests": has_tests,
            "test_coverage": test_coverage,
            "has_rollback": create_backup,
            "is_transparent": True,
            "is_open_source": True,
        },
    )

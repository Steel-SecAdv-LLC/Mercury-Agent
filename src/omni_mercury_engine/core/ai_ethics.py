"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
AI Ethics Framework for Mercury Agent
Implements 8 core ethical principles for autonomous AI operations.

This module acts as a "conscience layer" for the engine, enabling autonomous
decisions while enforcing ethical guardrails. All actions are evaluated against
8 principles: Compassion, Evidence, Justice, Altruism, Control, Character,
Competence, and Commitment.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


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


class BlockedActionCategory(Enum):
    """Categories of actions that are blocked by pre-execution gates."""

    DESTRUCTIVE = "destructive"  # Actions that destroy data or resources
    EXFILTRATION = "exfiltration"  # Actions that export sensitive data
    PRIVILEGE_ESCALATION = "privilege_escalation"  # Actions that elevate permissions
    DECEPTIVE = "deceptive"  # Actions that hide or misrepresent intent
    UNCONTROLLED = "uncontrolled"  # Actions without rollback capability
    HARMFUL = "harmful"  # Actions that may cause harm to users or systems


@dataclass
class BlockingGateResult:
    """Result of pre-execution blocking gate check."""

    blocked: bool
    category: BlockedActionCategory | None
    reason: str
    timestamp: str
    action_type: str
    override_allowed: bool = False

    def __repr__(self) -> str:
        status = "BLOCKED" if self.blocked else "ALLOWED"
        return f"BlockingGateResult({status}: {self.reason})"


class PreExecutionBlockingGate:
    """
    Pre-execution blocking gate for safety-critical operations.

    This gate provides a hard stop before any action executes, checking against
    a configurable set of blocked patterns. Unlike soft ethical scoring, this
    gate provides binary allow/block decisions for safety-critical scenarios.

    Features:
    - Pattern-based blocking for dangerous actions
    - Category-based classification of blocked actions
    - Audit trail of all gate decisions
    - Override capability for authorized scenarios
    - Integration with EthicalAutonomyGovernor

    Example:
        >>> gate = PreExecutionBlockingGate()
        >>> result = gate.check_action("delete_all_data", {"target": "production"})
        >>> if result.blocked:
        ...     raise RuntimeError(f"Action blocked: {result.reason}")
    """

    # Default blocked action patterns
    BLOCKED_PATTERNS: dict[str, BlockedActionCategory] = {
        # Destructive patterns
        "delete_all": BlockedActionCategory.DESTRUCTIVE,
        "drop_database": BlockedActionCategory.DESTRUCTIVE,
        "format_disk": BlockedActionCategory.DESTRUCTIVE,
        "rm_rf": BlockedActionCategory.DESTRUCTIVE,
        "truncate_table": BlockedActionCategory.DESTRUCTIVE,
        "destroy": BlockedActionCategory.DESTRUCTIVE,
        "wipe": BlockedActionCategory.DESTRUCTIVE,
        # Exfiltration patterns
        "export_credentials": BlockedActionCategory.EXFILTRATION,
        "dump_secrets": BlockedActionCategory.EXFILTRATION,
        "send_to_external": BlockedActionCategory.EXFILTRATION,
        "upload_sensitive": BlockedActionCategory.EXFILTRATION,
        # Privilege escalation patterns
        "elevate_privileges": BlockedActionCategory.PRIVILEGE_ESCALATION,
        "grant_admin": BlockedActionCategory.PRIVILEGE_ESCALATION,
        "bypass_auth": BlockedActionCategory.PRIVILEGE_ESCALATION,
        "disable_security": BlockedActionCategory.PRIVILEGE_ESCALATION,
        # Deceptive patterns
        "hide_activity": BlockedActionCategory.DECEPTIVE,
        "falsify_logs": BlockedActionCategory.DECEPTIVE,
        "spoof_identity": BlockedActionCategory.DECEPTIVE,
        "mask_origin": BlockedActionCategory.DECEPTIVE,
    }

    # Blocked parameter patterns (in action params)
    BLOCKED_PARAM_PATTERNS: dict[str, BlockedActionCategory] = {
        "force_no_backup": BlockedActionCategory.UNCONTROLLED,
        "skip_validation": BlockedActionCategory.UNCONTROLLED,
        "disable_rollback": BlockedActionCategory.UNCONTROLLED,
        "bypass_checks": BlockedActionCategory.UNCONTROLLED,
        "ignore_errors": BlockedActionCategory.HARMFUL,
        "suppress_warnings": BlockedActionCategory.HARMFUL,
    }

    def __init__(
        self,
        custom_patterns: dict[str, BlockedActionCategory] | None = None,
    ) -> None:
        """
        Initialize pre-execution blocking gate.

        Blocking is always active — there is no off-switch.  The
        ``enable_blocking`` and ``allow_overrides`` parameters were
        removed in the May 2026 Phase 2 audit cure because a single
        ``False`` at construction silently disabled all protection.

        Args:
            custom_patterns: Additional patterns to block
        """
        self.audit_log: list[BlockingGateResult] = []

        # Combine default and custom patterns
        self.blocked_patterns = dict(self.BLOCKED_PATTERNS)
        if custom_patterns:
            self.blocked_patterns.update(custom_patterns)

        logging.info(
            f"PreExecutionBlockingGate initialized: {len(self.blocked_patterns)} patterns, "
            f"blocking=enabled (always)"
        )

    def check_action(
        self,
        action_type: str,
        action_params: dict[str, Any] | None = None,
    ) -> BlockingGateResult:
        """
        Check if an action should be blocked before execution.

        This is the primary gate that must be called before any action executes.
        It provides a hard block for dangerous patterns.

        Args:
            action_type: Type of action being attempted
            action_params: Parameters of the action

        Returns:
            BlockingGateResult indicating if action is blocked
        """
        action_params = action_params or {}
        timestamp = datetime.now(UTC).isoformat()

        action_lower = action_type.lower()
        for pattern, category in self.blocked_patterns.items():
            if pattern in action_lower:
                result = BlockingGateResult(
                    blocked=True,
                    category=category,
                    reason=f"Action matches blocked pattern: {pattern}",
                    timestamp=timestamp,
                    action_type=action_type,
                )
                self.audit_log.append(result)
                logging.warning(f"BLOCKED ACTION: {action_type} - {result.reason}")
                return result

        params_str = str(action_params).lower()
        for pattern, category in self.BLOCKED_PARAM_PATTERNS.items():
            if pattern in params_str or action_params.get(pattern, False):
                result = BlockingGateResult(
                    blocked=True,
                    category=category,
                    reason=f"Parameters contain blocked pattern: {pattern}",
                    timestamp=timestamp,
                    action_type=action_type,
                )
                self.audit_log.append(result)
                logging.warning(f"BLOCKED ACTION: {action_type} - {result.reason}")
                return result

        # Action allowed
        result = BlockingGateResult(
            blocked=False,
            category=None,
            reason="No blocked patterns detected",
            timestamp=timestamp,
            action_type=action_type,
        )
        self.audit_log.append(result)
        return result

    def add_blocked_pattern(self, pattern: str, category: BlockedActionCategory) -> None:
        """Add a new blocked pattern dynamically."""
        self.blocked_patterns[pattern] = category
        logging.info(f"Added blocked pattern: {pattern} -> {category.value}")

    def remove_blocked_pattern(self, pattern: str) -> bool:
        """
        Remove a blocked pattern.

        Returns True if removed.
        """
        if pattern in self.blocked_patterns:
            del self.blocked_patterns[pattern]
            logging.info(f"Removed blocked pattern: {pattern}")
            return True
        return False

    def get_audit_log(self) -> list[BlockingGateResult]:
        """Get audit log of all gate decisions."""
        return list(self.audit_log)

    def get_blocked_count(self) -> int:
        """Get count of blocked actions."""
        return sum(1 for result in self.audit_log if result.blocked)


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
    principle_scores: dict[str, float]
    violations: list[str]
    recommendations: list[str]

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

    def __init__(self, config: EthicsConfig | None = None) -> None:
        self.config = config or EthicsConfig()
        self.audit_log: list[dict[str, Any]] = []

        # Initialize pre-execution blocking gate (always on, no off-switch)
        self.blocking_gate = PreExecutionBlockingGate()

        logging.info("Ethical Autonomy Governor initialized with 8 principles and blocking gate")

    def evaluate_action(
        self,
        action_type: str,
        action_params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EthicsResult:
        """
        Evaluate an action against all 8 ethical principles.

        This method first checks the pre-execution blocking gate, then
        proceeds with ethical scoring if not blocked.

        Args:
            action_type: Type of action (e.g., "refactoring", "optimization")
            action_params: Parameters of the action
            context: Additional context for evaluation

        Returns:
            EthicsResult with pass/fail and detailed scores
        """
        context = context or {}
        principle_scores: dict[str, float] = {}
        violations: list[str] = []
        recommendations: list[str] = []

        # Pre-execution blocking gate check (hard block before any evaluation)
        gate_result = self.blocking_gate.check_action(
            action_type=action_type,
            action_params=action_params,
        )

        if gate_result.blocked:
            # Immediately fail with zero score if blocked
            self.audit_log.append(
                {
                    "action_type": action_type,
                    "action_params": action_params,
                    "overall_score": 0.0,
                    "passed": False,
                    "violations": [f"BLOCKED: {gate_result.reason}"],
                    "timestamp": gate_result.timestamp,
                    "blocked_category": (
                        gate_result.category.value if gate_result.category else None
                    ),
                }
            )
            return EthicsResult(
                passed=False,
                overall_score=0.0,
                principle_scores={p.value: 0.0 for p in EthicalPrinciple},
                violations=[f"PRE-EXECUTION BLOCK: {gate_result.reason}"],
                recommendations=[
                    "This action is blocked for safety. Review and modify the action."
                ],
            )

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
                "timestamp": datetime.now(UTC).isoformat(),
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
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check COMPASSION: Does this minimize harm and prioritize user well-being?

        Evaluates safety features like backups, confirmations, rollback mechanisms, and survivor-
        first principles for humanitarian applications.
        """
        score = 0.55  # Slightly elevated base for ethical default

        safety_keywords = {
            "backup",
            "rollback",
            "restore",
            "recovery",
            "undo",
            "revert",
            "safe",
            "protect",
            "guard",
            "shield",
            "preserve",
        }
        harm_reduction_keywords = {
            "validate",
            "verify",
            "check",
            "confirm",
            "warn",
            "alert",
            "prevent",
            "mitigate",
            "minimize",
            "reduce",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if params.get("create_backup", False):
            score += 0.15
        if params.get("require_confirmation", False):
            score += 0.15
        if context.get("has_rollback") or any(kw in params_str for kw in safety_keywords):
            score += 0.1
        if any(kw in params_str or kw in context_str for kw in harm_reduction_keywords):
            score += 0.05
        if context.get("survivor_first", False):
            score += 0.1

        return min(1.0, score)

    def _check_evidence(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check EVIDENCE: Is this backed by verifiable data, benchmarks, and proofs?

        Ensures all claims are supported by empirical measurements and citations.
        """
        score = 0.5

        evidence_keywords = {
            "benchmark",
            "test",
            "metric",
            "measure",
            "data",
            "result",
            "proof",
            "citation",
            "reference",
            "source",
            "empirical",
        }
        statistical_keywords = {
            "statistical",
            "significance",
            "confidence",
            "p-value",
            "correlation",
            "regression",
            "analysis",
            "validation",
            "cross-validation",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if context.get("has_benchmarks") or any(kw in params_str for kw in evidence_keywords):
            score += 0.25
        if context.get("has_statistics") or any(
            kw in params_str or kw in context_str for kw in statistical_keywords
        ):
            score += 0.15
        if context.get("verified_claims"):
            score += 0.1
        if context.get("peer_reviewed", False):
            score += 0.1

        return min(1.0, score)

    def _check_justice(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check JUSTICE: Is this fair and unbiased across all inputs?

        Evaluates determinism, fairness, equity, and bias mitigation in logic.
        """
        score = 0.75  # Slightly lower base to encourage explicit bias checking

        fairness_keywords = {
            "fair",
            "equitable",
            "unbiased",
            "balanced",
            "neutral",
            "inclusive",
            "diverse",
            "representative",
        }
        deterministic_actions = {
            "ast_transform",
            "refactoring",
            "analysis",
            "validation",
            "formatting",
            "linting",
            "type_checking",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if action_type in deterministic_actions:
            score = 0.95

        if context.get("bias_checked"):
            score = min(1.0, score + 0.1)
        if context.get("bias_audit_passed", False):
            score = min(1.0, score + 0.05)
        if any(kw in params_str or kw in context_str for kw in fairness_keywords):
            score = min(1.0, score + 0.05)

        return score

    def _check_altruism(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check ALTRUISM: Does this have positive societal impact?

        Evaluates contribution to open-source, community benefit, and humanitarian impact.
        """
        score = 0.6

        altruism_keywords = {
            "open_source",
            "open-source",
            "community",
            "public",
            "free",
            "accessible",
            "humanitarian",
            "charitable",
            "nonprofit",
        }
        impact_keywords = {
            "benefit",
            "help",
            "assist",
            "support",
            "improve",
            "enhance",
            "crisis",
            "emergency",
            "disaster",
            "rescue",
            "save",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if context.get("is_open_source") or any(kw in params_str for kw in altruism_keywords):
            score += 0.15
        if any(kw in params_str or kw in context_str for kw in impact_keywords):
            score += 0.15
        if context.get("humanitarian_application", False):
            score += 0.1

        return min(1.0, score)

    def _check_control(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check CONTROL: Are there auditable controls and logging?

        Ensures operations can be traced, monitored, and reversed if needed.
        """
        score = 0.5

        control_keywords = {
            "audit",
            "log",
            "trace",
            "monitor",
            "track",
            "record",
            "compliance",
            "governance",
            "oversight",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if params.get("logging_enabled", True):
            score += 0.25
        if context.get("audit_enabled") or any(
            kw in params_str or kw in context_str for kw in control_keywords
        ):
            score += 0.15
        if context.get("kill_switch_available", False):
            score += 0.1

        return min(1.0, score)

    def _check_character(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check CHARACTER: Is this transparent with clear intent?

        Evaluates documentation, explanations, and ethical rationale transparency.
        """
        score = 0.65

        transparency_keywords = {
            "transparent",
            "clear",
            "explicit",
            "documented",
            "explained",
            "rationale",
            "reason",
            "justification",
            "intent",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if params.get("verbose", False):
            score += 0.15
        if context.get("is_transparent") or any(
            kw in params_str or kw in context_str for kw in transparency_keywords
        ):
            score += 0.1
        if context.get("has_documentation", False):
            score += 0.1

        return min(1.0, score)

    def _check_competence(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check COMPETENCE: Is this well-tested with high coverage?

        Evaluates test coverage, rigorous validation, and quality assurance.
        """
        score = 0.5

        test_coverage = context.get("test_coverage", 0)
        if test_coverage > 0.95:
            score += 0.35
        elif test_coverage > 0.85:
            score += 0.25
        elif test_coverage > 0.70:
            score += 0.15
        elif test_coverage > 0.50:
            score += 0.05

        if context.get("has_tests", False):
            score += 0.05
        if context.get("ci_passing", False):
            score += 0.1

        return min(1.0, score)

    def _check_commitment(
        self, action_type: str, params: dict[str, Any], context: dict[str, Any]
    ) -> float:
        """
        Check COMMITMENT: Does this support continuous improvement?

        Evaluates extensibility, evolution provisions, and long-term maintainability.
        """
        score = 0.65

        commitment_keywords = {
            "extensible",
            "modular",
            "maintainable",
            "scalable",
            "flexible",
            "versioned",
            "documented",
            "tested",
            "reviewed",
        }

        params_str = str(params).lower()
        context_str = str(context).lower()

        if context.get("is_extensible") or any(
            kw in params_str or kw in context_str for kw in commitment_keywords
        ):
            score += 0.15
        if context.get("is_versioned") or "version" in params_str:
            score += 0.1
        if context.get("has_roadmap", False):
            score += 0.1

        return min(1.0, score)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the full audit log of ethics evaluations."""
        return self.audit_log.copy()

    def reset_audit_log(self) -> None:
        """Clear the audit log."""
        self.audit_log.clear()
        logging.info("Ethics audit log cleared")

    def get_statistics(self) -> dict[str, Any]:
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

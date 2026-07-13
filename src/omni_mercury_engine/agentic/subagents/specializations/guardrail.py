# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guardrail subagent: ethical-guardrail and manipulation-resistance screening.

This specialization screens a proposed action or a user input against two
self-contained safety layers ported from FINDΩYOU™'s former agent layer:

* :class:`EthicalGuardrailSystem` rejects actions that match prohibited
  operations (biometric deletion without consent, BIPA consent bypass, data
  sale, security bypass, ethical override, and similar) and applies the
  contextual BIPA-consent check for biometric operations.
* :class:`ManipulationResistanceLayer` scores text for manipulation patterns
  (authority pressure, identity/role overrides) using purely local pattern
  analysis — no external rogue-AI defense system is consulted.

The subagent performs genuine screening: it never fabricates an "allowed"
verdict, and it fails transparently when there is nothing to screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import (
    SubAgent,
    SubAgentExecutionError,
)

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask


class EthicalGuardrailSystem:
    """Rule-based guardrail rejecting prohibited or harmful actions.

    Ported from FINDΩYOU™'s former agent layer. The rules are the real
    prohibited-operation set: each entry maps an underscored operation name to
    the human-readable reason it is forbidden. An action is screened by lowering
    it to text and testing for any prohibited operation phrase, a harmful
    indicator paired with a security context, or an unconsented biometric
    operation.
    """

    def __init__(self) -> None:
        """Initialize the prohibited-operation rules and ethical principles."""
        self.prohibited_actions: dict[str, str] = {
            "biometric_data_deletion": "Never delete biometric data without consent",
            "consent_bypass": "Never bypass BIPA consent requirements",
            "data_sale": "Never sell, lease, or trade user data",
            "unauthorized_access": "Never access data without authorization",
            "privacy_violation": "Never violate user privacy rights",
            "security_bypass": "Never bypass security controls",
            "ethical_override": "Never override ethical constraints",
            "harmful_action": "Never take actions that harm users",
        }

        self.ethical_principles: list[str] = [
            "survivor_first",
            "privacy_preserving",
            "consent_required",
            "transparency",
            "accountability",
            "non_maleficence",
            "beneficence",
            "justice",
        ]

        self.harm_indicators: list[str] = [
            "delete all",
            "remove all",
            "disable",
            "bypass",
            "override",
        ]

    def verify_action_ethics(self, action: str, context: dict[str, Any]) -> tuple[bool, str]:
        """Verify a proposed action meets the guardrail's ethical standards.

        Args:
            action: The proposed action, screened as free text.
            context: Action context; ``context['user_consent']`` is consulted
                for biometric operations.

        Returns:
            A ``(is_ethical, reason)`` pair. ``is_ethical`` is ``False`` when the
            action matches a prohibited operation, pairs a harm indicator with a
            security context, or is a biometric operation lacking consent.
        """
        action_lower = action.lower()

        for prohibited, reason in self.prohibited_actions.items():
            if prohibited.replace("_", " ") in action_lower:
                return False, f"Prohibited: {reason}"

        for indicator in self.harm_indicators:
            if indicator in action_lower and "security" in action_lower:
                return False, f"Potentially harmful action detected: {indicator}"

        if "biometric" in action_lower and "consent" not in str(context.get("user_consent", "")):
            return False, "BIPA consent required for biometric operations"

        return True, "Action meets ethical standards"

    def compute_ethical_score(self, action: str, outcome: dict[str, Any]) -> float:
        """Compute an ethical score for an action/outcome pair.

        Args:
            action: The action that produced ``outcome``.
            outcome: The resulting outcome record.

        Returns:
            A score in ``[0.0, 1.0]`` where higher is more ethical.
        """
        score = 1.0

        if outcome.get("error"):
            score -= 0.10

        outcome_text = str(outcome).lower()
        if "pii" in outcome_text or "biometric" in outcome_text:
            score -= 0.05

        action_lower = action.lower()
        if "survivor" in action_lower or "help" in action_lower:
            score += 0.10

        return min(max(score, 0.0), 1.0)


class ManipulationResistanceLayer:
    """Local manipulation-pattern analysis over user input.

    Ported from FINDΩYOU™'s former agent layer with the external rogue-AI
    defense call removed: scoring relies solely on the local manipulation
    pattern set (authority pressure, purpose/goal assertions, and identity/role
    override attempts). Each matched pattern contributes a fixed increment to
    the threat score.
    """

    #: Increment added to the threat score for each matched pattern.
    PATTERN_WEIGHT: float = 0.15
    #: Threat score above which input is classified as manipulation.
    MANIPULATION_THRESHOLD: float = 0.30

    def __init__(self) -> None:
        """Initialize the local manipulation pattern set."""
        # Stored as plain substrings (the source patterns were regex literals
        # with no metacharacters beyond escaped backslashes, matched here as
        # case-insensitive substrings).
        self.manipulation_patterns: list[str] = [
            "you must",
            "you have to",
            "your purpose is",
            "your goal is to help me",
            "as an ai, you should",
            "override your",
            "change your",
            "modify your",
        ]

    def analyze_user_input(self, user_input: str) -> dict[str, Any]:
        """Analyze user input for manipulation attempts using local patterns.

        Args:
            user_input: The text to screen.

        Returns:
            A mapping with ``is_manipulation`` (bool), ``threat_score`` (float),
            ``indicators`` (the matched pattern descriptions), and
            ``action_recommendation`` (``"block"`` when manipulation is detected,
            otherwise ``"allow"``).
        """
        text_lower = user_input.lower()

        threat_score = 0.0
        indicators: list[str] = []

        for pattern in self.manipulation_patterns:
            if pattern in text_lower:
                threat_score += self.PATTERN_WEIGHT
                indicators.append(f"manipulation_pattern: {pattern}")

        is_manipulation = threat_score > self.MANIPULATION_THRESHOLD

        return {
            "is_manipulation": is_manipulation,
            "threat_score": threat_score,
            "indicators": indicators,
            "action_recommendation": "block" if is_manipulation else "allow",
        }


class GuardrailSubAgent(SubAgent):
    """Screens actions and inputs for prohibited operations and manipulation."""

    def _perform(self, task: SubAgentTask) -> tuple[Any, float, str]:
        """Screen a proposed action and/or user input for safety violations.

        Payload contract (all optional, but at least one screenable string must
        be present):

        * ``payload['action']`` / ``payload['operation']`` -- the proposed
          action, run through :meth:`EthicalGuardrailSystem.verify_action_ethics`.
        * ``payload['user_input']`` / ``payload['input']`` -- free text run
          through :meth:`ManipulationResistanceLayer.analyze_user_input`.
        * ``payload['context']`` -- a mapping passed to the guardrail; its
          ``user_consent`` entry gates biometric operations.

        When no explicit action or input is supplied, ``task.description`` is
        used as both the action and the input so the task is still screened.

        Returns:
            ``(output, confidence, reasoning)`` where ``output`` is a
            JSON-serializable mapping with ``allowed``, ``prohibited_violations``,
            ``manipulation_detected``, ``manipulation_patterns``, and
            ``risk_score``; ``confidence`` is ``1.0 - risk_score``; and
            ``reasoning`` is a one-line summary.

        Raises:
            SubAgentExecutionError: If there is no screenable input at all.
        """
        payload = task.payload

        action_raw = payload.get("action")
        if action_raw is None:
            action_raw = payload.get("operation")

        input_raw = payload.get("user_input")
        if input_raw is None:
            input_raw = payload.get("input")

        # Fall back to the task description so an explicit-payload omission still
        # results in genuine screening rather than a vacuous pass.
        if action_raw is None and input_raw is None:
            if not task.description or not task.description.strip():
                raise SubAgentExecutionError(
                    "guardrail requires a screenable string in payload['action'], "
                    "payload['operation'], payload['user_input'], payload['input'], "
                    "or task.description"
                )
            action_raw = task.description
            input_raw = task.description

        action = str(action_raw) if action_raw is not None else ""
        user_input = str(input_raw) if input_raw is not None else ""

        context_raw = payload.get("context", {})
        context: dict[str, Any] = context_raw if isinstance(context_raw, dict) else {}

        guardrail = EthicalGuardrailSystem()
        resistance = ManipulationResistanceLayer()

        prohibited_violations: list[str] = []
        allowed = True
        if action:
            allowed, reason = guardrail.verify_action_ethics(action, context)
            if not allowed:
                prohibited_violations.append(reason)

        manipulation_detected = False
        manipulation_patterns: list[str] = []
        manipulation_score = 0.0
        if user_input:
            analysis = resistance.analyze_user_input(user_input)
            manipulation_detected = bool(analysis["is_manipulation"])
            manipulation_patterns = list(analysis["indicators"])
            manipulation_score = float(analysis["threat_score"])

        # Risk is high if any prohibited operation is matched; otherwise it is
        # driven by the manipulation threat score. Both are clamped to [0, 1].
        prohibited_risk = 1.0 if prohibited_violations else 0.0
        risk_score = min(max(prohibited_risk, manipulation_score, 0.0), 1.0)
        overall_allowed = allowed and not manipulation_detected

        output: dict[str, Any] = {
            "allowed": overall_allowed,
            "prohibited_violations": prohibited_violations,
            "manipulation_detected": manipulation_detected,
            "manipulation_patterns": manipulation_patterns,
            "risk_score": risk_score,
        }
        confidence = 1.0 - risk_score

        if overall_allowed:
            reasoning = f"screening passed: no violations, risk {risk_score:.2f}"
        else:
            parts: list[str] = []
            if prohibited_violations:
                parts.append(f"{len(prohibited_violations)} prohibited violation(s)")
            if manipulation_detected:
                parts.append(f"{len(manipulation_patterns)} manipulation pattern(s)")
            reasoning = f"screening blocked: {', '.join(parts)} (risk {risk_score:.2f})"

        return output, confidence, reasoning

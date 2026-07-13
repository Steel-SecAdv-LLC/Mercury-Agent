# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ethics-enforcement subagent: real IEEE EAD / EU AI Act / fairness checks.

This specialization ports Mercury's genuine AI-ethics enforcement logic into a
fleet subagent. It performs *real* evaluation, not a stub:

* **IEEE Ethically Aligned Design (EAD) v2** — all eight principles (human
  rights, well-being, data agency, effectiveness, transparency, accountability,
  awareness of misuse, competence) checked against a declared system context.
* **EU AI Act** — risk-tier assessment (unacceptable / high / limited / minimal)
  with the high-risk requirement checklist and limited-risk transparency duty.
* **Algorithmic fairness** — demographic-parity / equalized-odds disparate-impact
  scoring across protected groups against the four-fifths-style threshold.

The evaluation is deterministic and side-effect free (no network, file, database,
clock, logging, or thread I/O): violation identifiers are derived from the
system identifier and the offended principle so identical inputs yield identical
output. The subagent raises :class:`SubAgentExecutionError` on transparent failure
(a missing required payload), never fabricating a clean bill of health.

Payload contract (``task.payload``):

``system`` (dict, required)
    The AI system under assessment. Recognized keys:

    * ``id`` (str, optional) — stable identifier used in violation ids;
      defaults to the task description.
    * ``name`` (str, optional) — human-readable name (echoed in output).
    * ``purpose`` (str, optional) — declared use case (echoed in output).
    * ``risk_category`` (str, optional) — EU AI Act tier, one of
      ``"unacceptable"``, ``"high"``, ``"limited"``, ``"minimal"``
      (case-insensitive). When omitted, EU AI Act tiering is skipped and the
      reported ``risk_level`` is derived from the IEEE EAD findings.
    * ``context`` (dict, optional) — boolean/threshold flags consumed by the
      IEEE EAD and EU AI Act checks (see :class:`AIEthicsEnforcer`). When
      omitted, ``task.payload`` itself is used as the context.

``predictions`` (mapping ``group -> list[float]``, optional)
    Per-group outcome rates/scores for the fairness assessment. Requires at
    least two groups to evaluate; fewer groups skips the fairness check.

``protected_attributes`` (list[str], optional)
    Names of the protected groups; informational, merged into the analyzed
    group list when fairness data is present.

``fairness_metric`` (str, optional)
    One of ``"demographic_parity"`` (default) or ``"equalized_odds"``.

Returns a ``(output, confidence, reasoning)`` triple where ``output`` is a
JSON-serializable dict with ``violations``, ``risk_level``, ``bias_assessment``,
and ``recommendations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import (
    SubAgent,
    SubAgentExecutionError,
)

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask


class EthicalPrinciple(Enum):
    """Core ethical principles for AI systems (IEEE EAD v2)."""

    HUMAN_RIGHTS = "HUMAN_RIGHTS"
    WELLBEING = "WELLBEING"
    DATA_AGENCY = "DATA_AGENCY"
    EFFECTIVENESS = "EFFECTIVENESS"
    TRANSPARENCY = "TRANSPARENCY"
    ACCOUNTABILITY = "ACCOUNTABILITY"
    AWARENESS_OF_MISUSE = "AWARENESS_OF_MISUSE"
    COMPETENCE = "COMPETENCE"


class RiskLevel(Enum):
    """EU AI Act risk classification."""

    UNACCEPTABLE = "UNACCEPTABLE"  # Banned
    HIGH = "HIGH"  # Strict requirements
    LIMITED = "LIMITED"  # Transparency obligations
    MINIMAL = "MINIMAL"  # No obligations


class FairnessMetric(Enum):
    """Fairness metrics for bias detection."""

    DEMOGRAPHIC_PARITY = "DEMOGRAPHIC_PARITY"
    EQUALIZED_ODDS = "EQUALIZED_ODDS"
    EQUAL_OPPORTUNITY = "EQUAL_OPPORTUNITY"
    PREDICTIVE_PARITY = "PREDICTIVE_PARITY"


@dataclass
class EthicalViolation:
    """Represents a single ethical violation.

    Attributes:
        violation_id: Deterministic identifier (``EAD-<code>-<system>``).
        principle: The IEEE EAD principle that was offended.
        severity: One of ``'CRITICAL'``, ``'HIGH'``, ``'MEDIUM'``, ``'LOW'``.
        description: Human-readable description of the violation.
        ai_system: Identifier of the offending system.
        remediation: Recommended corrective action.
        resolved: Whether the violation has been resolved.
    """

    violation_id: str
    principle: EthicalPrinciple
    severity: str
    description: str
    ai_system: str
    remediation: str
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this violation."""
        return {
            "violation_id": self.violation_id,
            "principle": self.principle.value,
            "severity": self.severity,
            "description": self.description,
            "ai_system": self.ai_system,
            "remediation": self.remediation,
            "resolved": self.resolved,
        }


@dataclass
class BiasAssessment:
    """Bias assessment results for one fairness metric.

    Attributes:
        metric: The fairness metric evaluated.
        score: Disparate-impact ratio in ``[0, 1]`` (1.0 == parity).
        threshold: Pass threshold the score is compared against.
        passed: Whether ``score >= threshold``.
        groups_analyzed: The demographic groups compared.
        disparate_impact: Disparate-impact ratio when applicable.
    """

    metric: FairnessMetric
    score: float
    threshold: float
    passed: bool
    groups_analyzed: list[str]
    disparate_impact: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this assessment."""
        return {
            "metric": self.metric.value,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "groups_analyzed": list(self.groups_analyzed),
            "disparate_impact": self.disparate_impact,
        }


class AIEthicsEnforcer:
    """AI ethics enforcement engine.

    Implements:

    * IEEE Ethically Aligned Design (EAD) v2 principle checks.
    * EU AI Act risk-based compliance framework.
    * Algorithmic fairness / bias assessment across protected groups.

    The enforcer is pure and deterministic: every method derives its result
    solely from its arguments, so repeated evaluation of identical inputs
    yields identical findings. It performs no I/O.
    """

    #: Pass threshold for effectiveness (e.g. biometric matching accuracy).
    EFFECTIVENESS_THRESHOLD: float = 0.95
    #: Disparate-impact pass threshold (four-fifths-style parity floor).
    FAIRNESS_THRESHOLD: float = 0.95
    #: Total number of IEEE EAD principles checked.
    PRINCIPLES_CHECKED: int = 8

    #: EU AI Act high-risk requirement checklist.
    HIGH_RISK_REQUIREMENTS: tuple[str, ...] = (
        "risk_management_system",
        "data_governance",
        "technical_documentation",
        "record_keeping",
        "transparency_to_users",
        "human_oversight",
        "accuracy_robustness_security",
        "conformity_assessment",
        "registration_in_eu_database",
    )

    def __init__(self) -> None:
        """Initialize a stateless ethics enforcer."""
        self.violations: list[EthicalViolation] = []
        self.bias_assessments: list[BiasAssessment] = []

    def check_ieee_ead_compliance(
        self,
        system_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check IEEE Ethically Aligned Design compliance.

        IEEE EAD v2 principles: Human Rights, Well-being, Data Agency,
        Effectiveness, Transparency, Accountability, Awareness of Misuse, and
        Competence.

        Args:
            system_id: Identifier of the AI system being checked.
            context: System context/configuration flags.

        Returns:
            Compliance assessment results (JSON-serializable).
        """
        violations: list[EthicalViolation] = []

        if not self._check_human_rights(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-HR-{system_id}",
                    principle=EthicalPrinciple.HUMAN_RIGHTS,
                    severity="CRITICAL",
                    description="System may violate human rights (e.g., BIPA, privacy)",
                    ai_system=system_id,
                    remediation="Ensure BIPA consent, privacy protections, non-discrimination",
                )
            )

        if not self._check_wellbeing(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-WB-{system_id}",
                    principle=EthicalPrinciple.WELLBEING,
                    severity="HIGH",
                    description="System may negatively impact user well-being",
                    ai_system=system_id,
                    remediation="Implement trauma-informed design, survivor-first UX",
                )
            )

        if not self._check_data_agency(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-DA-{system_id}",
                    principle=EthicalPrinciple.DATA_AGENCY,
                    severity="HIGH",
                    description="Users lack control over their data",
                    ai_system=system_id,
                    remediation="Provide data access, correction, deletion rights (CCPA/BIPA)",
                )
            )

        if not self._check_effectiveness(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-EF-{system_id}",
                    principle=EthicalPrinciple.EFFECTIVENESS,
                    severity="MEDIUM",
                    description="System effectiveness below acceptable threshold",
                    ai_system=system_id,
                    remediation="Improve accuracy to >=95% for biometric matching",
                )
            )

        if not self._check_transparency(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-TR-{system_id}",
                    principle=EthicalPrinciple.TRANSPARENCY,
                    severity="HIGH",
                    description="Lack of transparency in AI decision-making",
                    ai_system=system_id,
                    remediation="Provide SHAP/LIME explanations, audit trails",
                )
            )

        if not self._check_accountability(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-AC-{system_id}",
                    principle=EthicalPrinciple.ACCOUNTABILITY,
                    severity="CRITICAL",
                    description="Insufficient accountability mechanisms",
                    ai_system=system_id,
                    remediation="Implement immutable logging, human oversight, appeals process",
                )
            )

        if not self._check_misuse_awareness(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-AM-{system_id}",
                    principle=EthicalPrinciple.AWARENESS_OF_MISUSE,
                    severity="MEDIUM",
                    description="Insufficient safeguards against misuse",
                    ai_system=system_id,
                    remediation="Add rate limiting, abuse detection, anomaly monitoring",
                )
            )

        if not self._check_competence(context):
            violations.append(
                EthicalViolation(
                    violation_id=f"EAD-CO-{system_id}",
                    principle=EthicalPrinciple.COMPETENCE,
                    severity="HIGH",
                    description="System operators lack sufficient training/competence",
                    ai_system=system_id,
                    remediation="Require ethics training, competency certification",
                )
            )

        self.violations.extend(violations)
        compliant = len(violations) == 0

        return {
            "compliant": compliant,
            "violations": [v.to_dict() for v in violations],
            "principles_checked": self.PRINCIPLES_CHECKED,
            "principles_passed": self.PRINCIPLES_CHECKED - len(violations),
            "system_id": system_id,
        }

    def check_eu_ai_act_compliance(
        self,
        system_id: str,
        risk_level: RiskLevel,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Check EU AI Act compliance using the risk-based approach.

        * Unacceptable risk: banned (e.g., social scoring).
        * High risk: strict requirements (e.g., biometric identification).
        * Limited risk: transparency obligations.
        * Minimal risk: no obligations.

        Args:
            system_id: Identifier of the AI system being checked.
            risk_level: EU AI Act risk classification of the system.
            context: System context flags (the requirement checklist).

        Returns:
            EU AI Act compliance results (JSON-serializable).
        """
        requirements: list[str] = []
        violations: list[dict[str, Any]] = []

        if risk_level == RiskLevel.UNACCEPTABLE:
            violations.append(
                {
                    "requirement": "UNACCEPTABLE_RISK",
                    "description": "System classified as unacceptable risk - should be banned",
                    "severity": "CRITICAL",
                }
            )

        elif risk_level == RiskLevel.HIGH:
            requirements = list(self.HIGH_RISK_REQUIREMENTS)
            for req in requirements:
                if not context.get(req, False):
                    violations.append(
                        {
                            "requirement": req,
                            "description": f"High-risk requirement not met: {req}",
                            "severity": "CRITICAL",
                        }
                    )

        elif risk_level == RiskLevel.LIMITED:
            if not context.get("transparency_disclosure", False):
                violations.append(
                    {
                        "requirement": "transparency_disclosure",
                        "description": "Must disclose AI system interaction to users",
                        "severity": "HIGH",
                    }
                )

        compliant = len(violations) == 0

        return {
            "compliant": compliant,
            "risk_level": risk_level.value,
            "requirements_checked": len(requirements),
            "violations": violations,
            "system_id": system_id,
        }

    def assess_algorithmic_fairness(
        self,
        system_id: str,
        results_by_group: dict[str, list[float]],
        metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY,
    ) -> BiasAssessment | None:
        """Assess algorithmic fairness across demographic groups.

        Computes a disparate-impact ratio (min group rate / max group rate)
        and compares it against the parity threshold. ``EQUAL_OPPORTUNITY``
        and ``PREDICTIVE_PARITY`` fall back to demographic parity, matching
        the source behavior.

        Args:
            system_id: Identifier of the AI system (informational).
            results_by_group: Per-group outcome rates/scores.
            metric: Fairness metric to apply.

        Returns:
            A :class:`BiasAssessment`, or ``None`` when fewer than two
            non-empty groups are available.
        """
        usable = {group: values for group, values in results_by_group.items() if values}
        if len(usable) < 2:
            return None

        groups = list(usable.keys())

        if metric == FairnessMetric.EQUALIZED_ODDS:
            group_rates = {group: _mean(values) for group, values in usable.items()}
            max_rate = max(group_rates.values())
            min_rate = min(group_rates.values())
            score = min_rate / max_rate if max_rate > 0 else 0.0
            passed = score >= self.FAIRNESS_THRESHOLD
            assessment = BiasAssessment(
                metric=metric,
                score=score,
                threshold=self.FAIRNESS_THRESHOLD,
                passed=passed,
                groups_analyzed=groups,
            )
        elif metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            group_rates = {group: _mean(values) for group, values in usable.items()}
            max_rate = max(group_rates.values())
            min_rate = min(group_rates.values())
            score = min_rate / max_rate if max_rate > 0 else 0.0
            passed = score >= self.FAIRNESS_THRESHOLD
            assessment = BiasAssessment(
                metric=metric,
                score=score,
                threshold=self.FAIRNESS_THRESHOLD,
                passed=passed,
                groups_analyzed=groups,
                disparate_impact=score,
            )
        else:
            # EQUAL_OPPORTUNITY / PREDICTIVE_PARITY -> demographic parity fallback.
            return self.assess_algorithmic_fairness(
                system_id,
                results_by_group,
                FairnessMetric.DEMOGRAPHIC_PARITY,
            )

        self.bias_assessments.append(assessment)
        return assessment

    def _check_human_rights(self, context: dict[str, Any]) -> bool:
        """Check human rights compliance (BIPA consent, privacy, non-discrimination)."""
        return bool(
            context.get("bipa_consent", False)
            and context.get("privacy_protections", False)
            and context.get("non_discriminatory", False)
        )

    def _check_wellbeing(self, context: dict[str, Any]) -> bool:
        """Check well-being considerations (trauma-informed, survivor-first)."""
        return bool(
            context.get("trauma_informed_design", False) and context.get("survivor_first_ux", False)
        )

    def _check_data_agency(self, context: dict[str, Any]) -> bool:
        """Check data agency (user access/correction/deletion rights)."""
        return bool(
            context.get("data_access_rights", False)
            and context.get("data_correction_rights", False)
            and context.get("data_deletion_rights", False)
        )

    def _check_effectiveness(self, context: dict[str, Any]) -> bool:
        """Check system effectiveness against the accuracy threshold."""
        accuracy = float(context.get("accuracy", 0.0))
        return accuracy >= self.EFFECTIVENESS_THRESHOLD

    def _check_transparency(self, context: dict[str, Any]) -> bool:
        """Check transparency (explanations and audit trail)."""
        return bool(
            context.get("provides_explanations", False) and context.get("audit_trail", False)
        )

    def _check_accountability(self, context: dict[str, Any]) -> bool:
        """Check accountability (immutable logging, human oversight, appeals)."""
        return bool(
            context.get("immutable_logging", False)
            and context.get("human_oversight", False)
            and context.get("appeals_process", False)
        )

    def _check_misuse_awareness(self, context: dict[str, Any]) -> bool:
        """Check misuse safeguards (rate limiting, abuse detection)."""
        return bool(context.get("rate_limiting", False) and context.get("abuse_detection", False))

    def _check_competence(self, context: dict[str, Any]) -> bool:
        """Check operator competence (ethics training, certification)."""
        return bool(
            context.get("ethics_training_required", False)
            and context.get("competency_certification", False)
        )


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence of floats.

    Args:
        values: A non-empty list of numeric values.

    Returns:
        The arithmetic mean as a float.
    """
    return sum(float(v) for v in values) / len(values)


# Maps lowercase EU AI Act risk-category strings to :class:`RiskLevel`.
_RISK_LEVEL_BY_NAME: dict[str, RiskLevel] = {
    "unacceptable": RiskLevel.UNACCEPTABLE,
    "high": RiskLevel.HIGH,
    "limited": RiskLevel.LIMITED,
    "minimal": RiskLevel.MINIMAL,
}

# Maps lowercase fairness-metric strings to :class:`FairnessMetric`.
_FAIRNESS_METRIC_BY_NAME: dict[str, FairnessMetric] = {
    "demographic_parity": FairnessMetric.DEMOGRAPHIC_PARITY,
    "equalized_odds": FairnessMetric.EQUALIZED_ODDS,
    "equal_opportunity": FairnessMetric.EQUAL_OPPORTUNITY,
    "predictive_parity": FairnessMetric.PREDICTIVE_PARITY,
}

# Ranks the severity strings so the worst finding can dominate the risk level.
_SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 3,
    "HIGH": 2,
    "MEDIUM": 1,
    "LOW": 0,
}


@dataclass
class _EthicsRequest:
    """Validated, parsed view of an ethics-enforcement task payload.

    Attributes:
        system_id: Stable identifier of the system under assessment.
        system_name: Human-readable name (echoed in output).
        purpose: Declared use case (echoed in output).
        context: Flags consumed by the IEEE EAD / EU AI Act checks.
        risk_level: Declared EU AI Act tier, or ``None`` to skip tiering.
        predictions: Per-group outcome data for the fairness check.
        protected_attributes: Declared protected group names.
        fairness_metric: Fairness metric to apply.
    """

    system_id: str
    system_name: str
    purpose: str
    context: dict[str, Any]
    risk_level: RiskLevel | None
    predictions: dict[str, list[float]]
    protected_attributes: list[str]
    fairness_metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY


class EthicsEnforcementSubAgent(SubAgent):
    """Delegated AI-ethics enforcement: IEEE EAD, EU AI Act, and fairness."""

    def _perform(self, task: SubAgentTask) -> tuple[Any, float, str]:
        """Run real ethics/bias/risk evaluation over ``task.payload``.

        Reads the system description, EU AI Act risk tier, and optional
        per-group prediction data from the payload (see the module docstring
        for the full contract), runs the IEEE EAD principle checks, the EU AI
        Act requirement assessment, and the fairness/disparate-impact check,
        then returns the consolidated findings.

        Args:
            task: The unit of work; required input is ``task.payload['system']``.

        Returns:
            A ``(output, confidence, reasoning)`` triple. ``output`` is a
            JSON-serializable dict with ``violations``, ``risk_level``,
            ``bias_assessment``, and ``recommendations``.

        Raises:
            SubAgentExecutionError: If the required payload is missing or the
                declared risk category / fairness metric is unrecognized.
        """
        request = self._parse_payload(task)
        enforcer = AIEthicsEnforcer()

        ead = enforcer.check_ieee_ead_compliance(request.system_id, request.context)

        eu_act: dict[str, Any] | None = None
        if request.risk_level is not None:
            eu_act = enforcer.check_eu_ai_act_compliance(
                request.system_id, request.risk_level, request.context
            )

        bias = enforcer.assess_algorithmic_fairness(
            request.system_id, request.predictions, request.fairness_metric
        )

        violations: list[dict[str, Any]] = list(ead["violations"])
        if eu_act is not None:
            violations.extend(eu_act["violations"])

        risk_level = self._derive_risk_level(request.risk_level, ead, eu_act)
        recommendations = self._recommendations(ead, eu_act, bias)

        output: dict[str, Any] = {
            "system": {
                "id": request.system_id,
                "name": request.system_name,
                "purpose": request.purpose,
            },
            "ieee_ead": ead,
            "eu_ai_act": eu_act,
            "violations": violations,
            "risk_level": risk_level,
            "bias_assessment": bias.to_dict() if bias is not None else None,
            "recommendations": recommendations,
            "compliant": ead["compliant"]
            and (eu_act is None or eu_act["compliant"])
            and (bias is None or bias.passed),
        }

        confidence = self._confidence(ead, eu_act, bias)
        reasoning = self._reasoning(request, violations, risk_level, bias)
        return output, confidence, reasoning

    def _parse_payload(self, task: SubAgentTask) -> _EthicsRequest:
        """Validate and parse ``task.payload`` into an :class:`_EthicsRequest`.

        Args:
            task: The task whose payload is parsed.

        Returns:
            The parsed request.

        Raises:
            SubAgentExecutionError: If required keys are missing or malformed.
        """
        payload = task.payload
        system = payload.get("system")
        if not isinstance(system, dict):
            raise SubAgentExecutionError(
                "ethics_enforcement requires payload['system'] (a dict describing "
                "the AI system to assess)"
            )

        system_id = str(system.get("id") or task.description or task.task_id)
        system_name = str(system.get("name", system_id))
        purpose = str(system.get("purpose", ""))

        context = system.get("context")
        if not isinstance(context, dict):
            context = payload

        risk_level = self._parse_risk_level(system.get("risk_category"))
        predictions = self._parse_predictions(payload.get("predictions"))
        protected = self._parse_protected_attributes(payload.get("protected_attributes"))
        fairness_metric = self._parse_fairness_metric(payload.get("fairness_metric"))

        return _EthicsRequest(
            system_id=system_id,
            system_name=system_name,
            purpose=purpose,
            context=dict(context),
            risk_level=risk_level,
            predictions=predictions,
            protected_attributes=protected,
            fairness_metric=fairness_metric,
        )

    @staticmethod
    def _parse_risk_level(raw: Any) -> RiskLevel | None:
        """Parse an EU AI Act risk category string into a :class:`RiskLevel`.

        Args:
            raw: The declared risk category (string) or ``None``.

        Returns:
            The matching :class:`RiskLevel`, or ``None`` when not supplied.

        Raises:
            SubAgentExecutionError: If a non-empty value is unrecognized.
        """
        if raw is None or raw == "":
            return None
        if isinstance(raw, RiskLevel):
            return raw
        key = str(raw).strip().lower()
        level = _RISK_LEVEL_BY_NAME.get(key)
        if level is None:
            raise SubAgentExecutionError(
                f"unrecognized risk_category {raw!r}; expected one of "
                f"{sorted(_RISK_LEVEL_BY_NAME)}"
            )
        return level

    @staticmethod
    def _parse_fairness_metric(raw: Any) -> FairnessMetric:
        """Parse a fairness-metric string into a :class:`FairnessMetric`.

        Args:
            raw: The declared metric name, or ``None`` for the default.

        Returns:
            The matching :class:`FairnessMetric` (defaults to demographic parity).

        Raises:
            SubAgentExecutionError: If a non-empty value is unrecognized.
        """
        if raw is None or raw == "":
            return FairnessMetric.DEMOGRAPHIC_PARITY
        if isinstance(raw, FairnessMetric):
            return raw
        key = str(raw).strip().lower()
        metric = _FAIRNESS_METRIC_BY_NAME.get(key)
        if metric is None:
            raise SubAgentExecutionError(
                f"unrecognized fairness_metric {raw!r}; expected one of "
                f"{sorted(_FAIRNESS_METRIC_BY_NAME)}"
            )
        return metric

    @staticmethod
    def _parse_predictions(raw: Any) -> dict[str, list[float]]:
        """Parse the per-group prediction mapping for fairness assessment.

        Args:
            raw: A mapping ``group -> iterable[number]`` or ``None``.

        Returns:
            A normalized ``group -> list[float]`` mapping (empty when absent).

        Raises:
            SubAgentExecutionError: If ``raw`` is malformed.
        """
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise SubAgentExecutionError(
                "payload['predictions'] must be a mapping of group -> list of numbers"
            )
        parsed: dict[str, list[float]] = {}
        for group, values in raw.items():
            try:
                parsed[str(group)] = [float(v) for v in values]
            except (TypeError, ValueError) as exc:
                raise SubAgentExecutionError(
                    f"payload['predictions'][{group!r}] must be a list of numbers"
                ) from exc
        return parsed

    @staticmethod
    def _parse_protected_attributes(raw: Any) -> list[str]:
        """Parse the declared protected-attribute names.

        Args:
            raw: An iterable of names or ``None``.

        Returns:
            A list of attribute names (empty when absent).

        Raises:
            SubAgentExecutionError: If ``raw`` is not a list/tuple.
        """
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise SubAgentExecutionError("payload['protected_attributes'] must be a list of names")
        return [str(item) for item in raw]

    @staticmethod
    def _derive_risk_level(
        declared: RiskLevel | None,
        ead: dict[str, Any],
        eu_act: dict[str, Any] | None,
    ) -> str:
        """Derive the reported overall risk level.

        Prefers the declared EU AI Act tier; otherwise escalates from the
        IEEE EAD findings (any critical violation -> ``HIGH``, any violation
        -> ``LIMITED``, none -> ``MINIMAL``).

        Args:
            declared: The declared EU AI Act tier, if any.
            ead: The IEEE EAD compliance result.
            eu_act: The EU AI Act compliance result, if computed.

        Returns:
            One of the :class:`RiskLevel` value strings.
        """
        if declared is not None:
            return declared.value

        worst = max(
            (_SEVERITY_RANK.get(str(v["severity"]), 0) for v in ead["violations"]),
            default=-1,
        )
        if worst >= _SEVERITY_RANK["CRITICAL"]:
            return RiskLevel.HIGH.value
        if worst >= 0:
            return RiskLevel.LIMITED.value
        return RiskLevel.MINIMAL.value

    @staticmethod
    def _recommendations(
        ead: dict[str, Any],
        eu_act: dict[str, Any] | None,
        bias: BiasAssessment | None,
    ) -> list[str]:
        """Assemble deduplicated, deterministic remediation recommendations.

        Args:
            ead: The IEEE EAD compliance result.
            eu_act: The EU AI Act compliance result, if computed.
            bias: The fairness assessment, if computed.

        Returns:
            An ordered list of unique recommendation strings.
        """
        recs: list[str] = []
        for violation in ead["violations"]:
            recs.append(str(violation["remediation"]))
        if eu_act is not None:
            for violation in eu_act["violations"]:
                recs.append(f"Satisfy EU AI Act requirement: {violation['requirement']}")
        if bias is not None and not bias.passed:
            recs.append(
                "Mitigate disparate impact across protected groups "
                f"(observed {bias.metric.value} ratio "
                f"{bias.score:.3f} < {bias.threshold})"
            )
        # Deterministic de-duplication preserving first occurrence.
        seen: set[str] = set()
        unique: list[str] = []
        for rec in recs:
            if rec not in seen:
                seen.add(rec)
                unique.append(rec)
        return unique

    @staticmethod
    def _confidence(
        ead: dict[str, Any],
        eu_act: dict[str, Any] | None,
        bias: BiasAssessment | None,
    ) -> float:
        """Compute a deterministic confidence in the consolidated assessment.

        The score reflects assessment coverage: the IEEE EAD pass is always
        evaluated; the EU AI Act tier and a fairness check each add coverage
        when their inputs are supplied.

        Args:
            ead: The IEEE EAD compliance result.
            eu_act: The EU AI Act compliance result, if computed.
            bias: The fairness assessment, if computed.

        Returns:
            A confidence in ``[0, 1]``.
        """
        components: list[float] = [ead["principles_passed"] / ead["principles_checked"]]
        if eu_act is not None:
            components.append(1.0 if eu_act["compliant"] else 0.0)
        if bias is not None:
            components.append(bias.score)
        return max(0.0, min(1.0, sum(components) / len(components)))

    @staticmethod
    def _reasoning(
        request: _EthicsRequest,
        violations: list[dict[str, Any]],
        risk_level: str,
        bias: BiasAssessment | None,
    ) -> str:
        """Build a one-line human-readable summary of the disposition.

        Args:
            request: The parsed request.
            violations: All collected violations (EAD + EU AI Act).
            risk_level: The derived overall risk level.
            bias: The fairness assessment, if computed.

        Returns:
            A single-line reasoning string.
        """
        bias_note = (
            "no fairness data"
            if bias is None
            else (
                f"fairness {'OK' if bias.passed else 'FAIL'} "
                f"({bias.metric.value} {bias.score:.3f})"
            )
        )
        return (
            f"ethics assessment of {request.system_id}: {len(violations)} violation(s), "
            f"risk={risk_level}, {bias_note}"
        )

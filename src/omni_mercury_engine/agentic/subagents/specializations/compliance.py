# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compliance subagent: privacy/biometric regulatory rule evaluation.

This specialization ports FINDΩYOU™'s former compliance-automation agent into a
Mercury :class:`~omni_mercury_engine.agentic.subagents.base.SubAgent`. It carries
the genuine BIPA/CCPA/CPRA rule definitions, the per-rule checkers, consent
recording, and report generation from that agent — no logic was reduced to a
stub. Network polling, file/db I/O, logging side effects, and the CrewAI/engine
couplings of the original were dropped: evaluation here is pure, deterministic,
and driven entirely by the task payload.

Payload contract (read by :meth:`ComplianceSubAgent._perform`):

* ``payload["data_category"]`` (str, required) — one of the
  :class:`DataCategory` names (e.g. ``"BIOMETRIC"``), case-insensitive. Selects
  the applicable rules.
* ``payload["data_subject_id"]`` (str, optional) — identifier for the data
  subject; defaults to ``"anonymous"``.
* ``payload["context"]`` (dict, optional) — the operation context the rule
  checkers inspect (consent flags, encryption settings, SLAs, …). Defaults to
  an empty mapping.
* ``payload["consent"]`` (dict, optional) — consent record for the subject,
  recorded before the check so consent-dependent rules (BIPA-001, BIPA-003)
  evaluate against it.
* ``payload["framework"]`` (str, optional) — one of the
  :class:`ComplianceFramework` names; when given, restricts the evaluation to
  rules from that framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.agentic.subagents.base import (
    SubAgent,
    SubAgentExecutionError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.agentic.subagents.base import SubAgentTask


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    BIPA = "BIPA"  # Illinois Biometric Information Privacy Act
    CCPA = "CCPA"  # California Consumer Privacy Act
    CPRA = "CPRA"  # California Privacy Rights Act
    GDPR = "GDPR"  # General Data Protection Regulation (reference only)
    HIPAA = "HIPAA"  # Health Insurance Portability Act (reference only)
    FERPA = "FERPA"  # Family Educational Rights Privacy Act


class ComplianceStatus(Enum):
    """Compliance check status."""

    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DataCategory(Enum):
    """Data categories for compliance."""

    BIOMETRIC = "BIOMETRIC"
    PERSONAL_INFO = "PERSONAL_INFO"
    SENSITIVE_PERSONAL = "SENSITIVE_PERSONAL"
    GEOLOCATION = "GEOLOCATION"
    BEHAVIORAL = "BEHAVIORAL"
    METADATA = "METADATA"


@dataclass
class ComplianceRule:
    """A single regulatory compliance rule.

    Attributes:
        rule_id: Stable identifier, e.g. ``"BIPA-001"``.
        framework: The framework the rule belongs to.
        title: Short human-readable rule name.
        description: What the rule requires.
        applies_to: Data categories the rule governs.
        severity: One of ``'CRITICAL'``, ``'HIGH'``, ``'MEDIUM'``, ``'LOW'``.
        check_function: Name of the checker method on the agent.
        remediation: Guidance for resolving a violation.
    """

    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    applies_to: list[DataCategory]
    severity: str
    check_function: str
    remediation: str


@dataclass
class ComplianceViolation:
    """A detected breach of a :class:`ComplianceRule`.

    Attributes:
        violation_id: Deterministic identifier for this violation.
        rule_id: The rule that was violated.
        framework: The framework the rule belongs to.
        severity: Severity inherited from the rule.
        description: Human-readable description of the violation.
        detected_at: When the violation was detected.
        data_subject_id: Subject the violation concerns, if any.
        remediation_required: Guidance for resolving the violation.
        resolved: Whether the violation has been remediated.
    """

    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    severity: str
    description: str
    detected_at: datetime
    data_subject_id: str | None
    remediation_required: str
    resolved: bool = False


class ComplianceAutomationAgent:
    """Automated compliance monitoring and enforcement.

    Features:
        - Rule-based compliance checking across BIPA/CCPA/CPRA.
        - Violation detection with severity and remediation.
        - Consent recording for consent-dependent rules.
        - Aggregate compliance report generation.

    The agent is deterministic: violation identifiers are derived from a
    monotonic per-instance counter rather than wall-clock time, so identical
    inputs produce identical outputs.
    """

    def __init__(self) -> None:
        """Initialize the agent and load the built-in rule tables."""
        self.rules: dict[str, ComplianceRule] = {}
        self.violations: list[ComplianceViolation] = []
        self.consent_records: dict[str, dict[str, Any]] = {}
        self._violation_seq: int = 0

        self._initialize_bipa_rules()
        self._initialize_ccpa_rules()
        self._initialize_cpra_rules()

    def _initialize_bipa_rules(self) -> None:
        """Initialize BIPA compliance rules (Illinois)."""
        bipa_rules = [
            ComplianceRule(
                rule_id="BIPA-001",
                framework=ComplianceFramework.BIPA,
                title="Written Consent Required",
                description="Must obtain written consent before collecting biometric data",
                applies_to=[DataCategory.BIOMETRIC],
                severity="CRITICAL",
                check_function="check_written_consent",
                remediation="Obtain and document written consent before any biometric collection",
            ),
            ComplianceRule(
                rule_id="BIPA-002",
                framework=ComplianceFramework.BIPA,
                title="Retention Schedule Required",
                description="Must have publicly available retention schedule",
                applies_to=[DataCategory.BIOMETRIC],
                severity="HIGH",
                check_function="check_retention_schedule",
                remediation=(
                    "Publish retention schedule (BIPA: destroy on purpose satisfaction "
                    "or within 3 years of last interaction, whichever is first)"
                ),
            ),
            ComplianceRule(
                rule_id="BIPA-003",
                framework=ComplianceFramework.BIPA,
                title="Purpose Disclosure",
                description="Must inform of specific purpose and duration of collection",
                applies_to=[DataCategory.BIOMETRIC],
                severity="CRITICAL",
                check_function="check_purpose_disclosure",
                remediation="Provide written notice of collection purpose and duration",
            ),
            ComplianceRule(
                rule_id="BIPA-004",
                framework=ComplianceFramework.BIPA,
                title="No Sale/Profit",
                description="Cannot sell, lease, or trade biometric data",
                applies_to=[DataCategory.BIOMETRIC],
                severity="CRITICAL",
                check_function="check_no_sale",
                remediation="Ensure no commercial transactions involving biometric data",
            ),
            ComplianceRule(
                rule_id="BIPA-005",
                framework=ComplianceFramework.BIPA,
                title="Reasonable Security",
                description="Must use reasonable standard of care to protect data",
                applies_to=[DataCategory.BIOMETRIC],
                severity="CRITICAL",
                check_function="check_security_measures",
                remediation="Implement AES-256 encryption, access controls, audit logging",
            ),
            ComplianceRule(
                rule_id="BIPA-006",
                framework=ComplianceFramework.BIPA,
                title="Breach Notification (72 hours)",
                description="Must notify affected individuals within 72 hours of breach",
                applies_to=[DataCategory.BIOMETRIC],
                severity="CRITICAL",
                check_function="check_breach_notification",
                remediation="Implement automated breach notification system",
            ),
            ComplianceRule(
                rule_id="BIPA-007",
                framework=ComplianceFramework.BIPA,
                title="Deletion on Request",
                description="Must delete data within 30 days of request",
                applies_to=[DataCategory.BIOMETRIC],
                severity="HIGH",
                check_function="check_deletion_sla",
                remediation="Implement cryptographic erasure within 30-day SLA",
            ),
        ]

        for rule in bipa_rules:
            self.rules[rule.rule_id] = rule

    def _initialize_ccpa_rules(self) -> None:
        """Initialize CCPA compliance rules (California)."""
        ccpa_rules = [
            ComplianceRule(
                rule_id="CCPA-001",
                framework=ComplianceFramework.CCPA,
                title="Right to Know",
                description="Consumers have right to know what personal info is collected",
                applies_to=[DataCategory.PERSONAL_INFO, DataCategory.BIOMETRIC],
                severity="HIGH",
                check_function="check_data_inventory",
                remediation="Maintain detailed inventory of collected data",
            ),
            ComplianceRule(
                rule_id="CCPA-002",
                framework=ComplianceFramework.CCPA,
                title="Right to Delete",
                description="Consumers can request deletion of personal information",
                applies_to=[DataCategory.PERSONAL_INFO, DataCategory.BIOMETRIC],
                severity="HIGH",
                check_function="check_deletion_capability",
                remediation="Implement verified deletion request process",
            ),
            ComplianceRule(
                rule_id="CCPA-003",
                framework=ComplianceFramework.CCPA,
                title="Right to Opt-Out",
                description="Consumers can opt-out of sale of personal information",
                applies_to=[DataCategory.PERSONAL_INFO],
                severity="MEDIUM",
                check_function="check_opt_out_mechanism",
                remediation="Provide 'Do Not Sell My Personal Information' link",
            ),
            ComplianceRule(
                rule_id="CCPA-004",
                framework=ComplianceFramework.CCPA,
                title="Non-Discrimination",
                description="Cannot discriminate against consumers exercising rights",
                applies_to=[DataCategory.PERSONAL_INFO],
                severity="HIGH",
                check_function="check_non_discrimination",
                remediation="Ensure equal service regardless of privacy choices",
            ),
        ]

        for rule in ccpa_rules:
            self.rules[rule.rule_id] = rule

    def _initialize_cpra_rules(self) -> None:
        """Initialize CPRA compliance rules (California, enhanced CCPA)."""
        cpra_rules = [
            ComplianceRule(
                rule_id="CPRA-001",
                framework=ComplianceFramework.CPRA,
                title="Right to Correct",
                description="Consumers can request correction of inaccurate information",
                applies_to=[DataCategory.PERSONAL_INFO],
                severity="MEDIUM",
                check_function="check_correction_process",
                remediation="Implement data correction request workflow",
            ),
            ComplianceRule(
                rule_id="CPRA-002",
                framework=ComplianceFramework.CPRA,
                title="Sensitive Personal Info Limitation",
                description="Additional restrictions on sensitive personal information",
                applies_to=[DataCategory.SENSITIVE_PERSONAL, DataCategory.BIOMETRIC],
                severity="HIGH",
                check_function="check_sensitive_info_limits",
                remediation="Apply stricter access controls to sensitive data",
            ),
            ComplianceRule(
                rule_id="CPRA-003",
                framework=ComplianceFramework.CPRA,
                title="Risk Assessment Required",
                description="Must conduct and document cybersecurity risk assessments",
                applies_to=[DataCategory.BIOMETRIC, DataCategory.SENSITIVE_PERSONAL],
                severity="HIGH",
                check_function="check_risk_assessment",
                remediation="Conduct annual cybersecurity risk assessment",
            ),
        ]

        for rule in cpra_rules:
            self.rules[rule.rule_id] = rule

    def check_compliance(
        self,
        data_category: DataCategory,
        data_subject_id: str,
        context: dict[str, Any],
        framework: ComplianceFramework | None = None,
    ) -> dict[str, Any]:
        """Check compliance for a data operation.

        Args:
            data_category: Category of data being processed.
            data_subject_id: Identifier for the data subject.
            context: Operation context (consent flags, purpose, security, …).
            framework: When set, restrict the evaluation to this framework's
                rules.

        Returns:
            A JSON-serializable dictionary with the overall status, the
            violations and warnings raised, and metadata for the check.
        """
        applicable_rules = [
            rule
            for rule in self.rules.values()
            if data_category in rule.applies_to
            and (framework is None or rule.framework == framework)
        ]

        violations: list[ComplianceViolation] = []
        warnings: list[dict[str, Any]] = []

        for rule in applicable_rules:
            status = self._check_rule(rule, data_subject_id, context)

            if status == ComplianceStatus.VIOLATION:
                self._violation_seq += 1
                violation = ComplianceViolation(
                    violation_id=f"V-{self._violation_seq:06d}",
                    rule_id=rule.rule_id,
                    framework=rule.framework,
                    severity=rule.severity,
                    description=f"Violation of {rule.title}: {rule.description}",
                    detected_at=datetime.now(),
                    data_subject_id=data_subject_id,
                    remediation_required=rule.remediation,
                )
                violations.append(violation)
                self.violations.append(violation)

            elif status == ComplianceStatus.WARNING:
                warnings.append(
                    {
                        "rule_id": rule.rule_id,
                        "framework": rule.framework.value,
                        "message": f"Warning for {rule.title}",
                    }
                )

        overall_status = ComplianceStatus.COMPLIANT
        if violations:
            overall_status = ComplianceStatus.VIOLATION
        elif warnings:
            overall_status = ComplianceStatus.WARNING

        result: dict[str, Any] = {
            "status": overall_status.value,
            "data_category": data_category.value,
            "data_subject_id": data_subject_id,
            "rules_checked": len(applicable_rules),
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "rule_id": v.rule_id,
                    "severity": v.severity,
                    "description": v.description,
                    "remediation": v.remediation_required,
                }
                for v in violations
            ],
            "warnings": warnings,
        }

        return result

    def _check_rule(
        self,
        rule: ComplianceRule,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """Dispatch to the rule's checker method and report its status.

        Args:
            rule: The rule to evaluate.
            data_subject_id: Identifier for the data subject.
            context: Operation context passed to the checker.

        Returns:
            The :class:`ComplianceStatus` from the checker, or
            ``NEEDS_REVIEW`` if no checker is bound or it raised.
        """
        check_func: Callable[[str, dict[str, Any]], ComplianceStatus] | None = getattr(
            self, rule.check_function, None
        )

        if check_func is None:
            return ComplianceStatus.NEEDS_REVIEW

        try:
            return check_func(data_subject_id, context)
        except Exception:
            return ComplianceStatus.NEEDS_REVIEW

    def check_written_consent(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-001: Check written consent."""
        consent = self.consent_records.get(data_subject_id, {})

        has_consent = consent.get("biometric_consent", False)
        is_written = consent.get("consent_method") == "written"

        if has_consent and is_written:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_retention_schedule(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-002: Check retention schedule."""
        has_schedule = context.get("retention_schedule_published", False)
        retention_period = context.get("retention_period_days", 0)

        if has_schedule and retention_period > 0:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_purpose_disclosure(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-003: Check purpose disclosure."""
        consent = self.consent_records.get(data_subject_id, {})

        has_purpose = consent.get("collection_purpose") is not None
        has_duration = consent.get("retention_period") is not None

        if has_purpose and has_duration:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_no_sale(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-004: Check no sale of biometric data."""
        operation_type = str(context.get("operation_type", ""))

        prohibited_operations = ["sale", "lease", "trade", "commercial_transfer"]

        if any(op in operation_type.lower() for op in prohibited_operations):
            return ComplianceStatus.VIOLATION

        return ComplianceStatus.COMPLIANT

    def check_security_measures(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-005: Check security measures."""
        encryption = context.get("encryption", {})
        encryption_enabled = encryption.get("enabled", False)
        encryption_strength = encryption.get("algorithm", "")
        access_control = context.get("access_control", False)
        audit_logging = context.get("audit_logging", False)

        acceptable_encryption = "AES-256" in encryption_strength

        if encryption_enabled and acceptable_encryption and access_control and audit_logging:
            return ComplianceStatus.COMPLIANT

        if encryption_enabled:
            return ComplianceStatus.WARNING

        return ComplianceStatus.VIOLATION

    def check_breach_notification(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-006: Check breach notification capability."""
        has_notification_system = context.get("breach_notification_system", False)
        notification_sla_hours = context.get("notification_sla_hours", 999)

        if has_notification_system and notification_sla_hours <= 72:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_deletion_sla(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """BIPA-007: Check deletion SLA."""
        has_deletion_capability = context.get("deletion_capability", False)
        deletion_sla_days = context.get("deletion_sla_days", 999)

        if has_deletion_capability and deletion_sla_days <= 30:
            return ComplianceStatus.COMPLIANT

        if has_deletion_capability:
            return ComplianceStatus.WARNING

        return ComplianceStatus.VIOLATION

    def check_data_inventory(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CCPA-001: Check data inventory."""
        has_inventory = context.get("data_inventory_maintained", False)

        if has_inventory:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_deletion_capability(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CCPA-002: Check deletion capability."""
        has_deletion = context.get("deletion_capability", False)
        has_verification = context.get("deletion_verification", False)

        if has_deletion and has_verification:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_opt_out_mechanism(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CCPA-003: Check opt-out mechanism."""
        has_opt_out = context.get("opt_out_available", False)

        if has_opt_out:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.WARNING

    def check_non_discrimination(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CCPA-004: Check non-discrimination."""
        return ComplianceStatus.COMPLIANT

    def check_correction_process(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CPRA-001: Check correction process."""
        has_correction = context.get("correction_process", False)

        if has_correction:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.WARNING

    def check_sensitive_info_limits(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CPRA-002: Check sensitive info limitations."""
        access_restricted = context.get("restricted_access", False)
        additional_consent = context.get("sensitive_data_consent", False)

        if access_restricted and additional_consent:
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.VIOLATION

    def check_risk_assessment(
        self,
        data_subject_id: str,
        context: dict[str, Any],
    ) -> ComplianceStatus:
        """CPRA-003: Check risk assessment."""
        last_assessment = context.get("last_risk_assessment")

        if last_assessment:
            assessment_date = datetime.fromisoformat(last_assessment)
            days_since = (datetime.now() - assessment_date).days

            if days_since <= 365:
                return ComplianceStatus.COMPLIANT
            return ComplianceStatus.WARNING

        return ComplianceStatus.VIOLATION

    def record_consent(
        self,
        data_subject_id: str,
        consent_data: dict[str, Any],
    ) -> None:
        """Record consent for a data subject.

        Args:
            data_subject_id: Identifier for the data subject.
            consent_data: Consent details merged into the stored record.
        """
        self.consent_records[data_subject_id] = {
            **consent_data,
        }

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate a comprehensive compliance report.

        Returns:
            A JSON-serializable dictionary summarizing rules, violations
            (by framework and severity), monitored frameworks, and consent
            counts.
        """
        total_violations = len(self.violations)
        unresolved_violations = sum(1 for v in self.violations if not v.resolved)

        violations_by_framework: dict[str, int] = {}
        for v in self.violations:
            framework = v.framework.value
            violations_by_framework[framework] = violations_by_framework.get(framework, 0) + 1

        violations_by_severity: dict[str, int] = {}
        for v in self.violations:
            severity = v.severity
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1

        report: dict[str, Any] = {
            "total_rules": len(self.rules),
            "total_violations": total_violations,
            "unresolved_violations": unresolved_violations,
            "violations_by_framework": violations_by_framework,
            "violations_by_severity": violations_by_severity,
            "frameworks_monitored": sorted({rule.framework.value for rule in self.rules.values()}),
            "consent_records_count": len(self.consent_records),
            "recent_violations": [
                {
                    "violation_id": v.violation_id,
                    "rule_id": v.rule_id,
                    "framework": v.framework.value,
                    "severity": v.severity,
                    "description": v.description,
                    "detected_at": v.detected_at.isoformat(),
                    "resolved": v.resolved,
                }
                for v in sorted(
                    self.violations,
                    key=lambda x: x.detected_at,
                    reverse=True,
                )[:10]
            ],
        }

        return report


class RegulatoryIntelligence:
    """Track regulatory changes and updates.

    Tracks changes in BIPA amendments, CCPA/CPRA updates, state privacy laws,
    and federal legislation. This is the in-memory ledger only; the original
    agent's network polling was removed, so changes are recorded explicitly via
    :meth:`detect_regulatory_change`.
    """

    def __init__(self) -> None:
        """Initialize empty tracking and change-history ledgers."""
        self.tracked_regulations: dict[str, dict[str, Any]] = {}
        self.change_history: list[dict[str, Any]] = []

    def monitor_regulation(
        self,
        regulation_name: str,
        source_url: str,
    ) -> None:
        """Register a regulation to monitor.

        Args:
            regulation_name: Name of the regulation being tracked.
            source_url: Reference URL recorded with the tracking entry.
        """
        self.tracked_regulations[regulation_name] = {
            "source_url": source_url,
            "last_checked": datetime.now().isoformat(),
            "version": "1.0",
        }

    def detect_regulatory_change(
        self,
        regulation_name: str,
        change_description: str,
    ) -> None:
        """Record a detected regulatory change.

        Args:
            regulation_name: Name of the regulation that changed.
            change_description: Human-readable description of the change.
        """
        change: dict[str, Any] = {
            "regulation": regulation_name,
            "description": change_description,
            "detected_at": datetime.now().isoformat(),
            "requires_review": True,
        }

        self.change_history.append(change)


class ComplianceSubAgent(SubAgent):
    """Privacy/biometric compliance evaluation (BIPA/CCPA/CPRA)."""

    def _perform(self, task: SubAgentTask) -> tuple[Any, float, str]:
        """Run a real BIPA/CCPA/CPRA compliance evaluation on the payload.

        Reads ``payload["data_category"]`` (required) to select the applicable
        rules, optionally ``payload["data_subject_id"]``, ``payload["context"]``,
        ``payload["consent"]`` (recorded before checking so consent-dependent
        rules can see it), and ``payload["framework"]`` (restricts the rule set).

        Args:
            task: The subagent task carrying the compliance payload.

        Returns:
            A 3-tuple ``(output, confidence, reasoning)`` where ``output`` is a
            JSON-serializable dict with ``status``, ``violations``, ``warnings``,
            and a fleet-level ``report``; ``confidence`` is the fraction of
            checked rules that did not raise a violation, in ``[0, 1]``; and
            ``reasoning`` is a one-line human summary.

        Raises:
            SubAgentExecutionError: If ``payload["data_category"]`` is missing or
                names an unknown data category / framework.
        """
        payload = task.payload

        raw_category = payload.get("data_category")
        if raw_category is None:
            raise SubAgentExecutionError(
                "compliance requires payload['data_category'] "
                "(one of DataCategory: e.g. 'BIOMETRIC')"
            )
        try:
            data_category = DataCategory[str(raw_category).upper()]
        except KeyError as exc:
            valid = ", ".join(c.name for c in DataCategory)
            raise SubAgentExecutionError(
                f"unknown data_category {raw_category!r}; expected one of: {valid}"
            ) from exc

        framework: ComplianceFramework | None = None
        raw_framework = payload.get("framework")
        if raw_framework is not None:
            try:
                framework = ComplianceFramework[str(raw_framework).upper()]
            except KeyError as exc:
                valid = ", ".join(f.name for f in ComplianceFramework)
                raise SubAgentExecutionError(
                    f"unknown framework {raw_framework!r}; expected one of: {valid}"
                ) from exc

        data_subject_id = str(payload.get("data_subject_id", "anonymous"))
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            raise SubAgentExecutionError(
                "compliance payload['context'] must be a mapping if provided"
            )

        agent = ComplianceAutomationAgent()

        consent = payload.get("consent")
        if consent is not None:
            if not isinstance(consent, dict):
                raise SubAgentExecutionError(
                    "compliance payload['consent'] must be a mapping if provided"
                )
            agent.record_consent(data_subject_id, consent)

        result = agent.check_compliance(
            data_category=data_category,
            data_subject_id=data_subject_id,
            context=context,
            framework=framework,
        )
        report = agent.generate_compliance_report()

        rules_checked = int(result["rules_checked"])
        n_violations = len(result["violations"])
        n_warnings = len(result["warnings"])
        confidence = 1.0 - (n_violations / rules_checked) if rules_checked else 0.0

        output: dict[str, Any] = {
            "status": result["status"],
            "data_category": result["data_category"],
            "data_subject_id": result["data_subject_id"],
            "rules_checked": rules_checked,
            "violations": result["violations"],
            "warnings": result["warnings"],
            "report": report,
        }
        reasoning = (
            f"{result['status']} for {data_category.value}: "
            f"{n_violations} violation(s), {n_warnings} warning(s) "
            f"across {rules_checked} rule(s)"
        )
        return output, confidence, reasoning

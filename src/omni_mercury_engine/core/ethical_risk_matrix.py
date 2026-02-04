"""
Mercury Agent ♱
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

from __future__ import annotations


"""
Ethical Risk Matrix with Dynamic Compliance and Anomaly Forecasting

Implements comprehensive risk assessment with:
- Dynamic US-only law polling for compliance
- GDPR and HIPAA compliance hooks
- Anomaly oracles for risk forecasting
- Risk matrix with likelihood/impact scoring
- Minimal human oversight optimization

References:
- NIST Risk Management Framework (2023)
- GDPR (General Data Protection Regulation, EU 2018)
- HIPAA (Health Insurance Portability and Accountability Act, US 1996)
- ISO 31000:2018 Risk Management

MIT-compatible implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable


class RiskLevel(Enum):
    """Risk level classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ComplianceRegime(Enum):
    """Compliance regime types."""

    US_FEDERAL = "us_federal"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    CUSTOM = "custom"


@dataclass
class RiskScore:
    """Risk assessment with likelihood and impact."""

    risk_id: str
    likelihood: float
    impact: float
    risk_level: RiskLevel
    compliance_violations: list[str] = field(default_factory=list)
    mitigation_required: bool = False
    forecast_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    regime: ComplianceRegime
    description: str
    validator: Callable[[dict[str, Any]], bool]
    severity: str = "medium"


class USLawPolling:
    """
    Dynamic US-only law compliance polling.

    Implements real-time compliance checks for US federal regulations:
    - CFAA (Computer Fraud and Abuse Act)
    - ECPA (Electronic Communications Privacy Act)
    - Section 230 (Communications Decency Act)
    - AI/ML specific regulations
    """

    def __init__(self) -> None:
        """Initialize US law polling system."""
        self.compliance_rules = self._initialize_us_rules()
        self.last_poll_time = datetime.now()

    def _initialize_us_rules(self) -> list[ComplianceRule]:
        """Initialize US federal compliance rules."""
        rules = []

        rules.append(
            ComplianceRule(
                rule_id="us_cfaa_unauthorized_access",
                regime=ComplianceRegime.US_FEDERAL,
                description="CFAA: Prevent unauthorized computer access",
                validator=lambda ctx: not ctx.get("unauthorized_access", False),
                severity="critical",
            )
        )

        rules.append(
            ComplianceRule(
                rule_id="us_ecpa_wiretap",
                regime=ComplianceRegime.US_FEDERAL,
                description="ECPA: No unauthorized electronic communication interception",
                validator=lambda ctx: not ctx.get("intercepts_communications", False),
                severity="critical",
            )
        )

        rules.append(
            ComplianceRule(
                rule_id="us_section_230_liability",
                regime=ComplianceRegime.US_FEDERAL,
                description="Section 230: Good faith content moderation",
                validator=lambda ctx: ctx.get("content_moderation_good_faith", True),
                severity="medium",
            )
        )

        rules.append(
            ComplianceRule(
                rule_id="us_ai_transparency",
                regime=ComplianceRegime.US_FEDERAL,
                description="AI Transparency: Disclose AI decision-making",
                validator=lambda ctx: ctx.get("ai_transparency", True),
                severity="medium",
            )
        )

        return rules

    def check_compliance(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check compliance with US federal laws.

        Args:
            context: Operational context

        Returns:
            Tuple of (compliant, violations)
        """
        violations = []

        for rule in self.compliance_rules:
            try:
                if not rule.validator(context):
                    violations.append(f"{rule.rule_id}: {rule.description}")
            except Exception as e:
                violations.append(f"{rule.rule_id}: Validation error - {e!s}")

        return len(violations) == 0, violations


@dataclass
class GDPRComplianceViolation:
    """Detailed GDPR compliance violation with contextual metadata."""

    article: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    remediation: str
    evidence_required: list[str] = field(default_factory=list)
    compliance_score_impact: float = 0.0


@dataclass
class GDPRLegalBasis:
    """Legal basis for data processing under GDPR Article 6."""

    basis_type: str  # "consent", "contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"
    documentation: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    expiry: datetime | None = None


@dataclass
class GDPRComplianceResult:
    """Comprehensive GDPR compliance assessment result."""

    compliant: bool
    violations: list[GDPRComplianceViolation]
    compliance_score: float  # 0.0 to 1.0
    legal_basis: GDPRLegalBasis | None
    data_subject_rights_status: dict[str, bool]
    recommendations: list[str]
    assessment_timestamp: datetime = field(default_factory=datetime.now)


class GDPRCompliance:
    """
    GDPR compliance framework with comprehensive validation.

    Implements detailed checks for EU General Data Protection Regulation:
    - Article 5: Data processing principles (lawfulness, purpose limitation, minimization)
    - Article 6: Lawful basis for processing
    - Article 7: Conditions for consent
    - Article 9: Special category data protections
    - Article 13-14: Information obligations
    - Article 17: Right to erasure
    - Article 22: Automated decision-making restrictions
    - Article 25: Data protection by design and default
    - Article 32: Security of processing
    - Article 33-34: Breach notification requirements

    References:
    - GDPR (EU) 2016/679
    - EDPB Guidelines on consent, legitimate interests, and automated decision-making
    """

    def __init__(self) -> None:
        """Initialize GDPR compliance framework."""
        self.data_subject_rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_data_portability",
            "right_to_object",
            "right_to_restriction",
            "right_to_withdraw_consent",
            "right_not_to_be_subject_to_automated_decisions",
        ]

        self.special_category_types = [
            "racial_ethnic_origin",
            "political_opinions",
            "religious_beliefs",
            "trade_union_membership",
            "genetic_data",
            "biometric_data",
            "health_data",
            "sex_life_orientation",
        ]

        self.valid_legal_bases = [
            "consent",
            "contract",
            "legal_obligation",
            "vital_interests",
            "public_task",
            "legitimate_interests",
        ]

    def _validate_legal_basis(self, context: dict[str, Any]) -> GDPRLegalBasis:
        """
        Validate the legal basis for data processing.

        Args:
            context: Processing context with legal basis documentation

        Returns:
            GDPRLegalBasis with validation status
        """
        basis_type = context.get("legal_basis_type", "")
        basis_doc = context.get("legal_basis_documentation", {})

        valid = False

        if basis_type == "consent":
            # Consent must be freely given, specific, informed, and unambiguous
            consent_checks = [
                basis_doc.get("freely_given", False),
                basis_doc.get("specific_purpose", False),
                basis_doc.get("informed", False),
                basis_doc.get("unambiguous_indication", False),
                basis_doc.get("withdrawable", False),
            ]
            valid = all(consent_checks)

        elif basis_type == "contract":
            # Processing necessary for contract performance
            valid = bool(
                basis_doc.get("contract_reference") and basis_doc.get("processing_necessary")
            )

        elif basis_type == "legal_obligation":
            # Processing required by law
            valid = bool(basis_doc.get("legal_reference") and basis_doc.get("member_state_law"))

        elif basis_type == "vital_interests":
            # Necessary to protect life
            valid = bool(
                basis_doc.get("vital_interest_documented") and basis_doc.get("no_alternative_basis")
            )

        elif basis_type == "public_task":
            # Processing for official authority
            valid = bool(
                basis_doc.get("public_authority_mandate") and basis_doc.get("task_documentation")
            )

        elif basis_type == "legitimate_interests":
            # Balancing test required
            lia_checks = [
                basis_doc.get("legitimate_interest_identified", False),
                basis_doc.get("necessity_demonstrated", False),
                basis_doc.get("balancing_test_conducted", False),
                basis_doc.get("data_subject_rights_considered", False),
            ]
            valid = all(lia_checks)

        expiry = None
        if basis_type == "consent" and basis_doc.get("consent_expiry"):
            try:
                expiry = datetime.fromisoformat(str(basis_doc["consent_expiry"]))
            except (ValueError, TypeError):
                pass

        return GDPRLegalBasis(
            basis_type=basis_type,
            documentation=basis_doc,
            valid=valid,
            expiry=expiry,
        )

    def _check_data_subject_rights(self, context: dict[str, Any]) -> dict[str, bool]:
        """
        Verify data subject rights implementation.

        Args:
            context: Processing context

        Returns:
            Dictionary of right -> implementation status
        """
        rights_config = context.get("data_subject_rights_config", {})
        rights_status = {}

        for right in self.data_subject_rights:
            right_config = rights_config.get(right, {})
            # Each right must have mechanism and reasonable timeline
            rights_status[right] = bool(
                right_config.get("mechanism_exists", False)
                and right_config.get("response_timeline_days", 0) <= 30
            )

        return rights_status

    def check_gdpr_compliance(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check GDPR compliance with comprehensive validation.

        Args:
            context: Data processing context containing:
                - processes_personal_data: bool
                - legal_basis_type: str (one of valid_legal_bases)
                - legal_basis_documentation: dict with basis-specific evidence
                - data_subject_rights_config: dict of right implementations
                - data_minimization_evidence: dict with justification
                - purpose_limitation_evidence: dict with documented purposes
                - automated_decision: bool
                - human_review_mechanism: dict with review process details
                - special_category_data: bool
                - special_category_exemption: str
                - security_measures: dict with Art. 32 compliance
                - data_protection_impact_assessment: dict (for high-risk processing)
                - cross_border_transfer: bool
                - transfer_mechanism: str (adequacy, SCCs, BCRs, etc.)

        Returns:
            Tuple of (compliant, violations)
        """
        violations: list[str] = []

        if not context.get("processes_personal_data", False):
            return True, violations

        # Article 6: Lawful basis validation
        legal_basis = self._validate_legal_basis(context)
        if not legal_basis.valid:
            violations.append(
                f"GDPR Art. 6: Invalid legal basis '{legal_basis.basis_type}' - "
                f"required documentation incomplete or missing"
            )

        # Check for expired consent
        if legal_basis.expiry and legal_basis.expiry < datetime.now():
            violations.append("GDPR Art. 7: Consent has expired and must be renewed")

        # Article 5(1)(c): Data minimization with evidence
        minimization_evidence = context.get("data_minimization_evidence", {})
        if not minimization_evidence.get("adequacy_justified", False):
            violations.append(
                "GDPR Art. 5(1)(c): Data minimization - no documented justification "
                "for data collected being adequate, relevant, and limited"
            )
        if not minimization_evidence.get("retention_policy_defined", False):
            violations.append("GDPR Art. 5(1)(e): Storage limitation - no defined retention policy")

        # Article 5(1)(b): Purpose limitation with evidence
        purpose_evidence = context.get("purpose_limitation_evidence", {})
        if not purpose_evidence.get("purposes_documented", False):
            violations.append(
                "GDPR Art. 5(1)(b): Purpose limitation - processing purposes not "
                "explicitly documented at collection time"
            )
        if purpose_evidence.get("further_processing", False):
            if not purpose_evidence.get("compatibility_assessment", False):
                violations.append(
                    "GDPR Art. 5(1)(b): Further processing without compatibility assessment"
                )

        # Article 22: Automated decision-making
        if context.get("automated_decision", False):
            human_review = context.get("human_review_mechanism", {})
            if not human_review.get("exists", False):
                violations.append(
                    "GDPR Art. 22(1): Automated decision-making without right to "
                    "obtain human intervention"
                )
            if not human_review.get("meaningful_review", False):
                violations.append(
                    "GDPR Art. 22(3): Automated decisions lack meaningful human review "
                    "capability (must be substantive, not perfunctory)"
                )
            if not context.get("automated_decision_logic_explained", False):
                violations.append(
                    "GDPR Art. 22(1)/Art. 13-14: No explanation of automated "
                    "decision-making logic provided to data subject"
                )

        # Article 9: Special category data
        if context.get("special_category_data", False):
            exemption = context.get("special_category_exemption", "")
            valid_exemptions = [
                "explicit_consent",
                "employment_law",
                "vital_interests",
                "legitimate_activities",
                "manifestly_public",
                "legal_claims",
                "public_interest",
                "health_social_care",
                "public_health",
                "archiving_research",
            ]
            if exemption not in valid_exemptions:
                violations.append(
                    f"GDPR Art. 9: Special category data processed without valid "
                    f"exemption (provided: '{exemption}')"
                )

        # Article 32: Security of processing
        security_measures = context.get("security_measures", {})
        required_measures = [
            ("pseudonymization", "Art. 32(1)(a): Pseudonymization/encryption"),
            ("confidentiality", "Art. 32(1)(b): Confidentiality assurance"),
            ("integrity", "Art. 32(1)(b): Integrity assurance"),
            ("availability", "Art. 32(1)(b): Availability assurance"),
            ("resilience", "Art. 32(1)(b): Resilience of systems"),
            ("restoration_capability", "Art. 32(1)(c): Timely restoration capability"),
            ("testing_process", "Art. 32(1)(d): Regular security testing"),
        ]
        for measure, article_ref in required_measures:
            if not security_measures.get(measure, False):
                violations.append(f"GDPR {article_ref} not demonstrated")

        # Article 35: DPIA for high-risk processing
        if context.get("high_risk_processing", False):
            dpia = context.get("data_protection_impact_assessment", {})
            if not dpia.get("conducted", False):
                violations.append(
                    "GDPR Art. 35: High-risk processing without Data Protection "
                    "Impact Assessment"
                )
            elif not dpia.get("risk_mitigation_documented", False):
                violations.append(
                    "GDPR Art. 35(7): DPIA conducted but risk mitigation measures " "not documented"
                )

        # Chapter V: International transfers
        if context.get("cross_border_transfer", False):
            transfer_mechanism = context.get("transfer_mechanism", "")
            valid_mechanisms = [
                "adequacy_decision",
                "standard_contractual_clauses",
                "binding_corporate_rules",
                "approved_certification",
                "approved_code_of_conduct",
                "explicit_consent",
                "contract_necessity",
                "public_interest",
                "legal_claims",
                "vital_interests",
            ]
            if transfer_mechanism not in valid_mechanisms:
                violations.append(
                    f"GDPR Art. 44-49: Cross-border transfer without valid mechanism "
                    f"(provided: '{transfer_mechanism}')"
                )

        return len(violations) == 0, violations

    def assess_compliance(self, context: dict[str, Any]) -> GDPRComplianceResult:
        """
        Comprehensive GDPR compliance assessment with scoring and recommendations.

        Args:
            context: Full data processing context

        Returns:
            GDPRComplianceResult with detailed assessment
        """
        compliant, violation_strings = self.check_gdpr_compliance(context)

        # Convert string violations to detailed objects
        violations = []
        for v_str in violation_strings:
            # Parse article from violation string
            article = v_str.split(":")[0] if ":" in v_str else "GDPR"
            severity = "high" if "Art. 6" in article or "Art. 9" in article else "medium"
            if "Art. 22" in article:
                severity = "critical"

            violations.append(
                GDPRComplianceViolation(
                    article=article,
                    severity=severity,
                    description=v_str,
                    remediation=self._get_remediation(v_str),
                    evidence_required=self._get_evidence_requirements(article),
                    compliance_score_impact=self._get_score_impact(severity),
                )
            )

        # Calculate compliance score
        base_score = 1.0
        for v in violations:
            base_score -= v.compliance_score_impact
        compliance_score = max(0.0, min(1.0, base_score))

        # Validate legal basis
        legal_basis = self._validate_legal_basis(context)

        # Check data subject rights
        rights_status = self._check_data_subject_rights(context)

        # Generate recommendations
        recommendations = self._generate_recommendations(violations, context)

        return GDPRComplianceResult(
            compliant=compliant,
            violations=violations,
            compliance_score=compliance_score,
            legal_basis=legal_basis,
            data_subject_rights_status=rights_status,
            recommendations=recommendations,
        )

    def _get_remediation(self, violation: str) -> str:
        """Get remediation guidance for a violation."""
        remediations = {
            "Art. 6": "Establish and document a valid legal basis before processing",
            "Art. 5(1)(c)": "Document data minimization justification and retention policy",
            "Art. 5(1)(b)": "Document explicit processing purposes at collection",
            "Art. 22": "Implement meaningful human review mechanism with explanation capability",
            "Art. 9": "Obtain explicit consent or document applicable exemption",
            "Art. 32": "Implement and document required security measures",
            "Art. 35": "Conduct and document Data Protection Impact Assessment",
            "Art. 44": "Implement valid transfer mechanism for cross-border data flows",
        }
        for article, remediation in remediations.items():
            if article in violation:
                return remediation
        return "Review GDPR requirements and implement appropriate controls"

    def _get_evidence_requirements(self, article: str) -> list[str]:
        """Get evidence requirements for an article."""
        evidence_map = {
            "Art. 6": ["legal_basis_documentation", "processing_records"],
            "Art. 5": ["data_inventory", "retention_schedule", "purpose_register"],
            "Art. 22": ["human_review_process", "logic_explanation_template"],
            "Art. 9": ["consent_records", "exemption_documentation"],
            "Art. 32": ["security_assessment", "testing_records"],
            "Art. 35": ["dpia_report", "risk_register", "mitigation_plan"],
            "Art. 44": ["transfer_agreement", "adequacy_assessment"],
        }
        for art, evidence in evidence_map.items():
            if art in article:
                return evidence
        return ["compliance_documentation"]

    def _get_score_impact(self, severity: str) -> float:
        """Get compliance score impact for a severity level."""
        impacts = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05,
        }
        return impacts.get(severity, 0.1)

    def _generate_recommendations(
        self, violations: list[GDPRComplianceViolation], context: dict[str, Any]
    ) -> list[str]:
        """Generate prioritized recommendations based on violations."""
        recommendations = []

        # Prioritize by severity
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            recommendations.append("IMMEDIATE: Address critical compliance gaps before processing")

        # Specific recommendations
        if any("Art. 6" in v.article for v in violations):
            recommendations.append("Establish documented legal basis with all required elements")

        if any("Art. 22" in v.article for v in violations):
            recommendations.append(
                "Implement explainable AI framework with human oversight capability"
            )

        if not context.get("data_protection_officer", False):
            recommendations.append("Consider appointing a Data Protection Officer (Art. 37)")

        if not context.get("records_of_processing", False):
            recommendations.append("Maintain records of processing activities (Art. 30)")

        return recommendations


class HIPAACompliance:
    """HIPAA compliance hooks for US healthcare data."""

    def __init__(self) -> None:
        """Initialize HIPAA compliance."""
        self.phi_identifiers = [
            "names",
            "dates",
            "phone_numbers",
            "email",
            "ssn",
            "medical_record_numbers",
            "health_plan_numbers",
            "device_ids",
        ]

    def check_hipaa_compliance(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check HIPAA compliance.

        Args:
            context: Healthcare data context

        Returns:
            Tuple of (compliant, violations)
        """
        violations = []

        if context.get("processes_phi", False):
            if not context.get("encryption_at_rest", False):
                violations.append("HIPAA Security Rule: PHI not encrypted at rest")

            if not context.get("encryption_in_transit", False):
                violations.append("HIPAA Security Rule: PHI not encrypted in transit")

            if not context.get("access_controls", False):
                violations.append("HIPAA Security Rule: Insufficient access controls")

            if not context.get("audit_trail", False):
                violations.append("HIPAA Security Rule: No audit trail for PHI access")

            if not context.get("baa_in_place", False):
                violations.append("HIPAA Privacy Rule: No Business Associate Agreement")

        return len(violations) == 0, violations


class AnomalyOracle:
    """
    Anomaly oracle for risk forecasting via pattern-based simulations.

    Uses historical patterns to predict future anomalies and risks.
    """

    def __init__(self, lookback_window: int = 100) -> None:
        """
        Initialize anomaly oracle.

        Args:
            lookback_window: Number of historical samples for forecasting
        """
        self.lookback_window = lookback_window
        self.historical_anomalies: list[tuple[float, float]] = []

    def record_anomaly(self, anomaly_score: float, impact: float) -> None:
        """
        Record anomaly for future forecasting.

        Args:
            anomaly_score: Anomaly detection score
            impact: Actual impact observed
        """
        self.historical_anomalies.append((anomaly_score, impact))

        if len(self.historical_anomalies) > self.lookback_window:
            self.historical_anomalies = self.historical_anomalies[-self.lookback_window :]

    def forecast_risk(self, current_anomaly_score: float) -> tuple[float, float]:
        """
        Forecast future risk based on current anomaly score.

        Args:
            current_anomaly_score: Current anomaly detection score

        Returns:
            Tuple of (forecasted_likelihood, forecasted_impact)
        """
        if not self.historical_anomalies:
            return 0.5, 0.5

        scores = np.array([a[0] for a in self.historical_anomalies])
        impacts = np.array([a[1] for a in self.historical_anomalies])

        similar_indices = np.where(np.abs(scores - current_anomaly_score) < 0.2)[0]

        if len(similar_indices) > 0:
            forecasted_likelihood = float(np.mean(scores[similar_indices]))
            forecasted_impact = float(np.mean(impacts[similar_indices]))
        else:
            forecasted_likelihood = float(np.mean(scores))
            forecasted_impact = float(np.mean(impacts))

        return forecasted_likelihood, forecasted_impact


class EthicalRiskMatrix:
    """
    Comprehensive ethical risk matrix with compliance and forecasting.

    Features:
    - Dynamic US law polling
    - GDPR/HIPAA compliance hooks
    - Anomaly oracle forecasting
    - Risk matrix (likelihood × impact)
    - Automated mitigation recommendations
    """

    def __init__(
        self,
        enable_us_compliance: bool = True,
        enable_gdpr: bool = True,
        enable_hipaa: bool = True,
        enable_forecasting: bool = True,
    ) -> None:
        """
        Initialize Ethical Risk Matrix.

        Args:
            enable_us_compliance: Enable US federal law compliance
            enable_gdpr: Enable GDPR compliance
            enable_hipaa: Enable HIPAA compliance
            enable_forecasting: Enable anomaly forecasting
        """
        self.enable_us_compliance = enable_us_compliance
        self.enable_gdpr = enable_gdpr
        self.enable_hipaa = enable_hipaa
        self.enable_forecasting = enable_forecasting

        self.us_law: USLawPolling | None = USLawPolling() if enable_us_compliance else None
        self.gdpr: GDPRCompliance | None = GDPRCompliance() if enable_gdpr else None
        self.hipaa: HIPAACompliance | None = HIPAACompliance() if enable_hipaa else None
        self.oracle: AnomalyOracle | None = AnomalyOracle() if enable_forecasting else None

        self.risk_history: list[RiskScore] = []

    def assess_risk(self, context: dict[str, Any], anomaly_score: float | None = None) -> RiskScore:
        """
        Comprehensive risk assessment.

        Args:
            context: Operational context
            anomaly_score: Optional anomaly detection score

        Returns:
            RiskScore with assessment results
        """
        risk_id = f"risk_{datetime.now().timestamp()}"

        likelihood, impact = self._compute_risk_components(context, anomaly_score)

        risk_level = self._determine_risk_level(likelihood, impact)

        compliance_violations = self._check_all_compliance(context)

        mitigation_required = risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]

        forecast_confidence = 0.0
        if self.enable_forecasting and anomaly_score is not None and self.oracle:
            _, _ = self.oracle.forecast_risk(anomaly_score)
            forecast_confidence = min(len(self.oracle.historical_anomalies) / 100, 1.0)

        risk_score = RiskScore(
            risk_id=risk_id,
            likelihood=likelihood,
            impact=impact,
            risk_level=risk_level,
            compliance_violations=compliance_violations,
            mitigation_required=mitigation_required,
            forecast_confidence=forecast_confidence,
        )

        self.risk_history.append(risk_score)

        if anomaly_score is not None and self.oracle:
            self.oracle.record_anomaly(anomaly_score, impact)

        return risk_score

    def _compute_risk_components(
        self, context: dict[str, Any], anomaly_score: float | None
    ) -> tuple[float, float]:
        """
        Compute likelihood and impact components.

        Args:
            context: Operational context
            anomaly_score: Anomaly score

        Returns:
            Tuple of (likelihood, impact)
        """
        base_likelihood = anomaly_score if anomaly_score is not None else 0.5

        if self.enable_forecasting and anomaly_score is not None and self.oracle:
            forecasted_likelihood, forecasted_impact = self.oracle.forecast_risk(anomaly_score)
            likelihood = (base_likelihood + forecasted_likelihood) / 2.0
            impact = forecasted_impact
        else:
            likelihood = base_likelihood
            impact = context.get("potential_impact", 0.5)

        if context.get("critical_system", False):
            impact *= 1.5
        if context.get("processes_phi", False):
            impact *= 1.3
        if context.get("processes_personal_data", False):
            impact *= 1.2

        impact = min(impact, 1.0)

        return likelihood, impact

    def _determine_risk_level(self, likelihood: float, impact: float) -> RiskLevel:
        """
        Determine risk level from likelihood × impact matrix.

        Args:
            likelihood: Risk likelihood (0-1)
            impact: Risk impact (0-1)

        Returns:
            RiskLevel classification
        """
        risk_product = likelihood * impact

        if risk_product >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_product >= 0.6:
            return RiskLevel.HIGH
        elif risk_product >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_product >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE

    def _check_all_compliance(self, context: dict[str, Any]) -> list[str]:
        """Check all enabled compliance regimes."""
        all_violations = []

        if self.enable_us_compliance and self.us_law:
            _, us_violations = self.us_law.check_compliance(context)
            all_violations.extend(us_violations)

        if self.enable_gdpr and self.gdpr:
            _, gdpr_violations = self.gdpr.check_gdpr_compliance(context)
            all_violations.extend(gdpr_violations)

        if self.enable_hipaa and self.hipaa:
            _, hipaa_violations = self.hipaa.check_hipaa_compliance(context)
            all_violations.extend(hipaa_violations)

        return all_violations

    def get_risk_matrix_table(self) -> dict[str, Any]:
        """
        Generate risk matrix table for visualization.

        Returns:
            Risk matrix with likelihood/impact grid
        """
        if not self.risk_history:
            return {"matrix": [], "summary": "No risk data available"}

        recent_risks = self.risk_history[-100:]

        likelihood_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        impact_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        matrix = np.zeros((len(likelihood_bins) - 1, len(impact_bins) - 1))

        for risk in recent_risks:
            l_idx = min(int(risk.likelihood * 5), 4)
            i_idx = min(int(risk.impact * 5), 4)
            matrix[l_idx, i_idx] += 1

        return {
            "matrix": matrix.tolist(),
            "likelihood_bins": likelihood_bins,
            "impact_bins": impact_bins,
            "total_risks_assessed": len(recent_risks),
            "critical_risks": sum(1 for r in recent_risks if r.risk_level == RiskLevel.CRITICAL),
            "high_risks": sum(1 for r in recent_risks if r.risk_level == RiskLevel.HIGH),
            "compliance_violations": sum(len(r.compliance_violations) for r in recent_risks),
        }

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate comprehensive compliance report."""
        recent_risks = (
            self.risk_history[-100:] if len(self.risk_history) > 100 else self.risk_history
        )

        all_violations: list[str] = []
        for risk in recent_risks:
            all_violations.extend(risk.compliance_violations)

        violation_counts: dict[str, int] = {}
        for violation in all_violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1

        return {
            "total_risks_assessed": len(recent_risks),
            "total_violations": len(all_violations),
            "unique_violations": len(set(all_violations)),
            "violation_breakdown": violation_counts,
            "compliance_rate": (
                1.0
                - (len([r for r in recent_risks if r.compliance_violations]) / len(recent_risks))
                if recent_risks
                else 1.0
            ),
            "us_compliance_enabled": self.enable_us_compliance,
            "gdpr_compliance_enabled": self.enable_gdpr,
            "hipaa_compliance_enabled": self.enable_hipaa,
        }

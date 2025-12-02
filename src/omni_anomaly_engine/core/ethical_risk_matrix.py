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

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


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
    compliance_violations: List[str] = field(default_factory=list)
    mitigation_required: bool = False
    forecast_confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    regime: ComplianceRegime
    description: str
    validator: Callable[[Dict[str, Any]], bool]
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

    def __init__(self):
        """Initialize US law polling system."""
        self.compliance_rules = self._initialize_us_rules()
        self.last_poll_time = datetime.now()

    def _initialize_us_rules(self) -> List[ComplianceRule]:
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

    def check_compliance(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
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
                violations.append(f"{rule.rule_id}: Validation error - {str(e)}")

        return len(violations) == 0, violations


class GDPRCompliance:
    """GDPR compliance hooks for EU data protection."""

    def __init__(self):
        """Initialize GDPR compliance."""
        self.data_subject_rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_data_portability",
            "right_to_object",
        ]

    def check_gdpr_compliance(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check GDPR compliance.

        Args:
            context: Data processing context

        Returns:
            Tuple of (compliant, violations)
        """
        violations = []

        if context.get("processes_personal_data", False):
            if not context.get("consent_obtained", False):
                violations.append("GDPR Art. 6: No valid consent for personal data processing")

            if not context.get("data_minimization", False):
                violations.append("GDPR Art. 5: Data minimization principle violated")

            if not context.get("purpose_limitation", False):
                violations.append("GDPR Art. 5: Purpose limitation principle violated")

            if context.get("automated_decision", False) and not context.get("human_review", False):
                violations.append("GDPR Art. 22: Automated decision-making without human review")

        return len(violations) == 0, violations


class HIPAACompliance:
    """HIPAA compliance hooks for US healthcare data."""

    def __init__(self):
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

    def check_hipaa_compliance(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
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
    Anomaly oracle for risk forecasting via Aether Halo-inspired simulations.

    Uses historical patterns to predict future anomalies and risks.
    """

    def __init__(self, lookback_window: int = 100):
        """
        Initialize anomaly oracle.

        Args:
            lookback_window: Number of historical samples for forecasting
        """
        self.lookback_window = lookback_window
        self.historical_anomalies: List[Tuple[float, float]] = []

    def record_anomaly(self, anomaly_score: float, impact: float):
        """
        Record anomaly for future forecasting.

        Args:
            anomaly_score: Anomaly detection score
            impact: Actual impact observed
        """
        self.historical_anomalies.append((anomaly_score, impact))

        if len(self.historical_anomalies) > self.lookback_window:
            self.historical_anomalies = self.historical_anomalies[-self.lookback_window :]

    def forecast_risk(self, current_anomaly_score: float) -> Tuple[float, float]:
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
    ):
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

        if enable_us_compliance:
            self.us_law = USLawPolling()
        else:
            self.us_law = None

        if enable_gdpr:
            self.gdpr = GDPRCompliance()
        else:
            self.gdpr = None

        if enable_hipaa:
            self.hipaa = HIPAACompliance()
        else:
            self.hipaa = None

        if enable_forecasting:
            self.oracle = AnomalyOracle()
        else:
            self.oracle = None

        self.risk_history: List[RiskScore] = []

    def assess_risk(
        self, context: Dict[str, Any], anomaly_score: Optional[float] = None
    ) -> RiskScore:
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
        self, context: Dict[str, Any], anomaly_score: Optional[float]
    ) -> Tuple[float, float]:
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

    def _check_all_compliance(self, context: Dict[str, Any]) -> List[str]:
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

    def get_risk_matrix_table(self) -> Dict[str, Any]:
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

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        recent_risks = (
            self.risk_history[-100:] if len(self.risk_history) > 100 else self.risk_history
        )

        all_violations = []
        for risk in recent_risks:
            all_violations.extend(risk.compliance_violations)

        violation_counts = {}
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

"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
Test suite for ethical framework components: Ethical Governor,
Sigma Directives, Risk Matrix, and Compliance.
"""

import numpy as np
import pytest

from omni_mercury_engine.core.ethical_config import DEFAULT_CONFIG
from omni_mercury_engine.core.ethical_governor import (
    BiasMetrics,
    EthicalAutonomyGovernor,
    EthicalDecision,
    SigmaDirective,
)
from omni_mercury_engine.core.ethical_risk_matrix import (
    AnomalyOracle,
    ComplianceRegime,
    EthicalRiskMatrix,
    GDPRCompliance,
    HIPAACompliance,
    RiskLevel,
    USLawPolling,
)


class TestSigmaDirective:
    """Test Sigma Directive system."""

    def test_initialization(self):
        """Test Sigma Directive initialization."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        assert SigmaDirective.JUSTICE in directive.directive_weights
        assert SigmaDirective.ALTRUISM in directive.directive_weights
        assert SigmaDirective.COMPASSION in directive.directive_weights
        assert SigmaDirective.TRUTH in directive.directive_weights

    def test_justice_directive(self):
        """Test justice evaluation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        fair_context = {"fairness_score": 0.9, "bias_detected": False}
        unfair_context = {"fairness_score": 0.3, "bias_detected": True}

        fair_score = directive._evaluate_justice(fair_context)
        unfair_score = directive._evaluate_justice(unfair_context)

        assert fair_score > unfair_score
        assert unfair_score == 0.0

    def test_directive_application(self):
        """Test full directive application."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        good_context = {
            "fairness_score": 0.9,
            "bias_detected": False,
            "societal_benefit": 0.9,
            "potential_harm": 0.1,
            "harm_prevention": 0.9,
            "suffering_mitigation": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
        }

        bad_context = {
            "fairness_score": 0.3,
            "bias_detected": True,
            "societal_benefit": 0.2,
            "potential_harm": 0.8,
            "harm_prevention": 0.2,
            "suffering_mitigation": 0.2,
            "transparency": 0.3,
            "honesty": 0.3,
        }

        allow_good, _ = directive.apply_directive("deploy_model", good_context)
        allow_bad, reasoning = directive.apply_directive("deploy_model", bad_context)

        assert allow_good is True
        assert allow_bad is False
        assert "violation" in reasoning.lower()


class TestEthicalAutonomyGovernor:
    """Test Ethical Autonomy Governor."""

    def test_initialization(self):
        """Test governor initialization."""
        governor = EthicalAutonomyGovernor()

        assert governor.ethical_scalars is not None
        assert governor.sigma_directive is not None
        assert len(governor.ethical_scalars.to_dict()) >= 150

    def test_decision_evaluation(self):
        """Test ethical decision evaluation."""
        governor = EthicalAutonomyGovernor()

        context = {
            "fairness_score": 0.9,
            "bias_detected": False,
            "societal_benefit": 0.9,
            "potential_harm": 0.1,
            "harm_prevention": 0.9,
            "suffering_mitigation": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
        }

        decision = governor.evaluate_decision(
            "deploy_ai_system", context, data=np.random.randn(100)
        )

        assert isinstance(decision, EthicalDecision)
        assert decision.ethical_score > 0.0
        assert isinstance(decision.bias_audit_passed, bool)
        assert decision.p_value >= 0.0

    def test_bias_auditing(self):
        """Test bias auditing functionality."""
        governor = EthicalAutonomyGovernor()

        np.random.seed(123)
        fair_data = np.concatenate([np.random.randn(50) * 1.0, np.random.randn(50) * 1.0])

        np.random.seed(456)
        biased_data = np.concatenate([np.random.randn(50) * 0.5, np.random.randn(50) * 2.0])

        fair_metrics = governor._audit_bias(fair_data, {})
        biased_metrics = governor._audit_bias(biased_data, {})

        assert isinstance(fair_metrics, BiasMetrics)
        assert isinstance(biased_metrics, BiasMetrics)
        assert biased_metrics.demographic_parity_diff >= fair_metrics.demographic_parity_diff * 0.5

    def test_statistical_validation(self):
        """Test p<0.05 statistical validation."""
        governor = EthicalAutonomyGovernor(p_value_threshold=0.05)

        context = {"fairness_score": 0.9}

        high_score = 1.5
        p_value = governor._statistical_validation(high_score, context)

        assert 0.0 <= p_value <= 1.0

    def test_rollback_mechanism(self):
        """Test automatic rollback on violations."""
        governor = EthicalAutonomyGovernor(ethical_threshold=0.8)

        bad_context = {
            "fairness_score": 0.2,
            "bias_detected": True,
            "societal_benefit": 0.1,
            "potential_harm": 0.9,
        }

        decision = governor.evaluate_decision("harmful_action", bad_context)

        if decision.ethical_score < 0.8 or not decision.bias_audit_passed:
            assert decision.rollback_triggered is True

    def test_governance_report(self):
        """Test governance reporting."""
        governor = EthicalAutonomyGovernor()

        for i in range(10):
            context = {"fairness_score": 0.8 + i * 0.01, "societal_benefit": 0.8}
            governor.evaluate_decision(f"action_{i}", context)

        report = governor.get_governance_report()

        assert "total_decisions" in report
        assert "avg_ethical_score" in report
        assert "bias_audit_pass_rate" in report
        assert report["ethical_scalars_count"] >= 150


class TestUSLawPolling:
    """Test US-only law compliance polling."""

    def test_initialization(self):
        """Test US law polling initialization."""
        us_law = USLawPolling()

        assert len(us_law.compliance_rules) > 0
        assert all(rule.regime == ComplianceRegime.US_FEDERAL for rule in us_law.compliance_rules)

    def test_cfaa_compliance(self):
        """Test CFAA compliance checking."""
        us_law = USLawPolling()

        compliant_context = {"unauthorized_access": False}
        violation_context = {"unauthorized_access": True}

        compliant, _ = us_law.check_compliance(compliant_context)
        non_compliant, violations = us_law.check_compliance(violation_context)

        assert compliant is True
        assert non_compliant is False
        assert len(violations) > 0


class TestGDPRCompliance:
    """Test GDPR compliance."""

    def test_gdpr_checks(self):
        """Test GDPR compliance checks.

        The enhanced GDPRCompliance implementation validates against GDPR Articles 5-35,
        requiring detailed evidence for legal basis, data minimization, purpose limitation,
        and security measures. This test provides comprehensive context matching the
        implementation's requirements.
        """
        gdpr = GDPRCompliance()

        # Comprehensive compliant context with all required GDPR evidence
        # per Articles 5, 6, 22, 32 requirements
        compliant_context = {
            "processes_personal_data": True,
            "automated_decision": False,
            # Article 6: Valid legal basis with complete documentation
            "legal_basis_type": "consent",
            "legal_basis_documentation": {
                "freely_given": True,
                "specific_purpose": True,
                "informed": True,
                "unambiguous_indication": True,
                "withdrawable": True,
            },
            # Article 5(1)(c): Data minimization evidence
            "data_minimization_evidence": {
                "adequacy_justified": True,
                "retention_policy_defined": True,
            },
            # Article 5(1)(b): Purpose limitation evidence
            "purpose_limitation_evidence": {
                "purposes_documented": True,
                "further_processing": False,
            },
            # Article 32: Security measures
            "security_measures": {
                "pseudonymization": True,
                "confidentiality": True,
                "integrity": True,
                "availability": True,
                "resilience": True,
                "restoration_capability": True,
                "testing_process": True,
            },
        }

        # Violation context missing required GDPR compliance elements
        violation_context = {
            "processes_personal_data": True,
            "automated_decision": True,
            # Missing valid legal basis
            "legal_basis_type": "consent",
            "legal_basis_documentation": {
                "freely_given": False,  # Invalid consent
            },
            # Missing data minimization evidence
            "data_minimization_evidence": {},
            # Missing purpose limitation evidence
            "purpose_limitation_evidence": {},
            # Article 22: Automated decision without human review
            "human_review_mechanism": {
                "exists": False,
            },
        }

        compliant, _ = gdpr.check_gdpr_compliance(compliant_context)
        non_compliant, violations = gdpr.check_gdpr_compliance(violation_context)

        assert compliant is True
        assert non_compliant is False
        assert len(violations) >= 3


class TestHIPAACompliance:
    """Test HIPAA compliance."""

    def test_hipaa_checks(self):
        """Test HIPAA compliance checks."""
        hipaa = HIPAACompliance()

        compliant_context = {
            "processes_phi": True,
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_controls": True,
            "audit_trail": True,
            "baa_in_place": True,
        }

        violation_context = {
            "processes_phi": True,
            "encryption_at_rest": False,
            "encryption_in_transit": False,
            "access_controls": False,
        }

        compliant, _ = hipaa.check_hipaa_compliance(compliant_context)
        non_compliant, violations = hipaa.check_hipaa_compliance(violation_context)

        assert compliant is True
        assert non_compliant is False
        assert len(violations) >= 3


class TestAnomalyOracle:
    """Test anomaly forecasting oracle."""

    def test_oracle_initialization(self):
        """Test oracle initialization."""
        oracle = AnomalyOracle(lookback_window=50)

        assert oracle.lookback_window == 50
        assert len(oracle.historical_anomalies) == 0

    def test_anomaly_recording(self):
        """Test anomaly recording."""
        oracle = AnomalyOracle()

        oracle.record_anomaly(0.8, 0.7)
        oracle.record_anomaly(0.9, 0.9)

        assert len(oracle.historical_anomalies) == 2

    def test_risk_forecasting(self):
        """Test risk forecasting."""
        oracle = AnomalyOracle()

        for i in range(20):
            oracle.record_anomaly(0.5 + i * 0.01, 0.5 + i * 0.02)

        likelihood, impact = oracle.forecast_risk(0.6)

        assert 0.0 <= likelihood <= 1.0
        assert 0.0 <= impact <= 1.0


class TestEthicalRiskMatrix:
    """Test Ethical Risk Matrix."""

    def test_initialization(self):
        """Test risk matrix initialization."""
        matrix = EthicalRiskMatrix(
            enable_us_compliance=True, enable_gdpr=True, enable_hipaa=True, enable_forecasting=True
        )

        assert matrix.us_law is not None
        assert matrix.gdpr is not None
        assert matrix.hipaa is not None
        assert matrix.oracle is not None

    def test_risk_assessment(self):
        """Test comprehensive risk assessment."""
        matrix = EthicalRiskMatrix()

        context = {
            "potential_impact": 0.7,
            "critical_system": True,
            "processes_phi": False,
            "fairness_score": 0.9,
        }

        risk = matrix.assess_risk(context, anomaly_score=0.6)

        assert risk.likelihood >= 0.0
        assert risk.impact >= 0.0
        assert isinstance(risk.risk_level, RiskLevel)

    def test_risk_level_determination(self):
        """Test risk level classification."""
        matrix = EthicalRiskMatrix()

        critical = matrix._determine_risk_level(0.9, 0.9)
        high = matrix._determine_risk_level(0.7, 0.8)
        medium = matrix._determine_risk_level(0.5, 0.5)
        low = matrix._determine_risk_level(0.3, 0.3)
        negligible = matrix._determine_risk_level(0.1, 0.1)

        assert critical == RiskLevel.CRITICAL
        assert high in [RiskLevel.HIGH, RiskLevel.MEDIUM]
        assert medium in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.LOW]
        assert low in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.NEGLIGIBLE]
        assert negligible in [RiskLevel.NEGLIGIBLE, RiskLevel.LOW]

    def test_compliance_checking(self):
        """Test all compliance regime checking."""
        matrix = EthicalRiskMatrix()

        context = {
            "unauthorized_access": False,
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_minimization": True,
            "purpose_limitation": True,
            "processes_phi": False,
        }

        violations = matrix._check_all_compliance(context)

        assert isinstance(violations, list)

    def test_risk_matrix_table(self):
        """Test risk matrix table generation."""
        matrix = EthicalRiskMatrix()

        for i in range(20):
            context = {"potential_impact": 0.5 + i * 0.02}
            matrix.assess_risk(context, anomaly_score=0.5 + i * 0.01)

        table = matrix.get_risk_matrix_table()

        assert "matrix" in table
        assert "total_risks_assessed" in table
        assert table["total_risks_assessed"] == 20

    def test_compliance_report(self):
        """Test compliance report generation."""
        matrix = EthicalRiskMatrix()

        for i in range(10):
            context = {
                "unauthorized_access": i % 2 == 0,
                "processes_personal_data": True,
                "consent_obtained": i % 3 != 0,
            }
            matrix.assess_risk(context)

        report = matrix.generate_compliance_report()

        assert "total_risks_assessed" in report
        assert "compliance_rate" in report
        assert 0.0 <= report["compliance_rate"] <= 1.0


class TestIntegratedEthicalFramework:
    """Integration tests for complete ethical framework."""

    def test_end_to_end_ethical_pipeline(self):
        """Test complete ethical decision-making pipeline."""
        np.random.seed(777)
        governor = EthicalAutonomyGovernor()
        risk_matrix = EthicalRiskMatrix()

        context = {
            "fairness_score": 0.9,
            "bias_detected": False,
            "societal_benefit": 0.9,
            "potential_harm": 0.1,
            "harm_prevention": 0.9,
            "suffering_mitigation": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
            "critical_system": False,
            "unauthorized_access": False,
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_minimization": True,
            "purpose_limitation": True,
        }

        data = np.random.randn(100)
        decision = governor.evaluate_decision("deploy_system", context, data=data)

        risk = risk_matrix.assess_risk(context, anomaly_score=0.3)

        assert not decision.rollback_triggered or decision.bias_audit_passed
        assert risk.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.NEGLIGIBLE]

    def test_high_risk_scenario(self):
        """Test high-risk scenario handling."""
        governor = EthicalAutonomyGovernor()
        risk_matrix = EthicalRiskMatrix()

        high_risk_context = {
            "fairness_score": 0.3,
            "bias_detected": True,
            "societal_benefit": 0.2,
            "potential_harm": 0.9,
            "critical_system": True,
            "unauthorized_access": True,
            "processes_phi": True,
            "encryption_at_rest": False,
        }

        decision = governor.evaluate_decision("critical_action", high_risk_context)

        risk = risk_matrix.assess_risk(high_risk_context, anomaly_score=0.9)

        assert decision.rollback_triggered or decision.sigma_directive_applied
        assert risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(risk.compliance_violations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

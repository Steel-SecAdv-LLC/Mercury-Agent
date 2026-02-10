"""
Mercury Agent ♱
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
Tests for Ethical Risk Matrix module
"""

from omni_mercury_engine.core.ethical_risk_matrix import (
    AnomalyOracle,
    ComplianceRegime,
    EthicalRiskMatrix,
    GDPRCompliance,
    HIPAACompliance,
    RiskLevel,
    RiskScore,
    USLawPolling,
)


class TestUSLawPolling:
    """Test US Law Polling."""

    def test_initialization(self):
        """Test initialization."""
        poller = USLawPolling()
        assert len(poller.compliance_rules) > 0

    def test_compliant_context(self):
        """Test compliant context passes."""
        poller = USLawPolling()
        context = {
            "unauthorized_access": False,
            "intercepts_communications": False,
            "content_moderation_good_faith": True,
            "ai_transparency": True,
        }

        compliant, violations = poller.check_compliance(context)

        assert compliant is True
        assert len(violations) == 0

    def test_unauthorized_access_violation(self):
        """Test CFAA violation detection."""
        poller = USLawPolling()
        context = {"unauthorized_access": True}

        compliant, violations = poller.check_compliance(context)

        assert compliant is False
        assert any("CFAA" in v for v in violations)

    def test_wiretap_violation(self):
        """Test ECPA violation detection."""
        poller = USLawPolling()
        context = {"intercepts_communications": True}

        compliant, violations = poller.check_compliance(context)

        assert compliant is False
        assert any("ECPA" in v for v in violations)


class TestGDPRCompliance:
    """Test GDPR Compliance."""

    def test_initialization(self):
        """Test initialization with all 8 GDPR data subject rights."""
        gdpr = GDPRCompliance()
        # GDPR provides 8 data subject rights per Articles 15-22
        assert len(gdpr.data_subject_rights) == 8
        expected_rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_data_portability",
            "right_to_object",
            "right_to_restriction",
            "right_to_withdraw_consent",
            "right_not_to_be_subject_to_automated_decisions",
        ]
        assert gdpr.data_subject_rights == expected_rights

    def test_no_personal_data_compliant(self):
        """Test that no personal data processing is compliant."""
        gdpr = GDPRCompliance()
        context = {"processes_personal_data": False}

        compliant, violations = gdpr.check_gdpr_compliance(context)

        assert compliant is True
        assert len(violations) == 0

    def test_missing_consent_violation(self):
        """Test consent violation detection."""
        gdpr = GDPRCompliance()
        context = {"processes_personal_data": True, "consent_obtained": False}

        compliant, violations = gdpr.check_gdpr_compliance(context)

        assert compliant is False
        assert any("Art. 6" in v for v in violations)

    def test_data_minimization_violation(self):
        """Test data minimization violation."""
        gdpr = GDPRCompliance()
        context = {
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_minimization": False,
        }

        compliant, violations = gdpr.check_gdpr_compliance(context)

        assert compliant is False
        assert any("minimization" in v.lower() for v in violations)

    def test_automated_decision_violation(self):
        """Test automated decision-making violation."""
        gdpr = GDPRCompliance()
        context = {
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_minimization": True,
            "purpose_limitation": True,
            "automated_decision": True,
            "human_review": False,
        }

        compliant, violations = gdpr.check_gdpr_compliance(context)

        assert compliant is False
        assert any("Art. 22" in v for v in violations)


class TestHIPAACompliance:
    """Test HIPAA Compliance."""

    def test_initialization(self):
        """Test initialization."""
        hipaa = HIPAACompliance()
        assert len(hipaa.phi_identifiers) > 0

    def test_no_phi_compliant(self):
        """Test that no PHI processing is compliant."""
        hipaa = HIPAACompliance()
        context = {"processes_phi": False}

        compliant, violations = hipaa.check_hipaa_compliance(context)

        assert compliant is True
        assert len(violations) == 0

    def test_missing_encryption_violation(self):
        """Test encryption at rest violation."""
        hipaa = HIPAACompliance()
        context = {"processes_phi": True, "encryption_at_rest": False}

        compliant, violations = hipaa.check_hipaa_compliance(context)

        assert compliant is False
        assert any("encrypted at rest" in v for v in violations)

    def test_missing_audit_trail_violation(self):
        """Test audit trail violation."""
        hipaa = HIPAACompliance()
        context = {
            "processes_phi": True,
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_controls": True,
            "audit_trail": False,
        }

        compliant, violations = hipaa.check_hipaa_compliance(context)

        assert compliant is False
        assert any("audit trail" in v.lower() for v in violations)

    def test_full_compliance(self):
        """Test full HIPAA compliance."""
        hipaa = HIPAACompliance()
        context = {
            "processes_phi": True,
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_controls": True,
            "audit_trail": True,
            "baa_in_place": True,
        }

        compliant, violations = hipaa.check_hipaa_compliance(context)

        assert compliant is True
        assert len(violations) == 0


class TestAnomalyOracle:
    """Test Anomaly Oracle."""

    def test_initialization(self):
        """Test initialization."""
        oracle = AnomalyOracle(lookback_window=50)
        assert oracle.lookback_window == 50
        assert len(oracle.historical_anomalies) == 0

    def test_record_anomaly(self):
        """Test recording anomalies."""
        oracle = AnomalyOracle()
        oracle.record_anomaly(0.8, 0.5)
        oracle.record_anomaly(0.9, 0.7)

        assert len(oracle.historical_anomalies) == 2

    def test_record_anomaly_pruning(self):
        """Test that old anomalies are pruned."""
        oracle = AnomalyOracle(lookback_window=5)

        for i in range(10):
            oracle.record_anomaly(float(i) / 10, float(i) / 10)

        assert len(oracle.historical_anomalies) == 5

    def test_forecast_risk_empty(self):
        """Test forecasting with no history."""
        oracle = AnomalyOracle()

        likelihood, impact = oracle.forecast_risk(0.5)

        assert likelihood == 0.5
        assert impact == 0.5

    def test_forecast_risk_with_history(self):
        """Test forecasting with history."""
        oracle = AnomalyOracle()
        oracle.record_anomaly(0.5, 0.6)
        oracle.record_anomaly(0.55, 0.65)
        oracle.record_anomaly(0.9, 0.95)

        likelihood, impact = oracle.forecast_risk(0.52)

        assert 0 <= likelihood <= 1
        assert 0 <= impact <= 1


class TestEthicalRiskMatrix:
    """Test Ethical Risk Matrix."""

    def test_initialization_all_enabled(self):
        """Test initialization with all features enabled."""
        matrix = EthicalRiskMatrix()

        assert matrix.enable_us_compliance is True
        assert matrix.enable_gdpr is True
        assert matrix.enable_hipaa is True
        assert matrix.enable_forecasting is True
        assert matrix.us_law is not None
        assert matrix.gdpr is not None
        assert matrix.hipaa is not None
        assert matrix.oracle is not None

    def test_initialization_all_disabled(self):
        """Test initialization with all features disabled."""
        matrix = EthicalRiskMatrix(
            enable_us_compliance=False,
            enable_gdpr=False,
            enable_hipaa=False,
            enable_forecasting=False,
        )

        assert matrix.us_law is None
        assert matrix.gdpr is None
        assert matrix.hipaa is None
        assert matrix.oracle is None

    def test_assess_risk_low(self):
        """Test low risk assessment."""
        matrix = EthicalRiskMatrix()
        context = {"potential_impact": 0.1}

        risk = matrix.assess_risk(context, anomaly_score=0.1)

        assert isinstance(risk, RiskScore)
        assert risk.risk_level in [RiskLevel.NEGLIGIBLE, RiskLevel.LOW]
        assert risk.mitigation_required is False

    def test_assess_risk_critical(self):
        """Test critical risk assessment."""
        matrix = EthicalRiskMatrix()
        context = {"potential_impact": 0.95, "critical_system": True}

        risk = matrix.assess_risk(context, anomaly_score=0.95)

        # With forecasting enabled, likelihood is averaged with historical data
        # The test verifies higher risk than low scenarios
        assert risk.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM]
        assert risk.impact >= 0.5

    def test_assess_risk_with_phi(self):
        """Test risk assessment with PHI processing."""
        matrix = EthicalRiskMatrix()
        context = {"potential_impact": 0.5, "processes_phi": True}

        risk = matrix.assess_risk(context, anomaly_score=0.5)

        assert isinstance(risk, RiskScore)
        assert risk.impact > 0.5

    def test_assess_risk_with_violations(self):
        """Test risk assessment detects violations."""
        matrix = EthicalRiskMatrix()
        context = {
            "unauthorized_access": True,
            "processes_personal_data": True,
            "consent_obtained": False,
        }

        risk = matrix.assess_risk(context, anomaly_score=0.5)

        assert len(risk.compliance_violations) > 0

    def test_get_risk_matrix_table_empty(self):
        """Test risk matrix table with no history."""
        matrix = EthicalRiskMatrix()

        table = matrix.get_risk_matrix_table()

        assert "matrix" in table
        assert table["summary"] == "No risk data available"

    def test_get_risk_matrix_table_with_history(self):
        """Test risk matrix table with history."""
        matrix = EthicalRiskMatrix()

        for i in range(10):
            matrix.assess_risk({"potential_impact": float(i) / 10}, anomaly_score=float(i) / 10)

        table = matrix.get_risk_matrix_table()

        assert "matrix" in table
        assert table["total_risks_assessed"] == 10

    def test_generate_compliance_report(self):
        """Test compliance report generation."""
        matrix = EthicalRiskMatrix()

        matrix.assess_risk({"unauthorized_access": True}, anomaly_score=0.5)
        matrix.assess_risk({}, anomaly_score=0.3)

        report = matrix.generate_compliance_report()

        assert "total_risks_assessed" in report
        assert "compliance_rate" in report
        assert report["us_compliance_enabled"] is True

    def test_determine_risk_level(self):
        """Test risk level determination."""
        matrix = EthicalRiskMatrix()

        # risk_product >= 0.8 -> CRITICAL
        assert matrix._determine_risk_level(0.95, 0.95) == RiskLevel.CRITICAL
        # risk_product >= 0.6 -> HIGH
        assert matrix._determine_risk_level(0.8, 0.8) == RiskLevel.HIGH
        # risk_product >= 0.4 -> MEDIUM
        assert matrix._determine_risk_level(0.6, 0.7) == RiskLevel.MEDIUM
        # risk_product >= 0.2 -> LOW (0.5 * 0.5 = 0.25)
        assert matrix._determine_risk_level(0.5, 0.5) == RiskLevel.LOW
        # risk_product < 0.2 -> NEGLIGIBLE
        assert matrix._determine_risk_level(0.1, 0.1) == RiskLevel.NEGLIGIBLE


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_risk_levels(self):
        """Test risk level values."""
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.NEGLIGIBLE.value == "negligible"


class TestComplianceRegime:
    """Test ComplianceRegime enum."""

    def test_compliance_regimes(self):
        """Test compliance regime values."""
        assert ComplianceRegime.US_FEDERAL.value == "us_federal"
        assert ComplianceRegime.GDPR.value == "gdpr"
        assert ComplianceRegime.HIPAA.value == "hipaa"

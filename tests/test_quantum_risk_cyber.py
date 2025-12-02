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

"""Tests for Quantum Risk Cyber module"""

from omni_anomaly_engine.cyber.quantum_risk_cyber import QuantumRiskCyber, ThreatLevel, CryptoSystem


def test_quantum_risk_initialization():
    """Test quantum risk cyber system initialization"""
    system = QuantumRiskCyber(threat_timeline_years=8, preparedness_threshold=0.75)
    assert system.threat_timeline_years == 8
    assert system.preparedness_threshold == 0.75
    assert isinstance(system.vulnerability_scan_history, list)


def test_assess_quantum_threat_level():
    """Test quantum threat level assessment"""
    system = QuantumRiskCyber()

    assessment = system.assess_quantum_threat_level(
        current_year=2025, cryptosystem=CryptoSystem.RSA_2048
    )

    assert "threat_level" in assessment
    assert "years_until_vulnerable" in assessment
    assert "recommended_action" in assessment
    assert isinstance(assessment["threat_level"], ThreatLevel)


def test_post_quantum_readiness():
    """Test post-quantum cryptography readiness evaluation"""
    system = QuantumRiskCyber()

    current_crypto = {"RSA_2048": 0.6, "ECC_256": 0.4}

    readiness = system.evaluate_post_quantum_readiness(current_crypto)

    assert "readiness_score" in readiness
    assert "vulnerable_percentage" in readiness
    assert "recommendation" in readiness
    assert 0.0 <= readiness["readiness_score"] <= 1.0


def test_quantum_vulnerability_scan():
    """Test quantum vulnerability scanning"""
    system = QuantumRiskCyber()

    crypto_systems = [
        {"name": "RSA_2048", "usage": 0.5},
        {"name": "AES_256", "usage": 0.3},
        {"name": "ECC_256", "usage": 0.2},
    ]

    scan_result = system.scan_quantum_vulnerabilities(crypto_systems)

    assert "vulnerabilities_found" in scan_result
    assert "critical_count" in scan_result
    assert "total_scanned" in scan_result
    assert len(system.vulnerability_scan_history) > 0


def test_risk_timeline_modeling():
    """Test risk timeline modeling based on Bain report (95% see threats within 10 years)"""
    system = QuantumRiskCyber(threat_timeline_years=10)

    timeline = system.model_risk_timeline(current_year=2025)

    assert "critical_year" in timeline
    assert "risk_progression" in timeline
    assert timeline["critical_year"] >= 2025
    assert timeline["critical_year"] <= 2035


def test_threat_level_escalation():
    """Test that threat level escalates over time"""
    system = QuantumRiskCyber()

    early_assessment = system.assess_quantum_threat_level(2025, CryptoSystem.RSA_2048)
    late_assessment = system.assess_quantum_threat_level(2033, CryptoSystem.RSA_2048)

    threat_levels = {
        ThreatLevel.LOW: 1,
        ThreatLevel.MEDIUM: 2,
        ThreatLevel.HIGH: 3,
        ThreatLevel.CRITICAL: 4,
    }

    early_level = threat_levels[early_assessment["threat_level"]]
    late_level = threat_levels[late_assessment["threat_level"]]

    assert late_level >= early_level


def test_preparedness_gap_detection():
    """Test detection of preparedness gap (95% aware, only 10% have plans)"""
    system = QuantumRiskCyber(preparedness_threshold=0.15)

    low_preparedness = 0.10
    high_awareness = 0.95

    gap = system.detect_preparedness_gap(awareness=high_awareness, preparedness=low_preparedness)

    assert "has_gap" in gap
    assert "gap_size" in gap
    assert gap["has_gap"] is True
    assert gap["gap_size"] > 0.5


def test_crypto_system_prioritization():
    """Test prioritization of crypto systems for upgrade"""
    system = QuantumRiskCyber()

    systems = [
        {"name": "RSA_1024", "usage": 0.3, "quantum_resistant": False},
        {"name": "AES_256", "usage": 0.4, "quantum_resistant": True},
        {"name": "RSA_2048", "usage": 0.3, "quantum_resistant": False},
    ]

    priorities = system.prioritize_crypto_upgrades(systems)

    assert len(priorities) > 0
    assert all("priority_score" in p for p in priorities)

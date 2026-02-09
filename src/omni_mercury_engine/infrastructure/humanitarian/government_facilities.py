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

"""Government Facilities infrastructure monitoring.

Covers 16th CISA sector (Government Facilities) with focus on
democratic governance, transparency, and accountability.
"""

from typing import Any


class GovernmentFacilitiesMonitor:
    """Government facilities and public administration anomaly detector.

    Covers 16th CISA sector (Government Facilities) with focus on
    democratic governance, transparency, and accountability.
    """

    def __init__(self, ethical_config: dict[str, float] | None = None) -> None:
        """Initialize Government Facilities Monitor.

        Args:
            ethical_config: Ethical scalar configuration
        """
        self.facility_types = {
            "executive": {
                "facilities": ["presidential", "cabinet", "agencies", "executive_offices"],
                "processes": ["policy_implementation", "regulation", "enforcement"],
            },
            "legislative": {
                "facilities": ["parliament", "congress", "senate", "house", "state_legislatures"],
                "processes": ["lawmaking", "oversight", "budget_approval"],
            },
            "judicial": {
                "facilities": ["supreme_court", "appeals_courts", "district_courts", "tribunals"],
                "processes": [
                    "case_adjudication",
                    "legal_interpretation",
                    "justice_administration",
                ],
            },
            "electoral": {
                "facilities": ["polling_stations", "election_offices", "vote_counting_centers"],
                "processes": [
                    "voter_registration",
                    "voting",
                    "ballot_counting",
                    "result_certification",
                ],
            },
            "emergency": {
                "facilities": ["eoc", "fusion_centers", "coordination_centers"],
                "processes": [
                    "incident_response",
                    "interagency_coordination",
                    "resource_allocation",
                ],
            },
            "educational": {
                "facilities": ["public_schools", "universities", "libraries", "research_centers"],
                "processes": ["education_delivery", "research", "public_access"],
            },
        }

        self.ethical_scalars = {
            "omni_justitia": ethical_config.get("omni_justitia", 0.95) if ethical_config else 0.95,
            "transparency": ethical_config.get("transparency", 0.90) if ethical_config else 0.90,
            "accountability": (
                ethical_config.get("accountability", 0.90) if ethical_config else 0.90
            ),
            "democratic_norms": (
                ethical_config.get("democratic_norms", 0.92) if ethical_config else 0.92
            ),
        }

    def detect(self, data: dict[str, Any], facility_type: str) -> dict[str, Any]:
        """Detect anomalies in government facilities and processes.

        Args:
            data: Facility access logs, system availability, process metrics
            facility_type: Type of facility ('executive', 'legislative',
                'judicial', 'electoral', etc.)

        Returns:
            Detection results with threat assessment, democratic impact
        """
        if facility_type not in self.facility_types:
            raise ValueError(f"Unknown facility type: {facility_type}")

        access_violations = data.get("unauthorized_access_attempts", 0)
        system_downtime_pct = data.get("system_downtime_percent", 0)
        data_integrity_issues = data.get("data_integrity_compromised", False)
        process_disruptions = data.get("process_disruption_count", 0)

        access_anomaly = access_violations > 5
        availability_anomaly = system_downtime_pct > 5.0
        integrity_anomaly = data_integrity_issues
        process_anomaly = process_disruptions > 3

        anomaly_detected = (
            access_anomaly or availability_anomaly or integrity_anomaly or process_anomaly
        )

        threat_type = self._classify_threat(
            access_anomaly, availability_anomaly, integrity_anomaly, process_anomaly
        )

        democratic_impact = self._assess_democratic_impact(facility_type, threat_type)

        return {
            "facility_type": facility_type,
            "anomaly_detected": anomaly_detected,
            "threat_type": threat_type,
            "metrics": {
                "access_violations": access_violations,
                "system_downtime_percent": system_downtime_pct,
                "data_integrity_compromised": data_integrity_issues,
                "process_disruptions": process_disruptions,
            },
            "democratic_impact_score": democratic_impact,
            "ethical_compliance": self._calculate_ethical_compliance(
                facility_type, anomaly_detected
            ),
            "recommendations": self._generate_gov_recommendations(facility_type, threat_type),
        }

    def monitor_democratic_process(self, process_type: str, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor specific democratic processes for integrity.

        Args:
            process_type: 'voting', 'legislative', 'judicial', 'regulatory'
            data: Process-specific metrics

        Returns:
            Democratic process integrity assessment
        """
        if process_type == "voting":
            return self._monitor_electoral_integrity(data)
        elif process_type == "legislative":
            return self._monitor_legislative_process(data)
        elif process_type == "judicial":
            return self._monitor_judicial_process(data)
        elif process_type == "regulatory":
            return self._monitor_regulatory_process(data)
        else:
            raise ValueError(f"Unknown process type: {process_type}")

    def _monitor_electoral_integrity(self, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor electoral process integrity."""
        voter_turnout = data.get("voter_turnout_percent", 50)
        ballot_rejection_rate = data.get("ballot_rejection_percent", 1.0)
        counting_discrepancies = data.get("counting_discrepancies", 0)
        system_failures = data.get("system_failure_count", 0)

        turnout_anomaly = abs(voter_turnout - data.get("expected_turnout", 55)) > 15
        rejection_anomaly = ballot_rejection_rate > 3.0
        discrepancy_anomaly = counting_discrepancies > 0
        system_anomaly = system_failures > 0

        integrity_compromised = discrepancy_anomaly or (rejection_anomaly and system_anomaly)

        return {
            "process_type": "voting",
            "integrity_score": 1.0 if not integrity_compromised else 0.60,
            "anomalies_detected": {
                "turnout_unusual": turnout_anomaly,
                "high_ballot_rejection": rejection_anomaly,
                "counting_discrepancies": discrepancy_anomaly,
                "system_failures": system_anomaly,
            },
            "recommendations": [
                "Conduct audit of voting systems" if system_anomaly else None,
                "Investigate ballot rejection causes" if rejection_anomaly else None,
                "Recount ballots in affected precincts" if discrepancy_anomaly else None,
                "Ensure transparency in vote counting process",
            ],
        }

    def _monitor_legislative_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor legislative process integrity."""
        quorum_violations = data.get("quorum_violations", 0)
        procedural_irregularities = data.get("procedural_issues", 0)
        voting_system_errors = data.get("voting_errors", 0)

        return {
            "process_type": "legislative",
            "integrity_score": 1.0
            - (quorum_violations + procedural_irregularities + voting_system_errors) * 0.15,
            "issues": {
                "quorum_violations": quorum_violations,
                "procedural_irregularities": procedural_irregularities,
                "voting_errors": voting_system_errors,
            },
        }

    def _monitor_judicial_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor judicial process integrity."""
        case_backlog_days = data.get("case_backlog_days", 60)
        due_process_violations = data.get("due_process_issues", 0)

        return {
            "process_type": "judicial",
            "integrity_score": 1.0 if due_process_violations == 0 else 0.50,
            "backlog_severity": (
                "high" if case_backlog_days > 180 else "medium" if case_backlog_days > 90 else "low"
            ),
        }

    def _monitor_regulatory_process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Monitor regulatory process integrity."""
        comment_period_violations = data.get("comment_period_issues", 0)
        transparency_score = data.get("transparency_score", 0.85)

        return {
            "process_type": "regulatory",
            "integrity_score": transparency_score * (1.0 - comment_period_violations * 0.20),
        }

    def _classify_threat(
        self,
        access_anomaly: bool,
        availability_anomaly: bool,
        integrity_anomaly: bool,
        process_anomaly: bool,
    ) -> str:
        """Classify threat type based on anomaly patterns."""
        if integrity_anomaly:
            return "data_integrity_compromise"
        elif access_anomaly and process_anomaly:
            return "insider_threat_or_espionage"
        elif availability_anomaly:
            return "denial_of_service_or_sabotage"
        elif access_anomaly:
            return "unauthorized_access_attempt"
        elif process_anomaly:
            return "operational_disruption"
        else:
            return "none"

    def _assess_democratic_impact(self, facility_type: str, threat_type: str) -> float:
        """Assess impact on democratic processes."""
        facility_criticality = {
            "electoral": 1.00,
            "legislative": 0.95,
            "judicial": 0.95,
            "executive": 0.90,
            "emergency": 0.85,
            "educational": 0.75,
        }

        threat_severity = {
            "data_integrity_compromise": 1.00,
            "insider_threat_or_espionage": 0.90,
            "denial_of_service_or_sabotage": 0.85,
            "unauthorized_access_attempt": 0.70,
            "operational_disruption": 0.60,
            "none": 0.0,
        }

        base_impact = facility_criticality.get(facility_type, 0.50)
        threat_impact = threat_severity.get(threat_type, 0.0)

        return base_impact * threat_impact * self.ethical_scalars["democratic_norms"]

    def _calculate_ethical_compliance(self, facility_type: str, anomaly_detected: bool) -> float:
        """Calculate ethical compliance score."""
        if not anomaly_detected:
            return 0.95

        critical_facilities = {"electoral", "judicial"}
        if facility_type in critical_facilities:
            return 0.60
        else:
            return 0.75

    def _generate_gov_recommendations(self, facility_type: str, threat_type: str) -> list[Any]:
        """Generate recommendations for government facility threats."""
        if threat_type == "data_integrity_compromise":
            return [
                "CRITICAL: Isolate affected systems immediately",
                "Conduct forensic analysis of data alterations",
                "Notify law enforcement and oversight bodies",
                "Implement enhanced access controls and audit logging",
            ]
        elif threat_type == "insider_threat_or_espionage":
            return [
                "URGENT: Initiate counterintelligence investigation",
                "Review access privileges for all personnel",
                "Implement enhanced monitoring of sensitive areas",
                "Conduct security clearance reviews",
            ]
        elif threat_type == "denial_of_service_or_sabotage":
            return [
                "Activate backup systems and continuity plans",
                "Investigate source of service disruption",
                "Implement DDoS mitigation if cyber-based",
                "Ensure public communication about service status",
            ]
        elif threat_type == "unauthorized_access_attempt":
            return [
                "Review and strengthen access control policies",
                "Investigate identity and motive of unauthorized actors",
                "Enhance physical and cyber security measures",
            ]
        else:
            return [
                "Maintain enhanced situational awareness",
                "Continue monitoring for anomalies",
                "Ensure compliance with transparency and accountability standards",
            ]

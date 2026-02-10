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
Quantum Risk Cybersecurity Module

Inspired by Bain & Company Technology Report 2025 quantum cybersecurity insights:
- 95% of tech leaders believe quantum computing will pose security risks within 10 years
- Only 10% have a plan to address quantum threats
- Critical gap between awareness and preparedness
- Post-quantum cryptography transition urgency

Research sources:
- Bain & Company Technology Report 2025 (https://www.bain.com/insights/topics/technology-report/)
- Wikipedia - Quantum computing (https://en.wikipedia.org/wiki/Quantum_computing)
- Wikipedia - Post-quantum cryptography (https://en.wikipedia.org/wiki/Post-quantum_cryptography)

"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ThreatLevel(Enum):
    """Quantum threat severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class CryptoSystem(Enum):
    """Common cryptographic systems and their quantum vulnerability."""

    RSA_1024 = "RSA-1024"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"
    ECC_256 = "ECC-256"
    ECC_384 = "ECC-384"
    AES_128 = "AES-128"
    AES_256 = "AES-256"
    DH_2048 = "DH-2048"
    ECDH = "ECDH"


@dataclass
class QuantumThreat:
    """Represents a quantum cybersecurity threat."""

    threat_id: str
    threat_type: str
    severity: ThreatLevel
    estimated_timeline_years: float
    mitigation_status: str


class QuantumRiskCyber:
    """
    Quantum cybersecurity risk management system.

    Implements quantum threat detection, post-quantum cryptography readiness,
    and risk assessment inspired by Bain 2025 report findings.
    """

    def __init__(
        self, threat_timeline_years: float = 10.0, preparedness_threshold: float = 0.5
    ) -> None:
        """
        Initialize quantum risk cybersecurity system.

        Args:
            threat_timeline_years: Timeline for quantum threat realization
                (default 10 from Bain report)
            preparedness_threshold: Threshold for adequate preparedness (0-1)
        """
        self.threat_timeline_years = threat_timeline_years
        self.preparedness_threshold = preparedness_threshold
        self.threat_catalog: list[QuantumThreat] = []
        self.preparedness_score = 0.1
        self.vulnerability_scan_history: list[dict[str, Any]] = []

    def assess_quantum_vulnerability(
        self, system_components: list[str], encryption_methods: list[str]
    ) -> dict[str, float]:
        """
        Assess vulnerability to quantum attacks.

        Inspired by Bain finding: 95% see risks, 10% have plans.

        Args:
            system_components: List of system component identifiers
            encryption_methods: List of encryption methods used

        Returns:
            Vulnerability scores by component
        """
        vulnerabilities = {}

        quantum_vulnerable_methods = ["RSA", "ECC", "DH", "ECDH", "DSA", "ECDSA"]

        for component in system_components:
            vulnerability = 0.0

            for method in encryption_methods:
                if any(vuln in method.upper() for vuln in quantum_vulnerable_methods):
                    vulnerability += 0.3

            vulnerabilities[component] = min(1.0, vulnerability)

        return vulnerabilities

    def assess_quantum_threat_level(
        self, current_year: int, cryptosystem: CryptoSystem
    ) -> dict[str, Any]:
        """
        Assess quantum threat level for a specific cryptosystem.

        Args:
            current_year: Current year for assessment
            cryptosystem: Cryptographic system to assess

        Returns:
            Assessment with threat level, timeline, and recommendations
        """
        quantum_vulnerable_systems = {
            CryptoSystem.RSA_1024,
            CryptoSystem.RSA_2048,
            CryptoSystem.RSA_4096,
            CryptoSystem.ECC_256,
            CryptoSystem.ECC_384,
            CryptoSystem.DH_2048,
            CryptoSystem.ECDH,
        }

        if cryptosystem not in quantum_vulnerable_systems:
            return {
                "threat_level": ThreatLevel.LOW,
                "years_until_vulnerable": float("inf"),
                "recommended_action": "System is quantum-resistant",
            }

        years_until_threat = self.threat_timeline_years - (current_year - 2025)

        if years_until_threat <= 2:
            threat_level = ThreatLevel.CRITICAL
            action = "IMMEDIATE: Migrate to post-quantum cryptography"
        elif years_until_threat <= 5:
            threat_level = ThreatLevel.HIGH
            action = "URGENT: Begin post-quantum transition planning"
        elif years_until_threat <= 8:
            threat_level = ThreatLevel.MEDIUM
            action = "PLAN: Evaluate post-quantum alternatives"
        else:
            threat_level = ThreatLevel.LOW
            action = "MONITOR: Track quantum computing advances"

        return {
            "threat_level": threat_level,
            "years_until_vulnerable": max(0, years_until_threat),
            "recommended_action": action,
        }

    def evaluate_post_quantum_readiness(self, current_crypto: dict[str, float]) -> dict[str, Any]:
        """
        Evaluate readiness for post-quantum cryptography.

        Args:
            current_crypto: Dict mapping crypto system names to usage percentages

        Returns:
            Readiness assessment with score and recommendations
        """
        quantum_resistant = ["AES", "SHA", "LATTICE", "HASH_BASED", "CODE_BASED"]

        resistant_usage = sum(
            usage
            for name, usage in current_crypto.items()
            if any(qr in name.upper() for qr in quantum_resistant)
        )

        vulnerable_usage = 1.0 - resistant_usage
        readiness_score = resistant_usage

        if readiness_score >= 0.8:
            recommendation = "EXCELLENT: Well prepared for quantum threats"
        elif readiness_score >= 0.5:
            recommendation = "GOOD: Continue expanding post-quantum coverage"
        elif readiness_score >= 0.3:
            recommendation = "FAIR: Accelerate post-quantum adoption"
        else:
            recommendation = "POOR: Critical need for post-quantum transition"

        return {
            "readiness_score": readiness_score,
            "vulnerable_percentage": vulnerable_usage,
            "recommendation": recommendation,
        }

    def scan_quantum_vulnerabilities(self, crypto_systems: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Scan cryptographic systems for quantum vulnerabilities.

        Args:
            crypto_systems: List of dicts with 'name' and 'usage' keys

        Returns:
            Scan results with vulnerability count and details
        """
        vulnerable_keywords = ["RSA", "ECC", "DH", "ECDH", "DSA", "ECDSA"]

        vulnerabilities = []
        critical_count = 0

        for system in crypto_systems:
            name = system["name"]
            usage = system.get("usage", 0.0)

            is_vulnerable = any(vk in name.upper() for vk in vulnerable_keywords)

            if is_vulnerable:
                severity = "CRITICAL" if usage > 0.3 else "HIGH" if usage > 0.1 else "MEDIUM"
                vulnerabilities.append(
                    {"system": name, "usage": usage, "severity": severity, "vulnerable": True}
                )
                if severity == "CRITICAL":
                    critical_count += 1

        scan_result = {
            "vulnerabilities_found": len(vulnerabilities),
            "critical_count": critical_count,
            "total_scanned": len(crypto_systems),
            "details": vulnerabilities,
        }

        self.vulnerability_scan_history.append(scan_result)

        return scan_result

    def model_risk_timeline(self, current_year: int) -> dict[str, Any]:
        """
        Model quantum risk timeline based on Bain report (95% see threats within 10 years).

        Args:
            current_year: Current year for timeline modeling

        Returns:
            Timeline model with critical year and risk progression
        """
        critical_year = current_year + int(self.threat_timeline_years * 0.5)
        threat_year = current_year + int(self.threat_timeline_years)

        risk_progression = []
        for year in range(current_year, threat_year + 1):
            years_passed = year - current_year
            risk_level = min(1.0, years_passed / self.threat_timeline_years)
            risk_progression.append({"year": year, "risk_level": risk_level})

        return {
            "critical_year": critical_year,
            "threat_realization_year": threat_year,
            "risk_progression": risk_progression,
            "bain_statistic": "95% of tech leaders expect quantum risks within 10 years",
        }

    def detect_preparedness_gap(self, awareness: float, preparedness: float) -> dict[str, Any]:
        """
        Detect preparedness gap (Bain finding: 95% aware, only 10% have plans).

        Args:
            awareness: Organization's quantum threat awareness level (0-1)
            preparedness: Organization's quantum threat preparedness level (0-1)

        Returns:
            Gap analysis with has_gap flag and gap size
        """
        gap_size = awareness - preparedness
        has_gap = gap_size > self.preparedness_threshold

        return {
            "has_gap": has_gap,
            "gap_size": gap_size,
            "awareness": awareness,
            "preparedness": preparedness,
            "threshold": self.preparedness_threshold,
            "recommendation": (
                "URGENT: Close preparedness gap"
                if has_gap
                else "GOOD: Preparedness matches awareness"
            ),
        }

    def prioritize_crypto_upgrades(self, systems: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Prioritize cryptographic systems for upgrade.

        Args:
            systems: List of dicts with 'name', 'usage', and 'quantum_resistant' keys

        Returns:
            Prioritized list of systems with priority scores
        """
        priorities = []

        for system in systems:
            name = system["name"]
            usage = system.get("usage", 0.0)
            quantum_resistant = system.get("quantum_resistant", False)

            if not quantum_resistant:
                priority_score = usage * 100.0

                if "RSA_1024" in name or "DH_1024" in name:
                    priority_score *= 2.0

                priorities.append(
                    {
                        "system": name,
                        "usage": usage,
                        "priority_score": priority_score,
                        "urgency": (
                            "CRITICAL"
                            if priority_score > 50
                            else "HIGH" if priority_score > 20 else "MEDIUM"
                        ),
                    }
                )

        priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        return priorities

    def post_quantum_readiness_check(
        self, current_crypto: list[str], pqc_alternatives: list[str] | None = None
    ) -> tuple[float, list[str]]:
        """
        Check readiness for post-quantum cryptography transition.

        Post-quantum cryptography (PQC) includes:
        - Lattice-based (NTRU, LWE)
        - Code-based (McEliece)
        - Hash-based (SPHINCS+)
        - Multivariate (Rainbow)
        - Isogeny-based (SIKE)

        Args:
            current_crypto: Current cryptographic methods
            pqc_alternatives: Available post-quantum alternatives

        Returns:
            Tuple of (readiness_score, recommendations)
        """
        pqc_alternatives = pqc_alternatives or []

        pqc_methods = [
            "NTRU",
            "LWE",
            "McEliece",
            "SPHINCS",
            "Rainbow",
            "SIKE",
            "Kyber",
            "Dilithium",
        ]

        pqc_count = sum(
            1 for alt in pqc_alternatives if any(pqc in alt.upper() for pqc in pqc_methods)
        )

        readiness_score = min(1.0, pqc_count / max(1, len(current_crypto)))

        recommendations = []
        for crypto in current_crypto:
            if not any(pqc in crypto.upper() for pqc in pqc_methods):
                if "RSA" in crypto.upper() or "ECC" in crypto.upper():
                    recommendations.append(
                        f"Replace {crypto} with Kyber (lattice-based) or Dilithium"
                    )
                elif "DH" in crypto.upper():
                    recommendations.append(f"Replace {crypto} with post-quantum key exchange")

        self.preparedness_score = readiness_score

        return readiness_score, recommendations

    def threat_timeline_modeling(
        self, threat_type: str, current_year: int = 2025
    ) -> dict[str, Any]:
        """
        Model quantum threat timeline.

        Based on Bain report: 95% of leaders believe quantum threats
        will materialize within 10 years.

        Args:
            threat_type: Type of quantum threat
            current_year: Current year for timeline calculation

        Returns:
            Timeline model with key milestones
        """
        threat_realization_year = current_year + self.threat_timeline_years
        critical_year = current_year + (self.threat_timeline_years * 0.5)

        timeline = {
            "threat_type": threat_type,
            "current_year": current_year,
            "critical_preparedness_year": int(critical_year),
            "threat_realization_year": int(threat_realization_year),
            "years_remaining": self.threat_timeline_years,
            "urgency_level": "HIGH" if self.threat_timeline_years <= 5 else "MEDIUM",
            "bain_statistic": "95% of tech leaders expect quantum risks within 10 years",
        }

        return timeline

    def quantum_threat_detection(
        self, data_patterns: np.ndarray[Any, Any], anomaly_threshold: float = 0.7
    ) -> tuple[bool, float, str]:
        """
        Detect potential quantum-based attacks or anomalies.

        Args:
            data_patterns: Input data patterns to analyze
            anomaly_threshold: Threshold for threat detection

        Returns:
            Tuple of (threat_detected, confidence, threat_description)
        """
        pattern_variance = np.var(data_patterns)
        pattern_entropy = -np.sum(
            np.abs(data_patterns) * np.log(np.abs(data_patterns) + 1e-10)
        ) / len(data_patterns)

        anomaly_score = (pattern_variance + pattern_entropy) / 2.0
        anomaly_score = min(1.0, anomaly_score)

        threat_detected = anomaly_score > anomaly_threshold

        if threat_detected:
            if anomaly_score > 0.9:
                description = "Critical: Potential quantum algorithm attack pattern detected"
            elif anomaly_score > 0.8:
                description = "High: Suspicious quantum-like computational patterns"
            else:
                description = "Medium: Anomalous patterns requiring investigation"
        else:
            description = "Normal: No quantum threat patterns detected"

        return threat_detected, anomaly_score, description

    def preparedness_gap_analysis(self) -> dict[str, Any]:
        """
        Analyze the preparedness gap.

        Inspired by Bain finding: Critical gap between 95% awareness
        and 10% having plans.

        Returns:
            Gap analysis with recommendations
        """
        awareness_level = 0.95
        planning_level = 0.10

        gap = awareness_level - planning_level

        recommendations: list[str] = []
        analysis: dict[str, Any] = {
            "awareness_level": awareness_level,
            "planning_level": planning_level,
            "preparedness_gap": gap,
            "current_preparedness": self.preparedness_score,
            "target_preparedness": self.preparedness_threshold,
            "needs_improvement": self.preparedness_score < self.preparedness_threshold,
            "bain_insight": (
                "Only 10% of tech leaders have quantum threat plans despite 95% awareness"
            ),
            "recommendations": recommendations,
        }

        if self.preparedness_score < 0.3:
            recommendations.extend(
                [
                    "Immediate: Conduct quantum vulnerability assessment",
                    "Immediate: Begin post-quantum cryptography evaluation",
                    "Short-term: Develop quantum threat response plan",
                ]
            )
        elif self.preparedness_score < 0.6:
            recommendations.extend(
                [
                    "Short-term: Pilot post-quantum cryptography implementations",
                    "Medium-term: Train security team on quantum threats",
                    "Medium-term: Update security policies for quantum era",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Maintain: Continue monitoring quantum computing advances",
                    "Enhance: Expand post-quantum cryptography coverage",
                    "Lead: Share learnings with broader community",
                ]
            )

        return analysis


class PostQuantumMigrationPlanner:
    """Post-quantum cryptography migration planning system.

    Models quantum computing threats to current encryption and plans
    migration to NIST Post-Quantum Cryptography standards.

    Based on NIST PQC standardization (2024) and Bain 2025 findings:
    - 95% awareness of quantum threats within 10 years
    - Only 10% have concrete migration plans
    - Urgent need for structured migration planning

    NIST PQC Selected Algorithms (2024):
    - CRYSTALS-Kyber: Key encapsulation (lattice-based)
    - CRYSTALS-Dilithium: Digital signatures (lattice-based)
    - SPHINCS+: Digital signatures (hash-based)
    - FALCON: Digital signatures (lattice-based)
    """

    def __init__(self) -> None:
        """Initialize post-quantum migration planner."""
        self.nist_pqc_algorithms = {
            "key_encapsulation": {
                "CRYSTALS-Kyber": {
                    "type": "lattice-based",
                    "security_levels": [512, 768, 1024],
                    "use_case": "Key establishment, TLS, VPNs",
                    "maturity": "standardized_2024",
                },
            },
            "digital_signatures": {
                "CRYSTALS-Dilithium": {
                    "type": "lattice-based",
                    "security_levels": [2, 3, 5],
                    "use_case": "General-purpose signatures",
                    "maturity": "standardized_2024",
                },
                "FALCON": {
                    "type": "lattice-based",
                    "security_levels": [512, 1024],
                    "use_case": "Constrained environments",
                    "maturity": "standardized_2024",
                },
                "SPHINCS+": {
                    "type": "hash-based",
                    "security_levels": [128, 192, 256],
                    "use_case": "High-security, stateless signatures",
                    "maturity": "standardized_2024",
                },
            },
        }

        self.vulnerable_algorithms = {
            "RSA": "Broken by Shor's algorithm on quantum computer",
            "ECC": "Broken by modified Shor's algorithm",
            "DSA": "Broken by Shor's algorithm",
            "ECDSA": "Broken by modified Shor's algorithm",
            "DH": "Broken by Shor's algorithm",
            "ECDH": "Broken by modified Shor's algorithm",
        }

    def assess_algorithm_vulnerability(
        self, algorithm: str, key_size: int, usage_context: str = "general"
    ) -> dict[str, Any]:
        """Assess vulnerability of current cryptographic algorithm.

        Args:
            algorithm: Current algorithm (e.g., 'RSA', 'ECC', 'AES')
            key_size: Key size in bits
            usage_context: Usage context ('tls', 'vpn', 'signatures', 'general')

        Returns:
            Vulnerability assessment with threat level and timeline
        """
        algorithm_upper = algorithm.upper()

        is_vulnerable = any(vuln in algorithm_upper for vuln in self.vulnerable_algorithms)

        if not is_vulnerable and "AES" in algorithm_upper and key_size >= 256:
            return {
                "algorithm": algorithm,
                "vulnerable_to_quantum": False,
                "threat_level": "low",
                "explanation": (
                    "AES-256 provides adequate security against quantum attacks "
                    "(Grover's algorithm)"
                ),
                "action_required": "none",
            }

        threat_timeline = self._calculate_threat_timeline(algorithm, key_size)
        recommended_pqc = self._recommend_pqc_algorithm(algorithm, usage_context)

        return {
            "algorithm": algorithm,
            "key_size": key_size,
            "vulnerable_to_quantum": is_vulnerable,
            "threat_level": "critical" if threat_timeline < 5 else "high",
            "threat_description": self.vulnerable_algorithms.get(
                algorithm_upper.split("-")[0], "Unknown vulnerability"
            ),
            "years_until_vulnerable": threat_timeline,
            "recommended_pqc": recommended_pqc,
            "migration_urgency": (
                "immediate" if threat_timeline < 3 else "high" if threat_timeline < 7 else "medium"
            ),
        }

    def _calculate_threat_timeline(self, algorithm: str, key_size: int) -> float:
        """Calculate estimated years until quantum threat materializes."""
        base_timeline = 10.0

        if "RSA" in algorithm.upper():
            if key_size <= 1024:
                return base_timeline * 0.5
            elif key_size <= 2048:
                return base_timeline * 0.7
            else:
                return base_timeline * 0.9
        elif "ECC" in algorithm.upper():
            if key_size <= 256:
                return base_timeline * 0.6
            else:
                return base_timeline * 0.8

        return base_timeline

    def _recommend_pqc_algorithm(
        self, current_algorithm: str, usage_context: str
    ) -> dict[str, str]:
        """Recommend NIST PQC algorithm to replace current algorithm."""
        algorithm_upper = current_algorithm.upper()

        if "RSA" in algorithm_upper or "DH" in algorithm_upper:
            return {
                "algorithm": "CRYSTALS-Kyber",
                "type": "key_encapsulation",
                "reason": "NIST-standardized lattice-based key encapsulation",
                "security_level": "768 (equivalent to AES-192)",
            }
        elif (
            "DSA" in algorithm_upper or "ECDSA" in algorithm_upper or usage_context == "signatures"
        ):
            if usage_context == "constrained":
                return {
                    "algorithm": "FALCON",
                    "type": "digital_signature",
                    "reason": "Optimized for constrained environments",
                    "security_level": "512 (equivalent to AES-128)",
                }
            else:
                return {
                    "algorithm": "CRYSTALS-Dilithium",
                    "type": "digital_signature",
                    "reason": "NIST-standardized general-purpose signatures",
                    "security_level": "3 (equivalent to AES-192)",
                }

        return {
            "algorithm": "CRYSTALS-Kyber",
            "type": "key_encapsulation",
            "reason": "Default NIST PQC recommendation",
            "security_level": "768",
        }

    def create_migration_plan(
        self, current_systems: list[dict[str, Any]], timeline_months: int = 24
    ) -> dict[str, Any]:
        """Create comprehensive migration plan to post-quantum cryptography.

        Args:
            current_systems: List of dicts with 'algorithm', 'key_size', 'usage', 'criticality'
            timeline_months: Target timeline for migration (default 24 months)

        Returns:
            Detailed migration plan with phases, priorities, and recommendations
        """
        migration_phases = {
            "phase_1_assessment": {
                "duration_months": max(2, timeline_months * 0.1),
                "activities": [
                    "Inventory all cryptographic systems",
                    "Assess quantum vulnerability for each system",
                    "Identify dependencies and integration points",
                    "Evaluate NIST PQC algorithm options",
                ],
            },
            "phase_2_pilot": {
                "duration_months": max(3, timeline_months * 0.15),
                "activities": [
                    "Deploy PQC in isolated test environment",
                    "Conduct performance and compatibility testing",
                    "Train security team on PQC implementation",
                    "Develop migration procedures and documentation",
                ],
            },
            "phase_3_hybrid": {
                "duration_months": max(6, timeline_months * 0.35),
                "activities": [
                    "Implement hybrid classical/PQC cryptography",
                    "Migrate non-critical systems first",
                    "Monitor performance and security",
                    "Refine migration procedures based on learnings",
                ],
            },
            "phase_4_full_migration": {
                "duration_months": max(9, timeline_months * 0.30),
                "activities": [
                    "Migrate critical systems to PQC",
                    "Decommission vulnerable classical cryptography",
                    "Conduct comprehensive security audits",
                    "Document migration outcomes",
                ],
            },
            "phase_5_maintenance": {
                "duration_months": max(4, timeline_months * 0.10),
                "activities": [
                    "Monitor PQC system performance",
                    "Track quantum computing advances",
                    "Update PQC implementations as standards evolve",
                    "Continuous improvement and optimization",
                ],
            },
        }

        priority_systems = sorted(
            current_systems,
            key=lambda x: (x.get("criticality", 0.5) * x.get("usage", 0.5)),
            reverse=True,
        )

        migration_recommendations = []
        for system in priority_systems:
            assessment = self.assess_algorithm_vulnerability(
                system["algorithm"],
                system.get("key_size", 2048),
                system.get("usage_context", "general"),
            )

            if assessment["vulnerable_to_quantum"]:
                migration_recommendations.append(
                    {
                        "system": system.get("name", system["algorithm"]),
                        "current_algorithm": system["algorithm"],
                        "recommended_pqc": assessment["recommended_pqc"],
                        "priority": (
                            "critical" if assessment["migration_urgency"] == "immediate" else "high"
                        ),
                        "estimated_effort": self._estimate_migration_effort(system),
                    }
                )

        return {
            "timeline_months": timeline_months,
            "migration_phases": migration_phases,
            "total_systems": len(current_systems),
            "vulnerable_systems": len(migration_recommendations),
            "migration_recommendations": migration_recommendations,
            "estimated_cost_range": (
                f"${len(migration_recommendations) * 50000}"
                f"-${len(migration_recommendations) * 200000}"
            ),
            "success_metrics": [
                "All critical systems migrated to PQC",
                "Zero quantum-vulnerable systems in production",
                "Performance within 10% of classical baseline",
                "Security audit findings resolved",
            ],
        }

    def _estimate_migration_effort(self, system: dict[str, Any]) -> str:
        """Estimate effort required to migrate a system."""
        criticality = system.get("criticality", 0.5)
        usage = system.get("usage", 0.5)

        effort_score = criticality * usage

        if effort_score > 0.7:
            return "high (6-12 months)"
        elif effort_score > 0.4:
            return "medium (3-6 months)"
        else:
            return "low (1-3 months)"

    def monitor_migration_progress(
        self, plan: dict[str, Any], completed_milestones: list[str]
    ) -> dict[str, Any]:
        """Monitor progress of PQC migration plan.

        Args:
            plan: Migration plan from create_migration_plan()
            completed_milestones: List of completed milestone identifiers

        Returns:
            Progress report with completion percentage and next steps
        """
        total_activities = sum(
            len(phase["activities"]) for phase in plan["migration_phases"].values()
        )

        completed_activities = len(completed_milestones)
        progress_percentage = (completed_activities / total_activities) * 100

        current_phase = None
        for phase_name, phase_data in plan["migration_phases"].items():
            phase_activities = len(phase_data["activities"])
            if completed_activities < phase_activities:
                current_phase = phase_name
                break
            completed_activities -= phase_activities

        if current_phase is None:
            current_phase = "phase_5_maintenance"

        return {
            "progress_percentage": progress_percentage,
            "current_phase": current_phase,
            "completed_milestones": len(completed_milestones),
            "total_milestones": total_activities,
            "systems_migrated": len([m for m in completed_milestones if "migrated" in m.lower()]),
            "total_systems_to_migrate": plan["vulnerable_systems"],
            "on_track": progress_percentage
            >= (100 * len(completed_milestones) / total_activities * 0.9),
            "next_steps": plan["migration_phases"][current_phase]["activities"],
        }

    def explain_pqc_algorithms(self) -> dict[str, Any]:
        """Provide detailed explanation of NIST PQC algorithms.

        Returns:
            Educational content about post-quantum cryptography
        """
        return {
            "nist_pqc_standardization": {
                "year": 2024,
                "process": "NIST Post-Quantum Cryptography Standardization",
                "selected_algorithms": list(self.nist_pqc_algorithms.keys()),
            },
            "algorithm_families": {
                "lattice_based": {
                    "examples": ["CRYSTALS-Kyber", "CRYSTALS-Dilithium", "FALCON"],
                    "security_basis": (
                        "Shortest Vector Problem (SVP) and Learning With Errors (LWE)"
                    ),
                    "advantages": "Fast, efficient, well-studied",
                    "quantum_resistance": "High confidence in long-term security",
                },
                "hash_based": {
                    "examples": ["SPHINCS+"],
                    "security_basis": "Cryptographic hash functions",
                    "advantages": "Conservative security assumptions, stateless",
                    "quantum_resistance": "Highest confidence (based on hash functions)",
                },
                "code_based": {
                    "examples": ["Classic McEliece (alternate candidate)"],
                    "security_basis": "Error-correcting codes",
                    "advantages": "Long history, conservative",
                    "quantum_resistance": "High confidence",
                },
            },
            "quantum_threats": {
                "shors_algorithm": (
                    "Breaks RSA, ECC, DH by solving factoring and discrete log problems"
                ),
                "grovers_algorithm": ("Weakens symmetric crypto (doubles required key length)"),
                "timeline": (
                    "10 years (per Bain 2025: 95% of leaders expect threats within 10 years)"
                ),
            },
            "migration_importance": [
                "Harvest now, decrypt later attacks already occurring",
                "Data encrypted today may be vulnerable when quantum computers arrive",
                "Regulatory requirements emerging (EU Cyber Resilience Act, etc.)",
                "Industry leaders beginning migrations now (10% have plans per Bain)",
            ],
        }

"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

"""
Intelligence Preparation Engine (IPB)

Implements the Intelligence Preparation of the Battlefield process
adapted for multi-domain anomaly detection:

1. DEFINE: Define the operational environment
2. DESCRIBE: Describe environmental effects on operations
3. EVALUATE: Evaluate threat capabilities
4. DETERMINE: Determine threat courses of action

Research Sources:
- Army FM 2-0: Intelligence, Chapter 5
- CISA All-Source Intelligence
- DARPA ANSR: Situational awareness
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EnvironmentDomain(Enum):
    """Operational domains for IPB analysis."""

    CYBER = "cyber"
    PHYSICAL = "physical"
    COGNITIVE = "cognitive"
    FINANCIAL = "financial"
    SOCIAL = "social"
    INFRASTRUCTURE = "infrastructure"
    MEDICAL = "medical"
    ENVIRONMENTAL = "environmental"
    SPACE = "space"


class ThreatCategory(Enum):
    """Categories of threats."""

    STATE_ACTOR = "state_actor"
    NON_STATE_ACTOR = "non_state_actor"
    INSIDER = "insider"
    CRIMINAL = "criminal"
    NATURAL = "natural"
    SYSTEMIC = "systemic"
    EMERGENT = "emergent"


class CourseOfAction(Enum):
    """Threat courses of action."""

    MOST_LIKELY = "most_likely"
    MOST_DANGEROUS = "most_dangerous"
    MOST_DISRUPTIVE = "most_disruptive"


@dataclass
class EnvironmentDefinition:
    """Definition of the operational environment (Phase 1)."""

    domain: EnvironmentDomain
    area_of_interest: dict[str, Any]
    area_of_influence: dict[str, Any]
    key_terrain: list[str]
    critical_assets: list[str]
    constraints: list[str]
    assumptions: list[str]
    timeline: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "domain": self.domain.value,
            "aoi": self.area_of_interest,
            "influence_area": self.area_of_influence,
            "key_terrain": self.key_terrain,
            "critical_assets": self.critical_assets,
            "constraints": self.constraints,
            "assumptions": self.assumptions,
        }


@dataclass
class EnvironmentEffect:
    """Environmental effect on operations (Phase 2)."""

    effect_id: str
    description: str
    affected_operations: list[str]
    severity: float  # 0-1
    probability: float  # 0-1
    mitigation_options: list[str]
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "id": self.effect_id,
            "description": self.description,
            "severity": self.severity,
            "probability": self.probability,
            "affected_ops": self.affected_operations,
            "mitigations": self.mitigation_options,
        }


@dataclass
class ThreatCapability:
    """Threat capability assessment (Phase 3)."""

    threat_id: str
    threat_name: str
    category: ThreatCategory
    capabilities: list[str]
    intent: str
    resources: dict[str, float]
    historical_actions: list[str]
    indicators: list[str]
    overall_rating: float  # 0-1

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "id": self.threat_id,
            "name": self.threat_name,
            "category": self.category.value,
            "capabilities": self.capabilities,
            "intent": self.intent,
            "rating": self.overall_rating,
            "indicators": self.indicators,
        }


@dataclass
class ThreatCOA:
    """Threat Course of Action (Phase 4)."""

    coa_id: str
    coa_type: CourseOfAction
    threat: ThreatCapability
    description: str
    objectives: list[str]
    phases: list[dict[str, Any]]
    indicators_and_warnings: list[str]
    probability: float
    impact: float
    decision_points: list[str]
    countermeasures: list[str]

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "id": self.coa_id,
            "type": self.coa_type.value,
            "threat": self.threat.threat_name,
            "description": self.description,
            "objectives": self.objectives,
            "phases": self.phases,
            "probability": self.probability,
            "impact": self.impact,
            "iaw": self.indicators_and_warnings,
            "countermeasures": self.countermeasures,
        }


@dataclass
class BattlefieldAssessment:
    """Complete IPB assessment result."""

    assessment_id: str
    timestamp: float
    environment: EnvironmentDefinition
    effects: list[EnvironmentEffect]
    threats: list[ThreatCapability]
    courses_of_action: list[ThreatCOA]
    priority_intelligence_requirements: list[str]
    collection_priorities: list[str]
    running_estimate: dict[str, Any]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "id": self.assessment_id,
            "timestamp": self.timestamp,
            "environment": self.environment.to_dict(),
            "effects": [e.to_dict() for e in self.effects],
            "threats": [t.to_dict() for t in self.threats],
            "coas": [c.to_dict() for c in self.courses_of_action],
            "pirs": self.priority_intelligence_requirements,
            "collection": self.collection_priorities,
            "confidence": self.confidence,
        }


class IPBEngine:
    """
    Intelligence Preparation of the Battlefield Engine.

    Implements the military intelligence preparation process for
    multi-domain anomaly detection:

    1. Define the operational environment
    2. Describe environmental effects
    3. Evaluate threat capabilities
    4. Determine threat courses of action

    This enables proactive threat anticipation rather than
    purely reactive detection.
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        domains: list[EnvironmentDomain] | None = None,
        enable_running_estimates: bool = True,
        historical_window_days: int = 30,
    ):
        """
        Initialize IPB Engine.

        Args:
            domains: Operational domains to analyze
            enable_running_estimates: Continuously update estimates
            historical_window_days: Window for historical analysis
        """
        self.domains = domains or list(EnvironmentDomain)
        self.enable_running_estimates = enable_running_estimates
        self.historical_window_days = historical_window_days

        # Storage
        self._environments: dict[EnvironmentDomain, EnvironmentDefinition] = {}
        self._effects: dict[str, EnvironmentEffect] = {}
        self._threats: dict[str, ThreatCapability] = {}
        self._coas: dict[str, ThreatCOA] = {}
        self._running_estimates: defaultdict[EnvironmentDomain, dict[str, Any]] = defaultdict(dict)
        self._observation_history: list[dict[str, Any]] = []

        # Priority intelligence requirements
        self._pirs: list[str] = []

        # Statistics
        self._stats = {
            "assessments_conducted": 0,
            "threats_identified": 0,
            "coas_generated": 0,
        }

        self._initialize_domain_templates()

        logger.info(f"IPBEngine initialized for {len(self.domains)} domains")

    def _initialize_domain_templates(self) -> None:
        """Initialize domain-specific templates."""
        self._domain_templates = {
            EnvironmentDomain.CYBER: {
                "key_terrain": [
                    "network_perimeter",
                    "authentication_systems",
                    "data_stores",
                    "critical_applications",
                ],
                "critical_assets": [
                    "credentials",
                    "encryption_keys",
                    "customer_data",
                    "intellectual_property",
                ],
                "threat_indicators": [
                    "unusual_traffic",
                    "failed_logins",
                    "privilege_escalation",
                    "data_exfiltration",
                ],
            },
            EnvironmentDomain.INFRASTRUCTURE: {
                "key_terrain": ["power_grid", "water_systems", "communications", "transportation"],
                "critical_assets": [
                    "scada_systems",
                    "control_networks",
                    "backup_power",
                    "emergency_systems",
                ],
                "threat_indicators": [
                    "sensor_anomalies",
                    "control_deviations",
                    "communication_disruption",
                ],
            },
            EnvironmentDomain.MEDICAL: {
                "key_terrain": ["hospitals", "supply_chains", "data_systems", "emergency_response"],
                "critical_assets": ["patient_data", "medical_devices", "pharmaceuticals", "staff"],
                "threat_indicators": [
                    "patient_spikes",
                    "supply_shortages",
                    "system_failures",
                    "outbreak_patterns",
                ],
            },
            EnvironmentDomain.FINANCIAL: {
                "key_terrain": [
                    "payment_systems",
                    "trading_platforms",
                    "banking_networks",
                    "regulatory_interfaces",
                ],
                "critical_assets": [
                    "transaction_data",
                    "customer_accounts",
                    "trading_algorithms",
                    "audit_trails",
                ],
                "threat_indicators": [
                    "unusual_transactions",
                    "market_manipulation",
                    "fraud_patterns",
                    "insider_trading",
                ],
            },
        }

    def define_environment(
        self,
        domain: EnvironmentDomain,
        area_of_interest: dict[str, Any],
        critical_assets: list[str] | None = None,
        constraints: list[str] | None = None,
        assumptions: list[str] | None = None,
    ) -> EnvironmentDefinition:
        """
        Phase 1: Define the operational environment.

        Args:
            domain: Domain to define
            area_of_interest: Geographic/logical boundaries
            critical_assets: Assets requiring protection
            constraints: Operational constraints
            assumptions: Planning assumptions

        Returns:
            Environment definition
        """
        template = self._domain_templates.get(domain, {})

        env = EnvironmentDefinition(
            domain=domain,
            area_of_interest=area_of_interest,
            area_of_influence=self._compute_influence_area(area_of_interest),
            key_terrain=template.get("key_terrain", []),
            critical_assets=critical_assets or template.get("critical_assets", []),
            constraints=constraints or [],
            assumptions=assumptions or [],
            timeline={
                "analysis_start": time.time(),
                "forecast_horizon_hours": 24 * 7,
            },
        )

        self._environments[domain] = env
        logger.info(f"Defined environment for {domain.value}")
        return env

    def describe_effects(
        self,
        domain: EnvironmentDomain,
        observations: list[dict[str, Any]],
    ) -> list[EnvironmentEffect]:
        """
        Phase 2: Describe environmental effects on operations.

        Args:
            domain: Domain to analyze
            observations: Recent observations/data

        Returns:
            List of environmental effects
        """
        effects: list[EnvironmentEffect] = []
        env = self._environments.get(domain)

        if not env:
            logger.warning(f"Environment not defined for {domain.value}")
            return effects

        # Analyze observations for effects
        for i, obs in enumerate(observations):
            severity = obs.get("severity", 0.5)
            probability = obs.get("probability", 0.5)

            # Determine affected operations
            affected = self._determine_affected_operations(obs, env)

            if affected:
                effect = EnvironmentEffect(
                    effect_id=f"effect_{domain.value}_{i}",
                    description=obs.get("description", f"Effect from observation {i}"),
                    affected_operations=affected,
                    severity=severity,
                    probability=probability,
                    mitigation_options=self._suggest_mitigations(obs, env),
                    dependencies=obs.get("dependencies", []),
                )
                effects.append(effect)
                self._effects[effect.effect_id] = effect

        logger.info(f"Identified {len(effects)} effects for {domain.value}")
        return effects

    def evaluate_threats(
        self,
        domain: EnvironmentDomain,
        intelligence_reports: list[dict[str, Any]],
    ) -> list[ThreatCapability]:
        """
        Phase 3: Evaluate threat capabilities.

        Args:
            domain: Domain to analyze
            intelligence_reports: Intelligence on potential threats

        Returns:
            List of threat capability assessments
        """
        threats = []

        for i, report in enumerate(intelligence_reports):
            # Determine threat category
            category = self._categorize_threat(report)

            capabilities = report.get("capabilities", [])
            if not capabilities:
                capabilities = self._infer_capabilities(report, domain)

            rating = self._calculate_threat_rating(report, domain)

            threat = ThreatCapability(
                threat_id=f"threat_{domain.value}_{i}",
                threat_name=report.get("name", f"Threat Actor {i}"),
                category=category,
                capabilities=capabilities,
                intent=report.get("intent", "unknown"),
                resources=report.get(
                    "resources", {"technical": 0.5, "financial": 0.5, "human": 0.5}
                ),
                historical_actions=report.get("history", []),
                indicators=report.get("indicators", []),
                overall_rating=rating,
            )
            threats.append(threat)
            self._threats[threat.threat_id] = threat
            self._stats["threats_identified"] += 1

        logger.info(f"Evaluated {len(threats)} threats for {domain.value}")
        return threats

    def determine_coas(
        self,
        domain: EnvironmentDomain,
        threats: list[ThreatCapability],
    ) -> list[ThreatCOA]:
        """
        Phase 4: Determine threat courses of action.

        Generates Most Likely, Most Dangerous, and Most Disruptive COAs.

        Args:
            domain: Domain to analyze
            threats: Evaluated threats

        Returns:
            List of threat COAs
        """
        coas = []
        env = self._environments.get(domain)

        for threat in threats:
            # Generate three COA types
            for coa_type in CourseOfAction:
                coa = self._generate_coa(threat, coa_type, env)
                coas.append(coa)
                self._coas[coa.coa_id] = coa
                self._stats["coas_generated"] += 1

        # Sort by risk (probability * impact)
        coas.sort(key=lambda c: c.probability * c.impact, reverse=True)

        logger.info(f"Generated {len(coas)} COAs for {domain.value}")
        return coas

    def conduct_full_ipb(
        self,
        domain: EnvironmentDomain,
        area_of_interest: dict[str, Any],
        observations: list[dict[str, Any]],
        intelligence_reports: list[dict[str, Any]],
        critical_assets: list[str] | None = None,
    ) -> BattlefieldAssessment:
        """
        Conduct complete IPB process.

        Args:
            domain: Domain to analyze
            area_of_interest: Area boundaries
            observations: Current observations
            intelligence_reports: Threat intelligence
            critical_assets: Assets to protect

        Returns:
            Complete battlefield assessment
        """
        start_time = time.time()
        self._stats["assessments_conducted"] += 1

        # Phase 1: Define
        environment = self.define_environment(domain, area_of_interest, critical_assets)

        # Phase 2: Describe
        effects = self.describe_effects(domain, observations)

        # Phase 3: Evaluate
        threats = self.evaluate_threats(domain, intelligence_reports)

        # Phase 4: Determine
        coas = self.determine_coas(domain, threats)

        pirs = self._generate_pirs(threats, coas)
        collection = self._prioritize_collection(pirs, threats)

        running_estimate = {}
        if self.enable_running_estimates:
            running_estimate = self._update_running_estimate(domain, threats, coas)

        confidence = self._calculate_assessment_confidence(effects, threats, coas)

        assessment = BattlefieldAssessment(
            assessment_id=f"ipb_{domain.value}_{int(time.time())}",
            timestamp=time.time(),
            environment=environment,
            effects=effects,
            threats=threats,
            courses_of_action=coas,
            priority_intelligence_requirements=pirs,
            collection_priorities=collection,
            running_estimate=running_estimate,
            confidence=confidence,
        )

        elapsed = time.time() - start_time
        logger.info(f"Complete IPB for {domain.value} in {elapsed:.2f}s")

        return assessment

    def _compute_influence_area(
        self,
        area_of_interest: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute area of influence from area of interest."""
        # Influence area typically extends beyond AOI
        influence = dict(area_of_interest)
        influence["extended"] = True
        influence["extension_factor"] = self.PHI  # Golden ratio expansion
        return influence

    def _determine_affected_operations(
        self,
        observation: dict[str, Any],
        environment: EnvironmentDefinition,
    ) -> list[str]:
        """Determine which operations are affected by an observation."""
        affected = []
        obs_type = observation.get("type", "")

        for asset in environment.critical_assets:
            if asset.lower() in obs_type.lower() or obs_type.lower() in asset.lower():
                affected.append(f"operations_on_{asset}")

        for terrain in environment.key_terrain:
            if terrain.lower() in obs_type.lower():
                affected.append(f"access_to_{terrain}")

        return affected or ["general_operations"]

    def _suggest_mitigations(
        self,
        observation: dict[str, Any],
        environment: EnvironmentDefinition,
    ) -> list[str]:
        """Suggest mitigation options for an effect."""
        mitigations = ["monitor_closely", "increase_redundancy"]

        severity = observation.get("severity", 0.5)
        if severity > 0.7:
            mitigations.extend(["activate_contingency", "notify_leadership"])
        if severity > 0.9:
            mitigations.append("immediate_intervention")

        return mitigations

    def _categorize_threat(self, report: dict[str, Any]) -> ThreatCategory:
        """Categorize a threat based on report data."""
        category_map = {
            "state": ThreatCategory.STATE_ACTOR,
            "nation": ThreatCategory.STATE_ACTOR,
            "apt": ThreatCategory.STATE_ACTOR,
            "insider": ThreatCategory.INSIDER,
            "employee": ThreatCategory.INSIDER,
            "criminal": ThreatCategory.CRIMINAL,
            "ransomware": ThreatCategory.CRIMINAL,
            "natural": ThreatCategory.NATURAL,
            "weather": ThreatCategory.NATURAL,
            "systemic": ThreatCategory.SYSTEMIC,
        }

        report_str = str(report).lower()
        for key, category in category_map.items():
            if key in report_str:
                return category

        return ThreatCategory.EMERGENT

    def _infer_capabilities(
        self,
        report: dict[str, Any],
        domain: EnvironmentDomain,
    ) -> list[str]:
        """Infer threat capabilities from report and domain."""
        base_capabilities = ["reconnaissance", "initial_access"]

        if domain == EnvironmentDomain.CYBER:
            base_capabilities.extend(["credential_theft", "lateral_movement", "data_exfiltration"])
        elif domain == EnvironmentDomain.INFRASTRUCTURE:
            base_capabilities.extend(["physical_access", "supply_chain_compromise"])
        elif domain == EnvironmentDomain.FINANCIAL:
            base_capabilities.extend(["fraud", "market_manipulation"])

        return base_capabilities

    def _calculate_threat_rating(
        self,
        report: dict[str, Any],
        domain: EnvironmentDomain,
    ) -> float:
        """Calculate overall threat rating."""
        capability = report.get("capability_score", 0.5)
        intent = report.get("intent_score", 0.5)
        opportunity = report.get("opportunity_score", 0.5)

        # Threat = Capability * Intent * Opportunity
        return float(min(1.0, capability * intent * opportunity * self.PHI))

    def _generate_coa(
        self,
        threat: ThreatCapability,
        coa_type: CourseOfAction,
        environment: EnvironmentDefinition | None,
    ) -> ThreatCOA:
        """Generate a specific course of action."""
        # Adjust probability and impact based on COA type
        if coa_type == CourseOfAction.MOST_LIKELY:
            probability = 0.7
            impact = 0.5
        elif coa_type == CourseOfAction.MOST_DANGEROUS:
            probability = 0.3
            impact = 0.9
        else:  # Most Disruptive
            probability = 0.4
            impact = 0.8

        phases = [
            {
                "phase": "Preparation",
                "activities": threat.capabilities[:2] if threat.capabilities else [],
            },
            {
                "phase": "Execution",
                "activities": threat.capabilities[2:4] if len(threat.capabilities) > 2 else [],
            },
            {
                "phase": "Exploitation",
                "activities": threat.capabilities[4:] if len(threat.capabilities) > 4 else [],
            },
        ]

        # Generate indicators and warnings
        iaw = [
            *threat.indicators,
            f"changes_in_{threat.category.value}_behavior",
            f"targeting_of_{environment.domain.value if environment else 'assets'}",
        ]

        countermeasures = [f"block_{cap.replace(' ', '_')}" for cap in threat.capabilities[:3]] + [
            "increase_monitoring",
            "activate_defenses",
        ]

        return ThreatCOA(
            coa_id=f"coa_{threat.threat_id}_{coa_type.value}",
            coa_type=coa_type,
            threat=threat,
            description=f"{coa_type.value.replace('_', ' ').title()} COA for {threat.threat_name}",
            objectives=[f"exploit_{environment.domain.value if environment else 'target'}"],
            phases=phases,
            indicators_and_warnings=iaw,
            probability=probability * threat.overall_rating,
            impact=impact,
            decision_points=[f"indicator_{i}_observed" for i in range(min(3, len(iaw)))],
            countermeasures=countermeasures,
        )

    def _generate_pirs(
        self,
        threats: list[ThreatCapability],
        coas: list[ThreatCOA],
    ) -> list[str]:
        """Generate Priority Intelligence Requirements."""
        pirs = []

        # Based on highest-rated threats
        for threat in sorted(threats, key=lambda t: t.overall_rating, reverse=True)[:3]:
            pirs.append(f"What are the current activities of {threat.threat_name}?")
            pirs.append(f"What are the indicators of {threat.threat_name} preparation?")

        # Based on most dangerous COAs
        dangerous_coas = [c for c in coas if c.coa_type == CourseOfAction.MOST_DANGEROUS][:2]
        for coa in dangerous_coas:
            pirs.append(
                f"What would indicate {coa.threat.threat_name} executing {coa.coa_type.value}?"
            )

        return pirs[:10]

    def _prioritize_collection(
        self,
        pirs: list[str],
        threats: list[ThreatCapability],
    ) -> list[str]:
        """Prioritize intelligence collection based on PIRs."""
        priorities = []

        for i, pir in enumerate(pirs[:5]):
            priorities.append(f"Priority {i + 1}: Collect on {pir[:50]}...")

        # Add gap-filling collection
        priorities.append("Fill gaps: Emerging threat indicators")

        return priorities

    def _update_running_estimate(
        self,
        domain: EnvironmentDomain,
        threats: list[ThreatCapability],
        coas: list[ThreatCOA],
    ) -> dict[str, Any]:
        """Update the running estimate for a domain."""
        estimate = {
            "last_updated": time.time(),
            "threat_count": len(threats),
            "avg_threat_rating": np.mean([t.overall_rating for t in threats]) if threats else 0,
            "highest_risk_coa": coas[0].coa_id if coas else None,
            "trend": "stable",  # Could be: improving, deteriorating, stable
        }

        self._running_estimates[domain] = estimate
        return estimate

    def _calculate_assessment_confidence(
        self,
        effects: list[EnvironmentEffect],
        threats: list[ThreatCapability],
        coas: list[ThreatCOA],
    ) -> float:
        """Calculate overall confidence in the assessment."""
        # More data = higher confidence (up to a point)
        data_confidence = min(0.9, 0.3 + 0.1 * (len(effects) + len(threats)))

        # Consistency of threat ratings
        if threats:
            ratings = [t.overall_rating for t in threats]
            consistency = 1 - np.std(ratings) if len(ratings) > 1 else 0.8
        else:
            consistency = 0.5

        return float(data_confidence * consistency)

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "environments_defined": len(self._environments),
            "effects_tracked": len(self._effects),
            "threats_tracked": len(self._threats),
            "coas_tracked": len(self._coas),
            "domains": [d.value for d in self.domains],
        }

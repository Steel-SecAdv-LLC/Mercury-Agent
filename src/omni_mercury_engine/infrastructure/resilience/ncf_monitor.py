# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""CISA National Critical Functions (NCF) anomaly detection.

Monitors 55 CISA National Critical Functions for anomalies and models
interdependencies for cascading failure analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NCFMonitor:
    """National Critical Functions anomaly detector.

    Monitors 55 CISA National Critical Functions for anomalies and
    models interdependencies for cascading failure analysis.

    Reference: https://www.cisa.gov/national-critical-functions-set
    """

    def __init__(self, ethical_config: dict[str, float] | None = None) -> None:
        """Initialize NCF Monitor.

        Args:
            ethical_config: Ethical scalar configuration
        """
        self.ncf_categories = {
            "connect": [
                "operate_core_network",
                "provide_cable_access",
                "provide_internet_services",
                "provide_mobile_services",
                "provide_satellite_services",
                "provide_storage_computation",
                "provide_voice_data",
                "provide_radio_tv_broadcasting",
                "support_aeronautical_operations",
            ],
            "distribute": [
                "distribute_electricity",
                "distribute_natural_gas",
                "distribute_petroleum",
                "operate_passenger_rail",
                "operate_freight_rail",
                "operate_waterborne_transportation",
                "operate_highway_transportation",
                "operate_aviation_transportation",
                "deliver_postal_shipping",
            ],
            "manage": [
                "assess_threats_hazards",
                "clear_carry_settle_payments",
                "conduct_public_health",
                "conduct_research_development",
                "conduct_resource_planning",
                "control_air_traffic",
                "establish_physical_security",
                "generate_personal_identification",
                "issue_currency_instruments",
                "maintain_safety_security_comms",
                "maintain_situational_awareness",
                "manage_ballistic_missiles",
                "manage_critical_supply_chains",
                "manage_environmental_hazards",
                "manage_homeland_defenses",
                "manage_household_waste",
                "manage_it",
                "manage_water_resources",
                "operate_government",
                "operate_nuclear_weapons",
                "perform_law_enforcement",
                "provide_fire_search_rescue",
                "regulate_hazmat",
                "treat_municipal_wastewater",
            ],
            "supply": [
                "generate_electricity",
                "mine_minerals",
                "process_treat_water",
                "produce_agricultural_products",
                "produce_provide_energy",
                "produce_industrial_chemicals",
                "produce_manufacturing_services",
                "produce_medical_materials",
                "provide_defense_equipment",
                "provide_housing",
                "provide_wholesale_retail",
                "refine_biological_materials",
                "remove_debris",
            ],
        }

        self.dependencies = self._build_dependency_graph()
        self.ethical_config = ethical_config or {}

    def _build_dependency_graph(self) -> dict[str, list[str]]:
        """Build directed graph of NCF dependencies."""
        return {
            "distribute_electricity": ["generate_electricity"],
            "provide_internet_services": ["operate_core_network", "distribute_electricity"],
            "operate_passenger_rail": ["distribute_electricity", "maintain_safety_security_comms"],
            "clear_carry_settle_payments": ["manage_it", "distribute_electricity"],
            "conduct_public_health": [
                "provide_internet_services",
                "distribute_electricity",
                "produce_medical_materials",
            ],
            "provide_mobile_services": ["operate_core_network", "distribute_electricity"],
            "manage_critical_supply_chains": [
                "operate_highway_transportation",
                "provide_internet_services",
            ],
            "produce_manufacturing_services": [
                "distribute_electricity",
                "process_treat_water",
                "operate_highway_transportation",
            ],
        }

    def detect(
        self, data: np.ndarray[Any, Any], ncf_id: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect anomalies for specific NCF.

        Args:
            data: Time-series or sensor data for the NCF
            ncf_id: NCF identifier (e.g., 'distribute_electricity')
            context: Additional context (location, asset type, etc.)

        Returns:
            Detection results with anomaly score, affected functions, cascade risk
        """
        z_scores = self._calculate_z_scores(data)
        anomaly_detected = np.max(z_scores) > 3.0

        cascade_risk = self.analyze_cascading_failures([ncf_id]) if anomaly_detected else {}
        ethical_score = self._calculate_ethical_impact(ncf_id, context)
        ncf_category = self._get_ncf_category(ncf_id)

        return {
            "ncf_id": ncf_id,
            "category": ncf_category,
            "anomaly_detected": anomaly_detected,
            "anomaly_score": float(np.max(z_scores)),
            "confidence": 0.85,
            "cascade_risk": cascade_risk,
            "ethical_impact": ethical_score,
            "affected_population_est": self._estimate_population_impact(ncf_id),
            "economic_impact_est": self._estimate_economic_impact(ncf_id),
            "recovery_time_est": self._estimate_recovery_time(ncf_id),
            "details": {
                "anomaly_indices": np.where(z_scores > 3.0)[0].tolist(),
                "severity": "high" if np.max(z_scores) > 5.0 else "medium",
            },
        }

    def analyze_cascading_failures(self, initial_failures: list[str]) -> dict[str, Any]:
        """Model cascading impacts across dependent NCFs.

        Args:
            initial_failures: List of NCF IDs that initially failed

        Returns:
            Cascading failure analysis with affected NCFs, impact scores
        """
        affected = set(initial_failures)
        wave_impacts = {0: initial_failures}
        wave = 0

        while wave < 5:
            wave += 1
            new_failures = set()

            for ncf in affected:
                dependents = [k for k, v in self.dependencies.items() if ncf in v]
                new_failures.update(dependents)

            if not new_failures - affected:
                break

            wave_impacts[wave] = list(new_failures - affected)
            affected.update(new_failures)

        total_affected = len(affected)
        population_impact = sum(self._estimate_population_impact(ncf) for ncf in affected)
        economic_impact = sum(self._estimate_economic_impact(ncf) for ncf in affected)

        return {
            "initial_failures": initial_failures,
            "cascading_impacts": list(affected - set(initial_failures)),
            "total_affected_ncfs": total_affected,
            "cascade_waves": wave,
            "wave_impacts": wave_impacts,
            "population_affected": population_impact,
            "economic_impact_usd": economic_impact,
            "criticality_score": min(100, total_affected * 1.5 + wave * 10),
        }

    def _calculate_z_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calculate z-scores for anomaly detection."""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        return np.abs((data - mean) / std)

    def _calculate_ethical_impact(self, ncf_id: str, context: dict[str, Any] | None) -> float:
        """Score ethical impact, weighting the people most exposed to the outage."""
        vulnerable_ncfs = {
            "conduct_public_health": 0.95,
            "provide_housing": 0.90,
            "process_treat_water": 0.95,
            "produce_agricultural_products": 0.88,
            "provide_fire_search_rescue": 0.92,
        }
        return vulnerable_ncfs.get(ncf_id, 0.70)

    def _estimate_population_impact(self, ncf_id: str) -> int:
        """Estimate population affected if NCF fails."""
        impact_estimates = {
            "distribute_electricity": 330000000,
            "provide_internet_services": 300000000,
            "distribute_natural_gas": 180000000,
            "process_treat_water": 330000000,
            "conduct_public_health": 330000000,
            "generate_electricity": 330000000,
            "provide_mobile_services": 290000000,
        }
        return impact_estimates.get(ncf_id, 1000000)

    def _estimate_economic_impact(self, ncf_id: str) -> float:
        """Estimate economic impact per day if NCF fails (USD)."""
        impact_estimates = {
            "distribute_electricity": 10_000_000_000,
            "clear_carry_settle_payments": 5_000_000_000,
            "provide_internet_services": 3_000_000_000,
            "operate_aviation_transportation": 2_000_000_000,
            "generate_electricity": 10_000_000_000,
        }
        return impact_estimates.get(ncf_id, 100_000_000)

    def _get_ncf_category(self, ncf_id: str) -> str:
        """Get the category (connect/distribute/manage/supply) for an NCF."""
        for category, ncfs in self.ncf_categories.items():
            if ncf_id in ncfs:
                return category
        return "unknown"

    def _estimate_recovery_time(self, ncf_id: str) -> str:
        """Estimate typical recovery time if NCF fails."""
        recovery_times = {
            "distribute_electricity": "1-7 days",
            "provide_internet_services": "1-3 days",
            "conduct_public_health": "1-4 weeks",
            "generate_electricity": "1-14 days",
        }
        return recovery_times.get(ncf_id, "1-7 days")

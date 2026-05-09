"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
AI Scaling and Compute Optimization Module

Inspired by Bain & Company Technology Report 2025 insights on AI scaling,
compute power demands, and agentic AI transformation.

Key insights:
- Hyperscalers investing $140B-298B in AI infrastructure
- AI compute demand growing exponentially
- Need for efficient resource allocation and power management

Research source: Bain & Company Technology Report 2025
(https://www.bain.com/insights/topics/technology-report/)

"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ComputeResource:
    """Represents compute resources for AI operations."""

    cpu_cores: int
    gpu_count: int
    memory_gb: float
    power_watts: float
    cost_per_hour: float


class BainAIScaling:
    """
    AI scaling and compute optimization inspired by Bain 2025 report.

    Implements efficient resource allocation and power management for AI workloads based on industry
    trends.
    """

    def __init__(self, max_power_watts: float = 1000.0) -> None:
        self.max_power_watts = max_power_watts
        self.current_allocation: dict[str, ComputeResource] = {}

    def optimize_compute_allocation(
        self, workloads: list[dict[str, Any]], available_resources: ComputeResource
    ) -> dict[str, ComputeResource]:
        """
        Optimize compute resource allocation across workloads.

        Args:
            workloads: List of workload specifications
            available_resources: Total available compute resources

        Returns:
            Optimal resource allocation per workload
        """
        allocation = {}
        remaining = available_resources

        total_priority = sum(w.get("priority", 1.0) for w in workloads)

        for workload in workloads:
            workload_id = workload["id"]
            priority = workload.get("priority", 1.0)
            priority_ratio = priority / total_priority

            allocated = ComputeResource(
                cpu_cores=int(remaining.cpu_cores * priority_ratio),
                gpu_count=int(remaining.gpu_count * priority_ratio),
                memory_gb=remaining.memory_gb * priority_ratio,
                power_watts=remaining.power_watts * priority_ratio,
                cost_per_hour=remaining.cost_per_hour * priority_ratio,
            )

            allocation[workload_id] = allocated

        return allocation

    def estimate_power_consumption(
        self, model_size: int, batch_size: int, sequence_length: int
    ) -> float:
        """
        Estimate power consumption for AI inference/training.

        Based on typical power profiles from hyperscaler deployments.

        Args:
            model_size: Number of parameters
            batch_size: Batch size
            sequence_length: Sequence length for transformers

        Returns:
            Estimated power consumption in watts
        """
        base_power = 100.0
        model_power = model_size / 1e9 * 10.0
        batch_power = batch_size * 0.5
        sequence_power = sequence_length / 100.0

        total_power = base_power + model_power + batch_power + sequence_power
        return min(total_power, self.max_power_watts)

    def calculate_efficiency_score(
        self, accuracy: float, power_watts: float, latency_ms: float
    ) -> float:
        """
        Calculate efficiency score balancing accuracy, power, and latency.

        Inspired by Bain report emphasis on AI leaders achieving 10-25% EBITDA improvement.

        Args:
            accuracy: Model accuracy (0-1)
            power_watts: Power consumption
            latency_ms: Inference latency in milliseconds

        Returns:
            Efficiency score (higher is better)
        """
        power_efficiency = 1.0 / (1.0 + power_watts / 1000.0)
        latency_efficiency = 1.0 / (1.0 + latency_ms / 100.0)

        efficiency = accuracy * power_efficiency * latency_efficiency
        return efficiency

    def plan_infrastructure_scaling(
        self, current_investment_millions: float, growth_rate: float = 0.25, years: int = 5
    ) -> dict[str, Any]:
        """
        Plan AI infrastructure scaling based on Bain findings.

        Bain 2025: Hyperscalers investing $140-298B in AI infrastructure.
        Models exponential growth in compute demand.

        Args:
            current_investment_millions: Current infrastructure investment (USD millions)
            growth_rate: Annual growth rate (default 0.25 = 25% per year)
            years: Planning horizon in years

        Returns:
            Infrastructure scaling plan with investment projections
        """
        projections = []
        cumulative_investment = current_investment_millions

        for year in range(1, years + 1):
            year_investment = current_investment_millions * ((1 + growth_rate) ** year)
            cumulative_investment += year_investment

            projections.append(
                {
                    "year": year,
                    "annual_investment_millions": year_investment,
                    "cumulative_investment_millions": cumulative_investment,
                    "estimated_compute_capacity_petaflops": year_investment * 0.05,
                }
            )

        return {
            "current_investment_millions": current_investment_millions,
            "growth_rate": growth_rate,
            "planning_horizon_years": years,
            "projections": projections,
            "total_investment_millions": cumulative_investment,
            "bain_context": "Hyperscalers investing $140B-298B in AI infrastructure globally",
            "recommendations": [
                "Prioritize energy-efficient compute (power management critical)",
                "Plan for exponential growth in training and inference demands",
                "Consider hybrid cloud/on-premise strategies for cost optimization",
                "Invest in AI-specific hardware (GPUs, TPUs, custom accelerators)",
            ],
        }

    def estimate_agentic_ai_impact(
        self, current_workforce_size: int, process_automation_target: float = 0.30
    ) -> dict[str, Any]:
        """
        Estimate impact of agentic AI on operations.

        Bain 2025: "At full potential, agents will run complete processes and workflows."
        Models transformation from human-driven to agent-driven processes.

        Args:
            current_workforce_size: Current workforce size
            process_automation_target: Target automation percentage (0-1)

        Returns:
            Impact assessment with workforce transformation and efficiency gains
        """
        automated_processes_equivalent = current_workforce_size * process_automation_target
        augmented_workforce = current_workforce_size * (1 - process_automation_target * 0.5)

        productivity_multiplier = 1.0 + (process_automation_target * 1.5)

        ebitda_improvement_low = 0.10
        ebitda_improvement_high = 0.25

        return {
            "current_workforce_size": current_workforce_size,
            "automation_target": process_automation_target,
            "automated_process_equivalent_ftes": automated_processes_equivalent,
            "augmented_workforce_size": augmented_workforce,
            "productivity_multiplier": productivity_multiplier,
            "estimated_ebitda_improvement_range": (
                f"{ebitda_improvement_low * 100:.0f}%-{ebitda_improvement_high * 100:.0f}%"
            ),
            "transformation_timeline": "2025-2030 (per Bain agentic AI projections)",
            "key_capabilities": [
                "Agents running complete workflows autonomously",
                "Minimal human oversight for routine processes",
                "Adaptive learning from outcomes",
                "Cross-functional process integration",
            ],
            "implementation_priorities": [
                "Start with well-defined, repeatable processes",
                "Implement robust monitoring and safety controls",
                "Train workforce on agent collaboration",
                "Measure and optimize agent performance continuously",
            ],
        }

    def optimize_power_management(
        self, workload_schedule: list[dict[str, Any]], power_budget_watts: float
    ) -> dict[str, Any]:
        """
        Optimize power management for AI workloads.

        Critical for sustainable AI scaling given massive infrastructure investments.

        Args:
            workload_schedule: List of workloads with 'id', 'power_watts', 'priority', 'flexible'
            power_budget_watts: Maximum power budget

        Returns:
            Optimized schedule with power allocation
        """
        critical_workloads = [w for w in workload_schedule if not w.get("flexible", False)]
        flexible_workloads = [w for w in workload_schedule if w.get("flexible", False)]

        critical_power = sum(w["power_watts"] for w in critical_workloads)
        remaining_power = power_budget_watts - critical_power

        if remaining_power < 0:
            return {
                "status": "over_budget",
                "power_deficit_watts": abs(remaining_power),
                "recommendation": "Reduce critical workloads or increase power budget",
            }

        allocated_flexible = []
        for workload in sorted(
            flexible_workloads, key=lambda x: x.get("priority", 0.5), reverse=True
        ):
            if workload["power_watts"] <= remaining_power:
                allocated_flexible.append(workload)
                remaining_power -= workload["power_watts"]

        total_allocated_power = critical_power + sum(w["power_watts"] for w in allocated_flexible)
        power_efficiency = (total_allocated_power / power_budget_watts) * 100

        return {
            "status": "optimized",
            "critical_workloads": len(critical_workloads),
            "flexible_workloads_allocated": len(allocated_flexible),
            "flexible_workloads_deferred": len(flexible_workloads) - len(allocated_flexible),
            "total_power_allocated_watts": total_allocated_power,
            "power_budget_watts": power_budget_watts,
            "power_utilization_percent": power_efficiency,
            "power_headroom_watts": power_budget_watts - total_allocated_power,
        }

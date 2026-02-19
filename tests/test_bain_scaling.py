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
Tests for Bain AI Scaling module
"""

from omni_mercury_engine.scaling.bain_ai_scaling import BainAIScaling, ComputeResource


class TestComputeResource:
    """Test ComputeResource dataclass."""

    def test_creation(self):
        """Test creating ComputeResource."""
        resource = ComputeResource(
            cpu_cores=16,
            gpu_count=4,
            memory_gb=64.0,
            power_watts=500.0,
            cost_per_hour=2.50,
        )

        assert resource.cpu_cores == 16
        assert resource.gpu_count == 4
        assert resource.memory_gb == 64.0
        assert resource.power_watts == 500.0
        assert resource.cost_per_hour == 2.50


class TestBainAIScaling:
    """Test BainAIScaling class."""

    def test_initialization(self):
        """Test initialization."""
        scaler = BainAIScaling()
        assert scaler.max_power_watts == 1000.0
        assert scaler.current_allocation == {}

    def test_initialization_custom_power(self):
        """Test initialization with custom power limit."""
        scaler = BainAIScaling(max_power_watts=5000.0)
        assert scaler.max_power_watts == 5000.0

    def test_optimize_compute_allocation(self):
        """Test compute allocation optimization."""
        scaler = BainAIScaling()

        workloads = [
            {"id": "training", "priority": 2.0},
            {"id": "inference", "priority": 1.0},
        ]

        available = ComputeResource(
            cpu_cores=32, gpu_count=8, memory_gb=128.0, power_watts=1000.0, cost_per_hour=10.0
        )

        allocation = scaler.optimize_compute_allocation(workloads, available)

        assert "training" in allocation
        assert "inference" in allocation
        assert allocation["training"].cpu_cores > allocation["inference"].cpu_cores

    def test_estimate_power_consumption(self):
        """Test power consumption estimation."""
        scaler = BainAIScaling()

        power = scaler.estimate_power_consumption(
            model_size=1_000_000_000, batch_size=32, sequence_length=512
        )

        assert power > 0
        assert power > 100.0  # Reasonable minimum for 1B param model
        assert power <= scaler.max_power_watts

    def test_estimate_power_consumption_large_model(self):
        """Test power consumption capped at max."""
        scaler = BainAIScaling(max_power_watts=200.0)

        power = scaler.estimate_power_consumption(
            model_size=100_000_000_000, batch_size=128, sequence_length=2048
        )

        assert power == 200.0

    def test_calculate_efficiency_score(self):
        """Test efficiency score calculation."""
        scaler = BainAIScaling()

        score = scaler.calculate_efficiency_score(accuracy=0.95, power_watts=500.0, latency_ms=50.0)

        assert 0 < score <= 1

    def test_calculate_efficiency_high_accuracy(self):
        """Test efficiency with high accuracy, low power, low latency."""
        scaler = BainAIScaling()

        high_score = scaler.calculate_efficiency_score(
            accuracy=0.99, power_watts=100.0, latency_ms=10.0
        )

        low_score = scaler.calculate_efficiency_score(
            accuracy=0.70, power_watts=1000.0, latency_ms=500.0
        )

        assert high_score > low_score

    def test_plan_infrastructure_scaling(self):
        """Test infrastructure scaling plan."""
        scaler = BainAIScaling()

        plan = scaler.plan_infrastructure_scaling(
            current_investment_millions=100.0, growth_rate=0.25, years=5
        )

        assert plan["current_investment_millions"] == 100.0
        assert plan["growth_rate"] == 0.25
        assert plan["planning_horizon_years"] == 5
        assert len(plan["projections"]) == 5
        assert plan["total_investment_millions"] > 100.0
        assert "recommendations" in plan
        assert len(plan["recommendations"]) > 0

    def test_plan_infrastructure_scaling_projections(self):
        """Test infrastructure scaling projections."""
        scaler = BainAIScaling()

        plan = scaler.plan_infrastructure_scaling(
            current_investment_millions=100.0, growth_rate=0.20, years=3
        )

        for projection in plan["projections"]:
            assert "year" in projection
            assert "annual_investment_millions" in projection
            assert "cumulative_investment_millions" in projection
            assert "estimated_compute_capacity_petaflops" in projection

    def test_estimate_agentic_ai_impact(self):
        """Test agentic AI impact estimation."""
        scaler = BainAIScaling()

        impact = scaler.estimate_agentic_ai_impact(
            current_workforce_size=1000, process_automation_target=0.30
        )

        assert impact["current_workforce_size"] == 1000
        assert impact["automation_target"] == 0.30
        assert impact["automated_process_equivalent_ftes"] == 300.0
        assert impact["productivity_multiplier"] > 1.0
        assert "key_capabilities" in impact
        assert "implementation_priorities" in impact

    def test_estimate_agentic_ai_impact_augmented_workforce(self):
        """Test augmented workforce calculation."""
        scaler = BainAIScaling()

        impact = scaler.estimate_agentic_ai_impact(
            current_workforce_size=1000, process_automation_target=0.50
        )

        assert impact["augmented_workforce_size"] < 1000

    def test_optimize_power_management(self):
        """Test power management optimization."""
        scaler = BainAIScaling()

        workloads = [
            {"id": "critical_1", "power_watts": 200.0, "priority": 1.0, "flexible": False},
            {"id": "critical_2", "power_watts": 300.0, "priority": 1.0, "flexible": False},
            {"id": "flex_1", "power_watts": 200.0, "priority": 0.8, "flexible": True},
            {"id": "flex_2", "power_watts": 400.0, "priority": 0.5, "flexible": True},
        ]

        result = scaler.optimize_power_management(workloads, power_budget_watts=1000.0)

        assert result["status"] == "optimized"
        assert result["critical_workloads"] == 2
        assert result["total_power_allocated_watts"] <= 1000.0

    def test_optimize_power_management_over_budget(self):
        """Test power management when critical workloads exceed budget."""
        scaler = BainAIScaling()

        workloads = [
            {"id": "critical_1", "power_watts": 600.0, "flexible": False},
            {"id": "critical_2", "power_watts": 600.0, "flexible": False},
        ]

        result = scaler.optimize_power_management(workloads, power_budget_watts=1000.0)

        assert result["status"] == "over_budget"
        assert result["power_deficit_watts"] == 200.0
        assert "recommendation" in result

    def test_optimize_power_management_with_headroom(self):
        """Test power management with headroom calculation."""
        scaler = BainAIScaling()

        workloads = [
            {"id": "critical_1", "power_watts": 200.0, "flexible": False},
        ]

        result = scaler.optimize_power_management(workloads, power_budget_watts=1000.0)

        assert result["power_headroom_watts"] == 800.0
        assert result["power_utilization_percent"] == 20.0

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

"""Tests for Energy Optimization module"""

from omni_mercury_engine.energy.energy_optimization import (
    EnergyOptimization,
    EnergyProfile,
    EnergySource,
)


def test_energy_optimization_initialization():
    """Test energy optimization system initialization"""
    system = EnergyOptimization(max_power_budget=2000.0, efficiency_target=0.85, carbon_limit=150.0)
    assert system.max_power_budget == 2000.0
    assert system.efficiency_target == 0.85
    assert system.carbon_limit == 150.0
    assert len(system.energy_profiles) == 0


def test_efficiency_first_optimization():
    """Test efficiency-first optimization strategy"""
    system = EnergyOptimization()

    operations = [
        {"id": "op1", "power": 100, "performance": 10},
        {"id": "op2", "power": 200, "performance": 15},
        {"id": "op3", "power": 50, "performance": 8},
    ]

    optimized = system.efficiency_first_optimization(operations, available_power=300)

    assert len(optimized) > 0
    assert all("efficiency" in op for op in optimized)
    efficiencies = [op["efficiency"] for op in optimized]
    assert efficiencies == sorted(efficiencies, reverse=True)


def test_renewable_resource_allocation():
    """Test renewable resource allocation"""
    system = EnergyOptimization()

    allocation = system.renewable_resource_allocation(
        total_resources=1000.0, renewable_fraction=0.25
    )

    assert "renewable" in allocation
    assert "conventional" in allocation
    assert allocation["renewable"] == 250.0
    assert allocation["conventional"] == 750.0
    assert allocation["renewable"] + allocation["conventional"] == 1000.0


def test_carbon_footprint_tracking():
    """Test carbon footprint tracking"""
    system = EnergyOptimization()

    carbon = system.carbon_footprint_tracking(
        operation_name="detection_run",
        power_watts=500.0,
        duration_seconds=3600.0,
        energy_source=EnergySource.MEDIUM_POWER,
    )

    assert isinstance(carbon, float)
    assert carbon > 0.0
    assert len(system.energy_profiles) == 1
    assert isinstance(system.energy_profiles[0], EnergyProfile)


def test_carbon_renewable_vs_conventional():
    """Test that renewable sources have lower carbon footprint"""
    system = EnergyOptimization()

    renewable_carbon = system.carbon_footprint_tracking(
        "renewable_op", 1000.0, 3600.0, EnergySource.RENEWABLE
    )

    conventional_carbon = system.carbon_footprint_tracking(
        "conventional_op", 1000.0, 3600.0, EnergySource.HIGH_POWER
    )

    assert renewable_carbon < conventional_carbon


def test_transition_strategy():
    """Test transition from legacy to modern methods"""
    system = EnergyOptimization()

    legacy_consumption, modern_consumption = system.transition_strategy(
        legacy_power=1000.0, modern_power=600.0, transition_rate=0.3
    )

    assert legacy_consumption > modern_consumption
    assert legacy_consumption == 700.0
    assert modern_consumption == 180.0


def test_roi_analysis():
    """Test ROI analysis balancing accuracy vs cost"""
    system = EnergyOptimization()

    high_roi = system.roi_analysis(accuracy_gain=0.8, power_cost=100, time_cost=50)
    low_roi = system.roi_analysis(accuracy_gain=0.2, power_cost=500, time_cost=200)

    assert isinstance(high_roi, float)
    assert isinstance(low_roi, float)
    assert high_roi > low_roi


def test_efficiency_report():
    """Test comprehensive efficiency report generation"""
    system = EnergyOptimization(carbon_limit=100.0)

    system.carbon_footprint_tracking("op1", 200.0, 1800.0, EnergySource.LOW_POWER)
    system.carbon_footprint_tracking("op2", 300.0, 1800.0, EnergySource.RENEWABLE)

    report = system.get_efficiency_report()

    assert "total_energy_kwh" in report
    assert "total_carbon_kg" in report
    assert "average_efficiency" in report
    assert "operation_count" in report
    assert report["operation_count"] == 2
    assert isinstance(report["within_carbon_limit"], bool)

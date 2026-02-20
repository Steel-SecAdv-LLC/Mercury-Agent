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

"""Tests for Bain AI Scaling module"""

from omni_mercury_engine.scaling.bain_ai_scaling import BainAIScaling, ComputeResource


def test_bain_scaling_initialization():
    """Test Bain AI scaling system initialization"""
    system = BainAIScaling(max_power_watts=1500.0)
    assert system.max_power_watts == 1500.0
    assert isinstance(system.current_allocation, dict)
    assert len(system.current_allocation) == 0


def test_optimize_compute_allocation():
    """Test compute resource optimization"""
    system = BainAIScaling()

    workloads = [
        {"id": "detection_1", "priority": 1.0},
        {"id": "detection_2", "priority": 0.5},
        {"id": "detection_3", "priority": 0.3},
    ]

    available = ComputeResource(
        cpu_cores=16, gpu_count=4, memory_gb=64.0, power_watts=800.0, cost_per_hour=10.0
    )

    allocation = system.optimize_compute_allocation(workloads, available)

    assert len(allocation) == 3
    assert "detection_1" in allocation
    assert isinstance(allocation["detection_1"], ComputeResource)


def test_estimate_power_consumption():
    """Test power consumption estimation"""
    system = BainAIScaling(max_power_watts=2000.0)

    model_size = 1_000_000_000
    batch_size = 32
    sequence_length = 512

    power = system.estimate_power_consumption(model_size, batch_size, sequence_length)

    assert isinstance(power, float)
    assert 0 < power <= system.max_power_watts
    assert power > 100.0


def test_power_budget_limit():
    """Test that power estimates respect max budget"""
    system = BainAIScaling(max_power_watts=500.0)

    large_model = 100_000_000_000
    large_batch = 1000
    long_sequence = 10000

    power = system.estimate_power_consumption(large_model, large_batch, long_sequence)

    assert power <= system.max_power_watts


def test_compute_allocation_priorities():
    """Test that higher priority workloads get more resources"""
    system = BainAIScaling()

    workloads = [{"id": "high_priority", "priority": 2.0}, {"id": "low_priority", "priority": 0.5}]

    available = ComputeResource(
        cpu_cores=8, gpu_count=2, memory_gb=32.0, power_watts=400.0, cost_per_hour=5.0
    )

    allocation = system.optimize_compute_allocation(workloads, available)

    high_cores = allocation["high_priority"].cpu_cores
    low_cores = allocation["low_priority"].cpu_cores

    assert high_cores >= low_cores

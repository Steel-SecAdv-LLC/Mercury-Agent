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

"""
Comprehensive integration tests for Mercury Agent ♱ enhancements.

Tests InfrastructureCoordinator with all 12 modules, synthetic data predictions,
and end-to-end workflow validation.
"""

import numpy as np

from omni_mercury_engine.infrastructure import InfrastructureCoordinator
from omni_mercury_engine.models.simulation import SimulationModule
from omni_mercury_engine.space.space_exploration_analyzer import SpaceExplorationAnalyzer


class TestComprehensiveIntegration:
    """Comprehensive integration test suite."""

    def test_coordinator_lists_12_modules(self):
        """Verify coordinator now lists 12 modules (was 11, +1 for SpaceExplorationAnalyzer)."""
        coord = InfrastructureCoordinator()
        modules = coord.list_all_modules()

        assert len(modules) == 12
        assert "space_exploration_analyzer" in modules

        module_info = modules["space_exploration_analyzer"]
        assert module_info["category"] == "scientific"
        assert module_info["priority"] == "high"

    def test_run_single_module(self):
        """Test running 1 module."""
        coord = InfrastructureCoordinator()
        modules = coord.instantiate_filtered_modules(module_names=["ncf_monitor"])

        assert len(modules) == 1
        assert "ncf_monitor" in modules

    def test_run_five_high_priority_modules(self):
        """Test running 5+ high-priority modules."""
        coord = InfrastructureCoordinator()
        modules = coord.instantiate_filtered_modules(priorities=["high"])

        assert len(modules) >= 5
        assert "space_exploration_analyzer" in modules

    def test_run_all_modules(self):
        """Test running all 12 modules."""
        coord = InfrastructureCoordinator()
        modules = coord.instantiate_filtered_modules()

        assert len(modules) == 12

    def test_run_scientific_category_modules(self):
        """Test running all scientific category modules."""
        coord = InfrastructureCoordinator()
        modules = coord.instantiate_filtered_modules(categories=["scientific"])

        assert len(modules) >= 2
        assert "space_exploration_analyzer" in modules
        assert "emerging_tech_monitor" in modules

    def test_space_exploration_cosmic_ray_prediction(self):
        """Run anomaly detection on synthetic cosmic ray data."""
        analyzer = SpaceExplorationAnalyzer()

        normal_data = np.random.randn(90, 5) * 0.5 + 1.0
        cosmic_ray_events = np.random.randn(10, 5) * 15 + 20.0
        data = np.vstack([normal_data, cosmic_ray_events])

        result = analyzer.detect(data, "cosmic_ray", {"telescope": "hubble_test"})

        assert result["anomaly_detected"] is True
        assert result["cosmic_ray_events"] >= 5
        assert result["severity"] in ["low", "medium", "high", "critical"]
        assert len(result["recommendations"]) > 0

        print("\n=== Cosmic Ray Prediction ===")
        print(f"Anomaly Score: {result['anomaly_score']:.2f}")
        print(f"Cosmic Ray Events: {result['cosmic_ray_events']}")
        print(f"Severity: {result['severity']}")
        print(
            "Insights: Detected high-energy particle events indicative of cosmic ray interference"
        )

    def test_space_exploration_satellite_position_prediction(self):
        """Run anomaly detection on synthetic satellite position data with deviations."""
        analyzer = SpaceExplorationAnalyzer()

        earth_radius = 6371.0
        leo_altitude = 400.0
        normal_orbit = np.random.randn(50, 3) * 5 + np.array([earth_radius + leo_altitude, 0, 0])

        anomalous_orbit = np.random.randn(50, 3) * 50 + np.array(
            [earth_radius + leo_altitude + 150, 100, 50]
        )

        data = np.vstack([normal_orbit, anomalous_orbit])

        result = analyzer.detect(
            data, "satellite_position", {"satellite_id": "TEST-SAT-001", "orbit_type": "leo"}
        )

        assert "anomaly_detected" in result
        assert "severity" in result
        assert "deviation_from_expected_km" in result
        assert len(result["insights"]) > 0

        print("\n=== Satellite Position Prediction ===")
        print(f"Anomaly Detected: {result['anomaly_detected']}")
        print(f"Severity: {result['severity']}")
        print(f"Deviation: {result['deviation_from_expected_km']:.2f} km")
        print(f"Insights: {result['insights'][0]}")
        rec = result["recommendations"][0] if result["recommendations"] else "Nominal orbit"
        print(f"Recommendation: {rec}")

    def test_simulation_module_collatz_prediction(self):
        """Run Collatz conjecture exploration and report insights."""
        sim = SimulationModule()

        result = sim.explore_conjecture("collatz", search_space=5000)

        assert result["conjecture"] == "collatz"
        assert result["all_reached_one"] is True
        assert result["viability_score"] == 1.0

        print("\n=== Collatz Conjecture Exploration ===")
        print(f"Cases Explored: {result['explored_cases']}")
        print(f"All Reached 1: {result['all_reached_one']}")
        print(f"Average Steps: {result['average_steps']:.2f}")
        print(f"Max Height: {result['max_height']}")
        print(f"Insights: {result['insights'][0]}")

    def test_simulation_module_p_vs_np_analysis(self):
        """Analyze P vs NP Millennium Prize Problem."""
        sim = SimulationModule()

        result = sim.analyze_millennium_problem("p_vs_np")

        assert result["problem"] == "p_vs_np"
        assert result["status"] == "unsolved"
        assert "complexity_gap" in result
        assert "ethical_flags" in result

        print("\n=== P vs NP Analysis ===")
        print(f"Status: {result['status']}")
        print(f"Prize Amount: {result['prize_amount']}")
        print(f"Complexity Gap (n=200): {result['complexity_gap']:.2e}x")
        print(f"Analysis: {result['analysis']}")
        print(f"Ethical Implications: {', '.join(result['ethical_flags'])}")

    def test_multiverse_prediction_with_simulation(self):
        """Test multiverse branching prediction."""
        sim = SimulationModule(config={"num_branches": 15})

        data = np.random.randn(10, 20)
        result = sim.predict(data)

        assert result["num_branches_explored"] == 15
        assert "anomaly_scores" in result
        assert "branch_variance" in result
        assert len(result["anomaly_scores"]) == 10

        print("\n=== Multiverse Branching Prediction ===")
        print(f"Branches Explored: {result['num_branches_explored']}")
        print(f"Mean Anomaly Score: {np.mean(result['anomaly_scores']):.3f}")
        print(f"Branch Variance: {np.mean(result['branch_variance']):.3f}")
        print(f"Ethical Risks Detected: {result['ethical_risk_detected']}")

    def test_orbital_debris_collision_risk_prediction(self):
        """Predict orbital debris collision risks with close approaches."""
        analyzer = SpaceExplorationAnalyzer(config={"debris_proximity_km": 15.0})

        safe_orbit = np.array([7000, 0, 0]) + np.random.randn(40, 3) * 50

        debris_1 = np.array([7100, 50, 20])
        debris_2 = np.array([7105, 52, 21])
        close_approach = np.vstack([debris_1, debris_2])

        positions = np.vstack([safe_orbit, close_approach])
        velocities = np.random.randn(len(positions), 3) * 0.5

        result = analyzer.predict_orbital_debris(
            positions, velocities, {"satellite_id": "ISS-TEST"}
        )

        assert result["analysis_type"] == "orbital_debris"
        assert len(result["proximity_warnings"]) > 0

        print("\n=== Orbital Debris Collision Risk ===")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Proximity Warnings: {len(result['proximity_warnings'])}")
        if result["proximity_warnings"]:
            min_sep = min(w["separation_km"] for w in result["proximity_warnings"])
            print(f"Minimum Separation: {min_sep:.2f} km")
        print(f"Recommendations: {result['recommendations'][0]}")

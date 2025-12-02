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

"""
Integration tests for SpaceExplorationAnalyzer with simulated NASA telemetry.
Tests end-to-end functionality with Hubble-inspired scenarios.
"""

import numpy as np
from omni_anomaly_engine.space import SpaceExplorationAnalyzer


class TestSpaceExplorationIntegration:

    def test_hubble_cosmic_ray_detection_realistic(self):
        """Test cosmic ray detection with Hubble-like energy distribution."""
        analyzer = SpaceExplorationAnalyzer(config={"cosmic_ray_threshold": 3.5})

        normal_exposure = np.random.randn(9500, 5) * 0.8 + 1.2
        cosmic_ray_events = np.random.randn(500, 5) * 18 + 25.0
        hubble_data = np.vstack([normal_exposure, cosmic_ray_events])

        result = analyzer.analyze_cosmic_rays(
            hubble_data, {"telescope": "HST", "instrument": "WFC3", "exposure_time": 600}
        )

        assert result["anomaly_detected"] is True
        assert result["cosmic_ray_events"] >= 400
        assert result["severity"] in ["medium", "high", "critical"]
        assert len(result["recommendations"]) > 0

    def test_iss_orbital_debris_prediction(self):
        """Test ISS orbital debris risk with close approach scenarios."""
        analyzer = SpaceExplorationAnalyzer()

        earth_radius = 6371.0
        iss_altitude = 408.0
        iss_position = np.array([earth_radius + iss_altitude, 0, 0])

        positions = np.random.randn(150, 3) * 15 + iss_position
        debris_close = np.array([iss_position + [5, 2, 1], iss_position + [3, 4, -1]])
        positions = np.vstack([positions, debris_close])
        velocities = np.random.randn(len(positions), 3) * 0.5 + [7.66, 0, 0]

        result = analyzer.predict_orbital_debris(
            positions, velocities, {"satellite_id": "ISS", "orbit_type": "leo"}
        )

        assert result["risk_level"] in ["low", "medium", "high", "critical"]
        assert len(result["proximity_warnings"]) >= 1
        if result["proximity_warnings"]:
            assert result["proximity_warnings"][0]["separation_km"] < 50.0

    def test_spectral_quasar_detection(self):
        """Test spectral analysis with quasar emission lines."""
        analyzer = SpaceExplorationAnalyzer()

        wavelengths = np.linspace(400, 700, 1000)
        continuum = 100 * np.exp(-((wavelengths - 550) ** 2) / (2 * 50**2))

        h_alpha = 50 * np.exp(-((wavelengths - 656.3) ** 2) / (2 * 2**2))
        h_beta = 30 * np.exp(-((wavelengths - 486.1) ** 2) / (2 * 2**2))

        spectrum = continuum + h_alpha + h_beta + np.random.randn(len(wavelengths)) * 2

        result = analyzer.detect_spectral_anomalies(
            wavelengths, spectrum, {"source": "quasar_candidate", "telescope": "HST"}
        )

        assert "identified_lines" in result
        assert len(result["identified_lines"]) >= 1
        identified_elements = [line["element"] for line in result["identified_lines"]]
        assert any("hydrogen" in elem.lower() for elem in identified_elements)

    def test_leo_satellite_deviation(self):
        """Test LEO satellite position deviation detection."""
        analyzer = SpaceExplorationAnalyzer()

        earth_radius = 6371.0
        leo_altitude = 550.0
        expected_radius = earth_radius + leo_altitude

        normal_positions = []
        for i in range(200):
            angle = i * 2 * np.pi / 200
            pos = (
                np.array([expected_radius * np.cos(angle), expected_radius * np.sin(angle), 0])
                + np.random.randn(3) * 2
            )
            normal_positions.append(pos)

        anomalous_positions = []
        for i in range(20):
            angle = (200 + i) * 2 * np.pi / 220
            pos = (
                np.array(
                    [
                        (expected_radius + 50) * np.cos(angle),
                        (expected_radius + 50) * np.sin(angle),
                        30,
                    ]
                )
                + np.random.randn(3) * 5
            )
            anomalous_positions.append(pos)

        all_positions = np.vstack([normal_positions, anomalous_positions])

        result = analyzer.detect(
            all_positions,
            "satellite_position",
            {"satellite_id": "STARLINK-1234", "orbit_type": "leo"},
        )

        assert "anomaly_detected" in result
        assert "position_outliers" in result
        assert result["position_outliers"] > 0

    def test_collision_avoidance_scenario(self):
        """Test debris collision avoidance with critical proximity."""
        analyzer = SpaceExplorationAnalyzer(config={"debris_proximity_km": 5.0})

        earth_radius = 6371.0
        satellite_pos = np.array([earth_radius + 400, 100, 50])

        positions = []
        velocities = []
        for i in range(80):
            pos = satellite_pos + np.random.randn(3) * 30
            vel = np.array([7.5, 0, 0]) + np.random.randn(3) * 0.3
            positions.append(pos)
            velocities.append(vel)

        positions.append(satellite_pos + np.array([2, 1, 0.5]))
        velocities.append(np.array([7.5, 0, 0]))
        positions.append(satellite_pos + np.array([1.5, 0.5, -1]))
        velocities.append(np.array([7.5, 0, 0]))

        all_positions = np.array(positions)
        all_velocities = np.array(velocities)

        result = analyzer.predict_orbital_debris(
            all_positions, all_velocities, {"satellite_id": "CRITICAL-SAT", "orbit_type": "leo"}
        )

        assert result["risk_level"] in ["high", "critical"]
        assert len(result["proximity_warnings"]) >= 1
        assert any(
            "maneuver" in r.lower() or "avoidance" in r.lower() for r in result["recommendations"]
        )

    def test_end_to_end_mission_monitoring(self):
        """Test complete mission monitoring scenario."""
        analyzer = SpaceExplorationAnalyzer()

        earth_radius = 6371.0
        positions = []
        for i in range(50):
            angle = i * 2 * np.pi / 50
            pos = (
                np.array(
                    [(earth_radius + 450) * np.cos(angle), (earth_radius + 450) * np.sin(angle), 0]
                )
                + np.random.randn(3) * 3
            )
            positions.append(pos)

        position_result = analyzer.detect(
            np.array(positions),
            "satellite_position",
            {"satellite_id": "MISSION-SAT", "orbit_type": "leo"},
        )

        cosmic_data = np.vstack(
            [np.random.randn(4000, 5) * 0.7 + 1.1, np.random.randn(1000, 5) * 20 + 30]
        )
        cosmic_result = analyzer.analyze_cosmic_rays(cosmic_data, {"mission": "monitoring"})

        assert "anomaly_detected" in position_result
        assert "anomaly_detected" in cosmic_result
        assert cosmic_result["cosmic_ray_events"] >= 700

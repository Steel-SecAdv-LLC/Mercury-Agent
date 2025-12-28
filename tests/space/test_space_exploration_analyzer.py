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

from __future__ import annotations

"""
Tests for SpaceExplorationAnalyzer (Hubble-inspired cosmic anomaly detection).
"""

import numpy as np

from omni_anomaly_engine.space.space_exploration_analyzer import SpaceExplorationAnalyzer


class TestSpaceExplorationAnalyzer:
    """Test suite for SpaceExplorationAnalyzer."""

    def test_analyzer_instantiation(self):
        """Test space exploration analyzer can be instantiated."""
        analyzer = SpaceExplorationAnalyzer()
        assert analyzer is not None
        assert analyzer.cosmic_ray_threshold == 3.0

    def test_analyzer_with_config(self):
        """Test analyzer with custom config."""
        config = {"cosmic_ray_threshold": 5.0, "debris_proximity_km": 5.0}
        analyzer = SpaceExplorationAnalyzer(config=config)
        assert analyzer.cosmic_ray_threshold == 5.0
        assert analyzer.debris_proximity_km == 5.0

    def test_analyze_cosmic_rays(self):
        """Test cosmic ray anomaly detection."""
        analyzer = SpaceExplorationAnalyzer()

        normal_data = np.random.randn(50, 5) * 0.5
        cosmic_ray_spike = np.random.randn(5, 5) * 10
        data = np.vstack([normal_data, cosmic_ray_spike])

        result = analyzer.analyze_cosmic_rays(data, {"telescope": "hubble"})

        assert "analysis_type" in result
        assert result["analysis_type"] == "cosmic_ray"
        assert "anomaly_detected" in result
        assert "anomaly_score" in result
        assert "cosmic_ray_events" in result
        assert "severity" in result
        assert "recommendations" in result

    def test_detect_spectral_anomalies(self):
        """Test spectral pattern matching."""
        analyzer = SpaceExplorationAnalyzer()

        wavelengths = np.linspace(400, 700, 300)
        intensities = np.random.randn(300) * 0.1 + 1.0
        intensities[150] = 5.0

        result = analyzer.detect_spectral_anomalies(
            wavelengths, intensities, {"target_object": "star_alpha_centauri"}
        )

        assert result["analysis_type"] == "spectral"
        assert "anomaly_detected" in result
        assert "total_peaks" in result
        assert "identified_lines" in result
        assert "unidentified_peaks" in result

    def test_predict_orbital_debris(self):
        """Test orbital debris risk prediction."""
        analyzer = SpaceExplorationAnalyzer()

        normal_positions = np.random.randn(50, 3) * 100 + np.array([7000, 0, 0])
        close_approach = np.array([[7000, 2, 0], [7000, 2.5, 0]])

        positions = np.vstack([normal_positions, close_approach])
        velocities = np.random.randn(len(positions), 3) * 0.1

        result = analyzer.predict_orbital_debris(positions, velocities, {"satellite_id": "ISS"})

        assert result["analysis_type"] == "orbital_debris"
        assert "risk_level" in result
        assert "proximity_warnings" in result
        assert "mean_orbital_altitude_km" in result

    def test_analyze_satellite_position(self):
        """Test satellite position deviation analysis."""
        analyzer = SpaceExplorationAnalyzer()

        earth_radius = 6371.0
        leo_altitude = 400.0
        positions = np.random.randn(100, 3) * 10 + np.array([earth_radius + leo_altitude, 0, 0])

        result = analyzer.analyze_satellite_position(
            positions, {"satellite_id": "SAT-001", "orbit_type": "leo"}
        )

        assert result["analysis_type"] == "satellite_position"
        assert "anomaly_detected" in result
        assert "severity" in result
        assert "mean_orbital_radius_km" in result
        assert "altitude_above_earth_km" in result
        assert "insights" in result

    def test_detect_with_cosmic_ray_type(self):
        """Test detect method with cosmic_ray analysis type."""
        analyzer = SpaceExplorationAnalyzer()
        data = np.random.randn(100, 5)

        result = analyzer.detect(data, "cosmic_ray")

        assert result["analysis_type"] == "cosmic_ray"
        assert "anomaly_score" in result

    def test_detect_with_spectral_type(self):
        """Test detect method with spectral analysis type."""
        analyzer = SpaceExplorationAnalyzer()
        data = np.random.randn(100, 2)

        result = analyzer.detect(data, "spectral")

        assert result["analysis_type"] == "spectral"

    def test_detect_with_satellite_position_type(self):
        """Test detect method with satellite_position analysis type."""
        analyzer = SpaceExplorationAnalyzer()
        data = np.random.randn(100, 3) * 100 + np.array([7000, 0, 0])

        result = analyzer.detect(data, "satellite_position", {"orbit_type": "leo"})

        assert result["analysis_type"] == "satellite_position"
        assert "severity" in result

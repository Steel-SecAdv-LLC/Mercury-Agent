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
Space Exploration Analyzer - Hubble-Inspired Cosmic Anomaly Detection

Inspired by Hubble Space Telescope anomaly detection in cosmic data streams
and NASA telemetry monitoring for orbital threats.

Key influences:
- Hubble Space Telescope: Deep field observations, cosmic ray detection
- NASA telemetry: Satellite health monitoring, orbital debris tracking
- Spectroscopy: Absorption/emission line analysis for cosmic phenomena
- Orbital mechanics: Keplerian elements, perturbation analysis

Research sources:
- NASA Technical Reports Server (NTRS): https://ntrs.nasa.gov/
- Hubble Space Telescope mission data
- Space debris monitoring systems (ESA, NASA, NORAD)

"""

import logging
from typing import Any

import numpy as np


class SpaceExplorationAnalyzer:
    """
    Hubble-inspired analyzer for cosmic anomalies and orbital threats.

    Features:
    - Cosmic ray anomaly detection (energetic particle events)
    - Spectral pattern matching (absorption/emission lines)
    - Orbital debris risk prediction (collision avoidance)
    - Satellite position deviation analysis (Keplerian orbit monitoring)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize space exploration analyzer.

        Args:
            config: Configuration dictionary with optional keys:
                - cosmic_ray_threshold: Threshold for cosmic ray detection
                - debris_proximity_km: Distance threshold for debris warnings
                - spectral_tolerance: Wavelength matching tolerance
        """
        self.config = config or {}
        self.cosmic_ray_threshold = self.config.get("cosmic_ray_threshold", 3.0)
        self.debris_proximity_km = self.config.get("debris_proximity_km", 10.0)
        self.spectral_tolerance = self.config.get("spectral_tolerance", 0.5)
        self.logger = logging.getLogger(__name__)

        self.spectral_reference_lines = self._initialize_spectral_database()
        self.orbital_parameters = self._initialize_orbital_parameters()

    def _initialize_spectral_database(self) -> dict[str, list[float]]:
        """
        Initialize reference database of known spectral lines.

        Returns:
            Dictionary mapping element/molecule to wavelength list (nm)
        """
        return {
            "hydrogen_balmer": [656.3, 486.1, 434.0, 410.2],
            "hydrogen_lyman": [121.6, 102.6, 97.3],
            "helium": [587.6, 501.6, 471.3, 447.1],
            "oxygen": [777.4, 844.6, 630.0],
            "carbon": [247.9, 283.6, 658.0],
            "nitrogen": [399.5, 500.5, 567.9],
            "iron": [438.4, 440.5, 527.0],
            "calcium_h_k": [396.8, 393.4],
            "sodium_d": [589.0, 589.6],
            "magnesium": [285.2, 517.3, 518.4],
        }

    def _initialize_orbital_parameters(self) -> dict[str, Any]:
        """
        Initialize standard orbital parameters for common orbits.

        Returns:
            Dictionary with orbital parameter ranges
        """
        return {
            "leo": {
                "altitude_km": (200, 2000),
                "velocity_km_s": (7.0, 7.8),
                "period_min": (88, 127),
            },
            "meo": {
                "altitude_km": (2000, 35786),
                "velocity_km_s": (3.0, 7.0),
                "period_min": (127, 1436),
            },
            "geo": {
                "altitude_km": (35786, 35786),
                "velocity_km_s": (3.07, 3.07),
                "period_min": (1436, 1436),
            },
        }

    def detect(
        self, data: np.ndarray, analysis_type: str, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Detect anomalies in space exploration data.

        Args:
            data: Telemetry, spectral, or positional data (numpy array)
            analysis_type: Type of analysis ('cosmic_ray', 'spectral',
                          'orbital_debris', 'satellite_position')
            context: Additional context dict with keys like:
                - telescope: Name of telescope/instrument
                - satellite_id: Satellite identifier
                - orbit_type: 'leo', 'meo', 'geo'
                - target_object: Astronomical object being observed

        Returns:
            Detection results with anomaly score, threat assessment, recommendations
        """
        context = context or {}
        analysis_type = analysis_type.lower()

        if analysis_type == "cosmic_ray":
            return self.analyze_cosmic_rays(data, context)
        elif analysis_type == "spectral":
            if data.shape[1] >= 2:
                wavelengths = data[:, 0]
                intensities = data[:, 1]
            else:
                wavelengths = np.arange(len(data))
                intensities = data.flatten()
            return self.detect_spectral_anomalies(wavelengths, intensities, context)
        elif analysis_type == "orbital_debris":
            if data.shape[1] >= 6:
                position_data = data[:, :3]
                velocity_data = data[:, 3:6]
            else:
                position_data = data
                velocity_data = np.zeros_like(position_data)
            return self.predict_orbital_debris(position_data, velocity_data, context)
        elif analysis_type == "satellite_position":
            return self.analyze_satellite_position(data, context)
        else:
            return {
                "analysis_type": analysis_type,
                "anomaly_detected": False,
                "anomaly_score": 0.0,
                "error": f"Unknown analysis type: {analysis_type}",
                "recommendations": [
                    "Use valid analysis_type: cosmic_ray, spectral, "
                    "orbital_debris, satellite_position"
                ],
            }

    def analyze_cosmic_rays(
        self, data: np.ndarray, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Detect cosmic ray anomalies in sensor data.

        Cosmic rays are high-energy particles from space that can
        interfere with telescopes and satellites.

        Args:
            data: Sensor readings (energy, count rate, etc.)
            context: Additional context (instrument, exposure time, etc.)

        Returns:
            Dictionary with cosmic ray detection results
        """
        context = context or {}

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        mean_energy = np.mean(data, axis=0)
        std_energy = np.std(data, axis=0)

        z_scores = np.abs((data - mean_energy) / (std_energy + 1e-8))

        cosmic_ray_events = np.sum(z_scores > self.cosmic_ray_threshold, axis=0)
        anomaly_score = float(np.max(z_scores))

        event_indices = np.where(np.max(z_scores, axis=1) > self.cosmic_ray_threshold)[0]

        severity = "low"
        if anomaly_score > 5.0:
            severity = "critical"
        elif anomaly_score > 4.0:
            severity = "high"
        elif anomaly_score > 3.0:
            severity = "medium"

        recommendations = []
        if len(event_indices) > 0:
            recommendations.append(f"Detected {len(event_indices)} cosmic ray events")
            recommendations.append("Apply cosmic ray rejection algorithms")
            recommendations.append("Consider re-observation if critical data affected")

        if severity in ["high", "critical"]:
            recommendations.append("Alert ground control for potential instrument damage")

        return {
            "analysis_type": "cosmic_ray",
            "anomaly_detected": len(event_indices) > 0,
            "anomaly_score": anomaly_score,
            "cosmic_ray_events": int(np.sum(cosmic_ray_events)),
            "event_indices": event_indices.tolist()[:10],
            "severity": severity,
            "mean_energy": float(np.mean(mean_energy)),
            "std_energy": float(np.mean(std_energy)),
            "recommendations": recommendations,
            "telescope": context.get("telescope", "unknown"),
        }

    def detect_spectral_anomalies(
        self, wavelengths: np.ndarray, intensities: np.ndarray, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Match spectral patterns against known absorption/emission lines.

        Spectroscopy is a key tool for identifying chemical composition
        and physical conditions of cosmic objects.

        Args:
            wavelengths: Wavelength array (nm)
            intensities: Intensity/flux array
            context: Additional context (target object, redshift, etc.)

        Returns:
            Dictionary with spectral analysis results
        """
        context = context or {}

        normalized_intensities = (intensities - np.min(intensities)) / (
            np.max(intensities) - np.min(intensities) + 1e-8
        )

        intensity_gradient = np.gradient(normalized_intensities)

        peaks = []
        for i in range(1, len(intensity_gradient) - 1):
            if intensity_gradient[i - 1] > 0 and intensity_gradient[i] < 0:
                if normalized_intensities[i] > 0.5:
                    peaks.append((wavelengths[i], intensities[i]))

        identified_lines = []
        unidentified_peaks = []

        for peak_wavelength, peak_intensity in peaks:
            identified = False
            for element, reference_lines in self.spectral_reference_lines.items():
                for ref_line in reference_lines:
                    if abs(peak_wavelength - ref_line) < self.spectral_tolerance:
                        identified_lines.append(
                            {
                                "element": element,
                                "wavelength_observed": float(peak_wavelength),
                                "wavelength_reference": ref_line,
                                "intensity": float(peak_intensity),
                            }
                        )
                        identified = True
                        break
                if identified:
                    break

            if not identified:
                unidentified_peaks.append(
                    {
                        "wavelength": float(peak_wavelength),
                        "intensity": float(peak_intensity),
                    }
                )

        anomaly_score = len(unidentified_peaks) / max(1, len(peaks))

        recommendations = []
        if unidentified_peaks:
            recommendations.append(
                f"Found {len(unidentified_peaks)} unidentified spectral features"
            )
            recommendations.append("Compare with high-resolution spectral atlases")
            recommendations.append("Consider redshift corrections or rare elements")

        if identified_lines:
            elements = set(line["element"] for line in identified_lines)
            recommendations.append(f'Identified elements: {", ".join(elements)}')

        return {
            "analysis_type": "spectral",
            "anomaly_detected": len(unidentified_peaks) > 0,
            "anomaly_score": float(anomaly_score),
            "total_peaks": len(peaks),
            "identified_lines": identified_lines[:10],
            "unidentified_peaks": unidentified_peaks[:5],
            "target_object": context.get("target_object", "unknown"),
            "recommendations": recommendations,
        }

    def predict_orbital_debris(
        self, position_data: np.ndarray, velocity_data: np.ndarray, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Predict orbital debris collision risks.

        Space debris poses significant threat to satellites and
        space stations. Early warning enables avoidance maneuvers.

        Args:
            position_data: Position vectors (km) shape (N, 3)
            velocity_data: Velocity vectors (km/s) shape (N, 3)
            context: Additional context (satellite_id, orbit_type, etc.)

        Returns:
            Dictionary with debris risk assessment
        """
        context = context or {}

        if position_data.ndim == 1:
            position_data = position_data.reshape(1, -1)
        if velocity_data.ndim == 1:
            velocity_data = velocity_data.reshape(1, -1)

        distances = np.linalg.norm(position_data, axis=1)

        mean_distance = np.mean(distances)

        proximity_warnings = []
        for i in range(len(distances) - 1):
            separation = np.linalg.norm(position_data[i] - position_data[i + 1])
            if separation < self.debris_proximity_km:
                proximity_warnings.append(
                    {
                        "time_index": i,
                        "separation_km": float(separation),
                    }
                )

        risk_level = "low"
        if proximity_warnings:
            min_separation = min(w["separation_km"] for w in proximity_warnings)
            if min_separation < 1.0:
                risk_level = "critical"
            elif min_separation < 5.0:
                risk_level = "high"
            else:
                risk_level = "medium"

        anomaly_score = len(proximity_warnings) / max(1, len(distances))

        recommendations = []
        if proximity_warnings:
            recommendations.append(
                f"Warning: {len(proximity_warnings)} close approach events detected"
            )
            recommendations.append(
                f'Minimum separation: {min(w["separation_km"] for w in proximity_warnings):.2f} km'
            )
            recommendations.append("Recommend debris avoidance maneuver planning")

        if risk_level in ["high", "critical"]:
            recommendations.append("URGENT: Coordinate with space traffic management")
            recommendations.append("Implement collision avoidance protocol")

        return {
            "analysis_type": "orbital_debris",
            "anomaly_detected": len(proximity_warnings) > 0,
            "anomaly_score": float(anomaly_score),
            "risk_level": risk_level,
            "proximity_warnings": proximity_warnings[:10],
            "mean_orbital_altitude_km": float(mean_distance),
            "satellite_id": context.get("satellite_id", "unknown"),
            "recommendations": recommendations,
        }

    def analyze_satellite_position(
        self, data: np.ndarray, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Analyze satellite position deviations from expected orbit.

        Monitors satellite health by detecting deviations from
        predicted Keplerian orbital elements.

        Args:
            data: Position data (x, y, z) in km, shape (N, 3) or (N, >3)
            context: Additional context (satellite_id, orbit_type, etc.)

        Returns:
            Dictionary with position deviation analysis
        """
        context = context or {}
        orbit_type = context.get("orbit_type", "leo")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        positions = data[:, :3]

        orbital_radii = np.linalg.norm(positions, axis=1)
        mean_radius = np.mean(orbital_radii)
        std_radius = np.std(orbital_radii)

        expected_params = self.orbital_parameters.get(orbit_type, self.orbital_parameters["leo"])
        expected_altitude_range = expected_params["altitude_km"]
        earth_radius_km = 6371.0
        expected_radius_range = (
            earth_radius_km + expected_altitude_range[0],
            earth_radius_km + expected_altitude_range[1],
        )

        in_range = (
            mean_radius >= expected_radius_range[0] and mean_radius <= expected_radius_range[1]
        )

        deviation_from_expected = 0.0
        if mean_radius < expected_radius_range[0]:
            deviation_from_expected = expected_radius_range[0] - mean_radius
        elif mean_radius > expected_radius_range[1]:
            deviation_from_expected = mean_radius - expected_radius_range[1]

        z_scores = np.abs((orbital_radii - mean_radius) / (std_radius + 1e-8))
        outlier_indices = np.where(z_scores > 3.0)[0]

        anomaly_score = deviation_from_expected / 100.0
        anomaly_score = min(1.0, max(0.0, anomaly_score))

        severity = "nominal"
        if deviation_from_expected > 100:
            severity = "critical"
        elif deviation_from_expected > 50:
            severity = "high"
        elif deviation_from_expected > 20:
            severity = "medium"
        elif len(outlier_indices) > 0:
            severity = "low"

        recommendations = []
        if not in_range:
            recommendations.append(
                f"Satellite altitude outside expected {orbit_type.upper()} range"
            )
            recommendations.append(
                f"Deviation: {deviation_from_expected:.2f} km from nominal orbit"
            )

        if len(outlier_indices) > 0:
            recommendations.append(f"Detected {len(outlier_indices)} position outliers")
            recommendations.append("Check attitude control and propulsion systems")

        if severity in ["high", "critical"]:
            recommendations.append("ALERT: Significant orbital deviation detected")
            recommendations.append("Recommend immediate satellite health check")
            recommendations.append("Assess need for orbit correction maneuver")

        alt_range = f"{expected_altitude_range[0]}-{expected_altitude_range[1]}"
        insights = [
            f"Mean orbital radius: {mean_radius:.2f} km",
            f"Altitude above Earth: {mean_radius - earth_radius_km:.2f} km",
            f"Expected {orbit_type.upper()} altitude: {alt_range} km",
        ]

        return {
            "analysis_type": "satellite_position",
            "anomaly_detected": not in_range or len(outlier_indices) > 0,
            "anomaly_score": float(anomaly_score),
            "severity": severity,
            "satellite_id": context.get("satellite_id", "unknown"),
            "orbit_type": orbit_type,
            "mean_orbital_radius_km": float(mean_radius),
            "altitude_above_earth_km": float(mean_radius - earth_radius_km),
            "deviation_from_expected_km": float(deviation_from_expected),
            "position_outliers": len(outlier_indices),
            "within_expected_range": in_range,
            "insights": insights,
            "recommendations": recommendations,
        }

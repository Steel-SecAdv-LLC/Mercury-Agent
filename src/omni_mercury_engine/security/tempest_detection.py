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
TEMPEST Detection Module - Electromagnetic Emanation Security

Electromagnetic eavesdropping countermeasures and TEMPEST monitoring:
- EM emanation detection and analysis
- Side-channel vulnerability assessment
- RF spectrum anomaly detection
- Compromising emanations identification
- EMSEC (Emissions Security) compliance monitoring

⚠️ SIMULATION-BASED: Research/development tool for TEMPEST analysis patterns.
Operational deployment requires specialized equipment and security clearance.

Research sources:
- TEMPEST/EMSEC standards (NSA NACSIM 5000)
- Van Eck phreaking research
- Side-channel attack literature
- RF spectrum analysis methodologies

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class EmanationType(Enum):
    """Electromagnetic emanation types"""

    VIDEO_DISPLAY = "video_display_emanation"
    KEYBOARD = "keyboard_emanation"
    PROCESSOR = "processor_emanation"
    NETWORK_CABLE = "network_cable_emanation"
    POWER_LINE = "power_line_emanation"
    ACOUSTIC = "acoustic_emanation"
    OPTICAL = "optical_emanation"


class TEMPESTThreatLevel(Enum):
    """TEMPEST threat classification"""

    NO_THREAT = "no_threat"
    LOW = "low_risk"
    MODERATE = "moderate_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_risk"


@dataclass
class TEMPESTAnalysisResult:
    """TEMPEST detection results"""

    emanation_detected: bool
    confidence: float
    threat_level: str
    risk_score: float

    emanation_types: list[str] = field(default_factory=list)
    frequency_bands: list[dict] = field(default_factory=list)
    signal_strength_dbm: float | None = None

    compromising_potential: float = 0.0
    reconstruction_feasibility: float = 0.0

    vulnerable_equipment: list[str] = field(default_factory=list)
    shielding_effectiveness: float | None = None

    countermeasures: list[str] = field(default_factory=list)
    compliance_status: dict[str, bool] = field(default_factory=dict)


class RFSpectrumAnalyzer:
    """
    RF spectrum analysis for EM emanation detection.

    Analyzes radio frequency spectrum for compromising emanations.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.tempest_frequency_bands = {
            "vga_video": (25e6, 200e6),
            "dvi_hdmi": (100e6, 1.5e9),
            "keyboard": (1e3, 100e3),
            "ethernet": (10e6, 100e6),
            "processor": (100e6, 3e9),
        }

    def analyze_spectrum(self, spectrum_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze RF spectrum for emanations.

        Args:
            spectrum_data: RF spectrum measurements

        Returns:
            Spectrum analysis with emanation detection
        """
        frequencies = spectrum_data.get("frequencies", [])
        power_levels = spectrum_data.get("power_dbm", [])

        if not frequencies or not power_levels:
            return {"emanation_detected": False}

        emanations = []
        frequency_bands_of_interest = []

        for equipment, (freq_min, freq_max) in self.tempest_frequency_bands.items():
            band_emanation = self._detect_band_emanation(
                frequencies, power_levels, freq_min, freq_max
            )

            if band_emanation["detected"]:
                emanations.append(equipment)
                frequency_bands_of_interest.append(
                    {
                        "equipment": equipment,
                        "frequency_range": (freq_min, freq_max),
                        "peak_power_dbm": band_emanation["peak_power"],
                        "compromising_potential": band_emanation["compromising_potential"],
                    }
                )

        max_signal_strength = max(power_levels) if power_levels else -120.0

        return {
            "emanation_detected": len(emanations) > 0,
            "emanation_sources": emanations,
            "frequency_bands": frequency_bands_of_interest,
            "max_signal_strength_dbm": max_signal_strength,
            "spectrum_occupancy": len(emanations) / len(self.tempest_frequency_bands),
        }

    def _detect_band_emanation(
        self, frequencies: list[float], power_levels: list[float], freq_min: float, freq_max: float
    ) -> dict[str, Any]:
        """Detect emanations in specific frequency band"""
        band_indices = [i for i, f in enumerate(frequencies) if freq_min <= f <= freq_max]

        if not band_indices:
            return {"detected": False, "peak_power": -120.0, "compromising_potential": 0.0}

        band_powers = [power_levels[i] for i in band_indices]
        peak_power = max(band_powers)
        avg_power = np.mean(band_powers)

        noise_floor = -100.0
        signal_to_noise = peak_power - noise_floor

        emanation_detected = signal_to_noise > 20.0

        compromising_potential = min(signal_to_noise / 60.0, 1.0) if emanation_detected else 0.0

        return {
            "detected": emanation_detected,
            "peak_power": peak_power,
            "avg_power": avg_power,
            "signal_to_noise": signal_to_noise,
            "compromising_potential": compromising_potential,
        }


class VideoEmanationDetector(nn.Module):
    """
    Neural network for video display emanation detection.

    Detects Van Eck phreaking vulnerabilities in video displays.
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()

        self.emanation_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.reconstruction_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.resolution_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, emanation_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for video emanation analysis.

        Args:
            emanation_features: Video emanation characteristics

        Returns:
            Tuple of (reconstruction_feasibility, resolution_category)
        """
        encoded = self.emanation_encoder(emanation_features)

        reconstruction_score = self.reconstruction_predictor(encoded)
        resolution_probs = self.resolution_estimator(encoded)

        return reconstruction_score, resolution_probs


class SideChannelVulnerabilityAssessor:
    """
    Side-channel vulnerability assessment.

    Evaluates equipment for side-channel attack susceptibility.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def assess_vulnerabilities(self, equipment_data: dict[str, Any]) -> dict[str, Any]:
        """
        Assess equipment for side-channel vulnerabilities.

        Args:
            equipment_data: Equipment characteristics and measurements

        Returns:
            Vulnerability assessment results
        """
        vulnerabilities = []
        vulnerability_scores = {}

        em_shielding = equipment_data.get("em_shielding_db", 0.0)
        if em_shielding < 40.0:
            vulnerabilities.append("insufficient_em_shielding")
            vulnerability_scores["em_shielding"] = 1.0 - (em_shielding / 80.0)

        power_filtering = equipment_data.get("power_line_filtering", False)
        if not power_filtering:
            vulnerabilities.append("unfiltered_power_lines")
            vulnerability_scores["power_filtering"] = 0.8

        cable_shielding = equipment_data.get("cable_shielding", False)
        if not cable_shielding:
            vulnerabilities.append("unshielded_cables")
            vulnerability_scores["cable_shielding"] = 0.7

        distance_to_boundary = equipment_data.get("distance_to_boundary_m", 100.0)
        if distance_to_boundary < 10.0:
            vulnerabilities.append("insufficient_control_zone")
            vulnerability_scores["control_zone"] = 1.0 - (distance_to_boundary / 20.0)

        overall_risk = np.mean(list(vulnerability_scores.values())) if vulnerability_scores else 0.0

        return {
            "vulnerabilities_detected": len(vulnerabilities) > 0,
            "vulnerabilities": vulnerabilities,
            "vulnerability_scores": vulnerability_scores,
            "overall_risk_score": overall_risk,
            "compliance_status": self._check_compliance(equipment_data),
        }

    def _check_compliance(self, equipment_data: dict[str, Any]) -> dict[str, bool]:
        """Check TEMPEST/EMSEC compliance"""
        compliance = {}

        em_shielding = equipment_data.get("em_shielding_db", 0.0)
        compliance["zone1_shielding"] = em_shielding >= 80.0
        compliance["zone2_shielding"] = em_shielding >= 60.0
        compliance["zone3_shielding"] = em_shielding >= 40.0

        compliance["power_line_filtering"] = equipment_data.get("power_line_filtering", False)
        compliance["cable_shielding"] = equipment_data.get("cable_shielding", False)

        distance = equipment_data.get("distance_to_boundary_m", 0.0)
        compliance["control_zone"] = distance >= 20.0

        return compliance


class EMSECCountermeasureGenerator:
    """
    Generate EMSEC countermeasures and mitigation strategies.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def generate_countermeasures(
        self, analysis_result: dict[str, Any], vulnerabilities: list[str]
    ) -> list[str]:
        """
        Generate TEMPEST countermeasures.

        Args:
            analysis_result: TEMPEST analysis results
            vulnerabilities: Identified vulnerabilities

        Returns:
            List of countermeasure recommendations
        """
        countermeasures = []

        if "video_display_emanation" in analysis_result.get("emanation_sources", []):
            countermeasures.append("Deploy TEMPEST-certified displays")
            countermeasures.append("Install video signal Faraday shielding")
            countermeasures.append("Use fiber optic video transmission")

        if "keyboard_emanation" in analysis_result.get("emanation_sources", []):
            countermeasures.append("Implement keyboard emanation shielding")
            countermeasures.append("Use encrypted keyboard connections")

        if "processor_emanation" in analysis_result.get("emanation_sources", []):
            countermeasures.append("Install RF-shielded equipment enclosures")
            countermeasures.append("Apply conductive coatings to chassis")

        if "insufficient_em_shielding" in vulnerabilities:
            countermeasures.append("Upgrade to 80dB minimum EM shielding")
            countermeasures.append("Install copper mesh Faraday cage")
            countermeasures.append("Seal all apertures with conductive gaskets")

        if "unfiltered_power_lines" in vulnerabilities:
            countermeasures.append("Install transient voltage suppressors")
            countermeasures.append("Deploy power line filters on all circuits")
            countermeasures.append("Use isolated power distribution")

        if "unshielded_cables" in vulnerabilities:
            countermeasures.append("Replace with double-shielded cables")
            countermeasures.append("Use fiber optic cables where possible")
            countermeasures.append("Install cable conduit shielding")

        if "insufficient_control_zone" in vulnerabilities:
            countermeasures.append("Expand TEMPEST control zone to 20m minimum")
            countermeasures.append("Install RF attenuation barriers")
            countermeasures.append("Deploy active EM noise generators")

        countermeasures.append("Conduct regular TEMPEST emissions testing")
        countermeasures.append("Maintain EMSEC equipment inventory")

        return countermeasures


class TEMPESTDetector:
    """
    Comprehensive TEMPEST detection system integrating RF spectrum analysis,
    video emanation detection, vulnerability assessment, and countermeasure generation.
    """

    def __init__(
        self,
        enable_rf_analysis: bool = True,
        enable_video_detection: bool = True,
        enable_vulnerability_assessment: bool = True,
    ):
        self.enable_rf_analysis = enable_rf_analysis
        self.enable_video_detection = enable_video_detection
        self.enable_vulnerability_assessment = enable_vulnerability_assessment

        self.rf_analyzer = RFSpectrumAnalyzer() if enable_rf_analysis else None
        self.video_detector = VideoEmanationDetector() if enable_video_detection else None
        self.vulnerability_assessor = (
            SideChannelVulnerabilityAssessor() if enable_vulnerability_assessment else None
        )
        self.countermeasure_generator = EMSECCountermeasureGenerator()

        self.logger = logging.getLogger(__name__)

    def detect_tempest_threats(self, tempest_data: dict[str, Any]) -> TEMPESTAnalysisResult:
        """
        Comprehensive TEMPEST threat detection.

        Args:
            tempest_data: TEMPEST monitoring data including:
                - spectrum_data: RF spectrum measurements
                - video_emanation_features: Video emanation characteristics
                - equipment_data: Equipment configuration and shielding

        Returns:
            TEMPEST analysis result with countermeasures
        """
        result = TEMPESTAnalysisResult(
            emanation_detected=False,
            confidence=0.0,
            threat_level="no_threat",
            risk_score=0.0,
        )

        if self.enable_rf_analysis and "spectrum_data" in tempest_data:
            rf_result = self.rf_analyzer.analyze_spectrum(tempest_data["spectrum_data"])

            if rf_result["emanation_detected"]:
                result.emanation_detected = True
                result.emanation_types = rf_result["emanation_sources"]
                result.frequency_bands = rf_result["frequency_bands"]
                result.signal_strength_dbm = rf_result["max_signal_strength_dbm"]
                result.confidence = max(result.confidence, rf_result["spectrum_occupancy"])

                compromising_potentials = [
                    band["compromising_potential"] for band in rf_result["frequency_bands"]
                ]
                if compromising_potentials:
                    result.compromising_potential = max(compromising_potentials)

        if self.enable_video_detection and "video_emanation_features" in tempest_data:
            video_result = self._analyze_video_emanation(tempest_data["video_emanation_features"])
            result.reconstruction_feasibility = video_result["reconstruction_feasibility"]

            if video_result["reconstruction_feasibility"] > 0.5:
                result.emanation_detected = True
                if "video_display_emanation" not in result.emanation_types:
                    result.emanation_types.append("video_display_emanation")

        if self.enable_vulnerability_assessment and "equipment_data" in tempest_data:
            vuln_result = self.vulnerability_assessor.assess_vulnerabilities(
                tempest_data["equipment_data"]
            )
            result.vulnerable_equipment = vuln_result["vulnerabilities"]
            result.compliance_status = vuln_result["compliance_status"]

            shielding = tempest_data["equipment_data"].get("em_shielding_db")
            if shielding is not None:
                result.shielding_effectiveness = shielding

            if vuln_result["vulnerabilities_detected"]:
                result.risk_score = max(result.risk_score, vuln_result["overall_risk_score"])

        result.threat_level = self._assess_threat_level(result)
        result.risk_score = self._calculate_risk_score(result)
        result.countermeasures = self.countermeasure_generator.generate_countermeasures(
            {
                "emanation_sources": result.emanation_types,
                "frequency_bands": result.frequency_bands,
            },
            result.vulnerable_equipment,
        )

        self.logger.info(
            f"TEMPEST detection: {result.threat_level}, "
            f"emanations={len(result.emanation_types)}, "
            f"compromising_potential={result.compromising_potential:.2f}"
        )

        return result

    def _analyze_video_emanation(self, features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze video emanation for reconstruction feasibility"""
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.video_detector.eval()
        with torch.no_grad():
            reconstruction_score, resolution_probs = self.video_detector(features_tensor)

        reconstruction_feasibility = float(reconstruction_score[0].item())
        resolution_class = int(torch.argmax(resolution_probs[0]).item())

        resolution_categories = ["low", "medium", "high"]
        estimated_resolution = resolution_categories[resolution_class]

        return {
            "reconstruction_feasibility": reconstruction_feasibility,
            "estimated_resolution": estimated_resolution,
        }

    def _assess_threat_level(self, result: TEMPESTAnalysisResult) -> str:
        """Assess overall TEMPEST threat level"""
        if result.compromising_potential > 0.8 or result.reconstruction_feasibility > 0.8:
            return TEMPESTThreatLevel.CRITICAL.value
        elif result.compromising_potential > 0.6 or result.reconstruction_feasibility > 0.6:
            return TEMPESTThreatLevel.HIGH.value
        elif result.compromising_potential > 0.4 or result.reconstruction_feasibility > 0.4:
            return TEMPESTThreatLevel.MODERATE.value
        elif result.emanation_detected or result.vulnerable_equipment:
            return TEMPESTThreatLevel.LOW.value
        else:
            return TEMPESTThreatLevel.NO_THREAT.value

    def _calculate_risk_score(self, result: TEMPESTAnalysisResult) -> float:
        """Calculate overall TEMPEST risk score"""
        base_score = result.confidence

        if result.compromising_potential > 0:
            base_score = max(base_score, result.compromising_potential)

        if result.reconstruction_feasibility > 0:
            base_score = max(base_score, result.reconstruction_feasibility)

        shielding_factor = 1.0
        if result.shielding_effectiveness is not None:
            shielding_factor = 1.0 - min(result.shielding_effectiveness / 100.0, 0.8)

        risk_score = base_score * shielding_factor

        if result.threat_level == TEMPESTThreatLevel.CRITICAL.value:
            risk_score *= 1.5

        return min(risk_score, 1.0)

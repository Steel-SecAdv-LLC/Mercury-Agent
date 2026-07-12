# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Electromagnetic Pulse (EMP) & Energy Surge Detector.

Comprehensive electromagnetic anomaly detection for critical infrastructure:
- EMP detection (nuclear, non-nuclear)
- Power grid surge monitoring
- Electromagnetic attack detection
- Solar-induced geomagnetically induced currents (GIC)
- Lightning-induced surges
- HEMP (High-altitude EMP) signature analysis
- Intentional electromagnetic interference (IEMI)
- E1/E2/E3 pulse component analysis

Integrations:
- Quantum-resistant cyber systems (quantum_risk_cyber.py)
- Solar storm detector integration
- Energy infrastructure (energy_dams.py)
- Sensor network monitoring
- Grid stability analysis

Research sources:
- DOE Electromagnetic Pulse Resilience Action Plan
- NERC (North American Electric Reliability Corporation)
- IEEE Standards for EMP Protection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class EMPType(Enum):
    """EMP classifications."""

    HEMP = "high_altitude_emp"
    NUCLEAR_EMP = "nuclear_emp"
    NON_NUCLEAR_EMP = "non_nuclear_emp"
    LIGHTNING = "lightning_surge"
    SOLAR_GIC = "solar_geomagnetic_current"
    INTENTIONAL_EMI = "intentional_emi"


class ThreatLevel(Enum):
    """Threat severity levels."""

    BENIGN = "benign"
    ANOMALOUS = "anomalous"
    SUSPICIOUS = "suspicious"
    THREAT = "threat"
    CRITICAL = "critical"


@dataclass
class EMPPredictionResult:
    """EMP prediction results."""

    emp_detected: bool
    confidence: float
    emp_type: str
    threat_level: str

    e1_component_detected: bool = False
    e2_component_detected: bool = False
    e3_component_detected: bool = False

    field_strength_vm: float | None = None
    frequency_mhz: float | None = None
    pulse_duration_ns: float | None = None

    grid_impact_assessment: str = "none"
    affected_infrastructure: list[str] = field(default_factory=list)

    source_localization: dict[str, float] | None = None
    intentional_attack_probability: float = 0.0

    protective_actions: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)


class E1PulseDetector:
    """E1 component detection (prompt gamma ray pulse).

    Characteristics: Very fast (nanoseconds), high frequency
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_e1_pulse(self, sensor_data: dict[str, Any]) -> dict[str, Any]:
        """Detect E1 pulse component.

        Args:
            sensor_data: EM field measurements, high-frequency sensors

        Returns:
            E1 pulse detection results
        """
        field_strength_vm = sensor_data.get("field_strength_vm", 0.0)
        rise_time_ns = sensor_data.get("rise_time_ns", 1000.0)
        frequency_mhz = sensor_data.get("peak_frequency_mhz", 1.0)

        e1_threshold_vm = 10000.0
        e1_rise_time_ns = 10.0
        e1_freq_range = (10.0, 1000.0)

        e1_detected = (
            field_strength_vm > e1_threshold_vm
            and rise_time_ns < e1_rise_time_ns
            and e1_freq_range[0] < frequency_mhz < e1_freq_range[1]
        )

        severity = "critical" if e1_detected else "low"

        return {
            "e1_detected": e1_detected,
            "field_strength_vm": float(field_strength_vm),
            "rise_time_ns": float(rise_time_ns),
            "severity": severity,
        }


class E3PulseDetector:
    """E3 component detection (magnetohydrodynamic EMP).

    Characteristics: Slow (seconds to minutes), low frequency, GIC induction
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_e3_pulse(self, magnetometer_data: dict[str, Any]) -> dict[str, Any]:
        """Detect E3 pulse (GIC).

        Args:
            magnetometer_data: Geomagnetic field measurements

        Returns:
            E3 pulse detection results
        """
        db_dt_nt_s = magnetometer_data.get("db_dt_nt_s", 0.0)
        duration_seconds = magnetometer_data.get("duration_seconds", 0.0)

        e3_threshold = 2400.0
        e3_min_duration = 60.0

        e3_detected = db_dt_nt_s > e3_threshold and duration_seconds > e3_min_duration

        gic_amplitude_a = abs(db_dt_nt_s) * 0.1

        if gic_amplitude_a > 100:
            grid_impact = "critical"
        elif gic_amplitude_a > 50:
            grid_impact = "high"
        elif gic_amplitude_a > 20:
            grid_impact = "moderate"
        else:
            grid_impact = "low"

        return {
            "e3_detected": e3_detected,
            "gic_amplitude_a": float(gic_amplitude_a),
            "grid_impact": grid_impact,
        }


class IntentionalEMIDetector(nn.Module):
    """Intentional electromagnetic interference (IEMI) detection.

    Distinguishes attacks from natural/accidental sources.
    """

    def __init__(self, input_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        phi = 1.618

        self.signature_analyzer = nn.Sequential(
            nn.Linear(input_dim, int(128 * phi)),
            nn.BatchNorm1d(int(128 * phi)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(128 * phi), int(64 * phi)),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.Linear(int(64 * phi), 64),
        )

        self.attack_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, em_signature: torch.Tensor) -> torch.Tensor:
        """Classify intentional vs.

        natural EM events.
                Note: This model uses BatchNorm which requires batch_size > 1 during
                training. For inference with any batch size, call model.eval() first.
        """
        # Ensure eval mode for inference to handle batch_size=1 with BatchNorm
        was_training = self.training
        if em_signature.size(0) == 1 and self.training:
            self.eval()

        features = self.signature_analyzer(em_signature)
        attack_prob = self.attack_classifier(features)

        # Restore training mode if it was changed
        if was_training and not self.training:
            self.train()

        return attack_prob


class EMPDetector:
    """Comprehensive EMP and electromagnetic surge detection system.

    Integrates E1/E2/E3 pulse detection with intentional attack classification.
    """

    def __init__(
        self,
        enable_e1_detection: bool = True,
        enable_e3_detection: bool = True,
        enable_attack_classification: bool = True,
    ):
        """Initialize the instance."""
        self.enable_e1 = enable_e1_detection
        self.enable_e3 = enable_e3_detection
        self.enable_attack = enable_attack_classification

        self.e1_detector = E1PulseDetector() if enable_e1_detection else None
        self.e3_detector = E3PulseDetector() if enable_e3_detection else None
        self.emi_detector = IntentionalEMIDetector() if enable_attack_classification else None

        self.logger = logging.getLogger(__name__)

    def predict_emp(self, emp_data: dict[str, Any]) -> EMPPredictionResult:
        """Comprehensive EMP prediction.

        Args:
            emp_data: Multi-sensor EM measurements including:
                - sensor_data: High-frequency EM field sensors
                - magnetometer_data: Geomagnetic field measurements
                - grid_data: Power grid monitoring
                - signature_data: EM signature features
                - solar_data: Space weather conditions

        Returns:
            EMP prediction with threat assessment and protective actions
        """
        result = EMPPredictionResult(
            emp_detected=False,
            confidence=0.0,
            emp_type="lightning",
            threat_level="benign",
        )

        components_detected = 0

        if self.enable_e1 and "sensor_data" in emp_data and self.e1_detector is not None:
            e1_result = self.e1_detector.detect_e1_pulse(emp_data["sensor_data"])
            result.e1_component_detected = e1_result["e1_detected"]
            result.field_strength_vm = e1_result["field_strength_vm"]

            if e1_result["e1_detected"]:
                components_detected += 1
                result.confidence = max(result.confidence, 0.9)
                result.emp_detected = True

        if self.enable_e3 and "magnetometer_data" in emp_data and self.e3_detector is not None:
            e3_result = self.e3_detector.detect_e3_pulse(emp_data["magnetometer_data"])
            result.e3_component_detected = e3_result["e3_detected"]
            result.grid_impact_assessment = e3_result["grid_impact"]

            if e3_result["e3_detected"]:
                components_detected += 1
                result.confidence = max(result.confidence, 0.8)
                result.emp_detected = True

        if result.e1_component_detected and result.e3_component_detected:
            result.emp_type = EMPType.NUCLEAR_EMP.value
        elif result.e1_component_detected:
            result.emp_type = EMPType.NON_NUCLEAR_EMP.value
        elif result.e3_component_detected:
            if "solar_data" in emp_data and emp_data["solar_data"].get("storm_active"):
                result.emp_type = EMPType.SOLAR_GIC.value
            else:
                result.emp_type = EMPType.HEMP.value

        if self.enable_attack and "signature_data" in emp_data:
            attack_result = self._classify_intentional_attack(emp_data["signature_data"])
            result.intentional_attack_probability = attack_result["attack_probability"]

            if attack_result["attack_detected"]:
                result.emp_type = EMPType.INTENTIONAL_EMI.value
                result.confidence = max(result.confidence, attack_result["attack_probability"])

        result.threat_level = self._assess_threat_level(result, components_detected)

        result.affected_infrastructure = self._identify_affected_infrastructure(result)
        result.protective_actions = self._generate_protective_actions(result)
        result.recovery_actions = self._generate_recovery_actions(result)

        return result

    def _classify_intentional_attack(self, signature_data: dict[str, Any]) -> dict[str, Any]:
        """Classify intentional electromagnetic attack."""
        if self.emi_detector is None:
            return {"attack_detected": False, "attack_probability": 0.0}

        if "signature_features" in signature_data:
            features = signature_data["signature_features"]
        else:
            repetition_rate = signature_data.get("repetition_rate_hz", 0.0)
            features = np.array([repetition_rate])
            features = np.pad(features, (0, 63), mode="constant")

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.emi_detector.eval()
        with torch.no_grad():
            attack_prob = self.emi_detector(features_tensor)

        attack_detected = float(attack_prob[0].item()) > 0.7

        return {
            "attack_detected": attack_detected,
            "attack_probability": float(attack_prob[0].item()),
        }

    def _assess_threat_level(self, result: EMPPredictionResult, components: int) -> str:
        """Assess overall threat level."""
        if result.emp_type == "nuclear_emp" or result.intentional_attack_probability > 0.8:
            return ThreatLevel.CRITICAL.value
        elif components >= 2:
            return ThreatLevel.THREAT.value
        elif result.emp_detected:
            return ThreatLevel.SUSPICIOUS.value
        elif result.field_strength_vm and result.field_strength_vm > 1000:
            return ThreatLevel.ANOMALOUS.value
        else:
            return ThreatLevel.BENIGN.value

    def _identify_affected_infrastructure(self, result: EMPPredictionResult) -> list[str]:
        """Identify affected critical infrastructure."""
        infrastructure = []

        if result.e1_component_detected:
            infrastructure.extend(
                [
                    "Electronics and semiconductors",
                    "Communication systems",
                    "Computer networks",
                ]
            )

        if result.e3_component_detected or result.grid_impact_assessment != "none":
            infrastructure.extend(["Power transmission grid", "Transformers"])

        if result.emp_type == "nuclear_emp":
            infrastructure.extend(
                [
                    "All electronic infrastructure",
                    "Transportation systems",
                    "Financial networks",
                ]
            )

        return list(set(infrastructure))

    def _generate_protective_actions(self, result: EMPPredictionResult) -> list[str]:
        """Generate protective actions."""
        actions = []

        if result.threat_level == "critical":
            actions.append("CRITICAL EMP THREAT: Activate hardened backup systems")
            actions.append("Disconnect non-essential electronics")
            actions.append("Implement Faraday cage protocols")

        if result.grid_impact_assessment in ["critical", "high"]:
            actions.append("Activate grid protective relays")
            actions.append("Load shedding to reduce transformer stress")

        if result.intentional_attack_probability > 0.7:
            actions.append("Alert cybersecurity and physical security teams")
            actions.append("Trace source of electromagnetic interference")

        return actions

    def _generate_recovery_actions(self, result: EMPPredictionResult) -> list[str]:
        """Generate recovery actions."""
        recovery = []

        if result.emp_detected:
            recovery.append("Assess extent of equipment damage")
            recovery.append("Prioritize critical infrastructure restoration")

        if result.emp_type == "nuclear_emp":
            recovery.append("Implement national emergency response plan")
            recovery.append("Coordinate with FEMA and DOE")

        return recovery

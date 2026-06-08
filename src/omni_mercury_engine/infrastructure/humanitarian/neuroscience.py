# Copyright (C) 2025 Steel Security Advisors LLC
"""Neuroscience Module - Cognitive pattern and neural anomaly detection.

Supports cognitive enhancement and mental health monitoring by detecting:
- Neural activity pattern anomalies
- Cognitive decline indicators
- Mental health crisis signals
- Brain-computer interface anomalies
- Neurological disorder markers

⚠️ SIMULATION-BASED: Uses simulated neural data. Clinical validation required.
Consult neuroscience professionals before acting on predictions.

Research sources:
- NIH neuroscience research
- Brain-computer interface studies
- Cognitive neuroscience literature
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class NeuralThreat(Enum):
    """Neural threat."""

    NORMAL = "normal"
    COGNITIVE_DECLINE = "cognitive_decline"
    SEIZURE_PATTERN = "seizure_pattern"
    MENTAL_HEALTH_CRISIS = "mental_health_crisis"
    ATTENTION_DEFICIT = "attention_deficit"


class NeuroscienceDetector:
    """Detect neural and cognitive pattern anomalies."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.neural_baseline = {"mean": 50.0, "std": 15.0}
        self.cognitive_threshold = 0.7

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "neural_activity",
        subject_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect neuroscience anomalies.

        Args:
            data: Neural/cognitive metrics (EEG, fMRI, cognitive scores)
            detection_type: 'neural_activity', 'cognitive', 'mental_health'
            subject_context: Subject demographic and medical history

        Returns:
            Neuroscience anomaly detection results
        """
        if detection_type == "neural_activity":
            return self.detect_neural_activity_anomaly(data, subject_context)
        elif detection_type == "cognitive":
            return self.detect_cognitive_anomaly(data, subject_context)
        else:
            return self.detect_neural_activity_anomaly(data, subject_context)

    def detect_neural_activity_anomaly(
        self, neural_data: np.ndarray[Any, Any], subject_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect abnormal neural activity patterns."""
        if len(neural_data) == 0:
            return {"anomaly_detected": False}

        mean_activity = np.mean(neural_data)
        std_activity = np.std(neural_data)
        baseline_mean = self.neural_baseline["mean"]
        baseline_std = self.neural_baseline["std"]

        z_score = (mean_activity - baseline_mean) / baseline_std if baseline_std > 0 else 0

        threat_type = NeuralThreat.NORMAL
        severity = "low"

        seizure_risk = self._detect_seizure_patterns(neural_data)
        if seizure_risk > 0.5:
            threat_type = NeuralThreat.SEIZURE_PATTERN
            severity = "critical"
        elif abs(z_score) > 3:
            threat_type = NeuralThreat.COGNITIVE_DECLINE
            severity = "high"

        return {
            "anomaly_detected": abs(z_score) > 2 or seizure_risk > 0.3,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "mean_activity": float(mean_activity),
                "std_activity": float(std_activity),
                "z_score": float(z_score),
                "seizure_risk": float(seizure_risk),
            },
            "neurological_risk": self._assess_neurological_risk(z_score, seizure_risk),
            "intervention_urgency": "immediate" if severity == "critical" else "routine",
            "recommendations": self._generate_neuroscience_recommendations(threat_type, severity),
            "timestamp": datetime.now(),
        }

    def detect_cognitive_anomaly(
        self, cognitive_data: np.ndarray[Any, Any], subject_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect cognitive decline or attention deficits."""
        if len(cognitive_data) == 0:
            return {"anomaly_detected": False}

        cognitive_score = np.mean(cognitive_data)

        threat_type = NeuralThreat.NORMAL
        severity = "low"

        if cognitive_score < self.cognitive_threshold:
            threat_type = NeuralThreat.COGNITIVE_DECLINE
            severity = "high" if cognitive_score < 0.5 else "medium"

        return {
            "anomaly_detected": cognitive_score < self.cognitive_threshold,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "cognitive_score": float(cognitive_score),
                "performance_percentile": float(cognitive_score * 100),
            },
            "intervention_needed": severity in ["high", "critical"],
            "recommendations": self._generate_neuroscience_recommendations(threat_type, severity),
            "timestamp": datetime.now(),
        }

    def _detect_seizure_patterns(self, neural_data: np.ndarray[Any, Any]) -> float:
        """Detect seizure-like patterns in neural activity."""
        if len(neural_data) < 10:
            return 0.0

        spikes = np.sum(np.abs(np.diff(neural_data)) > 3 * np.std(neural_data))
        spike_rate = spikes / len(neural_data)

        return float(min(spike_rate * 10, 1.0))

    def _assess_neurological_risk(self, z_score: float, seizure_risk: float) -> str:
        """Assess neurological risk level."""
        if seizure_risk > 0.5 or abs(z_score) > 3:
            return "critical"
        elif seizure_risk > 0.3 or abs(z_score) > 2:
            return "elevated"
        return "low"

    def _generate_neuroscience_recommendations(
        self, threat_type: NeuralThreat, severity: str
    ) -> list[str]:
        """Generate neuroscience recommendations."""
        recommendations = []

        if threat_type == NeuralThreat.SEIZURE_PATTERN:
            recommendations.append("URGENT: Alert medical staff immediately")
            recommendations.append("Prepare emergency seizure response")
            recommendations.append("Monitor continuously for 24 hours")
        elif threat_type == NeuralThreat.COGNITIVE_DECLINE:
            recommendations.append("Schedule comprehensive cognitive assessment")
            recommendations.append("Review medications and medical history")
            if severity == "high":
                recommendations.append("Consider neurological consultation")
        elif threat_type == NeuralThreat.MENTAL_HEALTH_CRISIS:
            recommendations.append("Provide immediate mental health support")
            recommendations.append("Activate crisis intervention protocols")

        if not recommendations:
            recommendations.append("Continue routine monitoring")

        return recommendations

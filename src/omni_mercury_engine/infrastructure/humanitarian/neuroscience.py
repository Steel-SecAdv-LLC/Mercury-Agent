# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neuroscience Module - Cognitive pattern and neural anomaly detection.

Supports cognitive enhancement and mental health monitoring by detecting:
- Neural activity pattern anomalies
- Cognitive decline indicators
- Mental health crisis signals
- Brain-computer interface anomalies
- Neurological disorder markers

⚠️ The neural-activity and cognitive paths are SIMULATION-BASED (simulated
neural data; clinical validation required). The mental-health path scores
published, validated screening instruments — PHQ-9 (Kroenke et al. 2001),
GAD-7 (Spitzer et al. 2006) and ASRS v1.1 Part A (Kessler et al. 2005) —
by their published rules, cutpoints and operating characteristics. Every
output is a screen, never a diagnosis, and defers to evaluation by a
licensed clinician.

Research sources:
- Kroenke, Spitzer & Williams (2001), J Gen Intern Med 16:606-613 (PHQ-9)
- Spitzer, Kroenke, Williams & Löwe (2006), Arch Intern Med 166:1092 (GAD-7)
- Kessler et al. (2005), Psychol Med 35:245-256 (ASRS v1.1)
- Simon et al. (2013), Psychiatr Serv 64:1195-1202 (PHQ-9 item 9 and risk)
"""

from __future__ import annotations

from datetime import UTC, datetime
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
            data: Neural/cognitive metrics (EEG, fMRI, cognitive scores), or
                for ``mental_health`` the item responses of a validated
                screening instrument (see
                :meth:`detect_mental_health_anomaly`).
            detection_type: 'neural_activity', 'cognitive', 'mental_health'
            subject_context: Subject demographic and medical history; for
                ``mental_health``, ``{"instrument": "phq9"|"gad7"|"asrs6"}``

        Returns:
            Neuroscience anomaly detection results
        """
        if detection_type == "neural_activity":
            return self.detect_neural_activity_anomaly(data, subject_context)
        elif detection_type == "cognitive":
            return self.detect_cognitive_anomaly(data, subject_context)
        elif detection_type == "mental_health":
            return self.detect_mental_health_anomaly(data, subject_context)
        else:
            return self.detect_neural_activity_anomaly(data, subject_context)

    def detect_neural_activity_anomaly(
        self, neural_data: np.ndarray[Any, Any], subject_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect abnormal neural activity patterns.

        Raises:
            ValueError: Non-finite samples — a NaN window must never read
                as normal neural activity.
        """
        if len(neural_data) == 0:
            return {"anomaly_detected": False}
        if not np.all(np.isfinite(np.asarray(neural_data, dtype=float))):
            raise ValueError("neural_data contains non-finite values")

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
            "anomaly_detected": bool(abs(z_score) > 2 or seizure_risk > 0.3),
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
            "timestamp": datetime.now(UTC),
        }

    def detect_cognitive_anomaly(
        self, cognitive_data: np.ndarray[Any, Any], subject_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect cognitive decline or attention deficits.

        Raises:
            ValueError: Non-finite scores — a NaN window must never read as
                normal cognition.
        """
        if len(cognitive_data) == 0:
            return {"anomaly_detected": False}
        if not np.all(np.isfinite(np.asarray(cognitive_data, dtype=float))):
            raise ValueError("cognitive_data contains non-finite values")

        cognitive_score = np.mean(cognitive_data)

        threat_type = NeuralThreat.NORMAL
        severity = "low"

        if cognitive_score < self.cognitive_threshold:
            threat_type = NeuralThreat.COGNITIVE_DECLINE
            severity = "high" if cognitive_score < 0.5 else "medium"

        return {
            "anomaly_detected": bool(cognitive_score < self.cognitive_threshold),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "cognitive_score": float(cognitive_score),
                "performance_percentile": float(cognitive_score * 100),
            },
            "intervention_needed": severity in ["high", "critical"],
            "recommendations": self._generate_neuroscience_recommendations(threat_type, severity),
            "timestamp": datetime.now(UTC),
        }

    #: Validated screening instruments, keyed by ``subject_context["instrument"]``.
    #: Primary sources:
    #: * PHQ-9: Kroenke, Spitzer & Williams (2001), J Gen Intern Med 16:606-613.
    #:   9 items scored 0-3; severity bands 5/10/15/20; the >= 10 cutpoint has
    #:   88% sensitivity and 88% specificity for major depression. Any
    #:   endorsement of item 9 (self-harm ideation) warrants an immediate
    #:   suicide-risk assessment (PHQ-9 instruction manual; Simon et al. 2013,
    #:   Psychiatr Serv 64:1195-1202).
    #: * GAD-7: Spitzer, Kroenke, Williams & Löwe (2006), Arch Intern Med
    #:   166:1092-1097. 7 items scored 0-3; bands 5/10/15; the >= 10 cutpoint
    #:   has 89% sensitivity and 82% specificity for generalized anxiety.
    #: * ASRS v1.1 Part A: Kessler et al. (2005), Psychol Med 35:245-256.
    #:   6 items scored 0-4; a positive screen is >= 4 items at or above the
    #:   shaded thresholds (items 1-3: >= 2 "sometimes"; items 4-6: >= 3
    #:   "often"); 68.7% sensitivity, 99.5% specificity for adult ADHD.
    SCREENING_INSTRUMENTS: dict[str, dict[str, Any]] = {
        "phq9": {
            "name": "PHQ-9",
            "items": 9,
            "item_max": 3,
            "bands": ((20, "severe"), (15, "moderately_severe"), (10, "moderate"), (5, "mild")),
            "minimal_band": "minimal",
            "cutpoint": 10,
            "sensitivity": 0.88,
            "specificity": 0.88,
            "crisis_item_index": 8,  # item 9: self-harm/suicidal ideation
            "citation": "Kroenke, Spitzer & Williams 2001, J Gen Intern Med 16:606-613",
        },
        "gad7": {
            "name": "GAD-7",
            "items": 7,
            "item_max": 3,
            "bands": ((15, "severe"), (10, "moderate"), (5, "mild")),
            "minimal_band": "minimal",
            "cutpoint": 10,
            "sensitivity": 0.89,
            "specificity": 0.82,
            "crisis_item_index": None,
            "citation": "Spitzer, Kroenke, Williams & Löwe 2006, Arch Intern Med 166:1092-1097",
        },
        "asrs6": {
            "name": "ASRS v1.1 Part A",
            "items": 6,
            "item_max": 4,
            # Per-item positive-screen thresholds (the shaded boxes).
            "item_thresholds": (2, 2, 2, 3, 3, 3),
            "positive_items_required": 4,
            "sensitivity": 0.687,
            "specificity": 0.995,
            "citation": "Kessler et al. 2005, Psychol Med 35:245-256",
        },
    }

    #: Every mental-health screening result carries this verbatim: the output
    #: is a screen, never a diagnosis, and always defers to clinical review.
    CLINICAL_DISCLAIMER = (
        "Screening output only, not a diagnosis; defer to evaluation by a "
        "licensed mental health clinician."
    )

    def detect_mental_health_anomaly(
        self, responses: np.ndarray[Any, Any], subject_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Score a validated mental-health screening instrument.

        Implements the published scoring rules and cutpoints of three
        validated instruments (see :data:`SCREENING_INSTRUMENTS` for the
        primary sources and measured sensitivity/specificity): PHQ-9
        (depression), GAD-7 (anxiety) and ASRS v1.1 Part A (adult ADHD).
        PHQ-9 item 9 endorsement escalates to
        :attr:`NeuralThreat.MENTAL_HEALTH_CRISIS` with immediate urgency, per
        the instrument's published guidance; a score in the instrument's
        severe band escalates likewise (urgent clinical attention). A
        positive ASRS screen reports :attr:`NeuralThreat.ATTENTION_DEFICIT`.

        Limits (stated, not implied away): these are self-report screens with
        the published operating characteristics above, not diagnoses; every
        result carries :data:`CLINICAL_DISCLAIMER` and defers to trained
        clinical review.

        Args:
            responses: The instrument's item scores, in item order.
            subject_context: ``{"instrument": "phq9" | "gad7" | "asrs6"}``;
                defaults to ``phq9``.

        Returns:
            Screening result with instrument metadata, score, severity band,
            threat type, recommendations and the clinical disclaimer.

        Raises:
            ValueError: Unknown instrument, wrong item count, or an item
                outside the instrument's published response range — a
                mis-scored instrument must refuse, not fabricate a screen.
        """
        if len(responses) == 0:
            return {"anomaly_detected": False}

        context = subject_context or {}
        key = str(context.get("instrument", "phq9")).strip().lower()
        spec = self.SCREENING_INSTRUMENTS.get(key)
        if spec is None:
            known = ", ".join(sorted(self.SCREENING_INSTRUMENTS))
            raise ValueError(f"unknown screening instrument {key!r}; supported: {known}")

        items = np.asarray(responses, dtype=float).ravel()
        if items.size != spec["items"]:
            raise ValueError(
                f"{spec['name']} takes exactly {spec['items']} item scores, got {items.size}"
            )
        if (
            not np.all(np.isfinite(items))
            or np.any(items < 0)
            or np.any(items > spec["item_max"])
            or not np.all(items == np.round(items))
        ):
            raise ValueError(f"{spec['name']} item scores are integers in 0..{spec['item_max']}")

        if key == "asrs6":
            return self._score_asrs(items, spec)
        return self._score_severity_banded(items, spec)

    def _score_severity_banded(
        self, items: np.ndarray[Any, Any], spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Score a total-score instrument (PHQ-9 / GAD-7) on its published bands."""
        total = int(np.sum(items))
        band = spec["minimal_band"]
        for threshold, name in spec["bands"]:
            if total >= threshold:
                band = name
                break
        severe_band = spec["bands"][0][1]

        crisis_index = spec["crisis_item_index"]
        crisis_item_positive = crisis_index is not None and items[crisis_index] > 0

        threat_type = NeuralThreat.NORMAL
        severity = "low"
        if crisis_item_positive:
            # Any item-9 endorsement warrants immediate suicide-risk
            # assessment (PHQ-9 manual; Simon et al. 2013).
            threat_type = NeuralThreat.MENTAL_HEALTH_CRISIS
            severity = "critical"
        elif band == severe_band:
            threat_type = NeuralThreat.MENTAL_HEALTH_CRISIS
            severity = "high"
        elif total >= spec["cutpoint"]:
            severity = "medium"

        recommendations = self._generate_neuroscience_recommendations(threat_type, severity)
        if threat_type is not NeuralThreat.MENTAL_HEALTH_CRISIS and total >= spec["cutpoint"]:
            recommendations.insert(
                0, "Refer for clinical assessment by a licensed mental health professional"
            )

        return {
            "anomaly_detected": bool(total >= spec["cutpoint"] or crisis_item_positive),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "instrument": spec["name"],
                "total_score": total,
                "severity_band": band,
                "cutpoint": spec["cutpoint"],
                "crisis_item_positive": bool(crisis_item_positive),
                "published_sensitivity": spec["sensitivity"],
                "published_specificity": spec["specificity"],
            },
            "intervention_urgency": "immediate" if severity == "critical" else "routine",
            "recommendations": recommendations,
            "clinical_disclaimer": self.CLINICAL_DISCLAIMER,
            "method_citation": spec["citation"],
            "timestamp": datetime.now(UTC),
        }

    def _score_asrs(self, items: np.ndarray[Any, Any], spec: dict[str, Any]) -> dict[str, Any]:
        """Score ASRS v1.1 Part A by its published per-item shaded thresholds."""
        thresholds = spec["item_thresholds"]
        positive_items = int(
            sum(1 for score, cut in zip(items, thresholds, strict=True) if score >= cut)
        )
        screen_positive = positive_items >= spec["positive_items_required"]

        threat_type = NeuralThreat.ATTENTION_DEFICIT if screen_positive else NeuralThreat.NORMAL
        severity = "medium" if screen_positive else "low"

        return {
            "anomaly_detected": bool(screen_positive),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "instrument": spec["name"],
                "positive_items": positive_items,
                "positive_items_required": spec["positive_items_required"],
                "published_sensitivity": spec["sensitivity"],
                "published_specificity": spec["specificity"],
            },
            "intervention_urgency": "routine",
            "recommendations": self._generate_neuroscience_recommendations(threat_type, severity),
            "clinical_disclaimer": self.CLINICAL_DISCLAIMER,
            "method_citation": spec["citation"],
            "timestamp": datetime.now(UTC),
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
            if severity == "critical":
                recommendations.append(
                    "URGENT: Conduct a structured suicide-risk assessment now "
                    "(e.g., Columbia Suicide Severity Rating Scale)"
                )
        elif threat_type == NeuralThreat.ATTENTION_DEFICIT:
            recommendations.append(
                "Refer for a comprehensive adult ADHD evaluation by a qualified clinician"
            )
            recommendations.append("Corroborate the screen with informant and clinical history")

        if not recommendations:
            recommendations.append("Continue routine monitoring")

        return recommendations

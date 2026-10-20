# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Education Equity Module - Learning bias and access anomaly detection.

Supports SDG 4 (Quality Education) by monitoring:
- Learning outcome disparities
- Resource allocation inequities
- Access barriers for marginalized groups
- Dropout risk prediction
- Curriculum bias detection

⚠️ SIMULATION-BASED: Uses simulated educational data. Real-world validation required.

Research sources:
- UNESCO Education data
- OECD education statistics
- EdTech research on learning analytics
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np


class EducationThreat(Enum):
    """Education threat."""

    NORMAL = "normal"
    ACHIEVEMENT_GAP = "achievement_gap"
    DROPOUT_RISK = "dropout_risk"
    ACCESS_BARRIER = "access_barrier"
    RESOURCE_INEQUITY = "resource_inequity"
    BIAS_DETECTED = "bias_detected"


class EducationEquityDetector:
    """Detect educational equity anomalies and learning barriers."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.achievement_baseline = {"mean": 75.0, "std": 10.0}
        self.dropout_risk_threshold = 0.3

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "achievement",
        demographic_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect educational equity anomalies.

        Args:
            data: Educational metrics (test scores, attendance, etc.)
            detection_type: 'achievement', 'dropout_risk', 'access',
                'resources', 'parity'
            demographic_data: Student demographic information

        Returns:
            Education equity anomaly detection results
        """
        if detection_type == "achievement":
            return self.detect_achievement_gap(data, demographic_data)
        elif detection_type == "dropout_risk":
            return self.detect_dropout_risk(data, demographic_data)
        elif detection_type == "access":
            return self.detect_access_barrier(data, demographic_data)
        elif detection_type == "resources":
            return self.detect_resource_inequity(data, demographic_data)
        elif detection_type == "parity":
            return self.detect_gender_parity_bias(data, demographic_data)
        else:
            return self.detect_achievement_gap(data, demographic_data)

    def detect_achievement_gap(
        self, achievement_data: np.ndarray[Any, Any], demographic_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect achievement gaps across student populations."""
        if len(achievement_data) == 0:
            return {"anomaly_detected": False}

        mean_achievement = np.mean(achievement_data)
        std_achievement = np.std(achievement_data)
        baseline_mean = self.achievement_baseline["mean"]
        baseline_std = self.achievement_baseline["std"]

        z_score = (mean_achievement - baseline_mean) / baseline_std if baseline_std > 0 else 0

        threat_type = EducationThreat.NORMAL
        severity = "low"

        if mean_achievement < baseline_mean - 2 * baseline_std:
            threat_type = EducationThreat.ACHIEVEMENT_GAP
            severity = "critical" if z_score < -3 else "high"

        disparity_score = std_achievement / mean_achievement if mean_achievement > 0 else 0

        return {
            "anomaly_detected": bool(z_score < -2 or disparity_score > 0.3),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "mean_achievement": float(mean_achievement),
                "std_achievement": float(std_achievement),
                "disparity_score": float(disparity_score),
                "z_score": float(z_score),
            },
            "equity_risk": self._assess_equity_risk(disparity_score, severity),
            "affected_students_estimate": self._estimate_affected_students(z_score),
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, disparity_score
            ),
            "timestamp": datetime.now(UTC),
        }

    def detect_dropout_risk(
        self, student_data: np.ndarray[Any, Any], demographic_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect students at risk of dropping out."""
        if len(student_data) == 0:
            return {"anomaly_detected": False}

        risk_indicators = np.mean(student_data, axis=0) if student_data.ndim > 1 else student_data
        dropout_risk = np.mean(risk_indicators < self.dropout_risk_threshold)

        threat_type = EducationThreat.NORMAL
        severity = "low"

        if dropout_risk > 0.2:
            threat_type = EducationThreat.DROPOUT_RISK
            severity = "critical" if dropout_risk > 0.4 else "high"

        return {
            "anomaly_detected": bool(dropout_risk > 0.15),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "dropout_risk_rate": float(dropout_risk),
                "at_risk_percentage": float(dropout_risk * 100),
            },
            "intervention_urgency": "immediate" if severity == "critical" else "standard",
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, dropout_risk  # type: ignore[arg-type, unused-ignore]
            ),
            "timestamp": datetime.now(UTC),
        }

    #: Measured world primary out-of-school rate, 10.3% (UNESCO UIS SDG4
    #: indicator ROFST.1.CP, geoUnit "SDG: World", 2023; recorded at
    #: tests/fixtures/humanitarian/uis_rofst1cp_sdg_world.json, sha256
    #: 49b32bef43425aaab82e65accdcb03c097654f07c917801e39acdc4996b6a3cc).
    WORLD_OUT_OF_SCHOOL_PCT = 10.3

    #: UNESCO EFA / Global Partnership for Education benchmark: a primary
    #: pupil-teacher ratio above 40:1 marks under-resourced provision
    #: (EFA Global Monitoring Report pupil/teacher benchmark).
    PUPIL_TEACHER_RATIO_BENCHMARK = 40.0

    #: UNESCO UIS adjusted gender parity index: values within
    #: [0.97, 1.03] count as parity (UIS GPI methodology; SDG 4.5.1).
    GPI_PARITY_LOW = 0.97
    GPI_PARITY_HIGH = 1.03

    def detect_access_barrier(
        self,
        out_of_school_pct: np.ndarray[Any, Any],
        demographic_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Access-barrier detection against the measured world benchmark.

        The latest out-of-school rate is compared with the UNESCO UIS world
        primary rate (:data:`WORLD_OUT_OF_SCHOOL_PCT`): above it marks an
        access barrier (medium); at twice the world rate the barrier grades
        high; at four times, critical.

        Args:
            out_of_school_pct: Out-of-school rates in percent of primary
                school age (a series grades its latest value).
            demographic_data: Unused; kept for signature parity.

        Returns:
            Detection result; ``threat_type`` is ``access_barrier`` above
            the world rate.
        """
        if len(out_of_school_pct) == 0:
            return {"anomaly_detected": False}
        rates = np.asarray(out_of_school_pct, dtype=float).ravel()
        if not np.all(np.isfinite(rates)) or np.any(rates < 0) or np.any(rates > 100):
            raise ValueError("out_of_school_pct values must be finite percentages in [0, 100]")

        latest = float(rates[-1])
        world = self.WORLD_OUT_OF_SCHOOL_PCT

        threat_type = EducationThreat.NORMAL
        severity = "low"
        if latest > 4 * world:
            threat_type = EducationThreat.ACCESS_BARRIER
            severity = "critical"
        elif latest > 2 * world:
            threat_type = EducationThreat.ACCESS_BARRIER
            severity = "high"
        elif latest > world:
            threat_type = EducationThreat.ACCESS_BARRIER
            severity = "medium"

        return {
            "anomaly_detected": bool(threat_type is EducationThreat.ACCESS_BARRIER),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "out_of_school_pct": latest,
                "world_out_of_school_pct": world,
                "ratio_to_world": latest / world,
            },
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, latest
            ),
            "timestamp": datetime.now(UTC),
        }

    def detect_resource_inequity(
        self,
        pupil_teacher_ratio: np.ndarray[Any, Any],
        demographic_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resource-inequity detection on the pupil-teacher ratio benchmark.

        A primary pupil-teacher ratio above 40:1
        (:data:`PUPIL_TEACHER_RATIO_BENCHMARK`, the UNESCO EFA benchmark)
        marks under-resourced provision; above 60:1 it grades critical.

        Args:
            pupil_teacher_ratio: Pupil-teacher ratios (a series grades its
                latest value).
            demographic_data: Unused; kept for signature parity.

        Returns:
            Detection result; ``threat_type`` is ``resource_inequity`` above
            the benchmark.
        """
        if len(pupil_teacher_ratio) == 0:
            return {"anomaly_detected": False}
        ratios = np.asarray(pupil_teacher_ratio, dtype=float).ravel()
        if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
            raise ValueError("pupil_teacher_ratio values must be finite and positive")

        latest = float(ratios[-1])
        over = latest > self.PUPIL_TEACHER_RATIO_BENCHMARK
        threat_type = EducationThreat.RESOURCE_INEQUITY if over else EducationThreat.NORMAL
        if latest > 1.5 * self.PUPIL_TEACHER_RATIO_BENCHMARK:
            severity = "critical"
        elif over:
            severity = "high"
        else:
            severity = "low"

        return {
            "anomaly_detected": bool(over),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "pupil_teacher_ratio": latest,
                "benchmark_ratio": self.PUPIL_TEACHER_RATIO_BENCHMARK,
            },
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, latest
            ),
            "timestamp": datetime.now(UTC),
        }

    def detect_gender_parity_bias(
        self,
        gender_parity_index: np.ndarray[Any, Any],
        demographic_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Gender-bias detection on the UIS adjusted parity band.

        A gender parity index outside [:data:`GPI_PARITY_LOW`,
        :data:`GPI_PARITY_HIGH`] (the UNESCO UIS parity band behind SDG
        4.5.1) marks disparity: below 0.97 at the expense of girls, above
        1.03 at the expense of boys.

        Args:
            gender_parity_index: GPI values (female/male ratio of the
                indicator; a series grades its latest value).
            demographic_data: Unused; kept for signature parity.

        Returns:
            Detection result; ``threat_type`` is ``bias_detected`` outside
            the parity band.
        """
        if len(gender_parity_index) == 0:
            return {"anomaly_detected": False}
        gpi = np.asarray(gender_parity_index, dtype=float).ravel()
        if not np.all(np.isfinite(gpi)) or np.any(gpi <= 0):
            raise ValueError("gender_parity_index values must be finite and positive")

        latest = float(gpi[-1])
        outside = latest < self.GPI_PARITY_LOW or latest > self.GPI_PARITY_HIGH
        threat_type = EducationThreat.BIAS_DETECTED if outside else EducationThreat.NORMAL
        if latest < 0.9 or latest > 1.1:
            severity = "high"
        elif outside:
            severity = "medium"
        else:
            severity = "low"

        disadvantaged = "none"
        if latest < self.GPI_PARITY_LOW:
            disadvantaged = "female"
        elif latest > self.GPI_PARITY_HIGH:
            disadvantaged = "male"

        return {
            "anomaly_detected": bool(outside),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "gender_parity_index": latest,
                "parity_band": [self.GPI_PARITY_LOW, self.GPI_PARITY_HIGH],
                "disadvantaged_group": disadvantaged,
            },
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, latest
            ),
            "timestamp": datetime.now(UTC),
        }

    def _assess_equity_risk(self, disparity_score: float, severity: str) -> str:
        """Assess educational equity risk."""
        if severity == "critical" or disparity_score > 0.4:
            return "severe_inequity"
        elif severity == "high" or disparity_score > 0.3:
            return "moderate_inequity"
        return "acceptable"

    def _estimate_affected_students(self, z_score: float) -> int:
        """Estimate number of students affected."""
        if z_score < -3:
            return 500
        elif z_score < -2:
            return 200
        return 0

    def _generate_education_recommendations(
        self, threat_type: EducationThreat, severity: str, metric: float
    ) -> list[str]:
        """Generate educational equity recommendations."""
        recommendations = []

        if threat_type == EducationThreat.ACHIEVEMENT_GAP:
            recommendations.append("Implement targeted interventions for underperforming groups")
            recommendations.append("Review curriculum for cultural bias")
            if severity == "critical":
                recommendations.append("URGENT: Deploy emergency tutoring programs")
        elif threat_type == EducationThreat.DROPOUT_RISK:
            recommendations.append("Activate student retention programs")
            recommendations.append("Provide counseling and mentorship")
            if severity == "critical":
                recommendations.append("CRITICAL: Immediate one-on-one interventions")
        elif threat_type == EducationThreat.ACCESS_BARRIER:
            recommendations.append("Remove enrollment barriers (fees, distance, documentation)")
            recommendations.append("Target out-of-school children with re-enrollment campaigns")
            if severity == "critical":
                recommendations.append("CRITICAL: Deploy emergency education access programs")
        elif threat_type == EducationThreat.RESOURCE_INEQUITY:
            recommendations.append("Recruit and deploy additional qualified teachers")
            recommendations.append("Rebalance resource allocation toward under-served schools")
        elif threat_type == EducationThreat.BIAS_DETECTED:
            recommendations.append("Investigate and remove gender-specific enrollment barriers")
            recommendations.append("Monitor parity progression against SDG 4.5.1")

        if not recommendations:
            recommendations.append("Continue routine educational monitoring")

        return recommendations

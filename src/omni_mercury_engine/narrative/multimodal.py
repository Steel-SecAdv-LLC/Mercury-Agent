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

Multi-Modal Support - Image and Audio Analysis Narration

Extends Mercury's narrative capabilities to describe and explain:
- Visual anomaly detection (images, video frames)
- Audio anomaly detection (spectrograms, waveforms)
- Biometric analysis results
- Industrial visual inspection

Architecture:
    Detection Result → Multi-Modal Analyzer → Narrative Engine → Human Description

This enables Mercury to explain what it "sees" and "hears" in human terms.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of data modalities."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SPECTROGRAM = "spectrogram"
    WAVEFORM = "waveform"
    POINT_CLOUD = "point_cloud"  # 3D data
    THERMAL = "thermal"  # Thermal imaging
    MIXED = "mixed"


class AnomalyVisualType(Enum):
    """Types of visual anomalies."""

    DEFECT = "defect"  # Manufacturing defect
    INTRUSION = "intrusion"  # Security intrusion
    MEDICAL = "medical"  # Medical imaging anomaly
    STRUCTURAL = "structural"  # Structural damage
    ENVIRONMENTAL = "environmental"  # Environmental hazard
    BIOMETRIC = "biometric"  # Biometric mismatch
    TEXTURE = "texture"  # Texture anomaly
    COLOR = "color"  # Color anomaly
    SHAPE = "shape"  # Shape anomaly
    UNKNOWN = "unknown"


class AudioAnomalyType(Enum):
    """Types of audio anomalies."""

    MACHINERY = "machinery"  # Machine fault sound
    VOICE = "voice"  # Voice anomaly
    ENVIRONMENTAL = "environmental"  # Environmental sound
    PATTERN = "pattern"  # Sound pattern anomaly
    FREQUENCY = "frequency"  # Frequency anomaly
    AMPLITUDE = "amplitude"  # Amplitude anomaly
    UNKNOWN = "unknown"


@dataclass
class RegionOfInterest:
    """Region of interest in visual data."""

    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    width: float  # Normalized 0-1
    height: float  # Normalized 0-1
    confidence: float
    label: str
    anomaly_score: float = 0.0


@dataclass
class AudioSegment:
    """Segment of interest in audio data."""

    start_time: float  # Seconds
    end_time: float  # Seconds
    confidence: float
    label: str
    anomaly_score: float = 0.0
    frequency_range: tuple[float, float] | None = None


@dataclass
class MultiModalDetection:
    """Detection result from multi-modal analysis."""

    modality: ModalityType
    anomaly_type: AnomalyVisualType | AudioAnomalyType
    anomaly_detected: bool
    anomaly_score: float
    confidence: float
    regions: list[RegionOfInterest] = field(default_factory=list)
    segments: list[AudioSegment] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality": self.modality.value,
            "anomaly_type": self.anomaly_type.value,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "regions": [
                {
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                    "label": r.label,
                    "score": r.anomaly_score,
                }
                for r in self.regions
            ],
            "segments": [
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "label": s.label,
                    "score": s.anomaly_score,
                }
                for s in self.segments
            ],
            "features": self.features,
            "timestamp": self.timestamp,
        }


@dataclass
class MultiModalNarration:
    """Narration of multi-modal detection."""

    summary: str
    visual_description: str | None = None
    audio_description: str | None = None
    location_description: str | None = None
    temporal_description: str | None = None
    severity_assessment: str = ""
    recommendations: list[str] = field(default_factory=list)
    confidence_statement: str = ""


class MultiModalNarrator:
    """
    Narrator for multi-modal detection results.

    Translates visual and audio anomaly detections into human-readable
    descriptions, enabling Mercury to explain what it "sees" and "hears".

    Capabilities:
        - Image anomaly description (defects, intrusions, medical)
        - Audio anomaly description (machinery faults, environmental)
        - Spatial localization narration (where in the image)
        - Temporal localization narration (when in the audio)
        - Domain-specific vocabulary adaptation

    Usage:
        narrator = MultiModalNarrator()

        detection = MultiModalDetection(
            modality=ModalityType.IMAGE,
            anomaly_type=AnomalyVisualType.DEFECT,
            anomaly_detected=True,
            anomaly_score=0.87,
            confidence=0.92,
            regions=[RegionOfInterest(x=0.3, y=0.4, width=0.1, height=0.1, ...)]
        )

        narration = narrator.narrate(detection, domain="manufacturing")
        print(narration.summary)
        # "Detected manufacturing defect in upper-left quadrant with high confidence..."
    """

    # Location descriptions for image regions
    POSITION_DESCRIPTIONS = {
        (0.0, 0.33, 0.0, 0.33): "upper-left",
        (0.33, 0.67, 0.0, 0.33): "upper-center",
        (0.67, 1.0, 0.0, 0.33): "upper-right",
        (0.0, 0.33, 0.33, 0.67): "center-left",
        (0.33, 0.67, 0.33, 0.67): "center",
        (0.67, 1.0, 0.33, 0.67): "center-right",
        (0.0, 0.33, 0.67, 1.0): "lower-left",
        (0.33, 0.67, 0.67, 1.0): "lower-center",
        (0.67, 1.0, 0.67, 1.0): "lower-right",
    }

    # Domain-specific vocabulary
    DOMAIN_VOCABULARY = {
        "manufacturing": {
            "defect": "production defect",
            "anomaly": "quality deviation",
            "region": "affected area",
            "action": "quality inspection",
        },
        "medical": {
            "defect": "lesion",
            "anomaly": "abnormality",
            "region": "region of interest",
            "action": "clinical review",
        },
        "security": {
            "defect": "threat indicator",
            "anomaly": "security concern",
            "region": "area of interest",
            "action": "security assessment",
        },
        "infrastructure": {
            "defect": "structural damage",
            "anomaly": "structural concern",
            "region": "affected section",
            "action": "engineering inspection",
        },
    }

    def __init__(self) -> None:
        """Initialize multi-modal narrator."""
        self.logger = logging.getLogger(__name__)

    def narrate(
        self,
        detection: MultiModalDetection,
        domain: str | None = None,
    ) -> MultiModalNarration:
        """
        Generate narration for multi-modal detection.

        Args:
            detection: Multi-modal detection result
            domain: Domain context for vocabulary

        Returns:
            MultiModalNarration with human-readable description
        """
        if detection.modality in [ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.THERMAL]:
            return self._narrate_visual(detection, domain)
        elif detection.modality in [
            ModalityType.AUDIO,
            ModalityType.SPECTROGRAM,
            ModalityType.WAVEFORM,
        ]:
            return self._narrate_audio(detection, domain)
        else:
            return self._narrate_generic(detection, domain)

    def _narrate_visual(
        self,
        detection: MultiModalDetection,
        domain: str | None = None,
    ) -> MultiModalNarration:
        """Generate narration for visual detection."""
        vocab = self.DOMAIN_VOCABULARY.get(
            domain or "manufacturing", self.DOMAIN_VOCABULARY["manufacturing"]
        )

        # Build summary
        if detection.anomaly_detected:
            anomaly_term = vocab.get("defect", "anomaly")
            summary = f"Visual analysis detected {anomaly_term} "
            summary += (
                f"(score: {detection.anomaly_score:.2f}, confidence: {detection.confidence:.0%}). "
            )
        else:
            summary = "Visual analysis found no significant anomalies. "
            summary += f"Confidence in assessment: {detection.confidence:.0%}."

        # Describe regions
        visual_description = None
        location_description = None

        if detection.regions:
            region_descriptions = []
            for region in detection.regions:
                pos = self._describe_position(region.x, region.y)
                region_descriptions.append(
                    f"{region.label} in {pos} region " f"(score: {region.anomaly_score:.2f})"
                )

            visual_description = f"Identified {len(detection.regions)} region(s) of interest: "
            visual_description += "; ".join(region_descriptions) + "."

            # Overall location
            if len(detection.regions) == 1:
                location_description = (
                    f"The detected {vocab.get('anomaly', 'anomaly')} is located in the "
                    f"{self._describe_position(detection.regions[0].x, detection.regions[0].y)} "
                    f"area of the image."
                )
            else:
                positions = [self._describe_position(r.x, r.y) for r in detection.regions]
                unique_positions = list(set(positions))
                location_description = (
                    f"Multiple {vocab.get('region', 'regions')}s identified across "
                    f"{', '.join(unique_positions)} areas."
                )

        # Severity assessment
        severity = self._assess_severity(detection.anomaly_score)

        # Recommendations
        recommendations = self._generate_visual_recommendations(detection, domain, severity)

        # Confidence statement
        confidence_stmt = self._confidence_statement(detection.confidence, "visual")

        return MultiModalNarration(
            summary=summary,
            visual_description=visual_description,
            location_description=location_description,
            severity_assessment=severity,
            recommendations=recommendations,
            confidence_statement=confidence_stmt,
        )

    def _narrate_audio(
        self,
        detection: MultiModalDetection,
        domain: str | None = None,
    ) -> MultiModalNarration:
        """Generate narration for audio detection."""
        # Build summary
        if detection.anomaly_detected:
            summary = "Audio analysis detected anomalous sound pattern "
            summary += (
                f"(score: {detection.anomaly_score:.2f}, confidence: {detection.confidence:.0%}). "
            )

            if isinstance(detection.anomaly_type, AudioAnomalyType):
                if detection.anomaly_type == AudioAnomalyType.MACHINERY:
                    summary += "Pattern consistent with machinery fault."
                elif detection.anomaly_type == AudioAnomalyType.ENVIRONMENTAL:
                    summary += "Unusual environmental sound detected."
        else:
            summary = "Audio analysis found no significant anomalies. "
            summary += "Sound patterns within normal parameters."

        # Describe segments
        audio_description = None
        temporal_description = None

        if detection.segments:
            segment_descriptions = []
            for segment in detection.segments:
                duration = segment.end_time - segment.start_time
                segment_descriptions.append(
                    f"{segment.label} from {segment.start_time:.1f}s to "
                    f"{segment.end_time:.1f}s ({duration:.1f}s duration)"
                )

            audio_description = f"Identified {len(detection.segments)} segment(s) of interest: "
            audio_description += "; ".join(segment_descriptions) + "."

            # Temporal summary
            if len(detection.segments) == 1:
                seg = detection.segments[0]
                temporal_description = (
                    f"The anomalous sound occurs at {seg.start_time:.1f} seconds, "
                    f"lasting {seg.end_time - seg.start_time:.1f} seconds."
                )
            else:
                total_duration = sum(s.end_time - s.start_time for s in detection.segments)
                temporal_description = (
                    f"Multiple anomalous segments totaling {total_duration:.1f} seconds."
                )

        # Severity and recommendations
        severity = self._assess_severity(detection.anomaly_score)
        recommendations = self._generate_audio_recommendations(detection, domain, severity)
        confidence_stmt = self._confidence_statement(detection.confidence, "audio")

        return MultiModalNarration(
            summary=summary,
            audio_description=audio_description,
            temporal_description=temporal_description,
            severity_assessment=severity,
            recommendations=recommendations,
            confidence_statement=confidence_stmt,
        )

    def _narrate_generic(
        self,
        detection: MultiModalDetection,
        domain: str | None = None,
    ) -> MultiModalNarration:
        """Generate generic narration for other modalities."""
        if detection.anomaly_detected:
            summary = (
                f"Multi-modal analysis detected anomaly in {detection.modality.value} data. "
                f"Score: {detection.anomaly_score:.2f}, Confidence: {detection.confidence:.0%}."
            )
        else:
            summary = f"No anomalies detected in {detection.modality.value} analysis."

        severity = self._assess_severity(detection.anomaly_score)
        confidence_stmt = self._confidence_statement(detection.confidence, "analysis")

        return MultiModalNarration(
            summary=summary,
            severity_assessment=severity,
            recommendations=["Review detection details for more information"],
            confidence_statement=confidence_stmt,
        )

    def _describe_position(self, x: float, y: float) -> str:
        """Describe position in human terms."""
        for (x_min, x_max, y_min, y_max), description in self.POSITION_DESCRIPTIONS.items():
            if x_min <= x < x_max and y_min <= y < y_max:
                return description
        return "unspecified region"

    def _assess_severity(self, score: float) -> str:
        """Assess severity from anomaly score."""
        if score >= 0.9:
            return "Critical: Immediate attention required."
        elif score >= 0.7:
            return "High: Prompt investigation recommended."
        elif score >= 0.5:
            return "Medium: Schedule review within normal cycle."
        elif score >= 0.3:
            return "Low: Monitor for changes."
        else:
            return "Minimal: No immediate action needed."

    def _confidence_statement(self, confidence: float, analysis_type: str) -> str:
        """Generate confidence statement."""
        if confidence >= 0.9:
            return f"High confidence in {analysis_type} assessment ({confidence:.0%})."
        elif confidence >= 0.7:
            return f"Moderate-high confidence ({confidence:.0%}). Results are reliable."
        elif confidence >= 0.5:
            return f"Moderate confidence ({confidence:.0%}). Consider additional verification."
        else:
            return f"Low confidence ({confidence:.0%}). Manual review strongly recommended."

    def _generate_visual_recommendations(
        self,
        detection: MultiModalDetection,
        domain: str | None,
        severity: str,
    ) -> list[str]:
        """Generate recommendations for visual detection."""
        recommendations = []

        if not detection.anomaly_detected:
            return ["Continue routine monitoring"]

        vocab = self.DOMAIN_VOCABULARY.get(
            domain or "manufacturing", self.DOMAIN_VOCABULARY["manufacturing"]
        )

        if detection.anomaly_score >= 0.7:
            recommendations.append(f"Immediate {vocab.get('action', 'inspection')} recommended")
        else:
            recommendations.append(f"Schedule {vocab.get('action', 'review')} for flagged area")

        if len(detection.regions) > 1:
            recommendations.append("Multiple regions detected - systematic review advised")

        if domain == "medical":
            recommendations.append("Correlate with patient history and additional imaging")
        elif domain == "security":
            recommendations.append("Review security footage for context")
        elif domain == "manufacturing":
            recommendations.append("Check production batch for similar defects")

        return recommendations

    def _generate_audio_recommendations(
        self,
        detection: MultiModalDetection,
        domain: str | None,
        severity: str,
    ) -> list[str]:
        """Generate recommendations for audio detection."""
        recommendations = []

        if not detection.anomaly_detected:
            return ["Continue routine audio monitoring"]

        if detection.anomaly_score >= 0.7:
            recommendations.append("Investigate sound source immediately")
        else:
            recommendations.append("Schedule equipment inspection")

        if isinstance(detection.anomaly_type, AudioAnomalyType):
            if detection.anomaly_type == AudioAnomalyType.MACHINERY:
                recommendations.append("Check mechanical components for wear or damage")
                recommendations.append("Review maintenance schedule")
            elif detection.anomaly_type == AudioAnomalyType.ENVIRONMENTAL:
                recommendations.append("Assess environmental factors")

        return recommendations


def create_multimodal_narrator() -> MultiModalNarrator:
    """
    Factory function to create multi-modal narrator.

    Returns:
        Configured MultiModalNarrator
    """
    return MultiModalNarrator()


def narrate_image_detection(
    detection_result: dict[str, Any],
    domain: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to narrate image detection result.

    Args:
        detection_result: Detection result dictionary
        domain: Domain context

    Returns:
        Narration as dictionary
    """
    narrator = MultiModalNarrator()

    # Convert dict to MultiModalDetection
    regions = []
    for r in detection_result.get("regions", []):
        regions.append(
            RegionOfInterest(
                x=r.get("x", 0),
                y=r.get("y", 0),
                width=r.get("width", 0),
                height=r.get("height", 0),
                confidence=r.get("confidence", 0.5),
                label=r.get("label", "anomaly"),
                anomaly_score=r.get("score", 0),
            )
        )

    detection = MultiModalDetection(
        modality=ModalityType.IMAGE,
        anomaly_type=(
            AnomalyVisualType(detection_result.get("anomaly_type", "unknown"))
            if detection_result.get("anomaly_type") in [e.value for e in AnomalyVisualType]
            else AnomalyVisualType.UNKNOWN
        ),
        anomaly_detected=detection_result.get("anomaly_detected", False),
        anomaly_score=detection_result.get("anomaly_score", 0),
        confidence=detection_result.get("confidence", 0.5),
        regions=regions,
    )

    narration = narrator.narrate(detection, domain)

    return {
        "summary": narration.summary,
        "visual_description": narration.visual_description,
        "location_description": narration.location_description,
        "severity": narration.severity_assessment,
        "recommendations": narration.recommendations,
        "confidence_statement": narration.confidence_statement,
    }


def narrate_audio_detection(
    detection_result: dict[str, Any],
    domain: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to narrate audio detection result.

    Args:
        detection_result: Detection result dictionary
        domain: Domain context

    Returns:
        Narration as dictionary
    """
    narrator = MultiModalNarrator()

    # Convert dict to MultiModalDetection
    segments = []
    for s in detection_result.get("segments", []):
        segments.append(
            AudioSegment(
                start_time=s.get("start", 0),
                end_time=s.get("end", 0),
                confidence=s.get("confidence", 0.5),
                label=s.get("label", "anomaly"),
                anomaly_score=s.get("score", 0),
            )
        )

    detection = MultiModalDetection(
        modality=ModalityType.AUDIO,
        anomaly_type=(
            AudioAnomalyType(detection_result.get("anomaly_type", "unknown"))
            if detection_result.get("anomaly_type") in [e.value for e in AudioAnomalyType]
            else AudioAnomalyType.UNKNOWN
        ),
        anomaly_detected=detection_result.get("anomaly_detected", False),
        anomaly_score=detection_result.get("anomaly_score", 0),
        confidence=detection_result.get("confidence", 0.5),
        segments=segments,
    )

    narration = narrator.narrate(detection, domain)

    return {
        "summary": narration.summary,
        "audio_description": narration.audio_description,
        "temporal_description": narration.temporal_description,
        "severity": narration.severity_assessment,
        "recommendations": narration.recommendations,
        "confidence_statement": narration.confidence_statement,
    }

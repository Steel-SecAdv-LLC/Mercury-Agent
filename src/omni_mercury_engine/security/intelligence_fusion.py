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
Security Intelligence Fusion Module

Comprehensive integration of multi-source intelligence disciplines for all-source
threat anomaly detection. Integrates OSINT, COMINT, HUMINT, GEOINT, IMINT, SIGINT,
ELINT, MASINT, CYBINT, FININT, and specialized intelligence modalities for unified
threat assessment and early warning.

Key Features:
- Multi-INT fusion with neurosymbolic correlation
- All-source intelligence synthesis
- Cryptanalysis-assisted pattern detection
- Operations security (OPSEC) anomaly detection
- Temporal threat progression modeling
- Golden ratio optimized fusion weights
- O(n) complexity for real-time threat assessment

Intelligence Disciplines:
- OSINT: Open-source intelligence (media, social, databases)
- COMINT: Communications intelligence (phone, email, radio)
- HUMINT: Human intelligence (interviews, espionage)
- GEOINT: Geospatial intelligence (location, terrain)
- IMINT: Imagery intelligence (satellite, drone)
- SIGINT: Signals intelligence (electronic intercepts)
- ELINT: Electronic intelligence (radar, sensors)
- MASINT: Measurement & signature intelligence (technical signatures)
- CYBINT: Cyber intelligence (network, malware)
- FININT: Financial intelligence (money laundering, terrorism financing)

⚠️ SIMULATION-BASED: Research/development tool for threat analysis patterns.
Operational deployment requires security clearance and legal authorization.

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class IntelligenceDiscipline(Enum):
    """Intelligence collection disciplines"""

    OSINT = "open_source"
    COMINT = "communications"
    HUMINT = "human"
    GEOINT = "geospatial"
    IMINT = "imagery"
    SIGINT = "signals"
    ELINT = "electronic"
    MASINT = "measurement_signature"
    CYBINT = "cyber"
    FININT = "financial"
    CRYPTANALYSIS = "cryptanalysis"
    METEOROLOGICAL = "meteorological"
    TRAFFIC_ANALYSIS = "traffic"


class ThreatLevel(Enum):
    """Threat assessment levels"""

    LOW = 1
    MODERATE = 2
    SUBSTANTIAL = 3
    SEVERE = 4
    CRITICAL = 5


@dataclass
class IntelligenceFusionResult:
    """Result from multi-INT fusion analysis"""

    threat_detected: bool
    threat_level: str
    confidence: float
    risk_score: float

    primary_intel_sources: list[str] = field(default_factory=list)
    corroborating_sources: list[str] = field(default_factory=list)

    threat_indicators: list[str] = field(default_factory=list)
    temporal_patterns: dict[str, Any] = field(default_factory=dict)
    geospatial_context: dict[str, Any] = field(default_factory=dict)

    recommended_actions: list[str] = field(default_factory=list)
    collection_priorities: list[str] = field(default_factory=list)

    neurosymbolic_assessment: dict[str, Any] | None = None
    cryptographic_indicators: dict[str, Any] | None = None


class AllSourceFusionNetwork(nn.Module):
    """
    Neural network for all-source intelligence fusion.

    Implements multi-head attention across INT disciplines with golden ratio
    architecture optimization for optimal information synthesis.
    """

    def __init__(self, input_dim: int = 128, num_int_types: int = 13) -> None:
        super().__init__()

        phi = 1.618
        hidden_1 = int(input_dim * phi)
        hidden_2 = int(hidden_1 * phi)
        hidden_3 = (
            round(int(hidden_2 / phi) / 13) * 13
        )  # Round to nearest multiple of 13 for attention

        self.int_encoders = nn.ModuleDict(
            {
                discipline.value: nn.Sequential(
                    nn.Linear(input_dim // num_int_types, hidden_1 // num_int_types),
                    nn.LayerNorm(hidden_1 // num_int_types),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                )
                for discipline in IntelligenceDiscipline
            }
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.ReLU(),
        )

        self.cross_int_attention = nn.MultiheadAttention(
            embed_dim=hidden_3, num_heads=13, dropout=0.1, batch_first=True
        )

        self.temporal_lstm = nn.LSTM(
            input_size=hidden_3,
            hidden_size=hidden_3 // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )

        self.threat_classifier = nn.Sequential(
            nn.Linear(hidden_3 + hidden_3 // 2, hidden_3),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_3, 5),
        )

        self.confidence_head = nn.Sequential(nn.Linear(hidden_3, 1), nn.Sigmoid())

    def forward(
        self,
        int_features: dict[str, torch.Tensor],
        temporal_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all-source fusion network.

        Args:
            int_features: Dictionary of features per INT discipline
            temporal_sequence: Optional temporal threat progression

        Returns:
            Tuple of (threat_class, confidence, attention_weights)
        """
        encoded_ints = []

        for discipline, encoder in self.int_encoders.items():
            if discipline in int_features:
                encoded = encoder(int_features[discipline])
                encoded_ints.append(encoded)

        if not encoded_ints:
            batch_size = 1
            return (
                torch.zeros(batch_size, 5),
                torch.zeros(batch_size, 1),
                torch.zeros(batch_size, 1),
            )

        fused = torch.cat(encoded_ints, dim=-1)

        encoded = self.fusion_encoder(fused)

        encoded_seq = encoded.unsqueeze(1)
        attended, attention_weights = self.cross_int_attention(
            encoded_seq, encoded_seq, encoded_seq
        )
        attended = attended.squeeze(1)

        temporal_features = torch.zeros_like(attended[:, : attended.shape[1] // 2])
        if temporal_sequence is not None:
            lstm_out, _ = self.temporal_lstm(temporal_sequence)
            temporal_features = lstm_out[:, -1, :]

        combined = torch.cat([attended, temporal_features], dim=-1)

        threat_logits = self.threat_classifier(combined)
        confidence = self.confidence_head(attended)

        return threat_logits, confidence, attention_weights.mean(dim=1)


class IntelligenceFusionEngine:
    """
    All-Source Intelligence Fusion Engine.

    Synthesizes multi-INT inputs for comprehensive threat assessment with
    neurosymbolic reasoning and cryptanalysis integration.
    """

    def __init__(
        self,
        enable_neurosymbolic: bool = True,
        enable_cryptanalysis: bool = True,
        golden_ratio_weights: bool = True,
    ):
        """
        Initialize intelligence fusion engine.

        Args:
            enable_neurosymbolic: Enable symbolic threat reasoning
            enable_cryptanalysis: Enable pattern-based cryptanalysis
            golden_ratio_weights: Use φ-optimized fusion weights
        """
        self.logger = logging.getLogger(__name__)
        self.enable_neurosymbolic = enable_neurosymbolic
        self.enable_cryptanalysis = enable_cryptanalysis
        self.golden_ratio = 1.618 if golden_ratio_weights else 1.0

        self.fusion_network = AllSourceFusionNetwork(input_dim=128, num_int_types=13)

        self.threat_knowledge_base = self._initialize_threat_kb()

        self.int_reliability_scores = self._initialize_reliability_scores()

        self.omni_intelligence_scalars = {
            "omni_source_credibility": 1.45 * self.golden_ratio,
            "omni_corroboration_strength": 1.42 * self.golden_ratio,
            "omni_temporal_correlation": 1.38 * self.golden_ratio,
            "omni_geospatial_precision": 1.40 * self.golden_ratio,
            "omni_threat_anticipation": 1.47 * self.golden_ratio,
            "omni_cryptographic_insight": 1.43 * self.golden_ratio,
            "omni_human_factor_analysis": 1.36 * self.golden_ratio,
            "omni_signal_clarity": 1.41 * self.golden_ratio,
            "omni_financial_tracing": 1.39 * self.golden_ratio,
            "omni_cyber_attribution": 1.44 * self.golden_ratio,
        }

        self.logger.info(
            f"Intelligence Fusion Engine initialized with {len(IntelligenceDiscipline)} disciplines"
        )

    def _initialize_threat_kb(self) -> dict[str, dict[str, Any]]:
        """Initialize threat pattern knowledge base"""
        return {
            "terrorism_indicators": {
                "patterns": ["recruitment", "training", "logistics", "finance", "communications"],
                "int_sources": [
                    IntelligenceDiscipline.HUMINT,
                    IntelligenceDiscipline.COMINT,
                    IntelligenceDiscipline.FININT,
                    IntelligenceDiscipline.CYBINT,
                ],
            },
            "cyber_attack_indicators": {
                "patterns": ["reconnaissance", "weaponization", "delivery", "exploitation", "c2"],
                "int_sources": [
                    IntelligenceDiscipline.CYBINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.OSINT,
                ],
            },
            "military_buildup": {
                "patterns": ["troop_movement", "equipment_positioning", "logistics_surge"],
                "int_sources": [
                    IntelligenceDiscipline.IMINT,
                    IntelligenceDiscipline.GEOINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.MASINT,
                ],
            },
            "espionage_indicators": {
                "patterns": ["insider_threat", "exfiltration", "covert_comms", "dead_drops"],
                "int_sources": [
                    IntelligenceDiscipline.HUMINT,
                    IntelligenceDiscipline.COMINT,
                    IntelligenceDiscipline.CYBINT,
                ],
            },
            "wmd_proliferation": {
                "patterns": ["procurement", "facility_construction", "testing", "delivery_systems"],
                "int_sources": [
                    IntelligenceDiscipline.IMINT,
                    IntelligenceDiscipline.MASINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.FININT,
                ],
            },
        }

    def _initialize_reliability_scores(self) -> dict[str, float]:
        """Initialize INT source reliability weights"""
        return {
            IntelligenceDiscipline.HUMINT.value: 0.85,
            IntelligenceDiscipline.IMINT.value: 0.92,
            IntelligenceDiscipline.SIGINT.value: 0.90,
            IntelligenceDiscipline.COMINT.value: 0.88,
            IntelligenceDiscipline.GEOINT.value: 0.93,
            IntelligenceDiscipline.MASINT.value: 0.87,
            IntelligenceDiscipline.ELINT.value: 0.89,
            IntelligenceDiscipline.CYBINT.value: 0.86,
            IntelligenceDiscipline.FININT.value: 0.84,
            IntelligenceDiscipline.OSINT.value: 0.75,
            IntelligenceDiscipline.CRYPTANALYSIS.value: 0.91,
            IntelligenceDiscipline.METEOROLOGICAL.value: 0.82,
            IntelligenceDiscipline.TRAFFIC_ANALYSIS.value: 0.83,
        }

    def fuse_intelligence(
        self, intel_reports: dict[str, Any], temporal_context: list[dict[str, Any]] | None = None
    ) -> IntelligenceFusionResult:
        """
        Fuse multi-source intelligence for threat assessment.

        Args:
            intel_reports: Dictionary of intelligence reports by discipline:
                - osint: Open-source findings
                - comint: Communications intercepts
                - humint: Human source reports
                - geoint: Geospatial data
                - imint: Imagery analysis
                - sigint: Signals intercepts
                - elint: Electronic signatures
                - masint: Technical measurements
                - cybint: Cyber indicators
                - finint: Financial transactions
                - cryptanalysis: Encrypted pattern analysis
            temporal_context: Optional historical threat timeline

        Returns:
            Fused intelligence assessment with threat level
        """
        int_features = self._extract_int_features(intel_reports)

        temporal_tensor = None
        if temporal_context:
            temporal_tensor = self._process_temporal_context(temporal_context)

        self.fusion_network.eval()
        with torch.no_grad():
            threat_logits, confidence, attention = self.fusion_network(
                int_features, temporal_tensor
            )

        threat_probs = torch.softmax(threat_logits[0], dim=0)
        threat_class = torch.argmax(threat_probs).item() + 1
        threat_level_enum = ThreatLevel(threat_class)
        confidence_score = float(confidence[0].item())

        risk_score = (
            confidence_score
            * self.omni_intelligence_scalars["omni_source_credibility"]
            * (threat_class / 5.0)
        )

        threat_detected = risk_score > (0.5 * self.golden_ratio)

        primary_sources, corroborating = self._identify_sources(intel_reports, attention[0].numpy())

        indicators = self._extract_threat_indicators(intel_reports, threat_level_enum)

        temporal_patterns = (
            self._analyze_temporal_patterns(temporal_context) if temporal_context else {}
        )

        geospatial_context = self._extract_geospatial_context(intel_reports)

        actions = self._recommend_actions(threat_level_enum, indicators)

        collection_priorities = self._prioritize_collection(
            threat_level_enum, primary_sources, indicators
        )

        neurosymbolic_assessment = None
        if self.enable_neurosymbolic:
            neurosymbolic_assessment = self._apply_symbolic_reasoning(
                intel_reports, threat_level_enum, indicators
            )

        crypto_indicators = None
        if self.enable_cryptanalysis:
            crypto_indicators = self._analyze_cryptographic_patterns(intel_reports)

        result = IntelligenceFusionResult(
            threat_detected=threat_detected,
            threat_level=threat_level_enum.name,
            confidence=confidence_score,
            risk_score=risk_score,
            primary_intel_sources=primary_sources,
            corroborating_sources=corroborating,
            threat_indicators=indicators,
            temporal_patterns=temporal_patterns,
            geospatial_context=geospatial_context,
            recommended_actions=actions,
            collection_priorities=collection_priorities,
            neurosymbolic_assessment=neurosymbolic_assessment,
            cryptographic_indicators=crypto_indicators,
        )

        self.logger.info(
            f"Intelligence fusion: {threat_level_enum.name} threat "
            f"(confidence={confidence_score:.3f}, risk={risk_score:.3f})"
        )

        return result

    def _extract_int_features(self, intel_reports: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract features from intelligence reports (O(n) complexity)"""
        int_features = {}
        feature_dim = 128 // len(IntelligenceDiscipline)

        for discipline in IntelligenceDiscipline:
            disc_key = discipline.value
            features = np.zeros(feature_dim, dtype=np.float32)

            if disc_key in intel_reports:
                report = intel_reports[disc_key]

                features[0] = float(report.get("confidence", 0.5))
                features[1] = float(report.get("timeliness", 0.5))
                features[2] = float(report.get("relevance", 0.5))
                features[3] = float(report.get("completeness", 0.5))

                if "indicators" in report:
                    features[4] = float(len(report["indicators"])) / 10.0

                if "threat_score" in report:
                    features[5] = float(report["threat_score"])

                reliability = self.int_reliability_scores.get(disc_key, 0.75)
                features = np.asarray(features * reliability)

            int_features[disc_key] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        return int_features

    def _process_temporal_context(self, temporal_context: list[dict[str, Any]]) -> torch.Tensor:
        """Process temporal threat progression"""
        sequence_length = min(len(temporal_context), 10)
        feature_dim = 165

        temporal_features = np.zeros((1, sequence_length, feature_dim), dtype=np.float32)

        for i, event in enumerate(temporal_context[-sequence_length:]):
            temporal_features[0, i, 0] = float(event.get("threat_level", 0)) / 5.0
            temporal_features[0, i, 1] = float(event.get("confidence", 0.5))
            temporal_features[0, i, 2] = float(event.get("num_sources", 1)) / 10.0

        return torch.tensor(temporal_features, dtype=torch.float32)

    def _identify_sources(
        self, intel_reports: dict[str, Any], attention_weights: np.ndarray[Any, Any]
    ) -> tuple[list[str], list[str]]:
        """Identify primary and corroborating intelligence sources"""
        source_scores = []

        for _i, discipline in enumerate(IntelligenceDiscipline):
            if discipline.value in intel_reports:
                score = intel_reports[discipline.value].get("confidence", 0.0)
                source_scores.append((discipline.value, score))

        source_scores.sort(key=lambda x: x[1], reverse=True)

        primary = [s[0] for s in source_scores[:3]]
        corroborating = [s[0] for s in source_scores[3:6]]

        return primary, corroborating

    def _extract_threat_indicators(
        self, intel_reports: dict[str, Any], threat_level: ThreatLevel
    ) -> list[str]:
        """Extract key threat indicators from reports"""
        indicators = set()

        for report in intel_reports.values():
            if "indicators" in report:
                indicators.update(report["indicators"][:5])

        if threat_level.value >= ThreatLevel.SUBSTANTIAL.value:
            indicators.add("elevated_threat_posture")

        return list(indicators)[:15]

    def _analyze_temporal_patterns(self, temporal_context: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze temporal threat progression patterns"""
        if not temporal_context:
            return {}

        threat_levels = [e.get("threat_level", 2) for e in temporal_context]

        return {
            "trend": "escalating" if threat_levels[-1] > threat_levels[0] else "stable",
            "mean_threat": np.mean(threat_levels),
            "volatility": np.std(threat_levels),
            "time_span_days": len(temporal_context),
        }

    def _extract_geospatial_context(self, intel_reports: dict[str, Any]) -> dict[str, Any]:
        """Extract geospatial threat context"""
        geospatial = {}

        if IntelligenceDiscipline.GEOINT.value in intel_reports:
            geo_report = intel_reports[IntelligenceDiscipline.GEOINT.value]
            geospatial["locations"] = geo_report.get("locations", [])
            geospatial["proximity_to_assets"] = geo_report.get("proximity_km", 0)

        if IntelligenceDiscipline.IMINT.value in intel_reports:
            img_report = intel_reports[IntelligenceDiscipline.IMINT.value]
            geospatial["imagery_confidence"] = img_report.get("confidence", 0.0)

        return geospatial

    def _recommend_actions(self, threat_level: ThreatLevel, indicators: list[str]) -> list[str]:
        """Recommend actions based on threat assessment"""
        actions = []

        if threat_level == ThreatLevel.CRITICAL:
            actions.extend(
                [
                    "Immediate senior leadership notification",
                    "Activate crisis response protocols",
                    "Deploy additional collection assets",
                    "Coordinate with operational elements",
                ]
            )
        elif threat_level == ThreatLevel.SEVERE:
            actions.extend(
                [
                    "Elevate monitoring posture",
                    "Increase collection frequency",
                    "Prepare contingency responses",
                    "Brief stakeholders",
                ]
            )
        elif threat_level == ThreatLevel.SUBSTANTIAL:
            actions.extend(
                [
                    "Enhanced monitoring recommended",
                    "Cross-reference with allied intelligence",
                    "Update threat assessments",
                ]
            )
        else:
            actions.append("Continue routine monitoring")

        return actions[:6]

    def _prioritize_collection(
        self, threat_level: ThreatLevel, primary_sources: list[str], indicators: list[str]
    ) -> list[str]:
        """Prioritize intelligence collection efforts"""
        priorities = []

        if threat_level.value >= ThreatLevel.SUBSTANTIAL.value:
            priorities.append("Priority 1: Real-time monitoring")

        for source in primary_sources:
            priorities.append(f"Sustain {source} collection")

        valid_disciplines = [d.value for d in IntelligenceDiscipline]
        source_disciplines: set[IntelligenceDiscipline] = {
            IntelligenceDiscipline(s) for s in primary_sources if s in valid_disciplines
        }
        gaps = set(IntelligenceDiscipline) - source_disciplines

        for gap in list(gaps)[:2]:
            priorities.append(f"Fill gap: {gap.value}")

        return priorities[:6]

    def _apply_symbolic_reasoning(
        self, intel_reports: dict[str, Any], threat_level: ThreatLevel, indicators: list[str]
    ) -> dict[str, Any]:
        """Apply neurosymbolic threat reasoning"""
        all_matched_patterns: list[str] = []
        deductions: list[str] = []
        confidence_factors: list[str] = []

        for threat_type, pattern_info in self.threat_knowledge_base.items():
            matched_patterns = [
                p for p in pattern_info["patterns"] if p in " ".join(indicators).lower()
            ]

            if matched_patterns:
                all_matched_patterns.append(f"{threat_type}: {', '.join(matched_patterns)}")
                deductions.append(f"Potential {threat_type.replace('_', ' ')} activity")

        if len(all_matched_patterns) > 1:
            confidence_factors.append("High confidence: Multiple threat pattern matches")

        return {
            "matched_patterns": all_matched_patterns,
            "deductions": deductions,
            "confidence_factors": confidence_factors,
        }

    def _analyze_cryptographic_patterns(self, intel_reports: dict[str, Any]) -> dict[str, Any]:
        """Analyze cryptographic and pattern indicators"""
        encrypted_comms_detected: bool = False
        pattern_strength: float = 0.0
        recommendations: list[str] = []

        if IntelligenceDiscipline.COMINT.value in intel_reports:
            comint = intel_reports[IntelligenceDiscipline.COMINT.value]
            encrypted_comms_detected = bool(comint.get("encryption_detected", False))

            if encrypted_comms_detected:
                recommendations.append("Prioritize cryptanalysis resources")
                pattern_strength = 0.7

        if IntelligenceDiscipline.CYBINT.value in intel_reports:
            cybint = intel_reports[IntelligenceDiscipline.CYBINT.value]
            if "encryption_algorithm" in cybint:
                recommendations.append(f"Target: {cybint['encryption_algorithm']}")

        return {
            "encrypted_comms_detected": encrypted_comms_detected,
            "pattern_strength": pattern_strength,
            "recommendations": recommendations,
        }

    def extract_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration"""
        int_features = self._extract_int_features(data)

        all_features = []
        for discipline in IntelligenceDiscipline:
            if discipline.value in int_features:
                all_features.append(int_features[discipline.value])

        if all_features:
            return torch.cat(all_features, dim=-1)
        else:
            return torch.zeros(1, 128, dtype=torch.float32)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict for engine integration"""
        result = self.fuse_intelligence(data)

        return {
            "anomaly_scores": np.array([result.risk_score], dtype=np.float32),
            "threat_level": result.threat_level,
            "confidence": result.confidence,
            "intel_sources": result.primary_intel_sources,
        }


def create_omni_intelligence_scalars() -> dict[str, float]:
    """
    Create doctorate-level intelligence scalars for truth deciphering.

    Returns:
        Dictionary of omni-intelligence scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_source_credibility": 1.45 * phi,
        "omni_corroboration_strength": 1.42 * phi,
        "omni_temporal_correlation": 1.38 * phi,
        "omni_geospatial_precision": 1.40 * phi,
        "omni_threat_anticipation": 1.47 * phi,
        "omni_cryptographic_insight": 1.43 * phi,
        "omni_human_factor_analysis": 1.36 * phi,
        "omni_signal_clarity": 1.41 * phi,
        "omni_financial_tracing": 1.39 * phi,
        "omni_cyber_attribution": 1.44 * phi,
        "omni_all_source_synthesis": 1.48 * phi,
        "omni_counterintelligence": 1.46 * phi,
        "omni_predictive_analysis": 1.43 * phi,
    }

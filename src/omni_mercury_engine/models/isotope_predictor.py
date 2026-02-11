"""
Mercury Agent ♱
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
"""

from __future__ import annotations

"""
Isotope Predictor - Nuclear Forensics & Isotope Ratio Analysis

Advanced isotope anomaly detection for humanitarian applications:
- Isotope ratio anomaly detection
- Nuclear forensics analysis
- Radiological threat assessment
- Medical isotope verification
- Environmental contamination detection

⚠️ SIMULATION-BASED: Research tool for isotope analysis patterns.
Operational use requires proper radiological safety protocols.

Research sources:
- IAEA nuclear forensics guidelines
- Isotope ratio mass spectrometry (IRMS) principles
- Environmental radiological monitoring standards

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class IsotopeType(Enum):
    """Isotope classifications"""

    NATURAL = "natural_isotope"
    ENRICHED = "enriched_uranium"
    DEPLETED = "depleted_uranium"
    PLUTONIUM = "plutonium"
    MEDICAL = "medical_isotope"
    INDUSTRIAL = "industrial_isotope"
    COSMOGENIC = "cosmogenic"


class ThreatLevel(Enum):
    """Radiological threat levels"""

    SAFE = "safe"
    LOW = "low_risk"
    MODERATE = "moderate_risk"
    HIGH = "high_risk"
    CRITICAL = "critical_threat"


@dataclass
class IsotopePredictionResult:
    """Isotope analysis results"""

    anomaly_detected: bool
    confidence: float
    isotope_type: str
    threat_level: str

    isotope_ratios: dict[str, float] = field(default_factory=dict)
    enrichment_level: float | None = None
    origin_indicators: list[str] = field(default_factory=list)

    nuclear_forensics: dict[str, Any] = field(default_factory=dict)
    contamination_detected: bool = False

    recommendations: list[str] = field(default_factory=list)
    regulatory_alerts: list[str] = field(default_factory=list)


class IsotopeRatioAnalyzer(nn.Module):
    """
    Neural network for isotope ratio analysis.

    Detects anomalous isotope signatures and enrichment patterns.
    """

    def __init__(self, input_dim: int = 64) -> None:
        super().__init__()

        self.ratio_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )

        self.isotope_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(IsotopeType))
        )

        self.enrichment_predictor = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        self.threat_assessor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(ThreatLevel))
        )

    def forward(
        self, isotope_ratios: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for isotope analysis.

        Args:
            isotope_ratios: Isotope ratio measurements

        Returns:
            Tuple of (isotope_classification, enrichment_level, threat_level)
        """
        encoded = self.ratio_encoder(isotope_ratios)

        isotope_logits = self.isotope_classifier(encoded)
        enrichment = self.enrichment_predictor(encoded)
        threat_logits = self.threat_assessor(encoded)

        return isotope_logits, enrichment, threat_logits


class NuclearForensicsAnalyzer:
    """
    Nuclear forensics analysis for isotope attribution.

    Determines origin and production methods of nuclear materials.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.uranium_isotope_ratios = {
            "natural": {"U235_U238": 0.00725, "U234_U238": 0.000055},
            "enriched_3pct": {"U235_U238": 0.0312, "U234_U238": 0.00024},
            "enriched_20pct": {"U235_U238": 0.25, "U234_U238": 0.002},
            "enriched_90pct": {"U235_U238": 9.0, "U234_U238": 0.009},
            "depleted": {"U235_U238": 0.0025, "U234_U238": 0.000018},
        }

    def analyze_uranium_signature(self, ratios: dict[str, float]) -> dict[str, Any]:
        """
        Analyze uranium isotope signature for forensics.

        Args:
            ratios: Measured isotope ratios

        Returns:
            Forensic analysis results
        """
        u235_u238 = ratios.get("U235_U238", 0.00725)
        u234_u238 = ratios.get("U234_U238", 0.000055)

        enrichment_category = self._classify_enrichment(u235_u238)

        production_method = self._infer_production_method(u235_u238, u234_u238)

        age_estimate = self._estimate_material_age(ratios)

        origin_indicators = self._identify_origin_indicators(ratios, enrichment_category)

        return {
            "enrichment_category": enrichment_category,
            "enrichment_percent": self._calculate_enrichment_percent(u235_u238),
            "production_method": production_method,
            "estimated_age_years": age_estimate,
            "origin_indicators": origin_indicators,
            "nuclear_forensic_score": self._calculate_forensic_confidence(ratios),
        }

    def _classify_enrichment(self, u235_u238_ratio: float) -> str:
        """Classify enrichment level"""

        if u235_u238_ratio < 0.004:
            return "depleted"
        elif u235_u238_ratio < 0.01:
            return "natural"
        elif u235_u238_ratio < 0.05:
            return "low_enriched"
        elif u235_u238_ratio < 0.5:
            return "medium_enriched"
        else:
            return "highly_enriched"

    def _calculate_enrichment_percent(self, u235_u238_ratio: float) -> float:
        """Calculate U-235 enrichment percentage"""

        u235_fraction = u235_u238_ratio / (1 + u235_u238_ratio)
        enrichment_pct = u235_fraction * 100

        return min(enrichment_pct, 100.0)

    def _infer_production_method(self, u235_u238: float, u234_u238: float) -> str:
        """Infer uranium enrichment method from isotope ratios"""

        theoretical_u234_u238 = u235_u238 * 0.0076

        ratio_deviation = abs(u234_u238 - theoretical_u234_u238) / theoretical_u234_u238

        if ratio_deviation < 0.1:
            return "gas_centrifuge"
        elif ratio_deviation < 0.3:
            return "gaseous_diffusion"
        else:
            return "chemical_or_unknown"

    def _estimate_material_age(self, ratios: dict[str, float]) -> float:
        """Estimate material age from decay products"""

        if "Pa231_U235" in ratios:
            pa231_u235 = ratios["Pa231_U235"]

            decay_constant = 2.116e-5
            age_years = -np.log(1 - pa231_u235) / decay_constant

            return float(min(age_years, 100.0))

        return 0.0

    def _identify_origin_indicators(self, ratios: dict[str, float], enrichment: str) -> list[str]:
        """Identify origin indicators from isotope signature"""

        indicators = []

        if "Pu239_Pu240" in ratios:
            pu_ratio = ratios["Pu239_Pu240"]

            if pu_ratio > 30:
                indicators.append("weapons_grade_plutonium")
            elif pu_ratio > 10:
                indicators.append("reactor_grade_plutonium")

        if enrichment in ["medium_enriched", "highly_enriched"]:
            indicators.append("potential_proliferation_concern")

        if "Kr85" in ratios and ratios["Kr85"] > 0:
            indicators.append("reprocessed_material")

        return indicators

    def _calculate_forensic_confidence(self, ratios: dict[str, float]) -> float:
        """Calculate confidence in forensic attribution"""

        confidence = 0.5

        if "U235_U238" in ratios:
            confidence += 0.2
        if "U234_U238" in ratios:
            confidence += 0.1
        if "Pu239_Pu240" in ratios:
            confidence += 0.2

        return min(confidence, 1.0)


class RadiologicalThreatAssessor:
    """
    Radiological threat assessment.

    Evaluates radiological hazards and security threats.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def assess_threat(self, isotope_data: dict[str, Any], enrichment: float) -> dict[str, Any]:
        """
        Assess radiological threat level.

        Args:
            isotope_data: Isotope analysis data
            enrichment: Enrichment level (0-1)

        Returns:
            Threat assessment
        """
        threat_indicators = []
        threat_score = 0.0

        if enrichment > 0.90:
            threat_indicators.append("weapons_grade_material")
            threat_score += 1.0
        elif enrichment > 0.20:
            threat_indicators.append("significant_enrichment")
            threat_score += 0.7

        if "weapons_grade_plutonium" in isotope_data.get("origin_indicators", []):
            threat_indicators.append("fissile_material")
            threat_score += 0.9

        if "potential_proliferation_concern" in isotope_data.get("origin_indicators", []):
            threat_indicators.append("proliferation_risk")
            threat_score += 0.6

        threat_score = min(threat_score, 1.0)

        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL.value
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH.value
        elif threat_score >= 0.4:
            threat_level = ThreatLevel.MODERATE.value
        elif threat_score >= 0.2:
            threat_level = ThreatLevel.LOW.value
        else:
            threat_level = ThreatLevel.SAFE.value

        regulatory_alerts = self._generate_regulatory_alerts(threat_indicators, enrichment)

        return {
            "threat_level": threat_level,
            "threat_score": threat_score,
            "threat_indicators": threat_indicators,
            "regulatory_alerts": regulatory_alerts,
        }

    def _generate_regulatory_alerts(self, indicators: list[str], enrichment: float) -> list[str]:
        """Generate regulatory compliance alerts"""

        alerts = []

        if enrichment > 0.20:
            alerts.append("IAEA Category I nuclear material - immediate reporting required")

        if "weapons_grade_material" in indicators:
            alerts.append("Notify national nuclear security authority")
            alerts.append("Implement enhanced physical protection measures")

        if "proliferation_risk" in indicators:
            alerts.append("Report to IAEA safeguards division")

        return alerts


class IsotopePredictor:
    """
    Comprehensive isotope prediction and nuclear forensics system.

    Integrates ratio analysis, forensics, and threat assessment.
    """

    def __init__(
        self,
        enable_ml: bool = True,
        enable_forensics: bool = True,
        enable_threat_assessment: bool = True,
    ):
        self.enable_ml = enable_ml
        self.enable_forensics = enable_forensics
        self.enable_threat_assessment = enable_threat_assessment

        self.ratio_analyzer = IsotopeRatioAnalyzer() if enable_ml else None
        self.forensics_analyzer = NuclearForensicsAnalyzer() if enable_forensics else None
        self.threat_assessor = RadiologicalThreatAssessor() if enable_threat_assessment else None

        self.logger = logging.getLogger(__name__)

    def predict_isotope_anomaly(self, isotope_data: dict[str, Any]) -> IsotopePredictionResult:
        """
        Comprehensive isotope anomaly prediction.

        Args:
            isotope_data: Isotope measurements including:
                - isotope_ratios: Measured isotope ratios
                - ratio_features: ML features (optional)

        Returns:
            Isotope prediction with forensics and threat assessment
        """
        result = IsotopePredictionResult(
            anomaly_detected=False,
            confidence=0.0,
            isotope_type="natural_isotope",
            threat_level="safe",
        )

        ratios = isotope_data.get("isotope_ratios", {})
        result.isotope_ratios = ratios

        if self.enable_ml and "ratio_features" in isotope_data:
            ml_result = self._analyze_with_ml(isotope_data["ratio_features"])
            result.isotope_type = ml_result["isotope_type"]
            result.enrichment_level = ml_result["enrichment"]
            result.threat_level = ml_result["threat_level"]
            result.confidence = ml_result["confidence"]

            if result.isotope_type != "natural_isotope":
                result.anomaly_detected = True

        if self.enable_forensics and ratios and self.forensics_analyzer is not None:
            forensics = self.forensics_analyzer.analyze_uranium_signature(ratios)
            result.nuclear_forensics = forensics
            result.enrichment_level = forensics["enrichment_percent"] / 100.0
            result.origin_indicators = forensics["origin_indicators"]

            if forensics["enrichment_category"] != "natural":
                result.anomaly_detected = True

        if self.enable_threat_assessment and self.threat_assessor is not None:
            threat_result = self.threat_assessor.assess_threat(
                result.nuclear_forensics, result.enrichment_level or 0.0
            )
            result.threat_level = threat_result["threat_level"]
            result.regulatory_alerts = threat_result["regulatory_alerts"]

            if threat_result["threat_score"] > 0.5:
                result.anomaly_detected = True

        result.recommendations = self._generate_recommendations(result)

        self.logger.info(
            f"Isotope prediction: {result.isotope_type}, "
            f"threat={result.threat_level}, enrichment={result.enrichment_level}"
        )

        return result

    def _analyze_with_ml(self, features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze isotopes with ML model"""
        if self.ratio_analyzer is None:
            return {
                "isotope_type": "natural_isotope",
                "enrichment": 0.0,
                "threat_level": "safe",
                "confidence": 0.0,
            }

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.ratio_analyzer.eval()
        with torch.no_grad():
            isotope_logits, enrichment, threat_logits = self.ratio_analyzer(features_tensor)

        isotope_probs = torch.softmax(isotope_logits[0], dim=0)
        isotope_idx = torch.argmax(isotope_probs).item()
        isotope_confidence = float(isotope_probs[isotope_idx].item())  # type: ignore[index]

        isotope_types = [e.value for e in IsotopeType]
        detected_type = isotope_types[isotope_idx]  # type: ignore[index]

        enrichment_level = float(enrichment[0].item())

        threat_probs = torch.softmax(threat_logits[0], dim=0)
        threat_idx = torch.argmax(threat_probs).item()

        threat_levels = [e.value for e in ThreatLevel]
        threat_level = threat_levels[threat_idx]  # type: ignore[index]

        return {
            "isotope_type": detected_type,
            "enrichment": enrichment_level,
            "threat_level": threat_level,
            "confidence": isotope_confidence,
        }

    def _generate_recommendations(self, result: IsotopePredictionResult) -> list[str]:
        """Generate recommendations based on isotope analysis"""

        recs = []

        if result.threat_level in ["high_risk", "critical_threat"]:
            recs.append("CRITICAL: Secure material immediately")
            recs.append("Notify radiation protection officer")
            recs.append("Implement emergency protocols")

        if result.enrichment_level and result.enrichment_level > 0.20:
            recs.append("Special nuclear material detected")
            recs.append("Enhanced security measures required")

        if "weapons_grade" in str(result.origin_indicators):
            recs.append("Contact national nuclear security authorities")
            recs.append("Maintain chain of custody documentation")

        if result.contamination_detected:
            recs.append("Initiate decontamination procedures")
            recs.append("Monitor personnel for exposure")

        recs.append("Continue regular radiological monitoring")

        return recs

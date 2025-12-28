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
Intelligence Source Sub-Modules

Specialized processors for each of the 13 intelligence collection disciplines.
Each sub-module provides domain-specific anomaly detection and pattern recognition
tailored to the unique characteristics and collection methodologies of each INT source.

Intelligence Disciplines Covered:
1. OSINT - Open Source Intelligence
2. COMINT - Communications Intelligence
3. HUMINT - Human Intelligence
4. GEOINT - Geospatial Intelligence
5. IM INT - Imagery Intelligence
6. SIGINT - Signals Intelligence
7. ELINT - Electronic Intelligence
8. MASINT - Measurement & Signature Intelligence
9. CYBINT - Cyber Intelligence
10. FININT - Financial Intelligence
11. CRYPTANALYSIS - Code Breaking & Pattern Analysis
12. METEOROLOGICAL - Weather Intelligence
13. TRAFFIC ANALYSIS - Communication Pattern Analysis

⚠️ SIMULATION-BASED: Research/development tool. Operational deployment requires
security clearance, legal authorization, and oversight by qualified intelligence
professionals.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol


class IntelligenceProcessor(Protocol):
    """Protocol for intelligence processors."""

    def analyze(self, data: dict[str, Any]) -> Any:
        """Analyze intelligence data."""
        ...


@dataclass
class OSINTAnalysisResult:
    """Open Source Intelligence analysis result"""

    source_credibility: float
    information_quality: float
    corroboration_level: float
    anomaly_indicators: list[str] = field(default_factory=list)
    sentiment_analysis: dict[str, float] | None = None
    entity_mentions: list[str] = field(default_factory=list)
    temporal_trends: dict[str, Any] | None = None


class OSINTProcessor:
    """
    Open Source Intelligence (OSINT) Processor.

    Analyzes publicly available information from media, social networks, academic
    publications, and open databases for threat indicators and anomalies.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.source_reliability = {
            "mainstream_media": 0.85,
            "social_media": 0.60,
            "academic": 0.90,
            "government_public": 0.95,
            "blogs": 0.55,
            "forums": 0.50,
        }

    def analyze(self, osint_data: dict[str, Any]) -> OSINTAnalysisResult:
        """Analyze open source intelligence data"""
        source_type = osint_data.get("source_type", "unknown")
        credibility = self.source_reliability.get(source_type, 0.50)

        content = osint_data.get("content", "")
        quality = self._assess_information_quality(content, osint_data)

        corroboration = self._check_corroboration(osint_data)

        anomalies = self._detect_anomalies(content, osint_data)

        sentiment = self._analyze_sentiment(content)

        entities = self._extract_entities(content)

        trends = self._analyze_temporal_trends(osint_data)

        return OSINTAnalysisResult(
            source_credibility=credibility,
            information_quality=quality,
            corroboration_level=corroboration,
            anomaly_indicators=anomalies,
            sentiment_analysis=sentiment,
            entity_mentions=entities,
            temporal_trends=trends,
        )

    def _assess_information_quality(self, content: str, data: dict[str, Any]) -> float:
        """Assess quality of information"""
        quality_score = 0.5

        if len(content) > 100:
            quality_score += 0.1

        if data.get("has_citations", False):
            quality_score += 0.2

        if data.get("author_verified", False):
            quality_score += 0.2

        return min(1.0, quality_score)

    def _check_corroboration(self, data: dict[str, Any]) -> float:
        """Check for corroborating sources"""
        num_corroborating = float(data.get("corroborating_sources", 0))
        return min(1.0, num_corroborating / 5.0)

    def _detect_anomalies(self, content: str, data: dict[str, Any]) -> list[str]:
        """Detect anomalous patterns"""
        anomalies = []

        threat_keywords = ["attack", "threat", "weapon", "explosive", "target"]
        for keyword in threat_keywords:
            if keyword.lower() in content.lower():
                anomalies.append(f"threat_keyword_{keyword}")

        if data.get("unusual_posting_time", False):
            anomalies.append("temporal_anomaly")

        if data.get("coordinated_campaign", False):
            anomalies.append("coordinated_information_operation")

        return anomalies[:10]

    def _analyze_sentiment(self, content: str) -> dict[str, float]:
        """Analyze sentiment (simplified)"""
        negative_words = ["threat", "danger", "attack", "crisis", "fear"]
        positive_words = ["peace", "cooperation", "success", "progress"]

        neg_count = sum(1 for word in negative_words if word in content.lower())
        pos_count = sum(1 for word in positive_words if word in content.lower())

        total = neg_count + pos_count + 1

        return {
            "negative": neg_count / total,
            "positive": pos_count / total,
            "neutral": 1.0 / total,
        }

    def _extract_entities(self, content: str) -> list[str]:
        """Extract named entities (simplified)"""
        words = content.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return capitalized[:20]

    def _analyze_temporal_trends(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze temporal posting patterns"""
        return {
            "posting_frequency": data.get("posting_frequency", 0.0),
            "trending": data.get("is_trending", False),
            "velocity": data.get("spread_velocity", 0.0),
        }


@dataclass
class COMINTAnalysisResult:
    """Communications Intelligence analysis result"""

    intercept_quality: float
    communication_pattern_score: float = 0.0
    encryption_detected: bool = False
    anomaly_indicators: list[str] = field(default_factory=list)
    participant_analysis: dict[str, Any] | None = None
    frequency_patterns: dict[str, float] | None = None
    anomaly_score: float = 0.0
    communication_type: str | None = None


class COMINTProcessor:
    """
    Communications Intelligence (COMINT) Processor.

    Analyzes intercepted communications (phone, email, radio) for patterns, anomalies,
    and threat indicators.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, comint_data: dict[str, Any]) -> COMINTAnalysisResult:
        """Analyze communications intelligence"""
        quality = self._assess_intercept_quality(comint_data)

        pattern_score = self._analyze_communication_patterns(comint_data)

        encryption = comint_data.get("encryption_detected", False)

        anomalies = self._detect_anomalies(comint_data)

        participants = self._analyze_participants(comint_data)

        frequencies = self._analyze_frequency_patterns(comint_data)

        return COMINTAnalysisResult(
            intercept_quality=quality,
            communication_pattern_score=pattern_score,
            encryption_detected=encryption,
            anomaly_indicators=anomalies,
            participant_analysis=participants,
            frequency_patterns=frequencies,
            anomaly_score=pattern_score,
            communication_type=comint_data.get("communication_type"),
        )

    def _assess_intercept_quality(self, data: dict[str, Any]) -> float:
        """Assess quality of intercept"""
        signal_strength = float(data.get("signal_strength", 0.5))
        clarity = float(data.get("audio_clarity", 0.5))
        completeness = float(data.get("message_completeness", 0.5))

        return (signal_strength + clarity + completeness) / 3.0

    def _analyze_communication_patterns(self, data: dict[str, Any]) -> float:
        """Analyze communication patterns"""
        frequency = float(data.get("communication_frequency", 0.0))
        regularity = float(data.get("temporal_regularity", 0.5))

        return min(1.0, (frequency * regularity) / 10.0)

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect communication anomalies"""
        anomalies = []

        if data.get("burst_communication", False):
            anomalies.append("communication_burst")

        if data.get("unusual_hours", False):
            anomalies.append("off_hours_communication")

        if data.get("code_words_detected", False):
            anomalies.append("potential_code_usage")

        if data.get("encryption_change", False):
            anomalies.append("encryption_protocol_change")

        return anomalies

    def _analyze_participants(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze communication participants"""
        return {
            "num_participants": data.get("num_participants", 2),
            "known_entities": data.get("known_entities", []),
            "network_centrality": data.get("network_centrality", 0.0),
        }

    def _analyze_frequency_patterns(self, data: dict[str, Any]) -> dict[str, float]:
        """Analyze frequency usage patterns"""
        return {
            "primary_frequency": data.get("primary_frequency_mhz", 0.0),
            "frequency_hopping": data.get("frequency_hopping_detected", 0.0),
            "signal_strength_avg": data.get("signal_strength", 0.0),
        }


@dataclass
class HUMINTAnalysisResult:
    """Human Intelligence analysis result"""

    source_reliability: float = 0.0
    information_criticality: float = 0.5
    corroboration_available: bool = False
    anomaly_indicators: list[str] = field(default_factory=list)
    source_motivation_assessment: dict[str, float] | None = None
    access_level: str = "unknown"
    source_reliability_score: float = 0.0
    information_credibility_score: float = 0.0
    report_assessment: str | None = None


class HUMINTProcessor:
    """
    Human Intelligence (HUMINT) Processor.

    Analyzes human source reports, including clandestine sources, interviews,
    and field observations.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.source_ratings = {
            "A": 1.0,  # Completely reliable
            "B": 0.85,  # Usually reliable
            "C": 0.70,  # Fairly reliable
            "D": 0.55,  # Not usually reliable
            "E": 0.40,  # Unreliable
            "F": 0.25,  # Cannot be judged
        }

    def analyze(self, humint_data: dict[str, Any]) -> HUMINTAnalysisResult:
        """Analyze human intelligence"""
        source_rating = humint_data.get("source_rating", "F")
        reliability = self.source_ratings.get(source_rating, 0.40)

        criticality = self._assess_criticality(humint_data)

        corroboration = humint_data.get("corroboration_available", False)

        anomalies = self._detect_anomalies(humint_data)

        motivation = self._assess_source_motivation(humint_data)

        access = humint_data.get("source_access_level", "unknown")

        return HUMINTAnalysisResult(
            source_reliability=reliability,
            information_criticality=criticality,
            corroboration_available=corroboration,
            anomaly_indicators=anomalies,
            source_motivation_assessment=motivation,
            access_level=access,
            source_reliability_score=reliability,
            information_credibility_score=criticality,
            report_assessment=humint_data.get("report_assessment"),
        )

    def _assess_criticality(self, data: dict[str, Any]) -> float:
        """Assess information criticality"""
        timeliness = float(data.get("timeliness_score", 0.5))
        relevance = float(data.get("relevance_score", 0.5))
        uniqueness = float(data.get("uniqueness_score", 0.5))

        return (timeliness + relevance + uniqueness) / 3.0

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect HUMINT anomalies"""
        anomalies = []

        if data.get("source_behavior_change", False):
            anomalies.append("source_behavior_anomaly")

        if data.get("information_inconsistency", False):
            anomalies.append("information_inconsistency")

        if data.get("potential_fabrication", False):
            anomalies.append("potential_fabrication")

        if data.get("coercion_indicators", False):
            anomalies.append("possible_coercion")

        return anomalies

    def _assess_source_motivation(self, data: dict[str, Any]) -> dict[str, float]:
        """Assess source motivation"""
        return {
            "financial": data.get("financial_motivation", 0.0),
            "ideological": data.get("ideological_motivation", 0.0),
            "coercion": data.get("coercion_likelihood", 0.0),
            "ego": data.get("ego_motivation", 0.0),
        }


@dataclass
class GEOINTAnalysisResult:
    """Geospatial Intelligence analysis result"""

    spatial_precision: float
    temporal_relevance: float = 1.0
    activity_density_score: float = 0.0
    anomaly_indicators: list[str] = field(default_factory=list)
    location_context: dict[str, Any] | None = None
    movement_patterns: dict[str, Any] | None = None
    spatial_anomalies: list[str] = field(default_factory=list)


class GEOINTProcessor:
    """
    Geospatial Intelligence (GEOINT) Processor.

    Analyzes geospatial data including terrain analysis, facility identification,
    and movement tracking.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, geoint_data: dict[str, Any]) -> GEOINTAnalysisResult:
        """Analyze geospatial intelligence"""
        precision = geoint_data.get("location_precision_meters", 1000.0)
        spatial_precision = min(1.0, 10.0 / precision)

        age_hours = geoint_data.get("data_age_hours", 24.0)
        temporal_relevance = min(1.0, 24.0 / age_hours)

        activity_density = self._assess_activity_density(geoint_data)

        anomalies = self._detect_anomalies(geoint_data)

        location_context = self._analyze_location_context(geoint_data)

        movement = self._analyze_movement_patterns(geoint_data)

        return GEOINTAnalysisResult(
            spatial_precision=spatial_precision,
            temporal_relevance=temporal_relevance,
            activity_density_score=activity_density,
            anomaly_indicators=anomalies,
            location_context=location_context,
            movement_patterns=movement,
            spatial_anomalies=anomalies,
        )

    def _assess_activity_density(self, data: dict[str, Any]) -> float:
        """Assess activity density in area"""
        num_entities = float(data.get("num_entities_tracked", 0))
        area_km2 = float(data.get("area_km2", 1.0))

        density = num_entities / area_km2
        return min(1.0, density / 100.0)

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect geospatial anomalies"""
        anomalies = []

        if data.get("unusual_concentration", False):
            anomalies.append("unusual_entity_concentration")

        if data.get("unexpected_movement", False):
            anomalies.append("unexpected_movement_pattern")

        if data.get("facility_construction", False):
            anomalies.append("new_facility_construction")

        if data.get("border_activity", False):
            anomalies.append("border_proximity_activity")

        return anomalies

    def _analyze_location_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze location context"""
        return {
            "terrain_type": data.get("terrain_type", "unknown"),
            "urban_density": data.get("urban_density", 0.0),
            "strategic_significance": data.get("strategic_significance", 0.0),
        }

    def _analyze_movement_patterns(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze movement patterns"""
        return {
            "average_speed": data.get("average_speed_kmh", 0.0),
            "direction_consistency": data.get("direction_consistency", 0.0),
            "stop_frequency": data.get("stop_frequency", 0.0),
        }


@dataclass
class IMINTAnalysisResult:
    """Imagery Intelligence analysis result"""

    image_quality: float
    resolution_score: float
    analysis_confidence: float
    anomaly_indicators: list[str] = field(default_factory=list)
    detected_objects: list[str] = field(default_factory=list)
    change_detection: dict[str, Any] | None = None


class IMINTProcessor:
    """
    Imagery Intelligence (IMINT) Processor.

    Analyzes satellite and aerial imagery for facility identification, change detection,
    and activity monitoring.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, imint_data: dict[str, Any]) -> IMINTAnalysisResult:
        """Analyze imagery intelligence"""
        quality = self._assess_image_quality(imint_data)

        resolution = imint_data.get("resolution_meters", 10.0)
        resolution_score = min(1.0, 1.0 / resolution)

        confidence = imint_data.get("analysis_confidence", 0.7)

        anomalies = self._detect_anomalies(imint_data)

        objects = imint_data.get("detected_objects", [])

        changes = self._analyze_change_detection(imint_data)

        return IMINTAnalysisResult(
            image_quality=quality,
            resolution_score=resolution_score,
            analysis_confidence=confidence,
            anomaly_indicators=anomalies,
            detected_objects=objects,
            change_detection=changes,
        )

    def _assess_image_quality(self, data: dict[str, Any]) -> float:
        """Assess imagery quality"""
        cloud_cover = float(data.get("cloud_cover_percent", 0.0))
        clarity = 1.0 - (cloud_cover / 100.0)

        lighting = float(data.get("lighting_quality", 0.7))

        return (clarity + lighting) / 2.0

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect imagery anomalies"""
        anomalies = []

        if data.get("new_construction", False):
            anomalies.append("new_construction_detected")

        if data.get("vehicle_concentration", False):
            anomalies.append("unusual_vehicle_concentration")

        if data.get("camouflage_netting", False):
            anomalies.append("camouflage_concealment")

        if data.get("thermal_signature", False):
            anomalies.append("thermal_anomaly")

        return anomalies

    def _analyze_change_detection(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze changes from previous imagery"""
        return {
            "change_detected": data.get("change_detected", False),
            "change_magnitude": data.get("change_magnitude", 0.0),
            "change_type": data.get("change_type", "none"),
        }


@dataclass
class CYBINTAnalysisResult:
    """Cyber Intelligence analysis result"""

    threat_severity: float
    attribution_confidence: float
    ioc_count: int
    anomaly_indicators: list[str] = field(default_factory=list)
    attack_vectors: list[str] = field(default_factory=list)
    ttps_identified: list[str] = field(default_factory=list)


class CYBINTProcessor:
    """
    Cyber Intelligence (CYBINT) Processor.

    Analyzes cyber threat indicators, malware, network anomalies, and attribution.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, cybint_data: dict[str, Any]) -> CYBINTAnalysisResult:
        """Analyze cyber intelligence"""
        severity = cybint_data.get("threat_severity_score", 0.5)

        attribution = cybint_data.get("attribution_confidence", 0.3)

        ioc_count = len(cybint_data.get("indicators_of_compromise", []))

        anomalies = self._detect_anomalies(cybint_data)

        attack_vectors = cybint_data.get("attack_vectors", [])

        ttps = cybint_data.get("ttps", [])

        return CYBINTAnalysisResult(
            threat_severity=severity,
            attribution_confidence=attribution,
            ioc_count=ioc_count,
            anomaly_indicators=anomalies,
            attack_vectors=attack_vectors,
            ttps_identified=ttps,
        )

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect cyber anomalies"""
        anomalies = []

        if data.get("zero_day_suspected", False):
            anomalies.append("potential_zero_day")

        if data.get("apt_indicators", False):
            anomalies.append("advanced_persistent_threat")

        if data.get("lateral_movement", False):
            anomalies.append("lateral_movement_detected")

        if data.get("data_exfiltration", False):
            anomalies.append("data_exfiltration_attempt")

        return anomalies


@dataclass
class FININTAnalysisResult:
    """Financial Intelligence analysis result"""

    transaction_risk_score: float
    pattern_anomaly_score: float = 0.0
    money_laundering_indicators: int = 0
    anomaly_indicators: list[str] = field(default_factory=list)
    entity_network: dict[str, Any] | None = None
    jurisdiction_risks: list[str] = field(default_factory=list)
    risk_score: float = 0.0


class FININTProcessor:
    """
    Financial Intelligence (FININT) Processor.

    Analyzes financial transactions, money laundering indicators, and terrorism financing.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.high_risk_jurisdictions = {"offshore", "sanctioned", "non_cooperative"}

    def analyze(self, finint_data: dict[str, Any]) -> FININTAnalysisResult:
        """Analyze financial intelligence"""
        risk_score = self._calculate_transaction_risk(finint_data)

        pattern_anomaly = self._analyze_transaction_patterns(finint_data)

        ml_indicators = self._count_money_laundering_indicators(finint_data)

        anomalies = self._detect_anomalies(finint_data)

        network = self._analyze_entity_network(finint_data)

        jurisdiction_risks = self._assess_jurisdiction_risks(finint_data)

        return FININTAnalysisResult(
            transaction_risk_score=risk_score,
            pattern_anomaly_score=pattern_anomaly,
            money_laundering_indicators=ml_indicators,
            anomaly_indicators=anomalies,
            entity_network=network,
            jurisdiction_risks=jurisdiction_risks,
            risk_score=risk_score,
        )

    def _calculate_transaction_risk(self, data: dict[str, Any]) -> float:
        """Calculate transaction risk score"""
        amount = float(data.get("transaction_amount", data.get("amount", 0)))
        frequency = float(data.get("transaction_frequency", 1))

        base_risk = min(1.0, (amount / 1000000.0) * (frequency / 100.0))

        structuring = data.get("structuring_pattern", False)
        if structuring:
            base_risk = max(base_risk, 0.7)

        return base_risk

    def _analyze_transaction_patterns(self, data: dict[str, Any]) -> float:
        """Analyze transaction patterns for anomalies"""
        structuring = bool(data.get("structuring_detected", False))
        smurfing = bool(data.get("smurfing_detected", False))
        round_amounts = bool(data.get("round_amounts", False))

        anomaly_count = sum([structuring, smurfing, round_amounts])
        return min(1.0, float(anomaly_count) / 3.0)

    def _count_money_laundering_indicators(self, data: dict[str, Any]) -> int:
        """Count money laundering red flags"""
        indicators = data.get("ml_indicators", [])
        return len(indicators)

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect financial anomalies"""
        anomalies = []

        if data.get("rapid_movement", False):
            anomalies.append("rapid_funds_movement")

        if data.get("shell_company", False):
            anomalies.append("shell_company_involvement")

        if data.get("cash_intensive", False):
            anomalies.append("cash_intensive_business")

        if data.get("trade_based_ml", False):
            anomalies.append("trade_based_money_laundering")

        return anomalies

    def _analyze_entity_network(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze financial entity network"""
        return {
            "network_size": data.get("network_size", 0),
            "hub_entities": data.get("hub_entities", []),
            "cross_border_links": data.get("cross_border_links", 0),
        }

    def _assess_jurisdiction_risks(self, data: dict[str, Any]) -> list[str]:
        """Assess jurisdiction-based risks"""
        jurisdictions = data.get("jurisdictions", [])
        return [j for j in jurisdictions if j.lower() in self.high_risk_jurisdictions]


@dataclass
class SIGINTAnalysisResult:
    """Signals Intelligence analysis result"""

    signal_strength: float
    intercept_confidence: float = 0.7
    decryption_success: bool = False
    anomaly_indicators: list[str] = field(default_factory=list)
    signal_characteristics: dict[str, Any] | None = None
    emitter_identification: str | None = None
    signal_classification: str | None = None


class SIGINTProcessor:
    """
    Signals Intelligence (SIGINT) Processor.

    Analyzes intercepted electronic signals (includes COMINT and ELINT).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, sigint_data: dict[str, Any]) -> SIGINTAnalysisResult:
        """Analyze signals intelligence"""
        strength = sigint_data.get("signal_strength_dbm", -80.0)
        signal_strength = min(1.0, (strength + 100.0) / 40.0)

        confidence = sigint_data.get("intercept_confidence", 0.7)

        decryption = sigint_data.get("decryption_successful", False)

        anomalies = self._detect_anomalies(sigint_data)

        characteristics = self._analyze_signal_characteristics(sigint_data)

        emitter = sigint_data.get("emitter_id")

        return SIGINTAnalysisResult(
            signal_strength=signal_strength,
            intercept_confidence=confidence,
            decryption_success=decryption,
            anomaly_indicators=anomalies,
            signal_characteristics=characteristics,
            emitter_identification=emitter,
            signal_classification=sigint_data.get("signal_classification"),
        )

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect signal anomalies"""
        anomalies = []

        if data.get("frequency_hopping", False):
            anomalies.append("frequency_hopping_detected")

        if data.get("burst_transmission", False):
            anomalies.append("burst_transmission_mode")

        if data.get("spread_spectrum", False):
            anomalies.append("spread_spectrum_detected")

        if data.get("unknown_protocol", False):
            anomalies.append("unknown_protocol")

        return anomalies

    def _analyze_signal_characteristics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze signal characteristics"""
        return {
            "frequency_mhz": data.get("frequency_mhz", 0.0),
            "bandwidth_khz": data.get("bandwidth_khz", 0.0),
            "modulation": data.get("modulation_type", "unknown"),
            "pulse_rate": data.get("pulse_rate_hz", 0.0),
        }


@dataclass
class ELINTAnalysisResult:
    """Electronic Intelligence analysis result"""

    radar_type_confidence: float
    emitter_threat_level: float = 0.5
    tracking_detected: bool = False
    anomaly_indicators: list[str] = field(default_factory=list)
    radar_parameters: dict[str, Any] | None = None
    targeting_assessment: dict[str, Any] | None = None
    emitter_classification: str | None = None


class ELINTProcessor:
    """
    Electronic Intelligence (ELINT) Processor.

    Analyzes non-communication electronic emissions (radar, sensors).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.radar_threat_levels = {
            "search": 0.3,
            "tracking": 0.7,
            "targeting": 0.9,
            "fire_control": 1.0,
        }

    def analyze(self, elint_data: dict[str, Any]) -> ELINTAnalysisResult:
        """Analyze electronic intelligence"""
        radar_confidence = elint_data.get("radar_type_confidence", 0.5)

        radar_type = elint_data.get("radar_type", "unknown")
        threat_level = self.radar_threat_levels.get(radar_type, 0.5)

        tracking = elint_data.get("tracking_detected", False)

        anomalies = self._detect_anomalies(elint_data)

        parameters = self._analyze_radar_parameters(elint_data)

        targeting = self._assess_targeting(elint_data)

        return ELINTAnalysisResult(
            radar_type_confidence=radar_confidence,
            emitter_threat_level=threat_level,
            tracking_detected=tracking,
            anomaly_indicators=anomalies,
            radar_parameters=parameters,
            targeting_assessment=targeting,
            emitter_classification=elint_data.get("emitter_classification"),
        )

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect ELINT anomalies"""
        anomalies = []

        if data.get("lock_on_detected", False):
            anomalies.append("radar_lock_on")

        if data.get("jamming_attempt", False):
            anomalies.append("electronic_jamming")

        if data.get("new_emitter", False):
            anomalies.append("unknown_emitter_type")

        if data.get("multi_mode", False):
            anomalies.append("multi_mode_radar")

        return anomalies

    def _analyze_radar_parameters(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze radar parameters"""
        return {
            "prf_hz": data.get("pulse_repetition_frequency", 0.0),
            "scan_rate_rpm": data.get("scan_rate", 0.0),
            "power_output_kw": data.get("power_output", 0.0),
        }

    def _assess_targeting(self, data: dict[str, Any]) -> dict[str, Any]:
        """Assess targeting indications"""
        return {
            "targeting_mode": data.get("targeting_mode", False),
            "illumination_detected": data.get("illumination", False),
            "fire_control_active": data.get("fire_control", False),
        }


@dataclass
class MASINTAnalysisResult:
    """Measurement & Signature Intelligence analysis result"""

    signature_match_confidence: float
    technical_specificity: float = 0.6
    collection_quality: float = 0.7
    anomaly_indicators: list[str] = field(default_factory=list)
    signature_type: str = "unknown"
    measurements: dict[str, float] | None = None
    signature_classification: str | None = None


class MASINTProcessor:
    """
    Measurement & Signature Intelligence (MASINT) Processor.

    Analyzes technical signatures (acoustic, seismic, chemical, radiation, thermal).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, masint_data: dict[str, Any]) -> MASINTAnalysisResult:
        """Analyze MASINT data"""
        match_confidence = masint_data.get("signature_match_confidence", 0.5)

        specificity = masint_data.get("technical_specificity", 0.6)

        quality = masint_data.get("collection_quality", 0.7)

        anomalies = self._detect_anomalies(masint_data)

        sig_type = masint_data.get("signature_type", "unknown")

        measurements = self._extract_measurements(masint_data)

        return MASINTAnalysisResult(
            signature_match_confidence=match_confidence,
            technical_specificity=specificity,
            collection_quality=quality,
            anomaly_indicators=anomalies,
            signature_type=sig_type,
            measurements=measurements,
            signature_classification=masint_data.get("signature_classification"),
        )

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect MASINT anomalies"""
        anomalies = []

        if data.get("radiation_spike", False):
            anomalies.append("radiation_anomaly")

        if data.get("seismic_event", False):
            anomalies.append("seismic_signature")

        if data.get("chemical_trace", False):
            anomalies.append("chemical_detection")

        if data.get("thermal_anomaly", False):
            anomalies.append("thermal_signature")

        return anomalies

    def _extract_measurements(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract technical measurements"""
        return {
            "amplitude": data.get("amplitude", 0.0),
            "frequency_hz": data.get("frequency", 0.0),
            "intensity": data.get("intensity", 0.0),
            "concentration": data.get("concentration_ppm", 0.0),
        }


@dataclass
class CryptanalysisResult:
    """Cryptanalysis result"""

    decryption_confidence: float
    algorithm_identification: str | None
    pattern_analysis_score: float
    anomaly_indicators: list[str] = field(default_factory=list)
    key_characteristics: dict[str, Any] | None = None
    plaintext_recovered: bool = False


class CryptanalysisProcessor:
    """
    Cryptanalysis Processor.

    Analyzes encrypted communications for patterns, vulnerabilities, and potential decryption.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, crypto_data: dict[str, Any]) -> CryptanalysisResult:
        """Analyze cryptographic patterns"""
        confidence = crypto_data.get("decryption_confidence", 0.1)

        algorithm = crypto_data.get("identified_algorithm")

        pattern_score = self._analyze_patterns(crypto_data)

        anomalies = self._detect_anomalies(crypto_data)

        key_chars = self._analyze_key_characteristics(crypto_data)

        recovered = crypto_data.get("plaintext_recovered", False)

        return CryptanalysisResult(
            decryption_confidence=confidence,
            algorithm_identification=algorithm,
            pattern_analysis_score=pattern_score,
            anomaly_indicators=anomalies,
            key_characteristics=key_chars,
            plaintext_recovered=recovered,
        )

    def _analyze_patterns(self, data: dict[str, Any]) -> float:
        """Analyze cryptographic patterns"""
        entropy = float(data.get("entropy_score", 0.5))
        repetition = data.get("repetition_detected", False)

        pattern_score = entropy
        if repetition:
            pattern_score -= 0.2

        return max(0.0, min(1.0, pattern_score))

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect cryptographic anomalies"""
        anomalies = []

        if data.get("weak_key", False):
            anomalies.append("weak_key_detected")

        if data.get("protocol_downgrade", False):
            anomalies.append("protocol_downgrade_attack")

        if data.get("replay_pattern", False):
            anomalies.append("replay_attack_pattern")

        if data.get("side_channel", False):
            anomalies.append("side_channel_vulnerability")

        return anomalies

    def _analyze_key_characteristics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Analyze encryption key characteristics"""
        return {
            "key_length_bits": data.get("key_length", 0),
            "key_rotation_detected": data.get("key_rotation", False),
            "perfect_forward_secrecy": data.get("pfs", False),
        }


@dataclass
class MeteorologicalIntelResult:
    """Meteorological Intelligence result"""

    weather_impact_score: float
    operational_feasibility: float
    forecast_confidence: float
    anomaly_indicators: list[str] = field(default_factory=list)
    conditions: dict[str, Any] | None = None
    windows_identified: list[dict[str, Any]] = field(default_factory=list)


class MeteorologicalProcessor:
    """
    Meteorological Intelligence Processor.

    Analyzes weather and atmospheric conditions for operational planning.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, meteo_data: dict[str, Any]) -> MeteorologicalIntelResult:
        """Analyze meteorological intelligence"""
        impact_score = self._assess_weather_impact(meteo_data)

        feasibility = self._assess_operational_feasibility(meteo_data)

        confidence = meteo_data.get("forecast_confidence", 0.7)

        anomalies = self._detect_anomalies(meteo_data)

        conditions = self._extract_conditions(meteo_data)

        windows = meteo_data.get("operational_windows", [])

        return MeteorologicalIntelResult(
            weather_impact_score=impact_score,
            operational_feasibility=feasibility,
            forecast_confidence=confidence,
            anomaly_indicators=anomalies,
            conditions=conditions,
            windows_identified=windows,
        )

    def _assess_weather_impact(self, data: dict[str, Any]) -> float:
        """Assess weather impact on operations"""
        visibility_km = float(data.get("visibility_km", 10.0))
        wind_speed = float(data.get("wind_speed_kmh", 0.0))
        precipitation = float(data.get("precipitation_mm", 0.0))

        impact = (
            (visibility_km / 20.0) * 0.4
            + (1.0 - wind_speed / 50.0) * 0.3
            + (1.0 - precipitation / 10.0) * 0.3
        )

        return min(1.0, max(0.0, impact))

    def _assess_operational_feasibility(self, data: dict[str, Any]) -> float:
        """Assess operational feasibility based on weather"""
        return self._assess_weather_impact(data) * float(data.get("forecast_confidence", 0.7))

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect meteorological anomalies"""
        anomalies = []

        if data.get("severe_weather", False):
            anomalies.append("severe_weather_event")

        if data.get("rapid_change", False):
            anomalies.append("rapid_weather_change")

        if data.get("unusual_pattern", False):
            anomalies.append("unusual_weather_pattern")

        return anomalies

    def _extract_conditions(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract current conditions"""
        return {
            "temperature_c": data.get("temperature_c", 20.0),
            "humidity_percent": data.get("humidity", 50.0),
            "cloud_cover_percent": data.get("cloud_cover", 50.0),
            "visibility_km": data.get("visibility_km", 10.0),
        }


@dataclass
class TrafficAnalysisResult:
    """Traffic Analysis result"""

    pattern_recognition_score: float
    network_structure_score: float
    temporal_correlation: float
    anomaly_indicators: list[str] = field(default_factory=list)
    communication_graph: dict[str, Any] | None = None
    key_nodes_identified: list[str] = field(default_factory=list)


class TrafficAnalysisProcessor:
    """
    Traffic Analysis Processor.

    Analyzes communication patterns without accessing content (metadata analysis).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def analyze(self, traffic_data: dict[str, Any]) -> TrafficAnalysisResult:
        """Analyze traffic patterns"""
        pattern_score = self._analyze_patterns(traffic_data)

        structure_score = self._analyze_network_structure(traffic_data)

        temporal = self._analyze_temporal_correlation(traffic_data)

        anomalies = self._detect_anomalies(traffic_data)

        graph = self._build_communication_graph(traffic_data)

        key_nodes = self._identify_key_nodes(traffic_data)

        return TrafficAnalysisResult(
            pattern_recognition_score=pattern_score,
            network_structure_score=structure_score,
            temporal_correlation=temporal,
            anomaly_indicators=anomalies,
            communication_graph=graph,
            key_nodes_identified=key_nodes,
        )

    def _analyze_patterns(self, data: dict[str, Any]) -> float:
        """Analyze communication patterns"""
        frequency = float(data.get("communication_frequency", 0.0))
        regularity = float(data.get("temporal_regularity", 0.5))

        return min(1.0, (frequency * regularity) / 10.0)

    def _analyze_network_structure(self, data: dict[str, Any]) -> float:
        """Analyze network structure"""
        centrality = float(data.get("network_centrality", 0.5))
        clustering = float(data.get("clustering_coefficient", 0.5))

        return (centrality + clustering) / 2.0

    def _analyze_temporal_correlation(self, data: dict[str, Any]) -> float:
        """Analyze temporal correlations"""
        return float(data.get("temporal_correlation_score", 0.5))

    def _detect_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect traffic anomalies"""
        anomalies = []

        if data.get("burst_activity", False):
            anomalies.append("burst_communication_pattern")

        if data.get("new_connections", False):
            anomalies.append("new_network_connections")

        if data.get("hub_emergence", False):
            anomalies.append("new_hub_node_emerged")

        if data.get("synchronized_activity", False):
            anomalies.append("synchronized_network_activity")

        return anomalies

    def _build_communication_graph(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build communication network graph"""
        return {
            "num_nodes": data.get("num_nodes", 0),
            "num_edges": data.get("num_edges", 0),
            "density": data.get("network_density", 0.0),
        }

    def _identify_key_nodes(self, data: dict[str, Any]) -> list[str]:
        """Identify key network nodes"""
        hub_nodes = data.get("hub_nodes", [])
        return list(hub_nodes) if hub_nodes else []


class IntelligenceSourceRegistry:
    """
    Registry for all intelligence source processors.

    Provides unified access to all 13 INT source sub-modules.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.processors: dict[str, IntelligenceProcessor] = {
            "osint": OSINTProcessor(),
            "comint": COMINTProcessor(),
            "humint": HUMINTProcessor(),
            "geoint": GEOINTProcessor(),
            "imint": IMINTProcessor(),
            "sigint": SIGINTProcessor(),
            "elint": ELINTProcessor(),
            "masint": MASINTProcessor(),
            "cybint": CYBINTProcessor(),
            "finint": FININTProcessor(),
            "cryptanalysis": CryptanalysisProcessor(),
            "meteorological": MeteorologicalProcessor(),
            "traffic": TrafficAnalysisProcessor(),
        }

        self.logger.info(
            f"Intelligence Source Registry initialized with {len(self.processors)} processors"
        )

    def process(self, int_type: str, data: dict[str, Any]) -> Any:
        """Process intelligence data with appropriate processor"""
        if int_type not in self.processors:
            self.logger.warning(f"No processor found for INT type: {int_type}")
            return None

        processor = self.processors[int_type]
        result = processor.analyze(data)

        return result

    def get_available_processors(self) -> list[str]:
        """Get list of available INT processors"""
        return list(self.processors.keys())


def create_int_source_registry() -> IntelligenceSourceRegistry:
    """
    Create intelligence source registry with all processors.

    Returns:
        Configured IntelligenceSourceRegistry
    """
    return IntelligenceSourceRegistry()

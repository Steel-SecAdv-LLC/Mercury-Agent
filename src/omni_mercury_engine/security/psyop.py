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
Psychological Operations (PSYOP) Analysis Module

Provides AI-powered analysis capabilities for psychological operations,
information warfare detection, and influence campaign analysis.

Based on principles from:
- U.S. Army Special Warfare Center and School (SWCS) PSYWAR doctrine
- National Counterintelligence concepts
- Information Operations frameworks

Key Capabilities:
1. Target Audience Analysis (TAA)
2. Influence Campaign Detection
3. Narrative/Message Analysis
4. Cognitive Vulnerability Assessment
5. Information Environment Monitoring
6. Counter-PSYOP Support

⚠️ SIMULATION-BASED: Research/development tool for authorized defensive
applications. Operational deployment requires proper legal authorization,
oversight, and adherence to applicable laws and regulations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class PSYOPCategory(Enum):
    """PSYOP category classifications."""

    STRATEGIC = "strategic"  # National-level objectives
    OPERATIONAL = "operational"  # Theater/campaign-level
    TACTICAL = "tactical"  # Immediate, localized objectives
    CONSOLIDATION = "consolidation"  # Post-conflict stabilization


class InfluenceVector(Enum):
    """Influence operation vectors."""

    SOCIAL_MEDIA = "social_media"
    TRADITIONAL_MEDIA = "traditional_media"
    INTERPERSONAL = "interpersonal"
    CULTURAL = "cultural"
    ECONOMIC = "economic"
    RELIGIOUS = "religious"
    POLITICAL = "political"
    MILITARY = "military"


class CognitiveBias(Enum):
    """Cognitive biases that can be exploited or detected."""

    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING = "anchoring"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    BANDWAGON_EFFECT = "bandwagon_effect"
    IN_GROUP_BIAS = "in_group_bias"
    AUTHORITY_BIAS = "authority_bias"
    FEAR_APPEAL = "fear_appeal"
    SOCIAL_PROOF = "social_proof"
    SCARCITY = "scarcity"
    RECIPROCITY = "reciprocity"


class NarrativeType(Enum):
    """Types of narratives in information operations."""

    DISINFORMATION = "disinformation"  # Intentionally false
    MISINFORMATION = "misinformation"  # Unintentionally false
    MALINFORMATION = "malinformation"  # True but harmful context
    PROPAGANDA = "propaganda"  # Biased promotion
    COUNTER_NARRATIVE = "counter_narrative"  # Opposing narrative
    NEUTRAL = "neutral"  # Factual reporting


@dataclass
class TargetAudienceProfile:
    """Profile of a target audience for PSYOP analysis.

    Attributes:
        audience_id: Unique identifier for the audience segment.
        demographics: Demographic characteristics.
        psychographics: Psychological characteristics and values.
        vulnerabilities: Identified cognitive/emotional vulnerabilities.
        influence_vectors: Most effective influence channels.
        sentiment_baseline: Baseline sentiment measurements.
        key_influencers: Identified opinion leaders.
    """

    audience_id: str
    demographics: dict[str, Any] = field(default_factory=dict)
    psychographics: dict[str, Any] = field(default_factory=dict)
    vulnerabilities: list[CognitiveBias] = field(default_factory=list)
    influence_vectors: list[InfluenceVector] = field(default_factory=list)
    sentiment_baseline: dict[str, float] = field(default_factory=dict)
    key_influencers: list[str] = field(default_factory=list)
    receptivity_score: float = 0.5


@dataclass
class NarrativeAnalysis:
    """Analysis of a narrative or message.

    Attributes:
        narrative_id: Unique identifier.
        content_summary: Summary of narrative content.
        narrative_type: Classification of narrative.
        themes: Key themes identified.
        emotional_appeals: Emotional triggers used.
        biases_exploited: Cognitive biases targeted.
        credibility_score: Assessment of credibility.
        reach_estimate: Estimated audience reach.
        amplification_indicators: Signs of artificial amplification.
    """

    narrative_id: str
    content_summary: str
    narrative_type: NarrativeType
    themes: list[str] = field(default_factory=list)
    emotional_appeals: list[str] = field(default_factory=list)
    biases_exploited: list[CognitiveBias] = field(default_factory=list)
    credibility_score: float = 0.5
    reach_estimate: int = 0
    amplification_indicators: list[str] = field(default_factory=list)
    source_attribution: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InfluenceCampaignDetection:
    """Detection results for influence campaign analysis.

    Attributes:
        campaign_id: Unique identifier.
        detection_confidence: Confidence level (0-1).
        category: PSYOP category classification.
        vectors: Identified influence vectors.
        narratives: Associated narratives.
        target_audiences: Identified target audiences.
        coordination_indicators: Signs of coordinated activity.
        attribution_assessment: Attribution analysis.
        threat_level: Overall threat assessment.
    """

    campaign_id: str
    detection_confidence: float
    category: PSYOPCategory
    vectors: list[InfluenceVector] = field(default_factory=list)
    narratives: list[str] = field(default_factory=list)
    target_audiences: list[str] = field(default_factory=list)
    coordination_indicators: list[str] = field(default_factory=list)
    attribution_assessment: dict[str, Any] = field(default_factory=dict)
    threat_level: str = "low"
    anomaly_indicators: list[str] = field(default_factory=list)


@dataclass
class InformationEnvironmentState:
    """State of the information environment.

    Attributes:
        environment_id: Unique identifier.
        timestamp: Timestamp of assessment.
        dominant_narratives: Currently dominant narratives.
        sentiment_distribution: Distribution of sentiment.
        polarization_index: Measure of polarization.
        influence_operations_active: Detected active operations.
        information_integrity_score: Overall integrity assessment.
    """

    environment_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    dominant_narratives: list[str] = field(default_factory=list)
    sentiment_distribution: dict[str, float] = field(default_factory=dict)
    polarization_index: float = 0.0
    influence_operations_active: list[str] = field(default_factory=list)
    information_integrity_score: float = 0.8


class PSYOPAnalyzer:
    """
    Psychological Operations Analysis Engine.

    Provides comprehensive PSYOP analysis capabilities including:
    - Target audience analysis
    - Influence campaign detection
    - Narrative analysis
    - Information environment monitoring

    Example:
        >>> analyzer = PSYOPAnalyzer()
        >>> result = analyzer.analyze_narrative({
        ...     "content": "Sample message content",
        ...     "source": "social_media",
        ...     "engagement_metrics": {"shares": 1000, "comments": 500}
        ... })
    """

    def __init__(self) -> None:
        """Initialize PSYOP analyzer."""
        self.logger = logging.getLogger(__name__)

        # Emotional trigger keywords
        self.fear_triggers = [
            "threat",
            "danger",
            "crisis",
            "urgent",
            "warning",
            "attack",
            "invasion",
            "collapse",
            "disaster",
        ]
        self.anger_triggers = [
            "outrage",
            "betrayal",
            "injustice",
            "corrupt",
            "enemy",
            "traitor",
            "exploit",
            "abuse",
            "oppression",
        ]
        self.hope_triggers = [
            "victory",
            "freedom",
            "justice",
            "unity",
            "strength",
            "future",
            "change",
            "progress",
            "liberation",
        ]

        # Propaganda technique indicators
        self.propaganda_indicators = {
            "loaded_language": [
                "radical",
                "extremist",
                "regime",
                "puppet",
                "freedom fighter",
                "terrorist",
                "patriot",
            ],
            "false_dilemma": ["either", "only option", "must choose", "no alternative"],
            "appeal_to_authority": [
                "experts say",
                "studies show",
                "scientists agree",
                "officials confirm",
            ],
            "bandwagon": [
                "everyone knows",
                "the people demand",
                "massive support",
                "overwhelming majority",
            ],
            "transfer": ["god-given", "constitutional", "patriotic duty", "national honor"],
        }

        # Coordination indicators
        self.coordination_patterns = [
            "identical_messaging",
            "synchronized_timing",
            "bot_network_behavior",
            "cross_platform_amplification",
            "artificial_trending",
            "sockpuppet_accounts",
        ]

        self.logger.info("PSYOP Analyzer initialized")

    def analyze_narrative(self, narrative_data: dict[str, Any]) -> NarrativeAnalysis:
        """
        Analyze a narrative or message for PSYOP indicators.

        Args:
            narrative_data: Dictionary containing narrative information:
                - content: Text content of the message
                - source: Source platform/channel
                - engagement_metrics: Engagement data (optional)
                - metadata: Additional metadata (optional)

        Returns:
            NarrativeAnalysis with detailed breakdown.
        """
        content = narrative_data.get("content", "")
        source = narrative_data.get("source", "unknown")
        metrics = narrative_data.get("engagement_metrics", {})

        # Detect narrative type
        narrative_type = self._classify_narrative_type(content, narrative_data)

        # Identify themes
        themes = self._extract_themes(content)

        # Detect emotional appeals
        emotional_appeals = self._detect_emotional_appeals(content)

        # Identify exploited biases
        biases = self._detect_exploited_biases(content)

        # Assess credibility
        credibility = self._assess_credibility(narrative_data)

        # Detect amplification
        amplification = self._detect_amplification_indicators(metrics, narrative_data)

        # Generate narrative ID
        narrative_id = f"nar_{hash(content[:100])}"

        return NarrativeAnalysis(
            narrative_id=narrative_id,
            content_summary=content[:200] if len(content) > 200 else content,
            narrative_type=narrative_type,
            themes=themes,
            emotional_appeals=emotional_appeals,
            biases_exploited=biases,
            credibility_score=credibility,
            reach_estimate=metrics.get("reach", metrics.get("shares", 0) * 10),
            amplification_indicators=amplification,
            source_attribution=source,
        )

    def _classify_narrative_type(self, content: str, data: dict[str, Any]) -> NarrativeType:
        """Classify the type of narrative."""
        content_lower = content.lower()

        # Check for disinformation indicators
        disinfo_score = 0

        # Check propaganda techniques
        for indicators in self.propaganda_indicators.values():
            for indicator in indicators:
                if indicator.lower() in content_lower:
                    disinfo_score += 1

        # Check source credibility
        source_credibility = data.get("source_credibility", 0.5)
        if source_credibility < 0.3:
            disinfo_score += 2

        # Check for factual claims without evidence
        if "reportedly" in content_lower or "sources say" in content_lower:
            if not data.get("has_citations", False):
                disinfo_score += 1

        # Classify based on score
        if disinfo_score >= 4:
            return NarrativeType.DISINFORMATION
        elif disinfo_score >= 2:
            return NarrativeType.PROPAGANDA
        elif disinfo_score >= 1:
            return NarrativeType.MISINFORMATION
        else:
            return NarrativeType.NEUTRAL

    def _extract_themes(self, content: str) -> list[str]:
        """Extract key themes from content."""
        themes = []
        content_lower = content.lower()

        theme_keywords = {
            "national_security": ["security", "defense", "military", "threat"],
            "economic": ["economy", "jobs", "trade", "inflation", "prices"],
            "political": ["government", "election", "democracy", "freedom"],
            "social": ["society", "community", "rights", "equality"],
            "health": ["health", "pandemic", "vaccine", "medicine"],
            "environmental": ["climate", "environment", "pollution"],
            "religious": ["faith", "religion", "god", "spiritual"],
            "cultural": ["culture", "tradition", "identity", "heritage"],
        }

        for theme, keywords in theme_keywords.items():
            if any(kw in content_lower for kw in keywords):
                themes.append(theme)

        return themes[:5]  # Limit to top 5 themes

    def _detect_emotional_appeals(self, content: str) -> list[str]:
        """Detect emotional appeals in content."""
        appeals = []
        content_lower = content.lower()

        if any(trigger in content_lower for trigger in self.fear_triggers):
            appeals.append("fear")
        if any(trigger in content_lower for trigger in self.anger_triggers):
            appeals.append("anger")
        if any(trigger in content_lower for trigger in self.hope_triggers):
            appeals.append("hope")

        # Check for urgency
        if any(word in content_lower for word in ["now", "immediately", "urgent", "act"]):
            appeals.append("urgency")

        # Check for nostalgia
        if any(word in content_lower for word in ["remember", "used to", "golden age", "past"]):
            appeals.append("nostalgia")

        return appeals

    def _detect_exploited_biases(self, content: str) -> list[CognitiveBias]:
        """Detect cognitive biases being exploited."""
        biases = []
        content_lower = content.lower()

        # Confirmation bias
        if any(
            phrase in content_lower for phrase in ["as we knew", "proves that", "just as expected"]
        ):
            biases.append(CognitiveBias.CONFIRMATION_BIAS)

        # Bandwagon effect
        if any(
            phrase in content_lower for phrase in ["everyone", "the people", "massive", "millions"]
        ):
            biases.append(CognitiveBias.BANDWAGON_EFFECT)

        # Authority bias
        if any(
            phrase in content_lower
            for phrase in ["experts", "scientists", "officials", "authorities"]
        ):
            biases.append(CognitiveBias.AUTHORITY_BIAS)

        # Fear appeal
        if any(trigger in content_lower for trigger in self.fear_triggers):
            biases.append(CognitiveBias.FEAR_APPEAL)

        # Social proof
        if any(
            phrase in content_lower for phrase in ["trending", "viral", "shared by", "liked by"]
        ):
            biases.append(CognitiveBias.SOCIAL_PROOF)

        # Scarcity
        if any(
            phrase in content_lower for phrase in ["limited time", "last chance", "running out"]
        ):
            biases.append(CognitiveBias.SCARCITY)

        return biases

    def _assess_credibility(self, data: dict[str, Any]) -> float:
        """Assess credibility of narrative source."""
        credibility = 0.5  # Baseline

        # Source reputation
        source_rep = data.get("source_reputation", 0.5)
        credibility = (credibility + source_rep) / 2

        # Has citations
        if data.get("has_citations", False):
            credibility += 0.15

        # Verified author
        if data.get("author_verified", False):
            credibility += 0.1

        # Multiple corroborating sources
        corroboration = data.get("corroborating_sources", 0)
        credibility += min(0.2, corroboration * 0.05)

        # Deductions
        if data.get("anonymous_source", False):
            credibility -= 0.1
        if data.get("sensationalist_headline", False):
            credibility -= 0.15

        return float(max(0.0, min(1.0, credibility)))

    def _detect_amplification_indicators(
        self, metrics: dict[str, Any], data: dict[str, Any]
    ) -> list[str]:
        """Detect artificial amplification indicators."""
        indicators = []

        shares = metrics.get("shares", 0)
        comments = metrics.get("comments", 0)
        _ = metrics.get("likes", 0)  # Reserved for future use

        # Unnatural engagement ratios
        if shares > 0 and comments > 0:
            ratio = shares / comments
            if ratio > 10:  # Very high share-to-comment ratio
                indicators.append("high_share_to_comment_ratio")

        # Rapid spread
        time_to_viral = data.get("time_to_viral_hours", float("inf"))
        if time_to_viral < 1:
            indicators.append("unusually_rapid_spread")

        # Bot-like patterns
        if data.get("suspected_bot_engagement_percent", 0) > 20:
            indicators.append("high_bot_engagement")

        # Cross-platform coordination
        if data.get("cross_platform_identical", False):
            indicators.append("cross_platform_coordination")

        # Suspicious account patterns
        if data.get("new_account_amplification", 0) > 30:
            indicators.append("new_account_amplification")

        return indicators

    def analyze_target_audience(self, audience_data: dict[str, Any]) -> TargetAudienceProfile:
        """
        Analyze a target audience for PSYOP susceptibility.

        Args:
            audience_data: Dictionary containing audience information:
                - segment_id: Audience segment identifier
                - demographics: Demographic data
                - behavioral_data: Behavioral patterns
                - media_consumption: Media consumption habits

        Returns:
            TargetAudienceProfile with analysis.
        """
        segment_id = audience_data.get("segment_id", "unknown")
        demographics = audience_data.get("demographics", {})
        behavioral = audience_data.get("behavioral_data", {})
        media = audience_data.get("media_consumption", {})

        # Identify vulnerabilities
        vulnerabilities = self._assess_cognitive_vulnerabilities(demographics, behavioral)

        # Determine effective influence vectors
        vectors = self._determine_influence_vectors(media, demographics)

        # Calculate receptivity score
        receptivity = self._calculate_receptivity(demographics, behavioral, media)

        # Identify key influencers
        influencers = audience_data.get("identified_influencers", [])

        # Baseline sentiment
        sentiment = {
            "positive": behavioral.get("positive_sentiment", 0.3),
            "negative": behavioral.get("negative_sentiment", 0.3),
            "neutral": behavioral.get("neutral_sentiment", 0.4),
        }

        return TargetAudienceProfile(
            audience_id=segment_id,
            demographics=demographics,
            psychographics=behavioral.get("psychographics", {}),
            vulnerabilities=vulnerabilities,
            influence_vectors=vectors,
            sentiment_baseline=sentiment,
            key_influencers=influencers[:10],
            receptivity_score=receptivity,
        )

    def _assess_cognitive_vulnerabilities(
        self, demographics: dict[str, Any], behavioral: dict[str, Any]
    ) -> list[CognitiveBias]:
        """Assess cognitive vulnerabilities of an audience."""
        vulnerabilities = []

        # Age-based vulnerabilities
        avg_age = demographics.get("average_age", 40)
        if avg_age > 60:
            vulnerabilities.append(CognitiveBias.AUTHORITY_BIAS)
        if avg_age < 25:
            vulnerabilities.append(CognitiveBias.BANDWAGON_EFFECT)
            vulnerabilities.append(CognitiveBias.SOCIAL_PROOF)

        # Education level
        education = demographics.get("education_level", "medium")
        if education == "low":
            vulnerabilities.append(CognitiveBias.AUTHORITY_BIAS)

        # Political polarization
        polarization = behavioral.get("political_polarization", 0.5)
        if polarization > 0.7:
            vulnerabilities.append(CognitiveBias.CONFIRMATION_BIAS)
            vulnerabilities.append(CognitiveBias.IN_GROUP_BIAS)

        # Economic anxiety
        if behavioral.get("economic_anxiety", 0.5) > 0.6:
            vulnerabilities.append(CognitiveBias.FEAR_APPEAL)
            vulnerabilities.append(CognitiveBias.SCARCITY)

        return list(set(vulnerabilities))  # Remove duplicates

    def _determine_influence_vectors(
        self, media: dict[str, Any], demographics: dict[str, Any]
    ) -> list[InfluenceVector]:
        """Determine effective influence vectors for audience."""
        vectors = []

        # Social media usage
        if media.get("social_media_hours_daily", 0) > 2:
            vectors.append(InfluenceVector.SOCIAL_MEDIA)

        # Traditional media consumption
        if media.get("tv_news_hours_daily", 0) > 1:
            vectors.append(InfluenceVector.TRADITIONAL_MEDIA)

        # Religious affiliation
        if demographics.get("religious_affiliation", False):
            vectors.append(InfluenceVector.RELIGIOUS)

        # Community engagement
        if media.get("community_participation", 0) > 0.5:
            vectors.append(InfluenceVector.INTERPERSONAL)

        # Political engagement
        if media.get("political_engagement", 0) > 0.5:
            vectors.append(InfluenceVector.POLITICAL)

        return vectors

    def _calculate_receptivity(
        self, demographics: dict[str, Any], behavioral: dict[str, Any], media: dict[str, Any]
    ) -> float:
        """Calculate receptivity score for influence operations."""
        receptivity = 0.5  # Baseline

        # High media consumption increases receptivity
        media_hours = media.get("total_media_hours_daily", 3)
        receptivity += min(0.2, media_hours * 0.03)

        # Critical thinking reduces receptivity
        critical_thinking = behavioral.get("critical_thinking_score", 0.5)
        receptivity -= critical_thinking * 0.2

        # Trust in institutions
        institutional_trust = behavioral.get("institutional_trust", 0.5)
        # Low trust can increase receptivity to alternative narratives
        if institutional_trust < 0.3:
            receptivity += 0.15

        # Information literacy
        info_literacy = behavioral.get("information_literacy", 0.5)
        receptivity -= info_literacy * 0.15

        return float(max(0.0, min(1.0, receptivity)))

    def detect_influence_campaign(
        self, campaign_data: dict[str, Any]
    ) -> InfluenceCampaignDetection:
        """
        Detect and analyze a potential influence campaign.

        Args:
            campaign_data: Dictionary containing:
                - messages: List of related messages/narratives
                - accounts: Associated accounts
                - timing_data: Temporal patterns
                - network_data: Network/relationship data

        Returns:
            InfluenceCampaignDetection with analysis results.
        """
        messages = campaign_data.get("messages", [])
        accounts = campaign_data.get("accounts", [])
        timing = campaign_data.get("timing_data", {})
        network = campaign_data.get("network_data", {})

        # Calculate detection confidence
        confidence = self._calculate_campaign_confidence(messages, accounts, timing, network)

        # Classify campaign category
        category = self._classify_campaign_category(campaign_data)

        # Identify coordination indicators
        coordination = self._detect_coordination(timing, network, accounts)

        # Determine influence vectors
        vectors = self._identify_campaign_vectors(campaign_data)

        # Assess threat level
        threat_level = self._assess_threat_level(confidence, len(messages), network)

        # Attribution assessment
        attribution = self._assess_attribution(campaign_data)

        # Anomaly indicators
        anomalies = self._detect_campaign_anomalies(campaign_data)

        campaign_id = f"camp_{hash(str(messages[:3]) if messages else 'unknown')}"

        return InfluenceCampaignDetection(
            campaign_id=campaign_id,
            detection_confidence=confidence,
            category=category,
            vectors=vectors,
            narratives=[
                m.get("id", str(i)) if isinstance(m, dict) else str(m)[:50]
                for i, m in enumerate(messages[:10])
            ],
            target_audiences=campaign_data.get("target_audiences", []),
            coordination_indicators=coordination,
            attribution_assessment=attribution,
            threat_level=threat_level,
            anomaly_indicators=anomalies,
        )

    def _calculate_campaign_confidence(
        self,
        messages: list[Any],
        accounts: list[Any],
        timing: dict[str, Any],
        network: dict[str, Any],
    ) -> float:
        """Calculate confidence that this is a coordinated campaign."""
        confidence = 0.0

        # Message similarity
        if len(messages) > 1:
            similarity = timing.get("message_similarity", 0.5)
            confidence += similarity * 0.25

        # Account coordination
        if len(accounts) > 5:
            account_correlation = network.get("account_correlation", 0.3)
            confidence += account_correlation * 0.25

        # Timing synchronization
        timing_sync = timing.get("synchronization_score", 0.3)
        confidence += timing_sync * 0.25

        # Network structure
        network_centrality = network.get("centrality_concentration", 0.3)
        confidence += network_centrality * 0.25

        return float(min(1.0, confidence))

    def _classify_campaign_category(self, data: dict[str, Any]) -> PSYOPCategory:
        """Classify the PSYOP category of a campaign."""
        scope = data.get("scope", "unknown")
        objectives = data.get("objectives", [])

        if "national_policy" in objectives or scope == "national":
            return PSYOPCategory.STRATEGIC
        elif "military_operations" in objectives or scope == "theater":
            return PSYOPCategory.OPERATIONAL
        elif "local_influence" in objectives or scope == "local":
            return PSYOPCategory.TACTICAL
        else:
            return PSYOPCategory.TACTICAL

    def _detect_coordination(
        self, timing: dict[str, Any], network: dict[str, Any], accounts: list[Any]
    ) -> list[str]:
        """Detect coordination indicators."""
        indicators = []

        # Synchronized timing
        if timing.get("synchronization_score", 0) > 0.7:
            indicators.append("synchronized_timing")

        # Identical messaging
        if timing.get("message_similarity", 0) > 0.9:
            indicators.append("identical_messaging")

        # Network clustering
        if network.get("clustering_coefficient", 0) > 0.8:
            indicators.append("high_network_clustering")

        # Bot indicators
        bot_percentage = sum(1 for a in accounts if a.get("is_bot", False)) / max(len(accounts), 1)
        if bot_percentage > 0.3:
            indicators.append("bot_network_behavior")

        # Cross-platform presence
        platforms = {a.get("platform") for a in accounts if a.get("platform")}
        if len(platforms) > 2:
            indicators.append("cross_platform_amplification")

        return indicators

    def _identify_campaign_vectors(self, data: dict[str, Any]) -> list[InfluenceVector]:
        """Identify influence vectors used in campaign."""
        vectors = []

        platforms = data.get("platforms", [])

        if any(p in platforms for p in ["twitter", "facebook", "telegram", "tiktok"]):
            vectors.append(InfluenceVector.SOCIAL_MEDIA)
        if any(p in platforms for p in ["tv", "radio", "newspaper"]):
            vectors.append(InfluenceVector.TRADITIONAL_MEDIA)

        themes = data.get("themes", [])
        if "political" in themes:
            vectors.append(InfluenceVector.POLITICAL)
        if "economic" in themes:
            vectors.append(InfluenceVector.ECONOMIC)
        if "religious" in themes:
            vectors.append(InfluenceVector.RELIGIOUS)

        return vectors

    def _assess_threat_level(
        self, confidence: float, message_count: int, network: dict[str, Any]
    ) -> str:
        """Assess overall threat level of campaign."""
        score = confidence * 0.4

        # Scale with message volume
        if message_count > 1000:
            score += 0.3
        elif message_count > 100:
            score += 0.2
        elif message_count > 10:
            score += 0.1

        # Network reach
        reach = network.get("estimated_reach", 0)
        if reach > 1000000:
            score += 0.3
        elif reach > 100000:
            score += 0.2
        elif reach > 10000:
            score += 0.1

        if score > 0.7:
            return "critical"
        elif score > 0.5:
            return "high"
        elif score > 0.3:
            return "medium"
        else:
            return "low"

    def _assess_attribution(self, data: dict[str, Any]) -> dict[str, Any]:
        """Assess attribution of campaign."""
        return {
            "suspected_origin": data.get("suspected_origin", "unknown"),
            "confidence": data.get("attribution_confidence", 0.3),
            "indicators": data.get("attribution_indicators", []),
            "known_actor_match": data.get("known_actor_match"),
        }

    def _detect_campaign_anomalies(self, data: dict[str, Any]) -> list[str]:
        """Detect anomalies in campaign patterns."""
        anomalies = []

        timing = data.get("timing_data", {})
        network = data.get("network_data", {})

        # Unusual posting times
        if timing.get("off_hours_percentage", 0) > 0.5:
            anomalies.append("unusual_posting_times")

        # Rapid account creation
        if network.get("new_account_percentage", 0) > 0.4:
            anomalies.append("rapid_account_creation")

        # Unusual engagement patterns
        if timing.get("engagement_anomaly_score", 0) > 0.7:
            anomalies.append("unusual_engagement_patterns")

        # Geographic inconsistencies
        if network.get("geo_inconsistency_score", 0) > 0.6:
            anomalies.append("geographic_inconsistencies")

        return anomalies

    def assess_information_environment(
        self, environment_data: dict[str, Any]
    ) -> InformationEnvironmentState:
        """
        Assess the current state of an information environment.

        Args:
            environment_data: Dictionary containing:
                - region: Geographic/topical region
                - active_narratives: Currently circulating narratives
                - sentiment_data: Sentiment measurements
                - detected_operations: Known active operations

        Returns:
            InformationEnvironmentState assessment.
        """
        region = environment_data.get("region", "global")
        narratives = environment_data.get("active_narratives", [])
        sentiment = environment_data.get("sentiment_data", {})
        operations = environment_data.get("detected_operations", [])

        # Calculate polarization index
        polarization = self._calculate_polarization(sentiment, narratives)

        # Assess information integrity
        integrity = self._assess_information_integrity(narratives, operations)

        # Identify dominant narratives
        dominant = self._identify_dominant_narratives(narratives)

        env_id = f"env_{region}_{datetime.now().strftime('%Y%m%d')}"

        return InformationEnvironmentState(
            environment_id=env_id,
            dominant_narratives=dominant,
            sentiment_distribution=sentiment,
            polarization_index=polarization,
            influence_operations_active=[op.get("id", str(i)) for i, op in enumerate(operations)],
            information_integrity_score=integrity,
        )

    def _calculate_polarization(self, sentiment: dict[str, Any], narratives: list[Any]) -> float:
        """Calculate polarization index."""
        # Sentiment extremity - positive/negative used for future enhancements
        _ = sentiment.get("positive", 0.33)
        _ = sentiment.get("negative", 0.33)
        neutral = sentiment.get("neutral", 0.34)

        # Higher polarization when less neutral sentiment
        sentiment_polarization = 1 - (neutral * 2)

        # Narrative opposition
        opposing_narratives = sum(
            1 for n in narratives if n.get("stance") in ["strongly_oppose", "strongly_support"]
        ) / max(len(narratives), 1)

        return float((sentiment_polarization + opposing_narratives) / 2)

    def _assess_information_integrity(self, narratives: list[Any], operations: list[Any]) -> float:
        """Assess overall information integrity."""
        integrity = 0.8  # Baseline

        # Reduce for each detected operation
        integrity -= len(operations) * 0.1

        # Check narrative quality
        disinfo_count = sum(
            1 for n in narratives if n.get("type") in ["disinformation", "propaganda"]
        )
        integrity -= (disinfo_count / max(len(narratives), 1)) * 0.3

        return max(0.1, min(1.0, integrity))

    def _identify_dominant_narratives(self, narratives: list[Any]) -> list[str]:
        """Identify dominant narratives by reach/engagement."""
        sorted_narratives = sorted(
            narratives, key=lambda n: n.get("reach", 0) + n.get("engagement", 0), reverse=True
        )
        return [n.get("title", f"narrative_{i}") for i, n in enumerate(sorted_narratives[:5])]

    def extract_features(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Extract PSYOP-relevant features for fusion with other detectors.

        Args:
            data: Input data (numeric array or PSYOP data dict)

        Returns:
            Feature vector for fusion pipeline
        """
        if isinstance(data, dict):
            # Extract features from PSYOP analysis
            features = np.zeros(32, dtype=np.float32)

            # Narrative analysis features
            if "content" in data:
                narrative = self.analyze_narrative(data)
                features[0] = narrative.credibility_score
                features[1] = len(narrative.emotional_appeals) / 5
                features[2] = len(narrative.biases_exploited) / 6
                features[3] = len(narrative.amplification_indicators) / 5
                features[4] = (
                    1.0 if narrative.narrative_type == NarrativeType.DISINFORMATION else 0.0
                )
                features[5] = 1.0 if narrative.narrative_type == NarrativeType.PROPAGANDA else 0.0

            # Audience features
            if "audience" in data:
                audience = self.analyze_target_audience(data["audience"])
                features[10] = audience.receptivity_score
                features[11] = len(audience.vulnerabilities) / 6
                features[12] = len(audience.influence_vectors) / 8

            # Campaign features
            if "campaign" in data:
                campaign = self.detect_influence_campaign(data["campaign"])
                features[20] = campaign.detection_confidence
                features[21] = len(campaign.coordination_indicators) / 6
                threat_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                features[22] = threat_map.get(campaign.threat_level, 0.25)

            return features.reshape(1, -1)

        else:
            # For numeric data, return statistical features
            if data.ndim == 1:
                data = data.reshape(1, -1)

            batch_size = data.shape[0]
            features = np.zeros((batch_size, 32), dtype=np.float32)

            for i in range(batch_size):
                sample = data[i]
                features[i, 0] = np.mean(sample)
                features[i, 1] = np.std(sample)
                features[i, 2] = np.min(sample)
                features[i, 3] = np.max(sample)
                features[i, 4] = np.median(sample)

            return features

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """
        Predict PSYOP activity from input data.

        Args:
            data: Input data for prediction

        Returns:
            Prediction results with anomaly scores
        """
        features = self.extract_features(data)

        # Simple anomaly scoring based on extracted features
        anomaly_scores = np.mean(features[:, :10], axis=1)

        return {
            "anomaly_scores": anomaly_scores.astype(np.float32),
            "psyop_features": features,
            "prediction_type": "psyop_analysis",
        }


# Convenience function for registry integration
def create_psyop_analyzer() -> PSYOPAnalyzer:
    """Create PSYOP analyzer instance for registry integration."""
    return PSYOPAnalyzer()

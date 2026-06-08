# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for PSYOP Analysis module."""

import numpy as np

from omni_mercury_engine.security.psyop import (
    CognitiveBias,
    InfluenceCampaignDetection,
    InfluenceVector,
    InformationEnvironmentState,
    NarrativeAnalysis,
    NarrativeType,
    PSYOPAnalyzer,
    PSYOPCategory,
    TargetAudienceProfile,
    create_psyop_analyzer,
)


class TestPSYOPCategory:
    """Tests for PSYOPCategory enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert PSYOPCategory.STRATEGIC.value == "strategic"
        assert PSYOPCategory.OPERATIONAL.value == "operational"
        assert PSYOPCategory.TACTICAL.value == "tactical"
        assert PSYOPCategory.CONSOLIDATION.value == "consolidation"


class TestInfluenceVector:
    """Tests for InfluenceVector enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert InfluenceVector.SOCIAL_MEDIA.value == "social_media"
        assert InfluenceVector.TRADITIONAL_MEDIA.value == "traditional_media"
        assert InfluenceVector.INTERPERSONAL.value == "interpersonal"
        assert InfluenceVector.POLITICAL.value == "political"


class TestCognitiveBias:
    """Tests for CognitiveBias enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert CognitiveBias.CONFIRMATION_BIAS.value == "confirmation_bias"
        assert CognitiveBias.BANDWAGON_EFFECT.value == "bandwagon_effect"
        assert CognitiveBias.AUTHORITY_BIAS.value == "authority_bias"
        assert CognitiveBias.FEAR_APPEAL.value == "fear_appeal"


class TestNarrativeType:
    """Tests for NarrativeType enum."""

    def test_enum_values(self) -> None:
        """Test enum values exist."""
        assert NarrativeType.DISINFORMATION.value == "disinformation"
        assert NarrativeType.MISINFORMATION.value == "misinformation"
        assert NarrativeType.PROPAGANDA.value == "propaganda"
        assert NarrativeType.NEUTRAL.value == "neutral"


class TestTargetAudienceProfile:
    """Tests for TargetAudienceProfile dataclass."""

    def test_init_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        profile = TargetAudienceProfile(audience_id="test_audience")
        assert profile.audience_id == "test_audience"
        assert profile.demographics == {}
        assert profile.vulnerabilities == []
        assert profile.receptivity_score == 0.5

    def test_init_full(self) -> None:
        """Test initialization with all parameters."""
        profile = TargetAudienceProfile(
            audience_id="test_audience",
            demographics={"age": "18-35"},
            psychographics={"values": ["freedom"]},
            vulnerabilities=[CognitiveBias.FEAR_APPEAL],
            influence_vectors=[InfluenceVector.SOCIAL_MEDIA],
            sentiment_baseline={"positive": 0.6},
            key_influencers=["influencer1"],
            receptivity_score=0.8,
        )
        assert profile.audience_id == "test_audience"
        assert len(profile.vulnerabilities) == 1
        assert profile.receptivity_score == 0.8


class TestNarrativeAnalysis:
    """Tests for NarrativeAnalysis dataclass."""

    def test_init_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        analysis = NarrativeAnalysis(
            narrative_id="nar_123",
            content_summary="Test content",
            narrative_type=NarrativeType.NEUTRAL,
        )
        assert analysis.narrative_id == "nar_123"
        assert analysis.narrative_type == NarrativeType.NEUTRAL
        assert analysis.credibility_score == 0.5

    def test_init_full(self) -> None:
        """Test initialization with all parameters."""
        analysis = NarrativeAnalysis(
            narrative_id="nar_123",
            content_summary="Test content",
            narrative_type=NarrativeType.DISINFORMATION,
            themes=["political"],
            emotional_appeals=["fear"],
            biases_exploited=[CognitiveBias.FEAR_APPEAL],
            credibility_score=0.2,
            reach_estimate=10000,
            amplification_indicators=["bot_activity"],
            source_attribution="unknown",
        )
        assert analysis.credibility_score == 0.2
        assert len(analysis.themes) == 1


class TestInfluenceCampaignDetection:
    """Tests for InfluenceCampaignDetection dataclass."""

    def test_init_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        detection = InfluenceCampaignDetection(
            campaign_id="camp_123",
            detection_confidence=0.8,
            category=PSYOPCategory.STRATEGIC,
        )
        assert detection.campaign_id == "camp_123"
        assert detection.detection_confidence == 0.8
        assert detection.threat_level == "low"


class TestInformationEnvironmentState:
    """Tests for InformationEnvironmentState dataclass."""

    def test_init_minimal(self) -> None:
        """Test initialization with minimal parameters."""
        state = InformationEnvironmentState(environment_id="env_123")
        assert state.environment_id == "env_123"
        assert state.polarization_index == 0.0
        assert state.information_integrity_score == 0.8


class TestPSYOPAnalyzer:
    """Tests for PSYOPAnalyzer class."""

    def test_init(self) -> None:
        """Test initialization."""
        analyzer = PSYOPAnalyzer()
        assert len(analyzer.fear_triggers) > 0
        assert len(analyzer.anger_triggers) > 0
        assert len(analyzer.hope_triggers) > 0
        assert len(analyzer.propaganda_indicators) > 0

    def test_analyze_narrative_neutral(self) -> None:
        """Test narrative analysis with neutral content."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "The weather today is sunny with clear skies.",
            "source": "weather_service",
            "engagement_metrics": {"shares": 10, "comments": 5},
        }
        result = analyzer.analyze_narrative(narrative_data)
        assert isinstance(result, NarrativeAnalysis)
        assert result.narrative_type == NarrativeType.NEUTRAL

    def test_analyze_narrative_with_fear(self) -> None:
        """Test narrative analysis with fear triggers."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "URGENT WARNING: A dangerous threat is imminent. Crisis looms.",
            "source": "unknown",
            "engagement_metrics": {"shares": 1000, "comments": 50},
        }
        result = analyzer.analyze_narrative(narrative_data)
        assert "fear" in result.emotional_appeals

    def test_analyze_narrative_with_propaganda(self) -> None:
        """Test narrative analysis with propaganda indicators."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "The radical extremist regime must be stopped. Everyone knows this.",
            "source": "unknown",
            "source_credibility": 0.2,
            "engagement_metrics": {"shares": 5000, "comments": 100},
        }
        result = analyzer.analyze_narrative(narrative_data)
        assert result.narrative_type in [NarrativeType.PROPAGANDA, NarrativeType.DISINFORMATION]

    def test_analyze_narrative_with_amplification(self) -> None:
        """Test narrative analysis with amplification indicators."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "Breaking news story.",
            "source": "social_media",
            "engagement_metrics": {"shares": 10000, "comments": 100},
            "time_to_viral_hours": 0.5,
            "suspected_bot_engagement_percent": 30,
        }
        result = analyzer.analyze_narrative(narrative_data)
        assert len(result.amplification_indicators) > 0

    def test_classify_narrative_type(self) -> None:
        """Test narrative type classification."""
        analyzer = PSYOPAnalyzer()

        neutral_type = analyzer._classify_narrative_type(
            "The sun rises in the east.", {"source_credibility": 0.8, "has_citations": True}
        )
        assert neutral_type == NarrativeType.NEUTRAL

    def test_extract_themes(self) -> None:
        """Test theme extraction."""
        analyzer = PSYOPAnalyzer()
        themes = analyzer._extract_themes("The government election affects democracy and freedom.")
        assert "political" in themes

    def test_detect_emotional_appeals(self) -> None:
        """Test emotional appeal detection."""
        analyzer = PSYOPAnalyzer()

        appeals = analyzer._detect_emotional_appeals("This is a dangerous threat!")
        assert "fear" in appeals

        appeals = analyzer._detect_emotional_appeals("Victory and freedom await!")
        assert "hope" in appeals

        appeals = analyzer._detect_emotional_appeals("Act now immediately!")
        assert "urgency" in appeals

    def test_detect_exploited_biases(self) -> None:
        """Test bias detection."""
        analyzer = PSYOPAnalyzer()

        biases = analyzer._detect_exploited_biases("Everyone knows this is true.")
        assert CognitiveBias.BANDWAGON_EFFECT in biases

        biases = analyzer._detect_exploited_biases("Experts and scientists agree.")
        assert CognitiveBias.AUTHORITY_BIAS in biases

    def test_assess_credibility(self) -> None:
        """Test credibility assessment."""
        analyzer = PSYOPAnalyzer()

        high_cred = analyzer._assess_credibility(
            {
                "source_reputation": 0.9,
                "has_citations": True,
                "author_verified": True,
                "corroborating_sources": 3,
            }
        )
        assert high_cred > 0.7

        low_cred = analyzer._assess_credibility(
            {
                "source_reputation": 0.2,
                "anonymous_source": True,
                "sensationalist_headline": True,
            }
        )
        assert low_cred < 0.5

    def test_detect_amplification_indicators(self) -> None:
        """Test amplification indicator detection."""
        analyzer = PSYOPAnalyzer()

        indicators = analyzer._detect_amplification_indicators(
            {"shares": 10000, "comments": 100}, {"time_to_viral_hours": 0.5}
        )
        assert "high_share_to_comment_ratio" in indicators
        assert "unusually_rapid_spread" in indicators

    def test_analyze_target_audience(self) -> None:
        """Test target audience analysis."""
        analyzer = PSYOPAnalyzer()
        audience_data = {
            "segment_id": "test_audience",
            "demographics": {"age_group": "18-35"},
            "media_consumption": {"social_media": 0.8},
            "behavioral_data": {"political_engagement": 0.7},
        }
        result = analyzer.analyze_target_audience(audience_data)
        assert isinstance(result, TargetAudienceProfile)
        assert result.audience_id == "test_audience"

    def test_detect_influence_campaign(self) -> None:
        """Test influence campaign detection."""
        analyzer = PSYOPAnalyzer()
        campaign_data = {
            "campaign_id": "camp_123",
            "narratives": [
                {"content": "Urgent threat warning!", "source": "unknown"},
                {"content": "Everyone knows the truth!", "source": "unknown"},
            ],
            "coordination_signals": {
                "synchronized_posting": True,
                "identical_messaging": True,
            },
            "target_audiences": ["general_public"],
        }
        result = analyzer.detect_influence_campaign(campaign_data)
        assert isinstance(result, InfluenceCampaignDetection)
        # campaign_id is generated from hash of narratives, not from input
        assert result.campaign_id.startswith("camp_")

    def test_assess_information_environment(self) -> None:
        """Test information environment assessment."""
        analyzer = PSYOPAnalyzer()
        environment_data = {
            "environment_id": "env_123",
            "narratives": [
                {"content": "Positive news", "sentiment": 0.8},
                {"content": "Negative news", "sentiment": 0.2},
            ],
            "active_campaigns": ["camp_1"],
        }
        result = analyzer.assess_information_environment(environment_data)
        assert isinstance(result, InformationEnvironmentState)
        # environment_id is generated with date, not from input
        assert result.environment_id.startswith("env_")

    def test_extract_features(self) -> None:
        """Test feature extraction."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "This is a test message with some content.",
            "source": "social_media",
            "engagement_metrics": {"shares": 100, "comments": 50},
        }
        features = analyzer.extract_features(narrative_data)
        # extract_features returns numpy array, not dict
        assert isinstance(features, np.ndarray)
        assert features.shape == (1, 32)  # 32 features, reshaped to (1, -1)

    def test_predict(self) -> None:
        """Test prediction."""
        analyzer = PSYOPAnalyzer()
        narrative_data = {
            "content": "This is a test message.",
            "source": "social_media",
        }
        prediction = analyzer.predict(narrative_data)
        assert isinstance(prediction, dict)


class TestCreatePsyopAnalyzer:
    """Tests for create_psyop_analyzer function."""

    def test_create_analyzer(self) -> None:
        """Test analyzer creation."""
        analyzer = create_psyop_analyzer()
        assert isinstance(analyzer, PSYOPAnalyzer)

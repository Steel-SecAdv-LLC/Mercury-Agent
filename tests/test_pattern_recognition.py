"""
Tests for Anti-Terrorism Pattern Recognition module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

from omni_mercury_engine.security.anti_terrorism.pattern_recognition import (
    TerrorismPatternDetector,
    TerrorismThreatResult,
)


class TestTerrorismThreatResult:
    """Tests for TerrorismThreatResult dataclass."""

    def test_init(self) -> None:
        """Test initialization."""
        result = TerrorismThreatResult(
            threat_detected=True,
            radicalization_stage="indoctrination",
            confidence=0.75,
            threat_indicators=["indicator1", "indicator2"],
            recommended_actions=["action1"],
        )
        assert result.threat_detected
        assert result.radicalization_stage == "indoctrination"
        assert result.confidence == 0.75
        assert len(result.threat_indicators) == 2
        assert len(result.recommended_actions) == 1


class TestTerrorismPatternDetector:
    """Tests for TerrorismPatternDetector class."""

    def test_init_default(self) -> None:
        """Test initialization with default config."""
        detector = TerrorismPatternDetector()
        assert detector.config == {}
        assert len(detector.radicalization_stages) == 5

    def test_init_with_config(self) -> None:
        """Test initialization with custom config."""
        config = {"threshold": 0.5}
        detector = TerrorismPatternDetector(config=config)
        assert detector.config == config

    def test_detect_radicalization_no_data(self) -> None:
        """Test detection with no data."""
        detector = TerrorismPatternDetector()
        result = detector.detect_radicalization()
        assert isinstance(result, TerrorismThreatResult)
        assert not result.threat_detected
        assert result.radicalization_stage == "pre_radicalization"

    def test_detect_radicalization_low_threat(self) -> None:
        """Test detection with low threat score."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.2, "indicators": ["suspicious_activity"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert not result.threat_detected
        assert result.radicalization_stage == "pre_radicalization"

    def test_detect_radicalization_identification_stage(self) -> None:
        """Test detection at identification stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.4, "indicators": ["extremist_content"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert not result.threat_detected
        assert result.radicalization_stage == "identification"

    def test_detect_radicalization_indoctrination_stage(self) -> None:
        """Test detection at indoctrination stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.6, "indicators": ["radicalization_signs"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "indoctrination"

    def test_detect_radicalization_action_planning_stage(self) -> None:
        """Test detection at action planning stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.8, "indicators": ["planning_activity"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "action_planning"

    def test_detect_radicalization_imminent_action_stage(self) -> None:
        """Test detection at imminent action stage."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.95, "indicators": ["imminent_threat"]}
        result = detector.detect_radicalization(osint_data=osint_data)
        assert result.threat_detected
        assert result.radicalization_stage == "imminent_action"

    def test_detect_radicalization_with_comint(self) -> None:
        """Test detection with COMINT data."""
        detector = TerrorismPatternDetector()
        comint_data = {"threat_score": 0.7, "indicators": ["encrypted_comms"]}
        result = detector.detect_radicalization(comint_data=comint_data)
        assert result.threat_detected
        assert "encrypted_comms" in result.threat_indicators

    def test_detect_radicalization_combined_data(self) -> None:
        """Test detection with combined OSINT and COMINT."""
        detector = TerrorismPatternDetector()
        osint_data = {"threat_score": 0.5, "indicators": ["osint_indicator"]}
        comint_data = {"threat_score": 0.8, "indicators": ["comint_indicator"]}
        result = detector.detect_radicalization(osint_data=osint_data, comint_data=comint_data)
        assert result.threat_detected
        assert result.confidence == 0.8  # Max of the two
        assert "osint_indicator" in result.threat_indicators
        assert "comint_indicator" in result.threat_indicators

    def test_classify_radicalization_stage(self) -> None:
        """Test radicalization stage classification."""
        detector = TerrorismPatternDetector()
        assert detector._classify_radicalization_stage(0.1, []) == "pre_radicalization"
        assert detector._classify_radicalization_stage(0.4, []) == "identification"
        assert detector._classify_radicalization_stage(0.6, []) == "indoctrination"
        assert detector._classify_radicalization_stage(0.8, []) == "action_planning"
        assert detector._classify_radicalization_stage(0.95, []) == "imminent_action"

    def test_recommend_actions_imminent(self) -> None:
        """Test action recommendations for imminent threat."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("imminent_action", 0.95)
        assert any("law enforcement" in a.lower() for a in actions)
        assert any("counter-terrorism" in a.lower() for a in actions)

    def test_recommend_actions_planning(self) -> None:
        """Test action recommendations for action planning."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("action_planning", 0.8)
        assert any("monitoring" in a.lower() for a in actions)
        assert any("federal" in a.lower() for a in actions)

    def test_recommend_actions_indoctrination(self) -> None:
        """Test action recommendations for indoctrination."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("indoctrination", 0.6)
        assert any("community" in a.lower() for a in actions)
        assert any("counter-narrative" in a.lower() for a in actions)

    def test_recommend_actions_low_threat(self) -> None:
        """Test action recommendations for low threat."""
        detector = TerrorismPatternDetector()
        actions = detector._recommend_actions("pre_radicalization", 0.2)
        assert any("monitoring" in a.lower() for a in actions)

    def test_indicator_limit(self) -> None:
        """Test that indicators are limited to 10."""
        detector = TerrorismPatternDetector()
        osint_data = {
            "threat_score": 0.6,
            "indicators": [f"indicator_{i}" for i in range(20)],
        }
        result = detector.detect_radicalization(osint_data=osint_data)
        assert len(result.threat_indicators) <= 10

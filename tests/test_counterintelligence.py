"""
Tests for Counterintelligence module.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import numpy as np
import torch

from omni_mercury_engine.security.counterintelligence import (
    OverwatchNexus,
    OverwatchNexusResult,
)
from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionResult


class TestOverwatchNexusResult:
    """Tests for OverwatchNexusResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = OverwatchNexusResult(
            threat_detected=False,
            threat_level="NONE",
            confidence=0.0,
            risk_score=0.0,
        )
        assert result.threat_detected is False
        assert result.ci_threat_type is None
        assert result.medical_interdiction_required is False
        assert result.ethical_compliance == 1.0

    def test_custom_values(self) -> None:
        """Test custom values."""
        result = OverwatchNexusResult(
            threat_detected=True,
            threat_level="SEVERE",
            confidence=0.9,
            risk_score=0.8,
            ci_threat_type="foreign_penetration",
            medical_interdiction_required=True,
            bio_threat_indicators=["pathogen_detected"],
        )
        assert result.threat_detected is True
        assert result.ci_threat_type == "foreign_penetration"
        assert len(result.bio_threat_indicators) == 1


class TestOverwatchNexus:
    """Tests for OverwatchNexus class."""

    def test_init_default(self) -> None:
        """Test default initialization (CI disabled by default in config)."""
        nexus = OverwatchNexus()
        assert nexus.enable_medical_interdiction is True

    def test_init_with_config_ci_enabled(self) -> None:
        """Test initialization with CI enabled."""
        config = {"enable_ci": True, "enable_medical_interdiction": True}
        nexus = OverwatchNexus(config=config)
        assert nexus.enable_ci is True
        assert nexus.enable_medical_interdiction is True

    def test_init_with_config_ci_disabled(self) -> None:
        """Test initialization with config."""
        config = {"enable_ci": False, "enable_medical_interdiction": False}
        nexus = OverwatchNexus(config=config)
        assert nexus.enable_ci is False
        assert nexus.enable_medical_interdiction is False

    def test_proactive_ci_disabled(self) -> None:
        """Test proactive CI when disabled."""
        config = {"enable_ci": False}
        nexus = OverwatchNexus(config=config)
        result = nexus.proactive_ci(np.array([1.0, 2.0, 3.0]))
        assert result.threat_detected is False
        assert result.threat_level == "NONE"
        assert "CI module disabled" in result.recommended_actions

    def test_generate_synthetic_intel(self) -> None:
        """Test synthetic intel generation."""
        nexus = OverwatchNexus()
        data_stream = np.random.randn(100)
        intel = nexus._generate_synthetic_intel(data_stream)
        assert "open_source" in intel
        assert "signals" in intel
        assert "cyber" in intel

    def test_detect_bifurcation_with_array(self) -> None:
        """Test bifurcation detection with numpy array."""
        nexus = OverwatchNexus()
        data_stream = np.random.randn(100) * 10
        chaos_score = nexus._detect_bifurcation(data_stream)
        assert 0 <= chaos_score <= 1

    def test_detect_bifurcation_without_array(self) -> None:
        """Test bifurcation detection without numpy array."""
        nexus = OverwatchNexus()
        chaos_score = nexus._detect_bifurcation("not an array")
        assert chaos_score == 0.1

    def test_detect_bio_threats(self) -> None:
        """Test bio threat detection."""
        nexus = OverwatchNexus()
        data_stream = np.ones(100) * 3.0
        intel_reports = {"masint": {"threat_score": 0.7}}
        indicators = nexus._detect_bio_threats(data_stream, intel_reports)
        assert len(indicators) > 0

    def test_detect_bio_threats_no_threats(self) -> None:
        """Test bio threat detection with no threats."""
        nexus = OverwatchNexus()
        data_stream = np.ones(100) * 0.5
        intel_reports = {}
        indicators = nexus._detect_bio_threats(data_stream, intel_reports)
        assert isinstance(indicators, list)

    def test_classify_ci_threat_terrorism(self) -> None:
        """Test CI threat classification for terrorism."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="SEVERE",
            confidence=0.9,
            risk_score=0.8,
            threat_indicators=["terrorism_related"],
        )
        threat_type = nexus._classify_ci_threat(fusion_result)
        assert threat_type == "foreign_penetration"

    def test_classify_ci_threat_insider(self) -> None:
        """Test CI threat classification for insider threat."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="MODERATE",
            confidence=0.7,
            risk_score=0.6,
            threat_indicators=["insider_activity"],
        )
        threat_type = nexus._classify_ci_threat(fusion_result)
        assert threat_type == "insider_threat"

    def test_classify_ci_threat_cyber(self) -> None:
        """Test CI threat classification for cyber intrusion."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="MODERATE",
            confidence=0.7,
            risk_score=0.6,
            threat_indicators=["cyber_attack"],
        )
        threat_type = nexus._classify_ci_threat(fusion_result)
        assert threat_type == "cyber_intrusion"

    def test_classify_ci_threat_espionage(self) -> None:
        """Test CI threat classification for espionage."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="CRITICAL",
            confidence=0.9,
            risk_score=0.9,
            threat_indicators=["unknown_activity"],
        )
        threat_type = nexus._classify_ci_threat(fusion_result)
        assert threat_type == "espionage"

    def test_classify_ci_threat_general(self) -> None:
        """Test CI threat classification for general anomaly."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="LOW",
            confidence=0.5,
            risk_score=0.3,
            threat_indicators=["unknown"],
        )
        threat_type = nexus._classify_ci_threat(fusion_result)
        assert threat_type == "general_anomaly"

    def test_identify_survivor_priorities(self) -> None:
        """Test survivor priority identification."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="SEVERE",
            confidence=0.9,
            risk_score=0.8,
            threat_indicators=["general_threat"],
        )
        priorities = nexus._identify_survivor_priorities(fusion_result)
        assert len(priorities) > 0
        assert "Civilian population protection" in priorities

    def test_identify_survivor_priorities_health(self) -> None:
        """Test survivor priority identification for health threats."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="SEVERE",
            confidence=0.9,
            risk_score=0.8,
            threat_indicators=["health_emergency"],
        )
        priorities = nexus._identify_survivor_priorities(fusion_result)
        assert any("Healthcare" in p for p in priorities)

    def test_assess_humanitarian_impact(self) -> None:
        """Test humanitarian impact assessment."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="MODERATE",
            confidence=0.7,
            risk_score=0.5,
            threat_indicators=[],
        )
        impact = nexus._assess_humanitarian_impact(fusion_result, [])
        assert "lives_at_risk" in impact
        assert "economic_impact_usd" in impact

    def test_assess_humanitarian_impact_with_bio_threats(self) -> None:
        """Test humanitarian impact with bio threats."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=True,
            threat_level="SEVERE",
            confidence=0.9,
            risk_score=0.8,
            threat_indicators=[],
        )
        bio_indicators = ["pathogen_detected"]
        impact = nexus._assess_humanitarian_impact(fusion_result, bio_indicators)
        assert impact["lives_at_risk"] > 0
        assert len(impact["vulnerable_populations"]) > 0

    def test_compute_purity_invariant(self) -> None:
        """Test purity invariant computation."""
        nexus = OverwatchNexus()
        fusion_result = IntelligenceFusionResult(
            threat_detected=False,
            threat_level="LOW",
            confidence=0.5,
            risk_score=0.2,
            threat_indicators=[],
        )
        purity = nexus._compute_purity_invariant(fusion_result)
        assert purity > 0

    def test_extract_features_numpy(self) -> None:
        """Test feature extraction with numpy array."""
        nexus = OverwatchNexus()
        data = np.random.randn(128)
        features = nexus.extract_features(data)
        assert isinstance(features, torch.Tensor)
        assert features.shape == (128,)

    def test_extract_features_non_array(self) -> None:
        """Test feature extraction with non-array data."""
        nexus = OverwatchNexus()
        features = nexus.extract_features("not an array")
        assert isinstance(features, torch.Tensor)
        assert features.shape == (128,)

    def test_predict_ci_disabled(self) -> None:
        """Test predict method with CI disabled."""
        config = {"enable_ci": False}
        nexus = OverwatchNexus(config=config)
        data = np.random.randn(100)
        result = nexus.predict(data)
        assert "anomaly_scores" in result
        assert "is_anomaly" in result
        assert "threat_level" in result
        assert result["model_type"] == "overwatch_nexus_ci"

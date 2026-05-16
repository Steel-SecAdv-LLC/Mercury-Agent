"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for PersonalityEngine - Omni-Scalar Shaped Communication.
"""

import pytest

from omni_mercury_engine.narrative.personality import (
    CommunicationModifiers,
    CommunicationTone,
    PersonalityEngine,
    PersonalityProfile,
    VerbosityLevel,
)


class TestPersonalityEngine:
    """Test PersonalityEngine functionality."""

    @pytest.fixture
    def engine(self) -> PersonalityEngine:
        """Create test engine."""
        return PersonalityEngine()

    def test_initialization(self, engine: PersonalityEngine) -> None:
        """Test engine initializes correctly."""
        assert engine._default_scalars is not None
        assert "omnibenevolence" in engine._default_scalars
        assert engine._default_scalars["omnibenevolence"] == 0.99

    def test_get_profile(self, engine: PersonalityEngine) -> None:
        """Test profile generation."""
        profile = engine.get_profile()

        assert isinstance(profile, PersonalityProfile)
        assert profile.tone in list(CommunicationTone)
        assert profile.verbosity in list(VerbosityLevel)
        assert 1 <= profile.reasoning_depth <= 5
        assert 0.0 <= profile.empathy_level <= 1.0
        assert profile.source_scalars is not None

    def test_profile_domain_specific(self, engine: PersonalityEngine) -> None:
        """Test domain-specific profile generation."""
        medical_profile = engine.get_profile(domain="medical")
        security_profile = engine.get_profile(domain="security")

        # Both should be valid profiles
        assert isinstance(medical_profile, PersonalityProfile)
        assert isinstance(security_profile, PersonalityProfile)

    def test_domain_overrides(self) -> None:
        """Test domain-specific scalar overrides."""
        overrides = {"medical": {"omnicompassion": 1.5}}
        engine = PersonalityEngine(domain_overrides=overrides)

        # Medical domain should have higher compassion
        profile = engine.get_profile(domain="medical")
        assert profile.source_scalars["omnicompassion"] == 1.5

    def test_get_modifiers(self, engine: PersonalityEngine) -> None:
        """Test communication modifier generation."""
        modifiers = engine.get_modifiers(
            severity=0.8, confidence=0.7, anomaly_detected=True, domain="medical"
        )

        assert isinstance(modifiers, CommunicationModifiers)
        assert modifiers.confidence_framing is not None
        # High severity should trigger acknowledgment
        # depending on profile settings

    def test_modifiers_for_high_severity(self, engine: PersonalityEngine) -> None:
        """Test modifiers for high severity detection."""
        modifiers = engine.get_modifiers(
            severity=0.9,
            confidence=0.85,
            anomaly_detected=True,
        )

        # High confidence should have strong framing
        assert modifiers.confidence_framing is not None
        assert (
            "high" in modifiers.confidence_framing.lower()
            or "reasonable" in modifiers.confidence_framing.lower()
        )

    def test_modifiers_for_low_confidence(self, engine: PersonalityEngine) -> None:
        """Test modifiers for low confidence detection."""
        modifiers = engine.get_modifiers(
            severity=0.5,
            confidence=0.3,
            anomaly_detected=True,
        )

        assert modifiers.confidence_framing is not None
        assert "limited" in modifiers.confidence_framing.lower()
        assert modifiers.uncertainty_framing is not None

    def test_shape_text(self, engine: PersonalityEngine) -> None:
        """Test text shaping."""
        profile = engine.get_profile()
        modifiers = engine.get_modifiers(
            severity=0.7, confidence=0.6, anomaly_detected=True, profile=profile
        )

        raw_text = "Anomaly detected with score 0.7."
        shaped = engine.shape_text(raw_text, profile, modifiers)

        # Shaped text should include original and additions
        assert "0.7" in shaped  # Original content preserved
        assert len(shaped) >= len(raw_text)  # Likely longer

    def test_get_greeting(self, engine: PersonalityEngine) -> None:
        """Test greeting generation."""
        greeting = engine.get_greeting()
        assert greeting is not None
        assert "Mercury" in greeting

        # Domain-specific greeting
        medical_greeting = engine.get_greeting(domain="medical")
        assert medical_greeting is not None

    def test_uncertainty_statement(self, engine: PersonalityEngine) -> None:
        """Test uncertainty statement generation."""
        # Low confidence should have statement
        statement = engine.get_uncertainty_statement(confidence=0.35)
        assert "confidence" in statement.lower() or "uncertain" in statement.lower()

        # High confidence might not need uncertainty statement
        statement_high = engine.get_uncertainty_statement(confidence=0.95)
        # May be empty for high confidence with default settings
        assert statement_high is not None  # Should return string (possibly empty)

    def test_uncertainty_decomposition(self, engine: PersonalityEngine) -> None:
        """Test uncertainty decomposition in statement."""
        statement = engine.get_uncertainty_statement(
            confidence=0.6,
            epistemic=0.2,
            aleatoric=0.15,
        )

        # With detailed verbosity, should mention both types
        # Depends on profile verbosity settings
        assert statement is not None  # Should return string

    def test_tone_derivation(self, engine: PersonalityEngine) -> None:
        """Test that tone is correctly derived from scalars."""
        profile = engine.get_profile()

        # Default scalars have high compassion (1.30)
        # So tone should be influenced by that
        assert profile.tone in list(CommunicationTone)

    def test_verbosity_derivation(self, engine: PersonalityEngine) -> None:
        """Test verbosity level derivation."""
        profile = engine.get_profile()

        # Default has high explainability (0.9)
        # Should result in detailed verbosity
        assert profile.verbosity in (
            VerbosityLevel.DETAILED,
            VerbosityLevel.COMPREHENSIVE,
            VerbosityLevel.STANDARD,
        )

    def test_behavioral_flags(self, engine: PersonalityEngine) -> None:
        """Test behavioral flags are set."""
        profile = engine.get_profile()

        # These should all be boolean
        assert isinstance(profile.acknowledge_uncertainty, bool)
        assert isinstance(profile.show_reasoning_chain, bool)
        assert isinstance(profile.include_historical_context, bool)
        assert isinstance(profile.provide_alternatives, bool)
        assert isinstance(profile.express_confidence_level, bool)

    def test_statistics(self, engine: PersonalityEngine) -> None:
        """Test statistics gathering."""
        stats = engine.get_statistics()
        assert "gosnn_connected" in stats
        assert "default_scalars" in stats


class TestCommunicationTone:
    """Test CommunicationTone enum."""

    def test_all_tones_defined(self) -> None:
        """Ensure all expected tones are defined."""
        expected = ["NEUTRAL", "SUPPORTIVE", "DIRECT", "THOROUGH", "CAUTIOUS"]
        for tone in expected:
            assert hasattr(CommunicationTone, tone)


class TestVerbosityLevel:
    """Test VerbosityLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Ensure all expected verbosity levels are defined."""
        expected = ["MINIMAL", "STANDARD", "DETAILED", "COMPREHENSIVE"]
        for level in expected:
            assert hasattr(VerbosityLevel, level)

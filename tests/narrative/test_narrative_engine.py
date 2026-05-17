"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for NarrativeEngine - Truth-Dense Communication Synthesis.
"""

from typing import Any

import pytest

from omni_mercury_engine.narrative.engine import (
    ConfidenceLevel,
    NarrativeEngine,
    NarrativeResult,
    NarrativeStyle,
    ReasoningChainNarrative,
)


class TestNarrativeEngine:
    """Test NarrativeEngine functionality."""

    @pytest.fixture
    def engine(self) -> NarrativeEngine:
        """Create test engine."""
        return NarrativeEngine()

    @pytest.fixture
    def sample_detection_result(self) -> dict[str, Any]:
        """Sample detection result for testing."""
        return {
            "anomaly_detected": True,
            "anomaly_score": 0.87,
            "severity": 0.75,
            "confidence": 0.82,
            "is_reliable": True,
            "reasoning_chain": [
                {
                    "rule": "statistical_threshold",
                    "conclusion": "z-score exceeded",
                    "confidence": 0.9,
                },
                {"rule": "temporal_pattern", "conclusion": "unusual timing", "confidence": 0.75},
                {
                    "rule": "correlation_check",
                    "conclusion": "correlated anomaly",
                    "confidence": 0.85,
                },
            ],
            "causal_factors": ["sensor_drift", "environmental_change"],
            "recommendations": ["Review sensor calibration", "Check environmental logs"],
            "warnings": [],
        }

    def test_engine_initialization(self, engine: NarrativeEngine) -> None:
        """Test engine initializes correctly."""
        assert engine.default_style == NarrativeStyle.EXPLANATORY
        assert engine.max_reasoning_steps == 10
        assert not engine.use_llm_enhancement

    def test_synthesize_with_anomaly(
        self, engine: NarrativeEngine, sample_detection_result: dict[str, Any]
    ) -> None:
        """Test narrative synthesis with anomaly detection."""
        result = engine.synthesize(sample_detection_result, domain="medical")

        assert isinstance(result, NarrativeResult)
        assert result.summary is not None
        assert "anomaly" in result.summary.lower() or "detected" in result.summary.lower()
        assert result.confidence_score == 0.82
        assert result.confidence_level in list(ConfidenceLevel)

    def test_synthesize_without_anomaly(self, engine: NarrativeEngine) -> None:
        """Test narrative synthesis without anomaly."""
        detection = {
            "anomaly_detected": False,
            "anomaly_score": 0.15,
            "severity": 0.1,
            "confidence": 0.95,
            "is_reliable": True,
        }

        result = engine.synthesize(detection)

        assert "no anomaly" in result.summary.lower() or "normal" in result.summary.lower()
        assert result.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_reasoning_chain_verbalization(
        self, engine: NarrativeEngine, sample_detection_result: dict[str, Any]
    ) -> None:
        """Test that reasoning chain is properly verbalized."""
        result = engine.synthesize(sample_detection_result)

        assert result.reasoning_chain is not None
        assert isinstance(result.reasoning_chain, ReasoningChainNarrative)
        assert len(result.reasoning_chain.steps) > 0
        assert result.reasoning_chain.final_conclusion is not None

    def test_uncertainty_disclosure(self, engine: NarrativeEngine) -> None:
        """Test explicit uncertainty disclosure."""
        low_confidence = {
            "anomaly_detected": True,
            "anomaly_score": 0.6,
            "severity": 0.5,
            "confidence": 0.35,
            "is_reliable": False,
        }

        result = engine.synthesize(low_confidence)

        assert (
            "uncertain" in result.uncertainty_disclosure.lower()
            or "confidence" in result.uncertainty_disclosure.lower()
        )
        assert (
            "unreliable" in result.uncertainty_disclosure.lower()
            or "warning" in result.uncertainty_disclosure.lower()
        )

    def test_confidence_levels(self, engine: NarrativeEngine) -> None:
        """Test confidence level classification."""
        test_cases = [
            (0.95, True, ConfidenceLevel.VERY_HIGH),
            (0.75, True, ConfidenceLevel.HIGH),
            (0.60, True, ConfidenceLevel.MODERATE),
            (0.40, True, ConfidenceLevel.LOW),
            (0.20, True, ConfidenceLevel.VERY_LOW),
            (0.85, False, ConfidenceLevel.MODERATE),  # Unreliable downgrades
        ]

        for conf, reliable, expected in test_cases:
            detection = {
                "anomaly_detected": True,
                "anomaly_score": 0.5,
                "severity": 0.5,
                "confidence": conf,
                "is_reliable": reliable,
            }
            result = engine.synthesize(detection)
            assert (
                result.confidence_level == expected
            ), f"Failed for conf={conf}, reliable={reliable}"

    def test_recommendations_proportional_to_confidence(self, engine: NarrativeEngine) -> None:
        """Test that recommendations are proportional to confidence."""
        high_conf = {
            "anomaly_detected": True,
            "anomaly_score": 0.9,
            "severity": 0.9,
            "confidence": 0.95,
            "is_reliable": True,
            "recommendations": ["Take immediate action"],
        }

        low_conf = {
            "anomaly_detected": True,
            "anomaly_score": 0.6,
            "severity": 0.5,
            "confidence": 0.30,
            "is_reliable": True,
        }

        high_result = engine.synthesize(high_conf)
        low_result = engine.synthesize(low_conf)

        # High confidence should have actionable recommendations
        assert high_result.recommendations is not None

        # Low confidence should have cautionary recommendations
        assert any(
            "verif" in r.lower() or "caution" in r.lower() or "additional" in r.lower()
            for r in low_result.recommendations
        )

    def test_urgency_levels(self, engine: NarrativeEngine) -> None:
        """Test urgency determination."""
        critical = {
            "anomaly_detected": True,
            "anomaly_score": 0.95,
            "severity": 0.95,
            "confidence": 0.9,
        }
        moderate = {
            "anomaly_detected": True,
            "anomaly_score": 0.6,
            "severity": 0.5,
            "confidence": 0.7,
        }
        low = {"anomaly_detected": True, "anomaly_score": 0.4, "severity": 0.3, "confidence": 0.5}

        assert engine.synthesize(critical).urgency_level in ("critical", "high")
        assert engine.synthesize(moderate).urgency_level in ("moderate", "high", "low")
        assert engine.synthesize(low).urgency_level in ("low", "informational")

    def test_style_determination(self, engine: NarrativeEngine) -> None:
        """Test automatic style determination."""
        urgent_case = {
            "anomaly_detected": True,
            "anomaly_score": 0.95,
            "severity": 0.95,
            "confidence": 0.9,
            "is_reliable": True,
        }

        result = engine.synthesize(urgent_case)
        assert result.style_used == NarrativeStyle.URGENT

    def test_style_override(self, engine: NarrativeEngine) -> None:
        """Test style can be overridden."""
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.5,
            "severity": 0.5,
            "confidence": 0.7,
        }

        result = engine.synthesize(detection, style_override=NarrativeStyle.CLINICAL)
        assert result.style_used == NarrativeStyle.CLINICAL

    def test_domain_specific_recommendations(self, engine: NarrativeEngine) -> None:
        """Test domain-specific recommendations."""
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.7,
            "severity": 0.6,
            "confidence": 0.8,
            "is_reliable": True,
        }

        medical_result = engine.synthesize(detection, domain="medical")
        security_result = engine.synthesize(detection, domain="security")

        assert any("clinical" in r.lower() for r in medical_result.recommendations)
        assert any(
            "log" in r.lower() or "document" in r.lower() for r in security_result.recommendations
        )

    def test_narrative_result_to_dict(
        self, engine: NarrativeEngine, sample_detection_result: dict[str, Any]
    ) -> None:
        """Test NarrativeResult serialization."""
        result = engine.synthesize(sample_detection_result)
        result_dict = result.to_dict()

        assert "summary" in result_dict
        assert "detailed_explanation" in result_dict
        assert "confidence" in result_dict
        assert result_dict["confidence"]["level"] in [c.value for c in ConfidenceLevel]
        assert "recommendations" in result_dict
        assert "metadata" in result_dict

    def test_generation_time_tracking(
        self, engine: NarrativeEngine, sample_detection_result: dict[str, Any]
    ) -> None:
        """Test that generation time is tracked."""
        result = engine.synthesize(sample_detection_result)
        assert result.generation_time_ms > 0
        assert result.generation_time_ms < 10000  # Should be fast

    def test_scalars_applied_tracking(
        self, engine: NarrativeEngine, sample_detection_result: dict[str, Any]
    ) -> None:
        """Test that applied scalars are tracked."""
        result = engine.synthesize(sample_detection_result)
        assert "omnibenevolence" in result.scalars_applied
        assert "omnitransparency" in result.scalars_applied

    def test_statistics(self, engine: NarrativeEngine) -> None:
        """Test statistics gathering."""
        # Generate a few syntheses
        for _ in range(3):
            engine.synthesize({"anomaly_detected": False, "confidence": 0.9})

        stats = engine.get_statistics()
        assert stats["synthesis_count"] == 3
        assert "default_style" in stats


class TestNarrativeStyle:
    """Test NarrativeStyle enum."""

    def test_all_styles_defined(self) -> None:
        """Ensure all expected styles are defined."""
        expected = ["CLINICAL", "EXPLANATORY", "URGENT", "ANALYTICAL", "SUPPORTIVE"]
        for style_name in expected:
            assert hasattr(NarrativeStyle, style_name)


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Ensure all confidence levels are defined."""
        expected = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]
        for level_name in expected:
            assert hasattr(ConfidenceLevel, level_name)

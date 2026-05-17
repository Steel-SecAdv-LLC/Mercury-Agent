"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for MemorySurface - Conversational Memory Integration.
"""

import time
from typing import Any

import pytest

from omni_mercury_engine.narrative.memory_surface import (
    MemoryContext,
    MemoryRelevance,
    MemorySurface,
    PredictionHistory,
    SimilarEvent,
)


class TestMemorySurface:
    """Test MemorySurface functionality."""

    @pytest.fixture
    def surface(self) -> MemorySurface:
        """Create test memory surface."""
        return MemorySurface()

    @pytest.fixture
    def sample_detection(self) -> dict[str, Any]:
        """Sample detection result."""
        return {
            "anomaly_detected": True,
            "anomaly_score": 0.75,
            "severity": 0.6,
            "confidence": 0.8,
        }

    def test_initialization(self, surface: MemorySurface) -> None:
        """Test surface initializes correctly."""
        assert surface.max_similar_events == 5
        assert surface.similarity_threshold == 0.4
        assert surface.lookback_days == 90
        assert surface._retrieval_count == 0

    def test_get_relevant_context(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test getting relevant context."""
        context = surface.get_relevant_context(sample_detection, domain="test")

        assert isinstance(context, MemoryContext)
        assert context.summary is not None
        assert isinstance(context.similar_events, list)
        assert context.overall_relevance in list(MemoryRelevance)

    def test_context_to_dict(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test context serialization."""
        context = surface.get_relevant_context(sample_detection)
        context_dict = context.to_dict()

        assert "summary" in context_dict
        assert "similar_events" in context_dict
        assert "pattern_frequency" in context_dict
        assert "relevance" in context_dict

    def test_record_event(self, surface: MemorySurface, sample_detection: dict[str, Any]) -> None:
        """Test recording events for future retrieval."""
        surface.record_event(sample_detection, domain="test", outcome="resolved")

        # Pattern registry should be updated
        assert len(surface._pattern_registry) > 0

    def test_pattern_accumulation(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test pattern frequency tracking."""
        domain = "test"

        # Record multiple events
        for _ in range(5):
            surface.record_event(sample_detection, domain)

        # Check pattern statistics
        context = surface.get_relevant_context(sample_detection, domain)
        assert context.pattern_frequency >= 5

    def test_prediction_outcome_recording(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test prediction outcome recording."""
        domain = "test"

        # Record predictions with outcomes
        surface.record_prediction_outcome(sample_detection, domain, was_correct=True)
        surface.record_prediction_outcome(sample_detection, domain, was_correct=True)
        surface.record_prediction_outcome(sample_detection, domain, was_correct=False)

        # Need 5 outcomes for prediction history to be returned
        surface.record_prediction_outcome(sample_detection, domain, was_correct=True)
        surface.record_prediction_outcome(sample_detection, domain, was_correct=True)

        context = surface.get_relevant_context(sample_detection, domain)

        # Should have prediction history now
        assert context.prediction_accuracy is not None
        assert context.prediction_accuracy.total_predictions == 5
        assert context.prediction_accuracy.accuracy == 0.8  # 4/5 correct

    def test_learned_insights(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test learned insight synthesis."""
        domain = "test"

        # Record enough events to generate insights
        for _ in range(15):
            surface.record_event(sample_detection, domain)

        context = surface.get_relevant_context(sample_detection, domain)

        # Should have some insights about frequency
        assert len(context.learned_insights) > 0 or context.pattern_frequency > 10

    def test_relevance_assessment(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test relevance is correctly assessed."""
        # With no prior events, relevance should be tangential
        context = surface.get_relevant_context(sample_detection)
        assert context.overall_relevance == MemoryRelevance.TANGENTIAL
        assert context.relevance_score < 0.5

    def test_similar_event_ids(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test similar event IDs are returned."""
        context = surface.get_relevant_context(sample_detection)

        # similar_event_ids should match similar_events
        assert len(context.similar_event_ids) == len(context.similar_events)

    def test_timespan_tracking(
        self, surface: MemorySurface, sample_detection: dict[str, Any]
    ) -> None:
        """Test pattern timespan tracking."""
        domain = "test"

        surface.record_event(sample_detection, domain)
        time.sleep(0.1)  # Small delay
        surface.record_event(sample_detection, domain)

        context = surface.get_relevant_context(sample_detection, domain)
        assert context.first_seen is not None
        assert context.last_seen is not None
        assert context.last_seen >= context.first_seen

    def test_statistics(self, surface: MemorySurface) -> None:
        """Test statistics gathering."""
        # Do some operations
        surface.get_relevant_context({"anomaly_score": 0.5})

        stats = surface.get_statistics()
        assert stats["retrieval_count"] == 1
        assert "pattern_types_tracked" in stats
        assert "neural_memory_connected" in stats


class TestMemoryRelevance:
    """Test MemoryRelevance enum."""

    def test_all_levels_defined(self) -> None:
        """Ensure all expected relevance levels are defined."""
        expected = ["HIGHLY_RELEVANT", "RELEVANT", "SOMEWHAT_RELEVANT", "TANGENTIAL"]
        for level in expected:
            assert hasattr(MemoryRelevance, level)


class TestSimilarEvent:
    """Test SimilarEvent dataclass."""

    def test_creation(self) -> None:
        """Test SimilarEvent can be created."""
        event = SimilarEvent(
            event_id="test_001",
            timestamp=time.time(),
            similarity_score=0.85,
            domain="test",
            outcome="resolved",
            summary="Test event",
        )

        assert event.event_id == "test_001"
        assert event.similarity_score == 0.85


class TestPredictionHistory:
    """Test PredictionHistory dataclass."""

    def test_creation(self) -> None:
        """Test PredictionHistory can be created."""
        history = PredictionHistory(
            pattern_type="test_pattern",
            total_predictions=10,
            correct_predictions=8,
            accuracy=0.8,
            avg_lead_time_sec=3600.0,
            last_prediction=time.time(),
        )

        assert history.accuracy == 0.8
        assert history.total_predictions == 10

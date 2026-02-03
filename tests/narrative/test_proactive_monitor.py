"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for ProactiveMonitor - Background Vigilance with Initiative.
"""

import time

import pytest

from omni_mercury_engine.narrative.proactive import (
    InitiativeEvent,
    InitiativeThreshold,
    InitiativeType,
    ProactiveMonitor,
    VigilanceLevel,
)


class TestProactiveMonitor:
    """Test ProactiveMonitor functionality."""

    @pytest.fixture
    def monitor(self) -> ProactiveMonitor:
        """Create test monitor (without scheduled reports for speed)."""
        return ProactiveMonitor(enable_scheduled_reports=False)

    @pytest.fixture
    def started_monitor(self, monitor: ProactiveMonitor):
        """Monitor that has been started and will be stopped after test."""
        monitor.start()
        yield monitor
        monitor.stop()

    def test_initialization(self, monitor: ProactiveMonitor) -> None:
        """Test monitor initializes correctly."""
        assert monitor.default_vigilance == VigilanceLevel.ATTENTIVE
        assert not monitor._running
        assert monitor._detections_processed == 0

    def test_start_stop(self, monitor: ProactiveMonitor) -> None:
        """Test monitor can be started and stopped."""
        assert not monitor._running

        monitor.start()
        assert monitor._running

        monitor.stop()
        assert not monitor._running

    def test_vigilance_levels(self, monitor: ProactiveMonitor) -> None:
        """Test setting vigilance levels."""
        # Global default
        monitor.set_vigilance(VigilanceLevel.VIGILANT)
        assert monitor.default_vigilance == VigilanceLevel.VIGILANT

        # Domain-specific
        monitor.set_vigilance(VigilanceLevel.CRITICAL, domain="security")
        assert monitor._vigilance_levels["security"] == VigilanceLevel.CRITICAL

    def test_threshold_configuration(self, monitor: ProactiveMonitor) -> None:
        """Test threshold configuration."""
        custom_threshold = InitiativeThreshold(
            min_anomaly_score=0.5,
            min_confidence=0.4,
            min_severity=0.3,
        )

        monitor.set_threshold(custom_threshold, domain="test")
        assert monitor._thresholds["test"].min_anomaly_score == 0.5

    def test_submit_detection(self, started_monitor: ProactiveMonitor) -> None:
        """Test submitting detection results."""
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.9,
            "severity": 0.85,
            "confidence": 0.8,
        }

        started_monitor.submit(detection, domain="test")

        # Wait for processing
        time.sleep(0.6)

        assert started_monitor._detections_processed >= 1

    def test_initiative_callback(self, started_monitor: ProactiveMonitor) -> None:
        """Test initiative callbacks are triggered."""
        events_received = []

        def callback(event: InitiativeEvent) -> None:
            events_received.append(event)

        started_monitor.on_initiative(callback)

        # Submit high-severity detection
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.95,
            "severity": 0.9,
            "confidence": 0.9,
        }

        started_monitor.submit(detection, domain="test")

        # Wait for processing
        time.sleep(0.8)

        assert len(events_received) >= 1
        assert events_received[0].initiative_type == InitiativeType.ANOMALY_ALERT

    def test_passive_vigilance_no_initiative(self, started_monitor: ProactiveMonitor) -> None:
        """Test passive vigilance doesn't generate initiatives."""
        events_received = []

        started_monitor.on_initiative(events_received.append)
        started_monitor.set_vigilance(VigilanceLevel.PASSIVE, domain="test")

        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.99,
            "severity": 0.99,
            "confidence": 0.99,
        }

        started_monitor.submit(detection, domain="test")
        time.sleep(0.6)

        # Passive mode should not generate initiative
        assert len(events_received) == 0

    def test_cooldown_prevents_duplicate_alerts(self, started_monitor: ProactiveMonitor) -> None:
        """Test cooldown mechanism prevents alert spam."""
        events_received = []
        started_monitor.on_initiative(events_received.append)

        # Set short cooldown for test
        started_monitor._thresholds["test"] = InitiativeThreshold(cooldown_sec=2.0)

        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.95,
            "severity": 0.9,
            "confidence": 0.9,
        }

        # Submit twice rapidly
        started_monitor.submit(detection, domain="test")
        time.sleep(0.3)
        started_monitor.submit(detection, domain="test")
        time.sleep(0.6)

        # Should only get one alert due to cooldown
        assert len(events_received) == 1

    def test_escalation_on_pattern_accumulation(self, started_monitor: ProactiveMonitor) -> None:
        """Test escalation triggers on pattern accumulation."""
        events_received = []
        started_monitor.on_initiative(events_received.append)

        # Configure for quick escalation
        threshold = InitiativeThreshold(
            pattern_accumulation_count=2,
            pattern_accumulation_window_sec=60.0,
            cooldown_sec=0.1,
        )
        started_monitor.set_threshold(threshold, domain="test")

        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.7,
            "severity": 0.6,
            "confidence": 0.75,
        }

        # Submit multiple times to trigger escalation
        for _ in range(3):
            started_monitor.submit(detection, domain="test")
            time.sleep(0.2)

        time.sleep(0.5)

        # Should have escalation event
        escalation_events = [
            e for e in events_received if e.initiative_type == InitiativeType.ESCALATION
        ]
        assert len(escalation_events) >= 1

    def test_initiative_event_structure(self, started_monitor: ProactiveMonitor) -> None:
        """Test initiative event has correct structure."""
        events_received = []
        started_monitor.on_initiative(events_received.append)

        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.9,
            "severity": 0.85,
            "confidence": 0.8,
        }

        started_monitor.submit(detection, domain="security")
        time.sleep(0.6)

        if events_received:
            event = events_received[0]
            assert event.event_id is not None
            assert event.timestamp > 0
            assert event.domain == "security"
            assert event.summary is not None
            assert event.recommendations is not None

            # Test to_dict
            event_dict = event.to_dict()
            assert "event_id" in event_dict
            assert "type" in event_dict
            assert "summary" in event_dict

    def test_statistics(self, started_monitor: ProactiveMonitor) -> None:
        """Test statistics gathering."""
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.5,
            "severity": 0.5,
            "confidence": 0.5,
        }

        started_monitor.submit(detection)
        time.sleep(0.6)

        stats = started_monitor.get_statistics()
        assert stats["running"] is True
        assert "detections_processed" in stats
        assert "initiatives_generated" in stats


class TestVigilanceLevel:
    """Test VigilanceLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Ensure all expected vigilance levels are defined."""
        expected = ["PASSIVE", "ATTENTIVE", "VIGILANT", "HEIGHTENED", "CRITICAL"]
        for level in expected:
            assert hasattr(VigilanceLevel, level)


class TestInitiativeType:
    """Test InitiativeType enum."""

    def test_all_types_defined(self) -> None:
        """Ensure all expected initiative types are defined."""
        expected = [
            "ANOMALY_ALERT",
            "PATTERN_EMERGENCE",
            "ESCALATION",
            "PREDICTION",
            "CALIBRATION",
            "MEMORY_INSIGHT",
            "SCHEDULED_REPORT",
        ]
        for type_name in expected:
            assert hasattr(InitiativeType, type_name)

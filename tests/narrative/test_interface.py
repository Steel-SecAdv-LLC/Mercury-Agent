# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for MercuryConversationInterface - Unified "Alive" Interface."""

from collections.abc import Iterator
from typing import Any

import pytest

from omni_mercury_engine.narrative.interface import (
    ConversationContext,
    MercuryConversationInterface,
    MercuryResponse,
    create_mercury_interface,
)
from omni_mercury_engine.narrative.proactive import VigilanceLevel


class TestMercuryConversationInterface:
    """Test MercuryConversationInterface functionality."""

    @pytest.fixture
    def interface(self) -> MercuryConversationInterface:
        """Create test interface (proactive disabled for speed)."""
        return MercuryConversationInterface(
            enable_proactive=False, enable_memory=True, default_domain="test"
        )

    @pytest.fixture
    def interface_with_proactive(self) -> Iterator[MercuryConversationInterface]:
        """Create interface with proactive monitoring."""
        iface = MercuryConversationInterface(
            enable_proactive=True,
            enable_memory=True,
        )
        yield iface
        if iface.proactive_monitor and iface.proactive_monitor._running:
            iface.stop_proactive_monitoring()

    @pytest.fixture
    def sample_detection(self) -> dict[str, Any]:
        """Sample detection result."""
        return {
            "anomaly_detected": True,
            "anomaly_score": 0.85,
            "severity": 0.7,
            "confidence": 0.82,
            "is_reliable": True,
            "reasoning_chain": [{"rule": "threshold", "conclusion": "exceeded", "confidence": 0.9}],
            "recommendations": ["Review data"],
        }

    def test_initialization(self, interface: MercuryConversationInterface) -> None:
        """Test interface initializes correctly."""
        assert interface.default_domain == "test"
        assert interface.narrative_engine is not None
        assert interface.personality_engine is not None
        assert interface.memory_surface is not None
        assert interface.proactive_monitor is None  # Disabled

    def test_create_session(self, interface: MercuryConversationInterface) -> None:
        """Test session creation."""
        ctx = interface.create_session(domain="medical")

        assert isinstance(ctx, ConversationContext)
        assert ctx.session_id is not None
        assert ctx.domain == "medical"
        assert len(ctx.conversation_history) == 0

    def test_get_greeting(self, interface: MercuryConversationInterface) -> None:
        """Test greeting generation."""
        greeting = interface.get_greeting()
        assert greeting is not None
        assert "Mercury" in greeting

        # With context
        ctx = interface.create_session(domain="security")
        greeting_ctx = interface.get_greeting(ctx)
        assert greeting_ctx is not None

    def test_process_detection(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test detection processing."""
        ctx = interface.create_session()
        response = interface.process_detection(sample_detection, ctx)

        assert isinstance(response, MercuryResponse)
        assert response.message is not None
        assert response.summary is not None
        assert response.confidence_statement is not None
        assert response.narrative is not None

    def test_response_structure(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test response has all expected fields."""
        response = interface.process_detection(sample_detection)

        # Check all required fields
        assert response.message is not None
        assert response.summary is not None
        assert response.response_time_ms > 0
        assert response.style is not None
        assert isinstance(response.historical_references, list)
        assert isinstance(response.follow_up_suggestions, list)
        assert isinstance(response.warnings, list)

    def test_response_to_dict(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test response serialization."""
        response = interface.process_detection(sample_detection)
        response_dict = response.to_dict()

        assert "message" in response_dict
        assert "summary" in response_dict
        assert "narrative" in response_dict
        assert "metadata" in response_dict

    def test_conversation_history(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test conversation history tracking."""
        ctx = interface.create_session()

        interface.process_detection(sample_detection, ctx)
        interface.process_detection(sample_detection, ctx)

        assert len(ctx.conversation_history) == 2

    def test_memory_context_in_response(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test memory context is included in response."""
        response = interface.process_detection(sample_detection)

        assert response.memory_context is not None
        assert response.historical_references is not None

    def test_follow_up_suggestions(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test follow-up suggestions are generated."""
        response = interface.process_detection(sample_detection)

        # With anomaly detected, should have some follow-ups
        assert len(response.follow_up_suggestions) > 0

    def test_warnings_propagation(self, interface: MercuryConversationInterface) -> None:
        """Test warnings are propagated from detection."""
        detection = {
            "anomaly_detected": True,
            "anomaly_score": 0.7,
            "severity": 0.6,
            "confidence": 0.5,
            "is_reliable": False,  # Should trigger warning
            "warnings": [{"message": "Test warning"}],
        }

        response = interface.process_detection(detection)
        assert len(response.warnings) > 0

    def test_ask_method(self, interface: MercuryConversationInterface) -> None:
        """Test ask method for simple questions."""
        response = interface.ask("What is my system status?")
        assert response is not None
        assert "Mercury" in response or "operational" in response.lower()

    def test_statistics(
        self, interface: MercuryConversationInterface, sample_detection: dict[str, Any]
    ) -> None:
        """Test statistics gathering."""
        interface.process_detection(sample_detection)
        interface.process_detection(sample_detection)

        stats = interface.get_statistics()
        assert stats["total_interactions"] == 2
        assert stats["total_detections"] == 2
        assert "narrative_engine" in stats
        assert "memory_surface" in stats


class TestMercuryConversationInterfaceProactive:
    """Test proactive monitoring features."""

    @pytest.fixture
    def interface(self) -> Iterator[MercuryConversationInterface]:
        """Create interface with proactive enabled."""
        iface = MercuryConversationInterface(
            enable_proactive=True,
            enable_memory=True,
        )
        yield iface
        if iface.proactive_monitor and iface.proactive_monitor._running:
            iface.stop_proactive_monitoring()

    def test_proactive_initialization(self, interface: MercuryConversationInterface) -> None:
        """Test proactive monitor is initialized."""
        assert interface.proactive_monitor is not None

    def test_start_stop_proactive(self, interface: MercuryConversationInterface) -> None:
        """Test starting and stopping proactive monitoring."""
        interface.start_proactive_monitoring()
        assert interface.proactive_monitor is not None
        assert interface.proactive_monitor._running is True

        interface.stop_proactive_monitoring()
        assert interface.proactive_monitor._running is False

    def test_set_vigilance(self, interface: MercuryConversationInterface) -> None:
        """Test setting vigilance level."""
        interface.set_vigilance(VigilanceLevel.HEIGHTENED, domain="security")

        assert interface.proactive_monitor is not None
        assert (
            interface.proactive_monitor._vigilance_levels["security"] == VigilanceLevel.HEIGHTENED
        )

    def test_proactive_callback_registration(self, interface: MercuryConversationInterface) -> None:
        """Test proactive alert callback registration."""
        events: list[Any] = []
        interface.on_proactive_alert(events.append)

        # Callback should be registered
        assert len(interface._proactive_callbacks) > 0


class TestCreateMercuryInterface:
    """Test factory function."""

    def test_factory_creates_interface(self) -> None:
        """Test factory function creates valid interface."""
        interface = create_mercury_interface()

        assert isinstance(interface, MercuryConversationInterface)
        assert interface.enable_proactive is True
        assert interface.enable_memory is True

    def test_factory_with_options(self) -> None:
        """Test factory with custom options."""
        interface = create_mercury_interface(
            enable_proactive=False, enable_memory=False, default_domain="medical"
        )

        assert interface.enable_proactive is False
        assert interface.enable_memory is False
        assert interface.default_domain == "medical"


class TestConversationContext:
    """Test ConversationContext dataclass."""

    def test_creation(self) -> None:
        """Test context creation."""
        ctx = ConversationContext(session_id="test_session", domain="medical")

        assert ctx.session_id == "test_session"
        assert ctx.domain == "medical"
        assert len(ctx.conversation_history) == 0
        assert ctx.active_since > 0

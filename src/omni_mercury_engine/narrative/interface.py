"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Conversation Interface - Unified "Alive" Interface for Mercury Agent

Integrates all narrative components into a cohesive conversation experience:
    - NarrativeEngine: Translates detection to human language
    - ProactiveMonitor: Background vigilance with initiative
    - MemorySurface: Historical context awareness
    - PersonalityEngine: Omni-scalar shaped communication

Philosophy:
    "An Agent that does not engage or communicate to retain users like almost
    all LLMs, but provides transparency and truth in every response."

This is the unified entry point for making Mercury "alive" - not performatively,
but through genuine transparency, truth density, and principled communication.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.narrative.engine import (
    NarrativeEngine,
    NarrativeResult,
)
from omni_mercury_engine.narrative.memory_surface import MemoryContext, MemorySurface
from omni_mercury_engine.narrative.personality import (
    CommunicationTone,
    PersonalityEngine,
    PersonalityProfile,
)
from omni_mercury_engine.narrative.proactive import (
    InitiativeEvent,
    ProactiveMonitor,
    VigilanceLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context for a conversation session."""

    session_id: str
    domain: str | None = None
    user_preferences: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    active_since: float = field(default_factory=time.time)


@dataclass
class MercuryResponse:
    """Complete response from Mercury's conversation interface."""

    # Primary response
    message: str  # Main response text
    summary: str  # One-line summary

    # Transparency components
    narrative: NarrativeResult | None  # Full narrative with reasoning
    confidence_statement: str  # Explicit confidence disclosure

    # Memory context
    memory_context: MemoryContext | None
    historical_references: list[str]

    # Metadata
    response_time_ms: float
    domain: str | None
    style: CommunicationTone
    personality_profile: PersonalityProfile | None

    # Proactive elements
    follow_up_suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-friendly dictionary."""
        return {
            "message": self.message,
            "summary": self.summary,
            "confidence_statement": self.confidence_statement,
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "memory_context": self.memory_context.to_dict() if self.memory_context else None,
            "historical_references": self.historical_references,
            "follow_up_suggestions": self.follow_up_suggestions,
            "warnings": self.warnings,
            "metadata": {
                "response_time_ms": self.response_time_ms,
                "domain": self.domain,
                "style": self.style.value,
            },
        }


class MercuryConversationInterface:
    """
    Unified Conversation Interface - Making Mercury "Alive".

    This is the primary interface for interacting with Mercury as an
    "alive" agent. It combines all narrative components to provide:

    1. Truth-Dense Responses: Every claim backed by evidence or uncertainty
    2. Transparent Reasoning: Full chains exposed, not hidden
    3. Historical Awareness: Memory-informed context
    4. Principled Communication: Shaped by ethical scalars
    5. Proactive Initiative: Speaks up when truth demands it

    Example Usage:
        # Initialize interface
        mercury = MercuryConversationInterface()

        # Create conversation context
        ctx = mercury.create_session(domain="medical")

        # Process detection and get alive response
        response = mercury.process_detection(
            detection_result=engine.detect_with_fusion(data),
            context=ctx
        )

        # Natural, truth-dense response
        print(response.message)
        # "Detected significant anomaly (score: 0.87, severity: 78%).
        #  Based on 3 similar historical cases, patterns like this
        #  escalated within 24 hours 67% of the time. Confidence: 82%.
        #  Recommendation: Clinical review within 4 hours."

        # Register for proactive alerts
        mercury.on_proactive_alert(lambda event: notify_user(event))
    """

    def __init__(
        self,
        enable_proactive: bool = True,
        enable_memory: bool = True,
        default_domain: str | None = None,
    ) -> None:
        """
        Initialize Mercury Conversation Interface.

        Args:
            enable_proactive: Enable proactive monitoring/alerts
            enable_memory: Enable memory-based context
            default_domain: Default domain for conversations
        """
        self.default_domain = default_domain
        self.enable_proactive = enable_proactive
        self.enable_memory = enable_memory

        # Initialize components
        self.narrative_engine = NarrativeEngine()
        self.personality_engine = PersonalityEngine()

        # Memory surface (if enabled)
        self.memory_surface: MemorySurface | None = None
        if enable_memory:
            self.memory_surface = MemorySurface()
            self.narrative_engine.set_memory_surface(self.memory_surface)

        # Proactive monitor (if enabled)
        self.proactive_monitor: ProactiveMonitor | None = None
        if enable_proactive:
            self.proactive_monitor = ProactiveMonitor()

        # Session management
        self._sessions: dict[str, ConversationContext] = {}
        self._session_counter = 0

        # Callbacks
        self._proactive_callbacks: list[Callable[[InitiativeEvent], None]] = []

        # Statistics
        self._total_interactions = 0
        self._total_detections_processed = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"MercuryConversationInterface initialized "
            f"(proactive={enable_proactive}, memory={enable_memory})"
        )

    def create_session(
        self,
        domain: str | None = None,
        user_preferences: dict[str, Any] | None = None,
    ) -> ConversationContext:
        """
        Create a new conversation session.

        Args:
            domain: Domain context for this session
            user_preferences: User preferences for communication

        Returns:
            ConversationContext for the session
        """
        self._session_counter += 1
        session_id = f"session_{self._session_counter}_{int(time.time())}"

        context = ConversationContext(
            session_id=session_id,
            domain=domain or self.default_domain,
            user_preferences=user_preferences or {},
        )

        self._sessions[session_id] = context
        return context

    def get_greeting(self, context: ConversationContext | None = None) -> str:
        """
        Get appropriate greeting for session start.

        Args:
            context: Conversation context

        Returns:
            Personality-appropriate greeting
        """
        domain = context.domain if context else self.default_domain
        return self.personality_engine.get_greeting(domain)

    def process_detection(
        self,
        detection_result: dict[str, Any],
        context: ConversationContext | None = None,
        raw_data: Any = None,
    ) -> MercuryResponse:
        """
        Process a detection result and generate alive response.

        This is the primary method for converting raw detections into
        truth-dense, transparent communication.

        Args:
            detection_result: Output from detection engine
            context: Conversation context
            raw_data: Optional raw data for deeper analysis

        Returns:
            MercuryResponse with full transparency
        """
        start_time = time.time()
        self._total_interactions += 1
        self._total_detections_processed += 1

        domain = context.domain if context else self.default_domain

        # Get personality profile
        profile = self.personality_engine.get_profile(domain)

        # Generate narrative
        narrative = self.narrative_engine.synthesize(
            detection_result=detection_result,
            domain=domain,
            context={"session": context.session_id if context else None},
        )

        # Get memory context
        memory_context = None
        historical_refs = []
        if self.memory_surface:
            memory_context = self.memory_surface.get_relevant_context(detection_result, domain)
            historical_refs = memory_context.similar_event_ids

            # Record this event for future memory
            self.memory_surface.record_event(detection_result, domain, outcome=None)

        # Get communication modifiers
        modifiers = self.personality_engine.get_modifiers(
            severity=detection_result.get("severity", 0.0),
            confidence=detection_result.get("confidence", 0.5),
            anomaly_detected=detection_result.get("anomaly_detected", False),
            profile=profile,
            domain=domain,
        )

        # Build main message
        message = self._build_message(
            narrative, memory_context, profile, modifiers, detection_result
        )

        # Generate confidence statement
        confidence_statement = self.personality_engine.get_uncertainty_statement(
            confidence=detection_result.get("confidence", 0.5),
            epistemic=detection_result.get("epistemic_uncertainty", 0.0),
            aleatoric=detection_result.get("aleatoric_uncertainty", 0.0),
            domain=domain,
        )

        # Submit to proactive monitor
        if self.proactive_monitor:
            self.proactive_monitor.submit(detection_result, domain)

        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(detection_result, profile, domain)

        # Collect warnings
        warnings = []
        if not detection_result.get("is_reliable", True):
            warnings.append("This prediction has been flagged as potentially unreliable.")
        for w in detection_result.get("warnings", [])[:3]:
            if isinstance(w, dict):
                warnings.append(w.get("message", str(w)))
            else:
                warnings.append(str(w))

        # Update conversation history
        if context:
            context.conversation_history.append(
                {
                    "type": "detection",
                    "timestamp": time.time(),
                    "summary": narrative.summary,
                }
            )

        response_time = (time.time() - start_time) * 1000

        return MercuryResponse(
            message=message,
            summary=narrative.summary,
            narrative=narrative,
            confidence_statement=confidence_statement,
            memory_context=memory_context,
            historical_references=historical_refs,
            response_time_ms=response_time,
            domain=domain,
            style=profile.tone,
            personality_profile=profile,
            follow_up_suggestions=follow_ups,
            warnings=warnings,
        )

    def _build_message(
        self,
        narrative: NarrativeResult,
        memory_context: MemoryContext | None,
        profile: PersonalityProfile,
        modifiers: Any,
        detection_result: dict[str, Any],
    ) -> str:
        """Build the main response message."""
        parts = []

        # Opening acknowledgment if appropriate
        if modifiers.opening_acknowledgment:
            parts.append(modifiers.opening_acknowledgment)

        # Main summary
        parts.append(narrative.summary)

        # Historical context (if relevant and personality allows)
        if memory_context and profile.include_historical_context:
            if memory_context.similar_events:
                n_similar = len(memory_context.similar_events)
                parts.append(f"This pattern is similar to {n_similar} previous observation(s).")

                # Add specific insight if highly relevant
                if memory_context.learned_insights:
                    parts.append(memory_context.learned_insights[0])

        # Reasoning chain (if transparency enabled)
        if profile.show_reasoning_chain and narrative.reasoning_chain:
            chain = narrative.reasoning_chain
            if len(chain.steps) > 0:
                parts.append(
                    f"Reasoning: {len(chain.steps)} inference steps. "
                    f"Final conclusion: {chain.final_conclusion}"
                )

        # Confidence disclosure
        parts.append(narrative.uncertainty_disclosure)

        # Top recommendation
        if narrative.recommendations:
            parts.append(f"Recommendation: {narrative.recommendations[0]}")

        # Support statement if supportive tone
        if modifiers.support_statement and profile.tone == CommunicationTone.SUPPORTIVE:
            parts.append(modifiers.support_statement)

        return " ".join(parts)

    def _generate_follow_ups(
        self,
        detection_result: dict[str, Any],
        profile: PersonalityProfile,
        domain: str | None,
    ) -> list[str]:
        """Generate follow-up suggestions based on personality."""
        follow_ups: list[str] = []

        if not profile.provide_alternatives:
            return follow_ups

        anomaly_detected = detection_result.get("anomaly_detected", False)
        confidence = detection_result.get("confidence", 0.5)

        if anomaly_detected:
            follow_ups.append("View detailed reasoning chain")
            if confidence < 0.8:
                follow_ups.append("Explore alternative interpretations")
            follow_ups.append("Check similar historical cases")

        if domain == "medical":
            follow_ups.append("Review clinical guidelines")
        elif domain == "security":
            follow_ups.append("View threat intelligence context")

        return follow_ups[:3]

    def on_proactive_alert(self, callback: Callable[[InitiativeEvent], None]) -> None:
        """
        Register callback for proactive alerts.

        Args:
            callback: Function to call when Mercury takes initiative
        """
        if self.proactive_monitor:
            self.proactive_monitor.on_initiative(callback)
        self._proactive_callbacks.append(callback)

    def set_vigilance(
        self,
        level: VigilanceLevel,
        domain: str | None = None,
    ) -> None:
        """
        Set vigilance level for proactive monitoring.

        Args:
            level: Vigilance level
            domain: Domain to apply (None for global)
        """
        if self.proactive_monitor:
            self.proactive_monitor.set_vigilance(level, domain)

    def start_proactive_monitoring(self) -> None:
        """Start proactive background monitoring."""
        if self.proactive_monitor:
            self.proactive_monitor.start()
            self.logger.info("Proactive monitoring started")

    def stop_proactive_monitoring(self) -> None:
        """Stop proactive background monitoring."""
        if self.proactive_monitor:
            self.proactive_monitor.stop()
            self.logger.info("Proactive monitoring stopped")

    def ask(
        self,
        question: str,
        context: ConversationContext | None = None,
    ) -> str:
        """
        Handle a natural language question.

        This is a simplified interface for questions. For full detection
        processing, use process_detection().

        Args:
            question: Natural language question
            context: Conversation context

        Returns:
            Response text
        """
        # Domain and profile available via context.domain and self.personality_engine for future use

        # Simple question handling
        question_lower = question.lower()

        if "status" in question_lower or "health" in question_lower:
            stats = self.get_statistics()
            return (
                f"Mercury Agent operational. "
                f"Processed {stats['total_detections']} detections. "
                f"Proactive monitoring: {'active' if stats['proactive_running'] else 'inactive'}."
            )

        if "help" in question_lower:
            return (
                "I can analyze data for anomalies and provide transparent explanations. "
                "Submit detection results via process_detection() for full analysis."
            )

        if "confidence" in question_lower or "certain" in question_lower:
            return (
                "I always disclose confidence levels and uncertainty. "
                "Every finding includes explicit confidence metrics and uncertainty quantification."
            )

        return (
            "I'm designed for anomaly detection analysis. "
            "Please submit data for detection, or ask about system status."
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive interface statistics."""
        stats: dict[str, Any] = {
            "total_interactions": self._total_interactions,
            "total_detections": self._total_detections_processed,
            "active_sessions": len(self._sessions),
            "proactive_enabled": self.enable_proactive,
            "memory_enabled": self.enable_memory,
            "proactive_running": (
                self.proactive_monitor._running if self.proactive_monitor else False
            ),
        }

        if self.narrative_engine:
            stats["narrative_engine"] = self.narrative_engine.get_statistics()

        if self.memory_surface:
            stats["memory_surface"] = self.memory_surface.get_statistics()

        if self.proactive_monitor:
            stats["proactive_monitor"] = self.proactive_monitor.get_statistics()

        return stats


# Factory function for easy instantiation
def create_mercury_interface(
    enable_proactive: bool = True,
    enable_memory: bool = True,
    default_domain: str | None = None,
) -> MercuryConversationInterface:
    """
    Create a Mercury Conversation Interface instance.

    Args:
        enable_proactive: Enable proactive monitoring
        enable_memory: Enable memory-based context
        default_domain: Default domain

    Returns:
        Configured MercuryConversationInterface
    """
    return MercuryConversationInterface(
        enable_proactive=enable_proactive,
        enable_memory=enable_memory,
        default_domain=default_domain,
    )

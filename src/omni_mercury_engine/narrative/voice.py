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
Mercury Voice - True Conversational Interface

This is Mercury's "voice" - the ability to understand queries, retrieve
information, reason about responses, and communicate with truth density.

Key Capabilities:
    1. Natural Language Understanding (intent classification, entity extraction)
    2. Information Retrieval (knowledge graph, memory, patterns)
    3. Response Generation (template-based + LLM-enhanced)
    4. Conversation Context (multi-turn awareness)
    5. Proactive Communication (alerts, insights)

Philosophy:
    Mercury speaks when truth demands it, not to retain engagement.
    Every response is backed by evidence or explicitly uncertain.
    Transparency over persuasion.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from omni_mercury_engine.narrative.engine import NarrativeEngine, NarrativeResult
from omni_mercury_engine.narrative.personality import (
    CommunicationTone,
    PersonalityEngine,
    PersonalityProfile,
)
from omni_mercury_engine.narrative.retriever import (
    KnowledgeRetriever,
    QueryIntent,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class ConversationType(Enum):
    """Types of conversation turns."""

    QUERY = "query"  # User asking a question
    COMMAND = "command"  # User giving instruction
    FEEDBACK = "feedback"  # User providing feedback
    DETECTION = "detection"  # Detection result to communicate
    ALERT = "alert"  # Proactive alert
    GREETING = "greeting"  # Session start/end


@dataclass
class ConversationTurn:
    """A single turn in conversation."""

    turn_id: str
    turn_type: ConversationType
    speaker: str  # "user" or "mercury"
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceResponse:
    """Response from Mercury's voice."""

    message: str  # Main response text
    confidence: float  # Confidence in response
    sources_cited: list[str]  # Evidence sources
    reasoning_summary: str | None  # Brief reasoning explanation

    # Metadata
    intent_understood: QueryIntent | None
    search_performed: bool
    search_results_count: int
    response_time_ms: float

    # Follow-up
    suggested_follow_ups: list[str] = field(default_factory=list)

    # Transparency
    uncertainty_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "confidence": self.confidence,
            "sources_cited": self.sources_cited,
            "reasoning_summary": self.reasoning_summary,
            "intent_understood": self.intent_understood.value if self.intent_understood else None,
            "search_performed": self.search_performed,
            "search_results_count": self.search_results_count,
            "response_time_ms": self.response_time_ms,
            "suggested_follow_ups": self.suggested_follow_ups,
            "uncertainty_note": self.uncertainty_note,
        }


class MercuryVoice:
    """
    Mercury's Voice - True Conversational Interface.

    This is the unified interface for all conversational interaction with Mercury.
    It understands queries, retrieves information, generates responses, and
    maintains conversation context.

    Not a chatbot - a truth-seeking, evidence-backed communication system.

    Usage:
        voice = MercuryVoice()

        # Configure voice
        voice.set_knowledge_graph(kg)
        voice.set_agent_memory(memory)

        # Converse
        response = voice.speak("What patterns preceded the last anomaly?")
        print(response.message)
        # "Based on detection logs, 3 similar patterns occurred in the past 24 hours.
        #  Historical analysis shows 67% escalation rate for this pattern type.
        #  Sources: detection_log (3), pattern_history (2). Confidence: 78%."

        # Process detection
        response = voice.process_detection(detection_result)
        print(response.message)
        # Full truth-dense narrative of detection

        # Proactive alert
        response = voice.alert(initiative_event)
        print(response.message)
        # Alert communication with evidence
    """

    def __init__(
        self,
        enable_llm: bool = False,
        default_domain: str | None = None,
    ) -> None:
        """
        Initialize Mercury Voice.

        Args:
            enable_llm: Whether to use LLM for response generation
            default_domain: Default domain context
        """
        self.enable_llm = enable_llm
        self.default_domain = default_domain

        # Core components
        self.narrative_engine = NarrativeEngine()
        self.personality_engine = PersonalityEngine()
        self.retriever = KnowledgeRetriever()

        # LLM adapter (if enabled)
        self._llm_adapter = None
        if enable_llm:
            self._init_llm()

        # Conversation state
        self._conversation_history: list[ConversationTurn] = []
        self._turn_counter = 0
        self._session_start = time.time()

        # Callbacks for external handlers
        self._command_handlers: dict[str, Callable[[str], str]] = {}

        # Statistics
        self._queries_handled = 0
        self._detections_narrated = 0
        self._alerts_communicated = 0

        self.logger = logging.getLogger(__name__)

    def _init_llm(self) -> None:
        """Initialize LLM adapter for enhanced responses."""
        try:
            from omni_mercury_engine.models.foundation.llm_adapter import (
                LLMConfig,
                MockLLMAdapter,
            )

            # Use mock adapter by default (can be configured for real LLM)
            config = LLMConfig()
            self._llm_adapter = MockLLMAdapter(config)
        except ImportError:
            self.logger.warning("LLM adapter not available")

    def set_knowledge_graph(self, kg: Any) -> None:
        """Set knowledge graph for retrieval."""
        self.retriever.set_knowledge_graph(kg)

    def set_agent_memory(self, memory: Any) -> None:
        """Set agent memory for retrieval."""
        self.retriever.set_agent_memory(memory)

    def set_memory_surface(self, surface: Any) -> None:
        """Set memory surface for retrieval and narrative."""
        self.retriever.set_memory_surface(surface)
        self.narrative_engine.set_memory_surface(surface)

    def register_command(self, command: str, handler: Callable[[str], str]) -> None:
        """Register a command handler."""
        self._command_handlers[command.lower()] = handler

    def speak(
        self,
        user_input: str,
        domain: str | None = None,
    ) -> VoiceResponse:
        """
        Process user input and generate response.

        This is the main entry point for conversational interaction.

        Args:
            user_input: Natural language input from user
            domain: Optional domain context

        Returns:
            VoiceResponse with truth-dense communication
        """
        start_time = time.time()
        self._queries_handled += 1
        domain = domain or self.default_domain

        # Record user turn
        self._record_turn(ConversationType.QUERY, "user", user_input)

        # Get personality profile
        profile = self.personality_engine.get_profile(domain)

        # Direct intent detection for special cases (before general search)
        input_lower = user_input.lower()

        # Check for status queries directly
        if any(kw in input_lower for kw in ["status", "health", "running", "operational"]):
            response = self._handle_status_query(profile)
            self._record_turn(ConversationType.QUERY, "mercury", response.message)
            return response

        # Check for help queries directly
        if any(
            kw in input_lower for kw in ["help", "how do you work", "what can you do", "guide me"]
        ):
            response = self._handle_help_query(user_input, profile)
            self._record_turn(ConversationType.QUERY, "mercury", response.message)
            return response

        # General search for other queries
        search_response = self.retriever.search(user_input, domain=domain)

        # General query handling with search
        response = self._handle_general_query(user_input, search_response, profile, domain)

        # Record Mercury's response
        self._record_turn(ConversationType.QUERY, "mercury", response.message)

        return response

    def process_detection(
        self,
        detection_result: dict[str, Any],
        domain: str | None = None,
    ) -> VoiceResponse:
        """
        Process a detection result and generate voice response.

        Args:
            detection_result: Detection output to communicate
            domain: Optional domain context

        Returns:
            VoiceResponse with full narrative
        """
        start_time = time.time()
        self._detections_narrated += 1
        domain = domain or self.default_domain

        # Log detection for future retrieval
        self.retriever.log_detection(detection_result, domain)

        # Generate narrative
        narrative = self.narrative_engine.synthesize(detection_result, domain=domain)

        # Build response
        message = self._build_detection_message(narrative, detection_result)

        response_time = (time.time() - start_time) * 1000

        # Record turn
        self._record_turn(ConversationType.DETECTION, "mercury", message)

        return VoiceResponse(
            message=message,
            confidence=narrative.confidence_score,
            sources_cited=["detection_engine", "narrative_synthesis"],
            reasoning_summary=self._summarize_reasoning(narrative),
            intent_understood=None,
            search_performed=False,
            search_results_count=0,
            response_time_ms=response_time,
            suggested_follow_ups=self._generate_detection_follow_ups(narrative),
            uncertainty_note=(
                narrative.uncertainty_disclosure if narrative.confidence_score < 0.7 else None
            ),
        )

    def alert(
        self,
        alert_content: dict[str, Any],
        alert_type: str = "anomaly",
    ) -> VoiceResponse:
        """
        Generate alert communication.

        Args:
            alert_content: Content of the alert
            alert_type: Type of alert

        Returns:
            VoiceResponse for the alert
        """
        start_time = time.time()
        self._alerts_communicated += 1

        profile = self.personality_engine.get_profile()

        # Build alert message
        severity = alert_content.get("severity", 0.5)
        confidence = alert_content.get("confidence", 0.5)

        if severity > 0.8:
            urgency = "CRITICAL"
        elif severity > 0.5:
            urgency = "HIGH"
        else:
            urgency = "MODERATE"

        message_parts = [
            f"[{urgency} ALERT]",
            alert_content.get("summary", "Proactive alert triggered."),
        ]

        if confidence < 0.7:
            message_parts.append(f"Note: Confidence is {confidence:.0%}. Verification recommended.")

        message = " ".join(message_parts)

        recommendations = alert_content.get("recommendations", [])
        if recommendations:
            message += f" Recommendation: {recommendations[0]}"

        response_time = (time.time() - start_time) * 1000

        self._record_turn(ConversationType.ALERT, "mercury", message)

        return VoiceResponse(
            message=message,
            confidence=confidence,
            sources_cited=["proactive_monitor"],
            reasoning_summary=f"Alert triggered by: {alert_content.get('triggered_by', 'threshold')}",
            intent_understood=None,
            search_performed=False,
            search_results_count=0,
            response_time_ms=response_time,
            suggested_follow_ups=["View alert details", "Acknowledge alert"],
            uncertainty_note=f"Alert confidence: {confidence:.0%}" if confidence < 0.8 else None,
        )

    def greet(self, domain: str | None = None) -> VoiceResponse:
        """
        Generate session greeting.

        Args:
            domain: Optional domain context

        Returns:
            VoiceResponse with greeting
        """
        domain = domain or self.default_domain
        greeting = self.personality_engine.get_greeting(domain)

        self._record_turn(ConversationType.GREETING, "mercury", greeting)

        return VoiceResponse(
            message=greeting,
            confidence=1.0,
            sources_cited=[],
            reasoning_summary=None,
            intent_understood=None,
            search_performed=False,
            search_results_count=0,
            response_time_ms=0.0,
        )

    def _handle_status_query(self, profile: PersonalityProfile) -> VoiceResponse:
        """Handle status query."""
        start_time = time.time()

        stats = self.get_statistics()

        message_parts = [
            "Mercury Agent operational.",
            f"Session active for {(time.time() - self._session_start) / 60:.1f} minutes.",
            f"Queries handled: {stats['queries_handled']}.",
            f"Detections narrated: {stats['detections_narrated']}.",
        ]

        retriever_stats = self.retriever.get_statistics()
        if retriever_stats["knowledge_graph_connected"]:
            message_parts.append("Knowledge graph connected.")
        if retriever_stats["agent_memory_connected"]:
            message_parts.append("Agent memory connected.")

        message = " ".join(message_parts)
        response_time = (time.time() - start_time) * 1000

        return VoiceResponse(
            message=message,
            confidence=1.0,
            sources_cited=["internal_state"],
            reasoning_summary="Status derived from internal metrics",
            intent_understood=QueryIntent.STATUS,
            search_performed=False,
            search_results_count=0,
            response_time_ms=response_time,
            suggested_follow_ups=["View detailed statistics", "Check component health"],
        )

    def _handle_help_query(
        self,
        user_input: str,
        profile: PersonalityProfile,
    ) -> VoiceResponse:
        """Handle help/guidance query."""
        start_time = time.time()

        message = (
            "I can help with anomaly detection, pattern analysis, and STEM exploration. "
            "You can ask me about: "
            "(1) Detection results and their meaning, "
            "(2) Historical patterns and trends, "
            "(3) Knowledge graph relationships, "
            "(4) System status and health. "
            "I provide transparent, evidence-backed responses with explicit confidence levels."
        )

        response_time = (time.time() - start_time) * 1000

        return VoiceResponse(
            message=message,
            confidence=1.0,
            sources_cited=[],
            reasoning_summary=None,
            intent_understood=QueryIntent.HELP,
            search_performed=False,
            search_results_count=0,
            response_time_ms=response_time,
            suggested_follow_ups=[
                "What anomalies have you detected?",
                "Show me recent patterns",
                "Explain your confidence levels",
            ],
        )

    def _handle_general_query(
        self,
        user_input: str,
        search_response: SearchResponse,
        profile: PersonalityProfile,
        domain: str | None,
    ) -> VoiceResponse:
        """Handle general query with search."""
        start_time = time.time()

        # Build response from search results
        if search_response.results:
            message = self._build_search_response(user_input, search_response, profile)
            sources = list(set(r.source.value for r in search_response.results))
            confidence = max(r.relevance_score for r in search_response.results)
        else:
            # No results found
            message = self._build_no_results_response(user_input, profile)
            sources = []
            confidence = 0.3

        # Add uncertainty note if low confidence
        uncertainty_note = None
        if confidence < 0.5:
            uncertainty_note = (
                f"Low confidence ({confidence:.0%}) in this response. "
                "Consider rephrasing your query or providing more context."
            )

        response_time = (time.time() - start_time) * 1000

        return VoiceResponse(
            message=message,
            confidence=confidence,
            sources_cited=sources,
            reasoning_summary=search_response.summary,
            intent_understood=search_response.intent,
            search_performed=True,
            search_results_count=len(search_response.results),
            response_time_ms=response_time,
            suggested_follow_ups=self._generate_follow_ups(search_response),
            uncertainty_note=uncertainty_note,
        )

    def _build_search_response(
        self,
        query: str,
        search_response: SearchResponse,
        profile: PersonalityProfile,
    ) -> str:
        """Build response from search results."""
        parts = []

        n_results = len(search_response.results)
        top_result = search_response.results[0]

        # Opening based on intent
        if search_response.intent == QueryIntent.SEARCH_EVENTS:
            parts.append(f"Found {n_results} relevant event(s) in memory.")
        elif search_response.intent == QueryIntent.SEARCH_PATTERNS:
            parts.append(f"Identified {n_results} pattern(s) related to your query.")
        elif search_response.intent == QueryIntent.SEARCH_FACTS:
            parts.append(f"Found {n_results} relevant fact(s).")
        else:
            parts.append(f"Found {n_results} result(s).")

        # Top result detail
        content = top_result.content
        if isinstance(content, dict):
            if "label" in content:
                parts.append(f"Top match: {content['label']}.")
            elif "fact" in content:
                parts.append(f"Top match: {content['fact']}.")
            elif "event" in content:
                parts.append(f"Top event: {content['event']}.")
            elif "anomaly_score" in content:
                parts.append(
                    f"Detection record: anomaly_score={content['anomaly_score']:.2f}, "
                    f"severity={content.get('severity', 0):.0%}."
                )
        else:
            parts.append(f"Top result: {str(content)[:100]}.")

        # Confidence note
        if profile.express_confidence_level:
            parts.append(f"Relevance: {top_result.relevance_score:.0%}.")

        # Additional context if multiple results
        if n_results > 1:
            sources = set(r.source.value for r in search_response.results)
            parts.append(f"Sources: {', '.join(sources)}.")

        return " ".join(parts)

    def _build_no_results_response(
        self,
        query: str,
        profile: PersonalityProfile,
    ) -> str:
        """Build response when no results found."""
        return (
            f"I couldn't find specific information matching '{query}'. "
            "This could mean: (1) no relevant data in my knowledge base, "
            "(2) the query could be rephrased, or "
            "(3) more context might help. "
            "Can you provide more details or rephrase your question?"
        )

    def _build_detection_message(
        self,
        narrative: NarrativeResult,
        detection_result: dict[str, Any],
    ) -> str:
        """Build message from detection narrative."""
        parts = []

        # Summary
        parts.append(narrative.summary)

        # Key reasoning if available
        if narrative.reasoning_chain and narrative.reasoning_chain.steps:
            parts.append(
                f"Reasoning: {len(narrative.reasoning_chain.steps)} inference steps applied."
            )

        # Historical context
        if narrative.historical_context:
            parts.append(narrative.historical_context)

        # Confidence
        parts.append(f"Confidence: {narrative.confidence_score:.0%}.")

        # Top recommendation
        if narrative.recommendations:
            parts.append(f"Recommendation: {narrative.recommendations[0]}")

        return " ".join(parts)

    def _summarize_reasoning(self, narrative: NarrativeResult) -> str | None:
        """Summarize reasoning chain."""
        if not narrative.reasoning_chain:
            return None

        chain = narrative.reasoning_chain
        return (
            f"{len(chain.steps)} steps, "
            f"{len(chain.hypotheses_considered)} hypotheses considered, "
            f"{len(chain.hypotheses_rejected)} rejected. "
            f"Conclusion: {chain.final_conclusion[:50]}..."
        )

    def _generate_detection_follow_ups(self, narrative: NarrativeResult) -> list[str]:
        """Generate follow-up suggestions for detection."""
        follow_ups = []

        if narrative.reasoning_chain:
            follow_ups.append("Show full reasoning chain")

        if narrative.historical_context:
            follow_ups.append("Show similar historical events")

        if narrative.confidence_score < 0.7:
            follow_ups.append("Explain uncertainty sources")

        follow_ups.append("View detailed analysis")

        return follow_ups[:3]

    def _generate_follow_ups(self, search_response: SearchResponse) -> list[str]:
        """Generate follow-up suggestions for search."""
        follow_ups = []

        if search_response.intent == QueryIntent.SEARCH_EVENTS:
            follow_ups.append("Show event timeline")
        elif search_response.intent == QueryIntent.SEARCH_PATTERNS:
            follow_ups.append("Analyze pattern frequency")

        if search_response.total_found > len(search_response.results):
            follow_ups.append(f"Show more results ({search_response.total_found} total)")

        follow_ups.append("Refine search")

        return follow_ups[:3]

    def _record_turn(
        self,
        turn_type: ConversationType,
        speaker: str,
        content: str,
    ) -> None:
        """Record a conversation turn."""
        self._turn_counter += 1

        turn = ConversationTurn(
            turn_id=f"turn_{self._turn_counter}",
            turn_type=turn_type,
            speaker=speaker,
            content=content[:500],  # Truncate for memory
            timestamp=time.time(),
        )

        self._conversation_history.append(turn)

        # Keep last 100 turns
        if len(self._conversation_history) > 100:
            self._conversation_history = self._conversation_history[-100:]

    def get_conversation_history(
        self,
        n_turns: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent conversation history."""
        return [
            {
                "turn_id": t.turn_id,
                "type": t.turn_type.value,
                "speaker": t.speaker,
                "content": t.content,
                "timestamp": t.timestamp,
            }
            for t in self._conversation_history[-n_turns:]
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get voice statistics."""
        return {
            "queries_handled": self._queries_handled,
            "detections_narrated": self._detections_narrated,
            "alerts_communicated": self._alerts_communicated,
            "conversation_turns": len(self._conversation_history),
            "session_duration_sec": time.time() - self._session_start,
            "llm_enabled": self._llm_adapter is not None,
            "retriever_stats": self.retriever.get_statistics(),
        }


# Factory function
def create_mercury_voice(
    enable_llm: bool = False,
    default_domain: str | None = None,
) -> MercuryVoice:
    """
    Create a Mercury Voice instance.

    Args:
        enable_llm: Whether to enable LLM for response generation
        default_domain: Default domain context

    Returns:
        Configured MercuryVoice
    """
    return MercuryVoice(
        enable_llm=enable_llm,
        default_domain=default_domain,
    )

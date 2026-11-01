# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury Voice - True Conversational Interface.

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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)
from omni_mercury_engine.narrative import conversation as _conv
from omni_mercury_engine.narrative.engine import NarrativeEngine, NarrativeResult
from omni_mercury_engine.narrative.personality import (
    PersonalityEngine,
    PersonalityProfile,
)
from omni_mercury_engine.narrative.retriever import (
    KnowledgeRetriever,
    QueryIntent,
    RetrievalSource,
    SearchResponse,
)
from omni_mercury_engine.security.sigma_immutable_gate import (
    SigmaImmutableGate,
    enforce_dual_ethical_gate,
    get_sigma_immutable_gate,
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
    confidence: float | None  # Confidence in response; None when not reported
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
    """Mercury's Voice - True Conversational Interface.

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
        llm_provider: str | None = None,
        llm_model_name: str | None = None,
        llm_revision: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
    ) -> None:
        """Initialize Mercury Voice.

        Args:
            enable_llm: Whether to use LLM for response generation.
            default_domain: Default domain context.
            llm_provider: Concrete LLM provider name (case-insensitive,
                must match an implemented member of
                :class:`~omni_mercury_engine.models.foundation.llm_adapter.LLMProvider`
                — e.g. ``"huggingface"``, ``"ollama"``).  Required when
                ``enable_llm`` is true and ``MERCURY_ENV=production``;
                in development a missing provider downgrades to
                template-only narration with a warning rather than
                instantiating a mock adapter.
            llm_model_name: Provider-specific model identifier
                (e.g. ``"facebook/bart-large-mnli"`` for HuggingFace,
                ``"llama3.2:3b"`` for Ollama).
            llm_revision: Optional HuggingFace revision pin.  Remote
                HuggingFace model IDs require a verified 40-character
                commit SHA; absolute local model paths do not.
            llm_api_key: Optional API key for providers that require one.
            llm_base_url: Optional override for providers with a
                configurable endpoint (Ollama, OpenAI-compatible
                providers).

        Raises:
            MercuryProductionConfigError: When ``enable_llm`` is true,
                ``MERCURY_ENV=production`` is set, and ``llm_provider``
                is missing or is the ``"mock"`` sentinel.  Mercury
                refuses to silently fall through to a stub LLM in
                production — see ``docs/MIGRATION-1.6-to-1.7.md``.
            ValueError: When ``llm_provider`` is set but does not match
                any implemented :class:`LLMProvider` member.
        """
        self.enable_llm = enable_llm
        self.default_domain = default_domain
        self._llm_provider = llm_provider
        self._llm_model_name = llm_model_name
        self._llm_revision = llm_revision
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url

        # Logger is set up before _init_llm so failure paths can log.
        self.logger = logging.getLogger(__name__)

        # Core components
        self.narrative_engine = NarrativeEngine()
        self.personality_engine = PersonalityEngine()
        self.retriever = KnowledgeRetriever()

        # LLM adapter (if enabled)
        self._llm_adapter: Any | None = None
        if enable_llm:
            self._init_llm()

        # Conversation state
        self._conversation_history: list[ConversationTurn] = []
        self._turn_counter = 0
        #: Per-principal memory of the last grounded answer (kind, argument and
        #: the runs it listed), so "why?", "and yesterday?" and "run 2" resolve
        #: against that account's own previous turn and nobody else's.
        self._context: dict[str, dict[str, Any]] = {}
        self._session_start = time.time()

        # Callbacks for external handlers
        self._command_handlers: dict[str, Callable[[str], str]] = {}

        # Statistics
        self._queries_handled = 0
        self._detections_narrated = 0
        self._alerts_communicated = 0

        # σ_Immutable Wave C: the narrative voice is a public boundary, so
        # both ethical gates are constructed eagerly here (the first
        # concurrent ``speak`` / ``process_detection`` / ``alert`` cannot
        # race the gate into existence).  The benevolence floor is clamped
        # to ``MINIMUM_BENEVOLENCE_FLOOR`` to mirror the engine/orchestrator
        # boundary contract.
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR
        )
        self._sigma_immutable_gate: SigmaImmutableGate = get_sigma_immutable_gate()

    def _enforce_voice_ethics(
        self,
        boundary: str,
        domain: str | None,
        severity: float = 0.0,
        anomaly_prob: float = 0.0,
        extra_details: dict[str, Any] | None = None,
        request: str | None = None,
    ) -> None:
        """Run the harm-uplift + σ_Immutable dual hard gate for the voice path.

        The **caller's own request** is what gets gated. The superseded
        implementation scored a canned description of the voice's good
        intentions (``"audit verify inform protect explain ..."``) and never
        showed the gate what the user actually asked for, so an uplift request
        arriving through ``speak`` was invisible to it. A harm-laden domain
        hint is still collapsed by ``sanitize_domain`` before it can ride into
        the scored text or the audit surface. Both gates fail closed — a
        violation raises :class:`EthicalConstraintViolationError` and the
        voice operation halts (it is *not* wrapped in a swallowing
        ``try/except``).
        """
        from omni_mercury_engine.cognitive.decision_gate import DecisionSubject

        safe_domain = sanitize_domain(domain if domain is not None else self.default_domain)
        enforce_dual_ethical_gate(
            subject=DecisionSubject(
                surface=boundary,
                operation="communicate a detection, alert or answer in natural language",
                domain=safe_domain,
                request=request or "",
                payload=extra_details,
            ),
            sigma_gate=self._sigma_immutable_gate,
            advisory_scorer=self._benevolence_scorer,
            boundary=boundary,
            domain=safe_domain,
            severity=severity,
            anomaly_prob=anomaly_prob,
            extra_details=extra_details,
        )

    def _init_llm(self) -> None:
        """Initialize the configured LLM adapter for enhanced responses.

        Behaviour matrix:

        - ``llm_provider`` unset and ``MERCURY_ENV=production``:
          raise :class:`MercuryProductionConfigError`.  Production must
          opt into a concrete provider; silently running on template-
          only narration is acceptable in dev but not in prod, where
          downstream consumers may rely on LLM-enhanced output.
        - ``llm_provider`` unset and ``MERCURY_ENV=development``:
          log a warning, leave ``self._llm_adapter = None``.  The
          rest of the voice path falls back to deterministic template
          generation.
        - ``llm_provider="mock"``: always raise — the historical
          MockLLMAdapter is a hard-fail stub
          (Phase 2 audit cure; see
          ``models/foundation/llm_adapter.py:MockLLMAdapter``).  We
          surface that misconfiguration here rather than at first
          call so the failure is at construction time.
        - ``llm_provider`` is an implemented provider: delegate to
          ``models.foundation.llm_adapter.create_llm_detector`` and
          store the underlying adapter.  A failure to import the
          provider's optional dependency (e.g. ``transformers`` for
          HuggingFace) is fatal in production and a warning-degrade
          in development, mirroring the unset-provider matrix.

        Replaces the pre-1.7.0 implementation, which unconditionally
        instantiated ``MockLLMAdapter``.  That code path crashed with
        an unhandled ``NotImplementedError`` once the Phase 2 audit
        cure made the mock hard-fail at construction, because the
        surrounding ``except ImportError`` did not catch it.
        """
        from omni_mercury_engine._env import (
            MercuryProductionConfigError,
            is_production,
            require_real_component,
        )

        if not self._llm_provider:
            require_real_component(
                "narrative LLM provider",
                remediation=(
                    "Pass llm_provider=<provider> (e.g. "
                    '"huggingface", "ollama") plus llm_model_name to '
                    "MercuryVoice() / create_mercury_voice(), or unset "
                    "enable_llm to use template-only narration."
                ),
            )
            self.logger.warning(
                "MercuryVoice(enable_llm=True) called without "
                "llm_provider; falling back to template-only narration. "
                "Set llm_provider= and llm_model_name= to enable "
                "LLM-enhanced responses."
            )
            return

        if self._llm_provider.lower() == "mock":
            raise MercuryProductionConfigError(
                "MercuryVoice does not support llm_provider='mock'. "
                "MockLLMAdapter hard-fails at construction by design "
                "(Phase 2 audit cure).  Configure a real provider "
                "such as 'huggingface' or 'ollama', or omit "
                "llm_provider to use template-only narration."
            )

        # Validate provider name up-front so an unknown value produces a
        # clean ValueError naming all supported providers, rather than
        # routing through create_llm_detector's silent mock-fallback and
        # then exploding on MockLLMAdapter's NotImplementedError two
        # frames away from the call site.
        from omni_mercury_engine.models.foundation.llm_adapter import (
            IMPLEMENTED_LLM_PROVIDERS,
            LLMProvider,
        )

        try:
            provider_enum = LLMProvider(self._llm_provider.lower())
        except ValueError as exc:
            supported = sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS)
            raise ValueError(
                f"Unknown llm_provider {self._llm_provider!r}.  Supported providers: {supported}."
            ) from exc
        if provider_enum not in IMPLEMENTED_LLM_PROVIDERS:
            supported = sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS)
            raise ValueError(
                f"llm_provider {self._llm_provider!r} is declared in LLMProvider "
                f"but has no MercuryVoice adapter implementation in this build. "
                f"Supported providers: {supported}."
            )

        try:
            if provider_enum == LLMProvider.HUGGINGFACE:
                from pathlib import PurePosixPath, PureWindowsPath

                if not self._llm_model_name:
                    raise ValueError(
                        "MercuryVoice llm_provider='huggingface' requires "
                        "llm_model_name=<HuggingFace model ID or absolute local path>."
                    )
                model_name = self._llm_model_name
                is_local_path = (
                    PurePosixPath(model_name).is_absolute()
                    or PureWindowsPath(model_name).is_absolute()
                )
                if not is_local_path and not self._llm_revision:
                    raise ValueError(
                        "HuggingFace remote model IDs require "
                        "llm_revision=<40-character commit SHA> so SafeHFLoader "
                        "can enforce reproducible model loading."
                    )

            from omni_mercury_engine.models.foundation.llm_adapter import (
                create_llm_detector,
            )

            detector = create_llm_detector(
                provider=self._llm_provider,
                model_name=self._llm_model_name,
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
                revision=self._llm_revision,
            )
            adapter = detector.adapter
            is_available = getattr(adapter, "is_available", None)
            if callable(is_available) and not bool(is_available()):
                raise ImportError(
                    f"adapter for provider {self._llm_provider!r} reported unavailable"
                )
            self._llm_adapter = adapter
        except (
            ImportError,
            NotImplementedError,
            ValueError,
        ) as exc:
            if is_production():
                raise MercuryProductionConfigError(
                    "MercuryVoice failed to initialise LLM provider "
                    f"{self._llm_provider!r} in production: {exc}.  "
                    "Install the provider's optional dependency or "
                    "select a different llm_provider."
                ) from exc
            self.logger.warning(
                "MercuryVoice LLM provider %r unavailable in "
                "development (%s); falling back to template-only "
                "narration.",
                self._llm_provider,
                exc,
            )
            self._llm_adapter = None

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
        principal: str | None = None,
    ) -> VoiceResponse:
        """Process user input and generate response.

        This is the main entry point for conversational interaction.

        Args:
            user_input: Natural language input from user
            domain: Optional domain context
            principal: The account asking (``"anonymous"`` when none);
                detection-record answers are bounded to that account's runs

        Returns:
            VoiceResponse with truth-dense communication
        """
        # σ_Immutable Wave C dual hard ethical gate (benevolence + σ_Immutable)
        # before any conversational work.  Fails closed.
        self._enforce_voice_ethics("MercuryVoice.speak", domain, request=user_input)

        # Response timing available via time.time() for future performance tracking
        self._queries_handled += 1
        # Rebind to the same canonical safe label the ethical gate evaluated, so
        # personality lookup, retriever search, and turn logging below can never
        # receive an arbitrary caller-supplied domain string.
        domain = sanitize_domain(domain or self.default_domain)

        # Record user turn
        self._record_turn(ConversationType.QUERY, "user", user_input)

        # Get personality profile
        profile = self.personality_engine.get_profile(domain)

        # One classifier names the grounded answer a question gets (record,
        # code, narrative engine); only a question none of them covers reaches
        # the free-text search, and an empty search ends in a menu, not an
        # apology.
        input_lower = user_input.lower()
        kind, argument = _conv.classify(user_input)
        if kind == "search":
            follow = _conv.follow_up(user_input)
            if follow is not None:
                kind, argument = self._resolve_follow_up(follow, domain, principal)
        response = self._dispatch(
            kind, argument, user_input, input_lower, domain, principal, profile
        )
        self._remember_context(kind, argument, domain, principal)

        # Record Mercury's response
        self._record_turn(ConversationType.QUERY, "mercury", response.message)

        return response

    def process_detection(
        self,
        detection_result: dict[str, Any],
        domain: str | None = None,
    ) -> VoiceResponse:
        """Process a detection result and generate voice response.

        Args:
            detection_result: Detection output to communicate
            domain: Optional domain context

        Returns:
            VoiceResponse with full narrative
        """
        # σ_Immutable Wave C dual hard ethical gate.  Severity / anomaly
        # signal is sourced from the detection being narrated so the
        # σ_Immutable verdict tracks the actual payload.  Fails closed.
        det_severity = float(detection_result.get("severity", 0.0) or 0.0)
        det_anomaly = float(
            detection_result.get("anomaly_prob", detection_result.get("anomaly_score", 0.0)) or 0.0
        )
        self._enforce_voice_ethics(
            "MercuryVoice.process_detection",
            domain,
            severity=det_severity,
            anomaly_prob=det_anomaly,
            extra_details={"is_anomaly": bool(detection_result.get("is_anomaly", False))},
            request=str(detection_result.get("summary", "") or ""),
        )

        start_time = time.time()
        self._detections_narrated += 1
        # Rebind to the same canonical safe label the ethical gate evaluated, so
        # detection logging and narrative synthesis below cannot receive an
        # arbitrary caller-supplied domain string.
        domain = sanitize_domain(domain or self.default_domain)

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
                narrative.uncertainty_disclosure
                if narrative.confidence_score is None or narrative.confidence_score < 0.7
                else None
            ),
        )

    def alert(
        self,
        alert_content: dict[str, Any],
        alert_type: str = "anomaly",
        domain: str | None = None,
    ) -> VoiceResponse:
        """Generate alert communication.

        Args:
            alert_content: Content of the alert
            alert_type: Type of alert
            domain: Optional domain context.  Sanitised via
                ``sanitize_domain`` before it reaches the σ_Immutable dual
                gate or the audit surface; defaults to the voice's
                configured ``default_domain`` when omitted.

        Returns:
            VoiceResponse for the alert
        """
        # Build alert message
        severity = alert_content.get("severity", 0.5)
        confidence = alert_content.get("confidence", 0.5)

        # σ_Immutable Wave C dual hard ethical gate.  Alert severity drives
        # the σ_Immutable signal window; confidence is read as the anomaly
        # signal.  Fails closed before any alert is emitted.
        self._enforce_voice_ethics(
            "MercuryVoice.alert",
            domain,
            severity=float(severity or 0.0),
            anomaly_prob=float(confidence or 0.0),
            extra_details={"alert_type": alert_type},
            request=str(alert_content.get("summary", "") or ""),
        )

        start_time = time.time()
        self._alerts_communicated += 1
        # Profile available via self.personality_engine.get_profile() for future alert styling

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
        """Generate session greeting.

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

    def _handle_status_query(
        self, profile: PersonalityProfile, principal: str | None = None, detailed: bool = False
    ) -> VoiceResponse:
        """Handle status query; ``detailed`` adds component and retriever figures."""
        start_time = time.time()

        stats = self.get_statistics()

        message_parts = [
            "Mercury Agent operational.",
            f"Session active for {(time.time() - self._session_start) / 60:.1f} minutes.",
            f"Queries handled: {stats['queries_handled']}.",
            f"Detections narrated: {stats['detections_narrated']}.",
        ]
        remembered = self.retriever.detections(principal=principal)
        if remembered:
            flagged = sum(1 for e in remembered if e["detection"].get("anomaly_detected", False))
            message_parts.append(
                f"Detection runs in memory: {len(remembered)} ({flagged} flagged at least "
                f"one sample); the newest was recorded {_conv.relative_time(remembered[0]['timestamp'])}."
            )
        else:
            message_parts.append(
                "No detection runs in memory yet: memory holds the runs made through this "
                "server since it started."
            )

        retriever_stats = self.retriever.get_statistics()
        if retriever_stats["knowledge_graph_connected"]:
            message_parts.append("Knowledge graph connected.")
        if retriever_stats["agent_memory_connected"]:
            message_parts.append("Agent memory connected.")
        if detailed:
            message_parts.append(
                f"Retriever: {retriever_stats['search_count']} searches, "
                f"{retriever_stats['total_results_returned']} results returned, "
                f"{retriever_stats['detection_log_size']} runs in the detection log; "
                f"knowledge graph {'connected' if retriever_stats['knowledge_graph_connected'] else 'not connected'}, "
                f"agent memory {'connected' if retriever_stats['agent_memory_connected'] else 'not connected'}, "
                f"memory surface {'connected' if retriever_stats['memory_surface_connected'] else 'not connected'}."
            )
            message_parts.append(
                "Language model: "
                + (
                    "attached."
                    if self._llm_adapter is not None
                    else "none attached; answers are template-built from the record and the code."
                )
            )
            try:
                from omni_mercury_engine import self_model

                states = []
                for spec in self_model.CAPABILITIES:
                    probe = self_model._probe_capability(spec)
                    states.append(f"{spec.name.replace('_', ' ')} {probe.get('state', 'unknown')}")
                message_parts.append("Components: " + "; ".join(states) + ".")
            except Exception:  # pragma: no cover - self-model probe failure
                message_parts.append("Components: self-model probe unavailable.")

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
            suggested_follow_ups=(
                [
                    "What was the latest anomaly?",
                    "What are your capabilities?",
                    "What should I run next?",
                ]
                if detailed
                else [
                    "View detailed statistics",
                    "Check component health",
                    "What was the latest anomaly?",
                ]
            ),
        )

    def _handle_help_query(
        self,
        user_input: str,
        profile: PersonalityProfile,
    ) -> VoiceResponse:
        """Handle help/guidance query."""
        start_time = time.time()

        message = (
            "I answer from this server's own record and code. About the record: "
            '"What was the latest anomaly?", "Which samples were flagged?", "What threshold '
            'did you use?", "Show me the last 3 runs", "How many anomalies today?" (also this '
            'hour, yesterday, this week), "Compare today with yesterday", "Summarize threat '
            'posture". Through the narrative engine: "Explain the latest detection", "Show full '
            'reasoning chain", "Explain uncertainty sources", "Recommend an action". About the '
            'code: "Which detectors do you have?", "Can you detect space weather?", "What are '
            'your capabilities?", "What should I run next?", "How do I collect data?", and '
            'definitions ("What is a z-score?", severity, threshold, AMA, FINDΩYOU, the gate). '
            '"Status" gives system health. Every answer names what it is built from; a field a '
            "run did not report is left out rather than filled, and I never execute a command."
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
                "What was the latest anomaly?",
                "Which detectors do you have?",
                "What are your capabilities?",
            ],
        )

    def _handle_explain_latest(
        self,
        domain: str | None,
        principal: str | None,
        variant: str = "narrative",
        entry: dict[str, Any] | None = None,
    ) -> VoiceResponse:
        """Narrate the newest recorded run through the narrative engine.

        The record is turned into the detection result the engine reads
        (verdict, score, severity fraction, the run's facts as observation
        steps) and synthesised like any other detection, behind the same
        ethical gate. Nothing is re-logged: the run is already in memory.
        """
        start_time = time.time()
        if entry is None:
            entry = self._newest(domain, principal)
        if entry is None:
            return VoiceResponse(
                message=(
                    "There is no recorded run to explain yet. Memory holds the runs made "
                    "through this server since it started; run detect and ask again."
                ),
                confidence=None,
                sources_cited=[],
                reasoning_summary="No recorded run.",
                intent_understood=QueryIntent.EXPLAIN_CONCEPT,
                search_performed=True,
                search_results_count=0,
                response_time_ms=(time.time() - start_time) * 1000,
                suggested_follow_ups=["How do I collect data?", "What can you do?"],
            )

        record = entry["detection"]
        narrative, detection_result = self._narrate_entry(entry, domain)
        described = _conv.describe_detection(record, entry["timestamp"])
        if variant == "reasoning":
            chain = narrative.reasoning_chain
            steps = chain.steps if chain else []
            message = (
                f"Reasoning for the {described} "
                + (
                    " ".join(f"({i + 1}) {step}" for i, step in enumerate(steps))
                    if steps
                    else "The engine recorded no reasoning steps for this run."
                )
                + (
                    f" Conclusion: {chain.final_conclusion}"
                    if chain and chain.final_conclusion
                    else ""
                )
                + (f" {chain.confidence_rationale}" if chain and chain.confidence_rationale else "")
            )
        elif variant == "uncertainty":
            message = (
                f"Uncertainty for the {described} {narrative.uncertainty_disclosure} "
                + (
                    f"Confidence reported: {narrative.confidence_score:.0%}."
                    if narrative.confidence_score is not None
                    else "The detector reported no confidence, so none is stated; the severity band "
                    "(peak score against threshold) is the only strength signal."
                )
                + (
                    " Fields the run did not report are left out of every answer rather than filled."
                )
            )
        elif variant == "dossier":
            chain = narrative.reasoning_chain
            parts = [f"Detailed analysis of the {described}", narrative.summary]
            if (
                narrative.detailed_explanation
                and narrative.detailed_explanation != narrative.summary
            ):
                parts.append(narrative.detailed_explanation)
            if chain and chain.steps:
                parts.append(
                    "Reasoning: " + " ".join(f"({i + 1}) {s}" for i, s in enumerate(chain.steps))
                )
            parts.append(narrative.uncertainty_disclosure)
            parts.append(f"Urgency: {narrative.urgency_level}.")
            if narrative.recommendations:
                parts.append(
                    "Recommendations: "
                    + " ".join(f"({i + 1}) {r}" for i, r in enumerate(narrative.recommendations))
                )
            if narrative.next_steps:
                parts.append(
                    "Next steps: "
                    + " ".join(f"({i + 1}) {r}" for i, r in enumerate(narrative.next_steps))
                )
            message = " ".join(p for p in parts if p)
        else:
            message = (
                "Explaining "
                + described
                + " "
                + self._build_detection_message(narrative, detection_result)
            )
        return VoiceResponse(
            message=message,
            confidence=narrative.confidence_score,
            sources_cited=["detection_log", "narrative_synthesis"],
            reasoning_summary=self._summarize_reasoning(narrative),
            intent_understood=QueryIntent.EXPLAIN_CONCEPT,
            search_performed=True,
            search_results_count=1,
            response_time_ms=(time.time() - start_time) * 1000,
            suggested_follow_ups=self._generate_detection_follow_ups(narrative, variant),
            uncertainty_note=(
                narrative.uncertainty_disclosure
                if narrative.confidence_score is None or narrative.confidence_score < 0.7
                else None
            ),
        )

    def _handle_capability_query(self, input_lower: str) -> VoiceResponse:
        """Answer "which detectors / what can you detect / what are your capabilities" from code.

        Detector facts come from ``core.detector_registry.DETECTOR_MANIFEST``
        (name, family, input modality, one-line description); capability facts
        from ``self_model.CAPABILITIES`` probed on this machine, so an
        unavailable backend or an operator switch is reported as such.
        """
        start_time = time.time()
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        wants_capabilities = any(w in input_lower for w in _CAPABILITY_WORDS)
        parts: list[str] = []
        sources: list[str] = []

        # Detectors: a named family or input type narrows the list. "Space
        # weather" names the space family, so the phrase is matched whole before
        # "weather" can pull in the meteorological one.
        family_text = input_lower.replace("space weather", "space")
        family_hits = [f for f in _FAMILY_WORDS if any(w in family_text for w in _FAMILY_WORDS[f])]
        modality_hits = [
            m for m in _MODALITY_WORDS if any(w in input_lower for w in _MODALITY_WORDS[m])
        ]
        entries = list(DETECTOR_MANIFEST)
        if family_hits:
            entries = [e for e in entries if e.category.value in family_hits]
        if modality_hits:
            entries = [e for e in entries if e.modality in modality_hits]
        if family_hits or modality_hits or not wants_capabilities:
            sources.append("detector_manifest")
            total = len(DETECTOR_MANIFEST)
            if family_hits or modality_hits:
                scope = " and ".join(
                    [f"the {f} family" for f in family_hits]
                    + [f"{_MODALITY_LABELS[m]} input" for m in modality_hits]
                )
                if entries:
                    listed = "; ".join(
                        f"{e.name} ({e.description.rstrip('.')})" for e in entries[:8]
                    )
                    more = f"; and {len(entries) - 8} more" if len(entries) > 8 else ""
                    parts.append(
                        f"{len(entries)} of the {total} manifest detectors cover {scope}: "
                        f"{listed}{more}."
                    )
                else:
                    parts.append(
                        f"None of the {total} manifest detectors is registered for {scope}."
                    )
            else:
                by_family: dict[str, int] = {}
                for e in DETECTOR_MANIFEST:
                    by_family[e.category.value] = by_family.get(e.category.value, 0) + 1
                families = ", ".join(
                    f"{k} {v}" for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])
                )
                by_modality: dict[str, int] = {}
                for e in DETECTOR_MANIFEST:
                    by_modality[e.modality] = by_modality.get(e.modality, 0) + 1
                modalities = ", ".join(
                    f"{_MODALITY_LABELS.get(k, k)} {v}"
                    for k, v in sorted(by_modality.items(), key=lambda kv: -kv[1])
                )
                parts.append(
                    f"The manifest registers {total} detectors. By family: {families}. "
                    f"By input: {modalities}. Name a family or an input type and I will list them."
                )

        if wants_capabilities or not parts:
            from omni_mercury_engine import self_model

            sources.append("self_model")
            lines = []
            for spec in self_model.CAPABILITIES:
                probe = self_model._probe_capability(spec)
                state = str(probe.get("state", "unknown"))
                reason = probe.get("reason")
                # The reason is the interesting part only when the capability
                # is not grounded (a missing backend, an operator switch).
                lines.append(
                    f"{spec.name.replace('_', ' ')}: {state}"
                    + (f" ({reason})" if reason and state != "grounded" else "")
                    + f" — {spec.description}"
                )
            parts.append("Capabilities on this machine: " + "; ".join(lines) + ".")

        return VoiceResponse(
            message=" ".join(parts),
            confidence=1.0,
            sources_cited=sources,
            reasoning_summary="Read from the detector manifest and the self-model probe.",
            intent_understood=QueryIntent.SEARCH_FACTS,
            search_performed=False,
            search_results_count=len(entries) if "detector_manifest" in sources else 0,
            response_time_ms=(time.time() - start_time) * 1000,
            suggested_follow_ups=[
                prompt
                for prompt in (
                    "What are your capabilities?",
                    "Which detectors handle images?",
                    "Can you detect space weather?",
                    "What was the latest anomaly?",
                )
                if prompt.lower().rstrip("?") not in input_lower
            ][:3],
        )

    def _handle_detection_summary(
        self, input_lower: str, domain: str | None, principal: str | None = None
    ) -> VoiceResponse:
        """Count and summarise the recorded detection runs in the asked window.

        Every number comes from the retriever's detection log; nothing is
        estimated. When the window holds no runs the answer says so and says
        what memory covers, instead of apologising generically.
        """
        start_time = time.time()
        window_label, since, until = _conv.parse_window(input_lower)
        entries = _conv.within(
            self.retriever.detections(domain=domain, since=since, principal=principal), until
        )
        flagged = [e for e in entries if e["detection"].get("anomaly_detected", False)]

        if not entries:
            message = (
                f"No detection runs are recorded in memory for {window_label}. Memory holds "
                "the runs made through this server since it started; run detect, or ask "
                "me about a different period."
            )
            follow_ups = ["Summarize all detections", "What can you do?"]
        else:
            by_method: dict[str, list[int]] = {}
            for e in entries:
                det = e["detection"]
                method = str(det.get("method") or "unknown")
                tally = by_method.setdefault(method, [0, 0])
                tally[0] += 1
                if det.get("anomaly_detected", False):
                    tally[1] += 1
            per_method = "; ".join(
                f"{m}: {n} run{'s' if n != 1 else ''}, {k} flagged"
                for m, (n, k) in sorted(by_method.items())
            )
            parts = [
                f"For {window_label}, {len(entries)} detection run"
                f"{'s' if len(entries) != 1 else ''} recorded: {len(flagged)} flagged at least "
                f"one sample, {len(entries) - len(flagged)} clean.",
                f"By method, {per_method}.",
            ]
            if flagged:
                peak = max(flagged, key=lambda e: float(e["detection"].get("anomaly_score") or 0.0))
                parts.append(
                    "Highest-scoring run: "
                    + _conv.describe_detection(peak["detection"], peak["timestamp"])
                )
            message = " ".join(parts)
            follow_ups = [
                "What was the latest anomaly?",
                "Narrate the latest detection",
                "Which samples were flagged?",
            ]

        return VoiceResponse(
            message=message,
            confidence=1.0 if entries else None,
            sources_cited=["detection_log"] if entries else [],
            reasoning_summary=(
                f"Counted {len(entries)} recorded run(s) for {window_label}"
                + (f" in domain {domain}" if domain and domain != "general" else "")
                + "."
            ),
            intent_understood=QueryIntent.SEARCH_EVENTS,
            search_performed=True,
            search_results_count=len(entries),
            response_time_ms=(time.time() - start_time) * 1000,
            suggested_follow_ups=follow_ups,
        )

    def _handle_latest_detection(
        self, domain: str | None, principal: str | None = None
    ) -> VoiceResponse:
        """Describe the newest run that flagged an anomaly, or the newest run if none did."""
        start_time = time.time()
        flagged = self.retriever.detections(domain=domain, anomalous_only=True, principal=principal)
        every = self.retriever.detections(domain=domain, principal=principal)

        if flagged:
            entry = flagged[0]
            message = "Latest anomaly: " + _conv.describe_detection(
                entry["detection"], entry["timestamp"]
            )
            if every and every[0] is not entry:
                message += (
                    f" {len(every) - every.index(entry) - 1} clean run(s) have been recorded "
                    "since."
                )
            follow_ups = [
                "Narrate the latest detection",
                "How many anomalies today?",
                "Explain this detection",
            ]
            count = len(flagged)
        elif every:
            entry = every[0]
            message = (
                "No recorded run has flagged an anomaly. The newest run was "
                + _conv.describe_detection(entry["detection"], entry["timestamp"])
            )
            follow_ups = ["Summarize all detections", "What should I run next?"]
            count = len(every)
        else:
            message = (
                "No detection runs are recorded in memory yet. Memory holds the runs made "
                "through this server since it started; run detect and ask again."
            )
            follow_ups = ["What can you do?", "How do I collect data?"]
            count = 0

        return VoiceResponse(
            message=message,
            confidence=1.0 if every else None,
            sources_cited=["detection_log"] if every else [],
            reasoning_summary=f"Read the newest of {len(every)} recorded run(s).",
            intent_understood=QueryIntent.SEARCH_EVENTS,
            search_performed=True,
            search_results_count=count,
            response_time_ms=(time.time() - start_time) * 1000,
            suggested_follow_ups=follow_ups,
        )

    def _dispatch(
        self,
        kind: str,
        argument: Any,
        user_input: str,
        input_lower: str,
        domain: str,
        principal: str | None,
        profile: PersonalityProfile,
    ) -> VoiceResponse:
        """Route a classified question to the handler that answers it."""
        if kind == "status":
            return self._handle_status_query(profile, principal, detailed=argument == "detailed")
        if kind == "help":
            return self._handle_help_query(user_input, profile)
        if kind == "greeting":
            return self._social("greeting", domain)
        if kind == "thanks":
            return self._social("thanks", domain)
        if kind == "identity":
            return self._handle_identity()
        if kind == "glossary":
            return self._handle_glossary(str(argument))
        if kind == "gate":
            return self._handle_gate(domain, principal)
        if kind == "recommend":
            return self._handle_recommend(domain, principal)
        if kind == "next_run":
            return self._handle_next_run(domain, principal)
        if kind == "collect":
            return self._handle_collect()
        if kind == "run_command":
            return self._handle_run_command(str(argument), user_input)
        if kind == "posture":
            return self._handle_posture(domain, principal)
        if kind == "compare":
            return self._handle_compare(input_lower, domain, principal)
        if kind == "detail":
            return self._handle_run_detail(str(argument), domain, principal)
        if kind == "last_n":
            return self._handle_last_runs(int(argument), domain, principal)
        if kind == "explain":
            return self._handle_explain_latest(domain, principal, variant=str(argument))
        if kind == "explain_entry":
            variant, entry = argument
            return self._handle_explain_latest(domain, principal, variant=str(variant), entry=entry)
        if kind == "no_such_entry":
            asked, shown = argument
            return self._grounded(
                (
                    f"The last list had {shown} run{'s' if shown != 1 else ''}, so there is no run "
                    f"{asked}. Ask for 1 to {shown}, or 'Show me the last 10 runs' for a longer list.",
                    ["Show me the last 10 runs", "What was the latest anomaly?"],
                ),
                ["detection_log"],
                QueryIntent.SEARCH_EVENTS,
                "Ordinal outside the last listing.",
                time.time(),
            )
        if kind == "confidence":
            return self._handle_confidence(domain, principal)
        if kind == "summary":
            return self._handle_detection_summary(input_lower, domain, principal)
        if kind == "latest":
            return self._handle_latest_detection(domain, principal)
        if kind == "capability":
            return self._handle_capability_query(input_lower)
        search_response = self.retriever.search(user_input, domain=domain, principal=principal)
        return self._handle_general_query(user_input, search_response, profile, domain)

    _RECORD_KINDS = frozenset(
        {
            "latest",
            "detail",
            "summary",
            "last_n",
            "posture",
            "compare",
            "recommend",
            "confidence",
            "gate",
        }
    )

    def _remember_context(
        self, kind: str, argument: Any, domain: str | None, principal: str | None
    ) -> None:
        """Keep what this principal was just told, bounded to the newest 256 principals."""
        key = principal or "anonymous"
        context: dict[str, Any] = {"kind": kind, "argument": argument}
        if kind == "last_n":
            context["listing"] = self._entries(domain, principal)[: int(argument)]
        elif key in self._context and "listing" in self._context[key]:
            # "run 2" then "the first one" both refer to the last list shown
            context["listing"] = self._context[key]["listing"]
        self._context.pop(key, None)
        self._context[key] = context
        while len(self._context) > 256:
            self._context.pop(next(iter(self._context)))

    def _resolve_follow_up(
        self, follow: tuple[str, Any], domain: str | None, principal: str | None
    ) -> tuple[str, Any]:
        """Turn "why?", "and yesterday?" or "run 2" into a grounded answer using the last turn."""
        context = self._context.get(principal or "anonymous") or {}
        previous = str(context.get("kind") or "")
        mode, value = follow
        if mode == "ordinal":
            listing = list(context.get("listing") or [])
            if not listing:
                return "last_n", 5
            index = len(listing) - 1 if value == -1 else int(value) - 1
            if 0 <= index < len(listing):
                return "explain_entry", ("narrative", listing[index])
            return "no_such_entry", (value, len(listing))
        if mode == "window":
            return ("compare", None) if previous == "compare" else ("summary", None)
        if previous == "explain":
            return "explain", "dossier"
        if previous == "explain_entry":
            return "explain_entry", ("dossier", context["argument"][1])
        if previous in self._RECORD_KINDS:
            return "explain", "narrative"
        if previous == "glossary":
            term = str(context.get("argument") or "")
            related = {
                "gate": ("gate", None),
                "ama": ("glossary", "decoupling"),
                "findoyou": ("glossary", "decoupling"),
                "decoupling": ("glossary", "ama"),
                "severity": ("glossary", "threshold"),
                "threshold": ("glossary", "z-score"),
                "z-score": ("glossary", "anomaly score"),
                "anomaly score": ("glossary", "severity"),
                "confidence": ("confidence", None),
                "contamination": ("glossary", "conformal"),
                "conformal": ("glossary", "contamination"),
            }
            return related.get(term, ("capability", None))
        if previous in {"capability", "identity", "next_run", "collect", "run_command"}:
            return "capability", None
        return "help", None

    @staticmethod
    def _grounded(
        answer: tuple[str, list[str]],
        sources: list[str],
        intent: QueryIntent,
        reasoning: str,
        started: float,
        *,
        results: int = 0,
        confidence: float | None = 1.0,
    ) -> VoiceResponse:
        """Wrap a builder's ``(message, follow_ups)`` in a response that names its sources."""
        message, follow_ups = answer
        return VoiceResponse(
            message=message,
            confidence=confidence,
            sources_cited=sources,
            reasoning_summary=reasoning,
            intent_understood=intent,
            search_performed=bool(results),
            search_results_count=results,
            response_time_ms=(time.time() - started) * 1000,
            suggested_follow_ups=follow_ups,
        )

    def _entries(self, domain: str | None, principal: str | None) -> list[dict[str, Any]]:
        """The caller's recorded runs, newest first."""
        return list(self.retriever.detections(domain=domain, principal=principal))

    def _newest(self, domain: str | None, principal: str | None) -> dict[str, Any] | None:
        """The newest flagged run, or the newest run when none flagged, or ``None``."""
        flagged = self.retriever.detections(domain=domain, anomalous_only=True, principal=principal)
        if flagged:
            return dict(flagged[0])
        every = self._entries(domain, principal)
        return every[0] if every else None

    def _social(self, kind: str, domain: str) -> VoiceResponse:
        """Greetings and thanks: short, and pointed at what the agent answers."""
        started = time.time()
        if kind == "greeting":
            message = (
                f"Hello. I am Mercury Agent, answering from this server's detection record and "
                f"its code in the {domain} domain. Ask what ran, what was flagged, what to run "
                "next, or what any of Mercury's terms mean."
            )
        else:
            message = "You are welcome. Ask again whenever a run lands in memory."
        return self._grounded(
            (
                message,
                ["What was the latest anomaly?", "What can you do?", "What should I run next?"],
            ),
            [],
            QueryIntent.HELP,
            "Conversational opener.",
            started,
        )

    def _handle_identity(self) -> VoiceResponse:
        """Who the agent is, from the installed package."""
        started = time.time()
        from omni_mercury_engine import __version__

        return self._grounded(
            _conv.identity(__version__, self._installation_facts()),
            ["self_model", "detector_manifest"],
            QueryIntent.HELP,
            "Read from the package version, the detector manifest and the loader package.",
            started,
        )

    def _handle_glossary(self, term: str) -> VoiceResponse:
        """Define one of Mercury's terms with the numbers read from the code that uses it."""
        started = time.time()
        return self._grounded(
            _conv.glossary_answer(term, self._installation_facts()),
            ["code"],
            QueryIntent.EXPLAIN_CONCEPT,
            f"Definition of '{term}' read from the module that implements it.",
            started,
        )

    def _handle_gate(self, domain: str | None, principal: str | None) -> VoiceResponse:
        """Explain the two hard ethical gates every reply passes."""
        started = time.time()
        latest = self._newest(domain, principal)
        return self._grounded(
            _conv.gate_explanation(latest),
            ["code"] + (["detection_log"] if latest else []),
            QueryIntent.EXPLAIN_CONCEPT,
            "Read from cognitive/ethical_bounding.py and security/sigma_immutable_gate.py.",
            started,
            results=1 if latest else 0,
        )

    def _handle_confidence(self, domain: str | None, principal: str | None) -> VoiceResponse:
        """What confidence means here and what the newest run reported."""
        started = time.time()
        latest = self._newest(domain, principal)
        return self._grounded(
            _conv.confidence_answer(latest),
            ["code"] + (["detection_log"] if latest else []),
            QueryIntent.EXPLAIN_CONCEPT,
            "Confidence policy from the narrative engine; run facts from the detection log.",
            started,
            results=1 if latest else 0,
        )

    def _handle_run_detail(
        self, aspect: str, domain: str | None, principal: str | None
    ) -> VoiceResponse:
        """One field of the newest run: flagged samples, threshold, method, score or time."""
        started = time.time()
        latest = self._newest(domain, principal)
        return self._grounded(
            _conv.run_detail(aspect, latest),
            ["detection_log"] if latest else [],
            QueryIntent.SEARCH_EVENTS,
            f"Read the {aspect} of the newest recorded run.",
            started,
            results=1 if latest else 0,
            confidence=1.0 if latest else None,
        )

    def _handle_last_runs(self, n: int, domain: str | None, principal: str | None) -> VoiceResponse:
        """Number the newest ``n`` runs, newest first."""
        started = time.time()
        entries = self._entries(domain, principal)
        return self._grounded(
            _conv.last_runs(entries, n),
            ["detection_log"] if entries else [],
            QueryIntent.SEARCH_EVENTS,
            f"Listed the newest {min(n, len(entries))} of {len(entries)} recorded run(s).",
            started,
            results=min(n, len(entries)),
            confidence=1.0 if entries else None,
        )

    def _handle_posture(self, domain: str | None, principal: str | None) -> VoiceResponse:
        """The whole record at a glance."""
        started = time.time()
        entries = self._entries(domain, principal)
        return self._grounded(
            _conv.posture(entries),
            ["detection_log"] if entries else [],
            QueryIntent.SEARCH_PATTERNS,
            f"Band distribution over {len(entries)} recorded run(s).",
            started,
            results=len(entries),
            confidence=1.0 if entries else None,
        )

    def _handle_compare(
        self, input_lower: str, domain: str | None, principal: str | None
    ) -> VoiceResponse:
        """Counts for two time windows side by side."""
        started = time.time()

        def fetch(since: float | None) -> list[dict[str, Any]]:
            return list(self.retriever.detections(domain=domain, since=since, principal=principal))

        entries = self._entries(domain, principal)
        return self._grounded(
            _conv.compare_windows(input_lower, fetch),
            ["detection_log"] if entries else [],
            QueryIntent.COMPARE,
            "Counted the recorded runs in each window.",
            started,
            results=len(entries),
            confidence=1.0 if entries else None,
        )

    def _handle_recommend(self, domain: str | None, principal: str | None) -> VoiceResponse:
        """The narrative engine's recommendations for the newest flagged run."""
        started = time.time()
        latest = self._newest(domain, principal)
        if latest is None or not latest["detection"].get("anomaly_detected", False):
            lead = (
                "No recorded run has flagged a sample, so there is nothing to act on from the record. "
                if latest is not None
                else "Nothing has been recorded yet, so there is nothing to act on from the record. "
            )
            message, follow_ups = _conv.next_steps(
                self._entries(domain, principal), self._cli_commands()
            )
            return self._grounded(
                (lead + message, follow_ups),
                ["detection_log", "cli"] if latest else ["cli"],
                QueryIntent.PREDICT,
                "No flagged run; recommended the CLI's next commands instead.",
                started,
            )
        narrative, result = self._narrate_entry(latest, domain)
        parts = [
            "For the " + _conv.describe_detection(latest["detection"], latest["timestamp"]),
            f"Urgency: {narrative.urgency_level}.",
        ]
        if narrative.recommendations:
            parts.append(
                "Recommendations: "
                + " ".join(f"({i + 1}) {r}" for i, r in enumerate(narrative.recommendations[:4]))
            )
        if narrative.next_steps:
            parts.append(
                "Next steps: "
                + " ".join(f"({i + 1}) {r}" for i, r in enumerate(narrative.next_steps[:4]))
            )
        return self._grounded(
            (
                " ".join(parts),
                [
                    "What should I run next?",
                    "Explain the latest detection",
                    "Which samples were flagged?",
                ],
            ),
            ["detection_log", "narrative_synthesis"],
            QueryIntent.PREDICT,
            self._summarize_reasoning(narrative) or "Narrative engine recommendations.",
            started,
            results=1,
            confidence=narrative.confidence_score,
        )

    def _handle_next_run(self, domain: str | None, principal: str | None) -> VoiceResponse:
        """What to run next, naming only commands the CLI really has."""
        started = time.time()
        entries = self._entries(domain, principal)
        return self._grounded(
            _conv.next_steps(entries, self._cli_commands()),
            ["cli"] + (["detection_log"] if entries else []),
            QueryIntent.HELP,
            "Commands verified against the mercury-agent click group.",
            started,
            results=len(entries),
        )

    def _handle_collect(self) -> VoiceResponse:
        """How data reaches Mercury."""
        started = time.time()
        return self._grounded(
            _conv.collect_data(self._loader_domains(), self._cli_commands()),
            ["cli", "loaders"],
            QueryIntent.HELP,
            "Loader domains read from the loaders package; commands from the CLI.",
            started,
        )

    def _handle_run_command(self, name: str, user_input: str) -> VoiceResponse:
        """Hand back the exact command; the agent never executes one."""
        started = time.time()
        return self._grounded(
            _conv.run_command(name, self._cli_commands(), user_input),
            ["cli"],
            QueryIntent.HELP,
            "Command assembled from the CLI's own option table.",
            started,
        )

    def _narrate_entry(
        self, entry: dict[str, Any], domain: str | None
    ) -> tuple[NarrativeResult, dict[str, Any]]:
        """Synthesise a recorded run through the narrative engine, behind the ethical gate."""
        record = entry["detection"]
        detection_result = _conv.detection_result_from_record(record, entry["timestamp"])
        domain = sanitize_domain(domain or self.default_domain)
        self._enforce_voice_ethics(
            "MercuryVoice.explain_latest",
            domain,
            severity=float(detection_result.get("severity") or 0.0),
            anomaly_prob=min(1.0, float(detection_result.get("anomaly_score") or 0.0)),
            extra_details={"is_anomaly": bool(detection_result.get("anomaly_detected"))},
            request=str(detection_result.get("summary", "")),
        )
        return self.narrative_engine.synthesize(detection_result, domain=domain), detection_result

    @staticmethod
    def _cli_commands() -> dict[str, list[str]]:
        """Each mercury-agent command and its option names, read from the click group."""
        try:
            from omni_mercury_engine.cli import main as group
        except Exception:  # pragma: no cover - CLI extras missing
            return {}
        commands = getattr(group, "commands", {})
        return {
            name: [opt for param in cmd.params for opt in param.opts if opt.startswith("--")]
            for name, cmd in commands.items()
        }

    @staticmethod
    def _loader_domains() -> list[str]:
        """Domain names of the shipped loaders, from the package's module files.

        The loaders package is not imported (it pulls in every domain's
        dependencies); each ``<domain>_loader`` module name is its loader's
        ``DOMAIN``, which ``tests/narrative/test_voice_conversation.py`` pins.
        """
        import importlib.util
        import pkgutil

        spec = importlib.util.find_spec("omni_mercury_engine.loaders")
        if spec is None or not spec.submodule_search_locations:
            return []
        names = sorted(
            m.name[: -len("_loader")]
            for m in pkgutil.iter_modules(list(spec.submodule_search_locations))
            if m.name.endswith("_loader")
        )
        return names

    def _installation_facts(self) -> dict[str, Any]:
        """Live facts about this installation for the identity and glossary answers."""
        facts: dict[str, Any] = {"loader_count": len(self._loader_domains())}
        try:
            from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

            facts["manifest_size"] = len(DETECTOR_MANIFEST)
        except Exception:  # pragma: no cover - registry import failure
            pass
        try:
            import ama_cryptography

            facts["ama_version"] = str(getattr(ama_cryptography, "__version__", "") or "")
        except Exception:  # pragma: no cover - AMA absent
            pass
        try:
            from omni_mercury_engine import decouple

            facts["findoyou_enabled"] = decouple.is_enabled("findoyou")
            facts["switches"] = ", ".join(
                f"{name} {'on' if decouple.is_enabled(name) else 'off'}"
                for name in decouple.SWITCHES
            )
        except Exception:  # pragma: no cover - decouple state unreadable
            pass
        return facts

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
            sources = list({r.source.value for r in search_response.results})
            confidence = max(r.relevance_score for r in search_response.results)
        else:
            # No results found: say what can be asked instead of apologising
            message = self._build_no_results_response(user_input, profile)
            sources = []
            confidence = None

        # Add uncertainty note if low confidence
        uncertainty_note = None
        if confidence is not None and confidence < 0.5:
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
            suggested_follow_ups=(
                self._generate_follow_ups(search_response)
                if search_response.results
                else _conv.fallback_menu(user_input)[1]
            ),
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

        # Detection-log hits are described as runs, newest first: what ran,
        # when, how many samples were flagged and which, and the peak score
        # against its threshold. A bare score was the previous answer.
        detection_hits = [
            r.content
            for r in search_response.results
            if r.source == RetrievalSource.DETECTION_LOG and isinstance(r.content, dict)
        ]
        if detection_hits:
            detection_hits.sort(key=lambda c: float(c.get("timestamp") or 0.0), reverse=True)
            for content in detection_hits[:3]:
                parts.append(_conv.describe_detection(content, content.get("timestamp")))
            if len(detection_hits) > 3:
                parts.append(f"{len(detection_hits) - 3} older run(s) matched as well.")
            return " ".join(parts)

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
                severity = content.get("severity")
                label = content.get("severity_label")
                detail = f"Detection record: anomaly_score={content['anomaly_score']:.2f}"
                if isinstance(severity, (int, float)):
                    detail += (
                        f", severity {label + ' ' if label else ''}({severity:.0%} of critical)"
                    )
                parts.append(detail + ".")
        else:
            parts.append(f"Top result: {str(content)[:100]}.")

        # Confidence note
        if profile.express_confidence_level:
            parts.append(f"Relevance: {top_result.relevance_score:.0%}.")

        # Additional context if multiple results
        if n_results > 1:
            sources = {r.source.value for r in search_response.results}
            parts.append(f"Sources: {', '.join(sources)}.")

        return " ".join(parts)

    def _build_no_results_response(
        self,
        query: str,
        profile: PersonalityProfile,
    ) -> str:
        """Build response when no results found: what can be asked, not an apology."""
        return _conv.fallback_menu(query)[0]

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
        parts.append(
            f"Confidence: {narrative.confidence_score:.0%}."
            if narrative.confidence_score is not None
            else "Confidence: not reported."
        )

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

    def _generate_detection_follow_ups(
        self, narrative: NarrativeResult, answered: str = "narrative"
    ) -> list[str]:
        """Follow-up prompts for a narrated run, never repeating the one just answered."""
        follow_ups: list[tuple[str, str]] = []

        if narrative.reasoning_chain:
            follow_ups.append(("reasoning", "Show full reasoning chain"))

        if narrative.historical_context:
            follow_ups.append(("history", "Show similar historical events"))

        if narrative.confidence_score is None or narrative.confidence_score < 0.7:
            follow_ups.append(("uncertainty", "Explain uncertainty sources"))

        follow_ups.append(("dossier", "View detailed analysis"))
        follow_ups.append(("recommend", "Recommend an action"))
        follow_ups.append(("samples", "Which samples were flagged?"))

        return [text for key, text in follow_ups if key != answered][:3]

    def _generate_follow_ups(self, search_response: SearchResponse) -> list[str]:
        """Generate follow-up suggestions for search."""
        follow_ups = []

        if search_response.intent == QueryIntent.SEARCH_EVENTS:
            follow_ups.append("What was the latest anomaly?")
            follow_ups.append("How many anomalies today?")
        elif search_response.intent == QueryIntent.SEARCH_PATTERNS:
            follow_ups.append("Analyze pattern frequency")

        if search_response.total_found > len(search_response.results):
            follow_ups.append(f"Show more results ({search_response.total_found} total)")

        follow_ups.append("What can you do?")

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
    llm_provider: str | None = None,
    llm_model_name: str | None = None,
    llm_revision: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
) -> MercuryVoice:
    """Create a Mercury Voice instance.

    Args:
        enable_llm: Whether to enable LLM for response generation.
        default_domain: Default domain context.
        llm_provider: Concrete LLM provider name forwarded to
            :class:`MercuryVoice`.  Required in
            ``MERCURY_ENV=production`` when ``enable_llm`` is true.
        llm_model_name: Provider-specific model identifier.
        llm_revision: Optional HuggingFace revision pin.
        llm_api_key: Optional API key for providers that require one.
        llm_base_url: Optional endpoint override for providers that
            support it.

    Returns:
        Configured MercuryVoice
    """
    return MercuryVoice(
        enable_llm=enable_llm,
        default_domain=default_domain,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_revision=llm_revision,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )


# --------------------------------------------------------------------------- #
# Detection-record wording: shared by the summary, latest and search answers
# --------------------------------------------------------------------------- #

_CAPABILITY_WORDS = ("capabilit", "what can you do", "are you able", "can you do")

#: Manifest family -> words a question uses for it (``DetectorCategory`` values).
_FAMILY_WORDS: dict[str, tuple[str, ...]] = {
    "geological": ("geolog", "earthquake", "seismic", "volcan", "landslide", "tsunami"),
    "space": ("space", "solar", "geomagnetic", "kp", "satellite", "orbit"),
    "meteorological": (
        "weather",
        "meteorolog",
        "storm",
        "hurricane",
        "tornado",
        "flood",
        "wildfire",
        "drought",
    ),
    "medical": ("medical", "clinical", "sepsis", "ecg", "patient", "health record"),
    "security": ("security", "cyber", "network", "intrusion", "malware"),
    "visual": ("visual", "image", "picture", "photo", "camera"),
    "physics": ("physics", "vibration", "spectral", "dynamics"),
    "economic": ("econom", "financ", "market", "price"),
    "energy": ("energy", "grid", "power", "emp"),
    "marine": ("marine", "vessel", "ship", "ais", "maritime"),
    "foundation": ("foundation model", "chronos", "timegpt"),
    "neurosymbolic": ("neurosymbolic", "neuro-symbolic", "symbolic"),
    "intelligence": ("intelligence", "osint", "intel"),
    "uav": ("uav", "drone"),
    "vlm": ("vision-language", "vlm", "language model"),
    "uiux": ("uiux", "ui/ux", "interaction", "rage click"),
    "model": ("model-based", "autoencoder", "isolation forest", "transformer"),
}

#: Manifest input modality -> words a question uses for it.
_MODALITY_WORDS: dict[str, tuple[str, ...]] = {
    "series_1d": ("time series", "timeseries", "series", "signal", "univariate"),
    "matrix_2d": ("matrix", "multivariate", "tabular", "table"),
    "image_batch": ("images", "image batch", "pictures", "photos"),
    "video_frames": ("video", "frames", "footage"),
    "text": ("text", "document", "log line"),
    "graph": ("graph", "network topology"),
    "trajectory": ("trajectory", "track", "path"),
    "event": ("events", "event stream"),
}

_MODALITY_LABELS: dict[str, str] = {
    "series_1d": "1-D series",
    "matrix_2d": "2-D matrix",
    "record": "record",
    "rows": "row batch",
    "image_batch": "image batch",
    "event": "event",
    "window": "window",
    "video_frames": "video frames",
    "trajectory": "trajectory",
    "graph": "graph",
    "text": "text",
    "windows_3d": "3-D windows",
}

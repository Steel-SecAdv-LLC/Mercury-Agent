"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Narrative Module - Truth-Dense Communication Synthesis

This module transforms Mercury Agent from a detection engine into an "alive"
system that communicates with genuine transparency and truth density.

Components:
    - NarrativeEngine: Translates detection results to human language
    - ProactiveMonitor: Background vigilance with initiative thresholds
    - MemorySurface: Historical context awareness in communication
    - PersonalityEngine: Omni-scalar shaped communication style
    - MercuryConversationInterface: Unified "alive" interface

Philosophy:
    "An Agent that does not engage or communicate to retain users like almost
    all LLMs, but provides transparency and truth in every response."

Example:
    from omni_mercury_engine.narrative import create_mercury_interface

    # Create conversational interface
    mercury = create_mercury_interface(domain="medical")
    ctx = mercury.create_session()

    # Process detection with full transparency
    response = mercury.process_detection(detection_result, ctx)
    print(response.message)
    # "Detected significant anomaly (score: 0.87). Based on 3 similar
    #  historical cases, this pattern escalated 67% of the time.
    #  Confidence: 82%. Recommendation: Clinical review within 4 hours."
"""

from omni_mercury_engine.narrative.engine import (
    ConfidenceLevel,
    NarrativeEngine,
    NarrativeResult,
    NarrativeStyle,
    ReasoningChainNarrative,
)
from omni_mercury_engine.narrative.external_retrieval import (
    DatabaseRetriever,
    ExternalInformationRetriever,
    ExternalResult,
    ExternalSearchConfig,
    ExternalSourceType,
    WebSearchProvider,
    WebSearchRetriever,
    create_external_retriever,
)
from omni_mercury_engine.narrative.interface import (
    ConversationContext,
    MercuryConversationInterface,
    MercuryResponse,
    create_mercury_interface,
)
from omni_mercury_engine.narrative.memory_surface import (
    MemoryContext,
    MemoryRelevance,
    MemorySurface,
    PredictionHistory,
    SimilarEvent,
)
from omni_mercury_engine.narrative.multimodal import (
    AnomalyVisualType,
    AudioAnomalyType,
    AudioSegment,
    ModalityType,
    MultiModalDetection,
    MultiModalNarration,
    MultiModalNarrator,
    RegionOfInterest,
    create_multimodal_narrator,
    narrate_audio_detection,
    narrate_image_detection,
)
from omni_mercury_engine.narrative.personality import (
    CommunicationModifiers,
    CommunicationTone,
    PersonalityEngine,
    PersonalityProfile,
    VerbosityLevel,
)
from omni_mercury_engine.narrative.proactive import (
    InitiativeEvent,
    InitiativeThreshold,
    InitiativeType,
    ProactiveMonitor,
    VigilanceLevel,
)
from omni_mercury_engine.narrative.retriever import (
    KnowledgeRetriever,
    QueryIntent,
    RetrievalResult,
    RetrievalSource,
    SearchResponse,
)
from omni_mercury_engine.narrative.voice import (
    ConversationTurn,
    ConversationType,
    MercuryVoice,
    VoiceResponse,
    create_mercury_voice,
)

__all__ = [
    # Core engine
    "NarrativeEngine",
    "NarrativeStyle",
    "NarrativeResult",
    "ConfidenceLevel",
    "ReasoningChainNarrative",
    # Proactive monitoring
    "ProactiveMonitor",
    "VigilanceLevel",
    "InitiativeEvent",
    "InitiativeType",
    "InitiativeThreshold",
    # Memory surface
    "MemorySurface",
    "MemoryContext",
    "MemoryRelevance",
    "SimilarEvent",
    "PredictionHistory",
    # Personality
    "PersonalityEngine",
    "PersonalityProfile",
    "CommunicationTone",
    "VerbosityLevel",
    "CommunicationModifiers",
    # Interface
    "MercuryConversationInterface",
    "MercuryResponse",
    "ConversationContext",
    "create_mercury_interface",
    # Retriever
    "KnowledgeRetriever",
    "RetrievalResult",
    "RetrievalSource",
    "QueryIntent",
    "SearchResponse",
    # Voice
    "MercuryVoice",
    "VoiceResponse",
    "ConversationTurn",
    "ConversationType",
    "create_mercury_voice",
    # External Retrieval
    "ExternalInformationRetriever",
    "ExternalResult",
    "ExternalSearchConfig",
    "ExternalSourceType",
    "WebSearchProvider",
    "WebSearchRetriever",
    "DatabaseRetriever",
    "create_external_retriever",
    # Multi-Modal Support
    "MultiModalNarrator",
    "MultiModalDetection",
    "MultiModalNarration",
    "ModalityType",
    "AnomalyVisualType",
    "AudioAnomalyType",
    "RegionOfInterest",
    "AudioSegment",
    "create_multimodal_narrator",
    "narrate_image_detection",
    "narrate_audio_detection",
]

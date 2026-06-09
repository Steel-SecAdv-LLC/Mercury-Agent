# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Voice Interface API - Mercury Agent Conversational Endpoints.

Provides REST API endpoints for Mercury's voice/conversational interface:
- Natural language queries
- Detection result narration
- Status inquiries
- Proactive alerts (via WebSocket)

Usage:
    curl -X POST "http://localhost:8000/api/v1/voice/speak" \
        -H "Content-Type: application/json" \
        -d '{"message": "What is my system status?"}'
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Request/Response Models
class SpeakRequest(BaseModel):
    """Request model for voice speak endpoint.

    Attributes:
        message: User's natural language message
        domain: Optional domain context (medical, security, etc.)
        session_id: Optional session ID for conversation continuity
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's natural language message",
        json_schema_extra={"example": "What anomalies have you detected recently?"},
    )
    domain: str | None = Field(
        default=None,
        description="Domain context for response tailoring",
        json_schema_extra={"example": "medical"},
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is my system status?",
                    "domain": "security",
                }
            ]
        }
    }


class VoiceResponseModel(BaseModel):
    """Response model for voice speak endpoint.

    Attributes:
        message: Mercury's response message
        confidence: Confidence level in the response
        intent: Detected intent of the user's message
        sources: Information sources used for the response
        metadata: Additional response metadata
    """

    message: str = Field(..., description="Mercury's response message")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the response",
    )
    intent: str = Field(..., description="Detected intent of user message")
    sources: list[str] = Field(
        default_factory=list,
        description="Information sources used",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Mercury Agent operational. All systems normal. "
                    "3 detectors active, 0 anomalies in the last hour.",
                    "confidence": 0.95,
                    "intent": "status",
                    "sources": ["system_status", "detection_log"],
                    "timestamp": 1704067200.0,
                    "metadata": {"session_id": "sess_abc123"},
                }
            ]
        }
    }


class DetectionNarrationRequest(BaseModel):
    """Request model for detection narration.

    Attributes:
        detection_result: The detection result to narrate
        domain: Domain context for appropriate language
        verbosity: Detail level (brief, normal, detailed)
    """

    detection_result: dict[str, Any] = Field(
        ...,
        description="Detection result to narrate",
        json_schema_extra={
            "example": {
                "anomaly_detected": True,
                "anomaly_score": 0.87,
                "confidence": 0.92,
                "severity": 0.75,
            }
        },
    )
    domain: str | None = Field(
        default=None,
        description="Domain context",
    )
    verbosity: str = Field(
        default="normal",
        description="Verbosity level: brief, normal, detailed",
    )


class NarrationResponse(BaseModel):
    """Response model for detection narration.

    Attributes:
        summary: Brief summary of the detection
        detailed_explanation: Full explanation with reasoning
        recommendations: Suggested actions
        confidence_statement: Statement about uncertainty
    """

    summary: str = Field(..., description="Brief summary")
    detailed_explanation: str = Field(..., description="Full explanation")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended actions",
    )
    confidence_statement: str = Field(
        ...,
        description="Statement about confidence/uncertainty",
    )
    severity: str = Field(..., description="Severity level")


class StatusResponse(BaseModel):
    """Response model for system status.

    Attributes:
        status: Overall system status
        message: Status description
        components: Component-level status
        statistics: Usage statistics
    """

    status: str = Field(..., description="Overall status")
    message: str = Field(..., description="Status description")
    components: dict[str, Any] = Field(
        default_factory=dict,
        description="Component statuses",
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict,
        description="Usage statistics",
    )


class GreetingResponse(BaseModel):
    """Response model for greeting endpoint."""

    message: str = Field(..., description="Greeting message")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Available capabilities",
    )


# Create router
router = APIRouter(prefix="/api/v1/voice", tags=["Voice Interface"])

# Global voice instance (lazy loaded) - guarded by lock for thread safety
_mercury_voice: Any = None
_narrative_engine: Any = None
_voice_lock = __import__("threading").Lock()
_narrative_lock = __import__("threading").Lock()


def _get_voice() -> Any:
    """Get or create Mercury Voice instance (thread-safe)."""
    global _mercury_voice

    if _mercury_voice is None:
        with _voice_lock:
            # Double-checked locking: re-test after acquiring the lock
            if _mercury_voice is None:
                try:
                    from omni_mercury_engine.narrative.voice import create_mercury_voice

                    _mercury_voice = create_mercury_voice()
                except ImportError:
                    logger.warning("Narrative module not available, using fallback")
                    _mercury_voice = _FallbackVoice()

    return _mercury_voice


def _get_narrative_engine() -> Any:
    """Get or create Narrative Engine instance (thread-safe)."""
    global _narrative_engine

    if _narrative_engine is None:
        with _narrative_lock:
            if _narrative_engine is None:
                try:
                    from omni_mercury_engine.narrative.engine import NarrativeEngine

                    _narrative_engine = NarrativeEngine()
                except ImportError:
                    logger.warning("Narrative engine not available")
                    _narrative_engine = None

    return _narrative_engine


class _FallbackVoice:
    """Fallback voice implementation when narrative module unavailable."""

    def speak(self, message: str, domain: str | None = None) -> dict[str, Any]:
        """Process user message."""
        message_lower = message.lower()

        if any(kw in message_lower for kw in ["status", "health", "running"]):
            return {
                "message": "Mercury Agent operational. Running in fallback mode. "
                "Full narrative capabilities require numpy installation.",
                "confidence": 0.9,
                "intent": "status",
                "sources": ["system_status"],
            }
        elif any(kw in message_lower for kw in ["hello", "hi", "hey"]):
            return {
                "message": "Hello. Mercury Agent at your service. "
                "Currently running with limited conversational capability.",
                "confidence": 0.95,
                "intent": "greeting",
                "sources": [],
            }
        elif any(kw in message_lower for kw in ["help", "what can"]):
            return {
                "message": "I can help with anomaly detection and monitoring. "
                "Ask about system status, recent detections, or submit data for analysis.",
                "confidence": 0.9,
                "intent": "help",
                "sources": [],
            }
        else:
            return {
                "message": f"I received your query: '{message[:100]}'. "
                "For detailed analysis, please use the detection endpoints.",
                "confidence": 0.7,
                "intent": "unknown",
                "sources": [],
            }

    def greet(self, domain: str | None = None) -> dict[str, Any]:
        """Generate greeting."""
        return {
            "message": "Mercury Agent online and monitoring. How can I assist you?",
            "confidence": 1.0,
            "intent": "greeting",
            "sources": [],
        }

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return []


# API Endpoints
@router.post(
    "/speak",
    response_model=VoiceResponseModel,
    summary="Send Message to Mercury",
    description="""
Send a natural language message to Mercury and receive a conversational response.

## Supported Intents

Mercury automatically detects the intent of your message:
- **status**: System health and operational status
- **search**: Information lookup in knowledge base
- **detection**: Queries about recent detections
- **help**: Guidance and assistance
- **greeting**: Conversational greetings

## Examples

```json
{"message": "What is my system status?"}
{"message": "Have there been any anomalies today?"}
{"message": "Explain the last detection"}
```
    """,
    responses={
        200: {"description": "Response generated successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Processing error"},
    },
)
async def speak(request: SpeakRequest) -> VoiceResponseModel:
    """Send a message to Mercury and receive a response.

    Args:
        request: SpeakRequest with user message

    Returns:
        VoiceResponseModel with Mercury's response
    """
    try:
        voice = _get_voice()
        response = voice.speak(request.message, domain=request.domain)

        # Handle both VoiceResponse objects and dicts
        if hasattr(response, "message"):
            return VoiceResponseModel(
                message=response.message,
                confidence=getattr(response, "confidence", 0.8),
                intent=getattr(response, "intent", "unknown"),
                sources=getattr(response, "sources", []),
                timestamp=time.time(),
                metadata={
                    "session_id": request.session_id,
                    "domain": request.domain,
                },
            )
        else:
            return VoiceResponseModel(
                message=response.get("message", ""),
                confidence=response.get("confidence", 0.8),
                intent=response.get("intent", "unknown"),
                sources=response.get("sources", []),
                timestamp=time.time(),
                metadata={
                    "session_id": request.session_id,
                    "domain": request.domain,
                },
            )

    except Exception as e:
        logger.error("Voice processing error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the message.",
        )


@router.post(
    "/narrate",
    response_model=NarrationResponse,
    summary="Narrate Detection Result",
    description="""
Convert a detection result into human-readable narrative with explanations.

The narration includes:
- Summary of what was detected
- Detailed explanation of the analysis
- Recommended actions
- Confidence statement

Verbosity levels:
- **brief**: One-sentence summary
- **normal**: Summary + key details
- **detailed**: Full explanation with reasoning chain
    """,
)
async def narrate_detection(request: DetectionNarrationRequest) -> NarrationResponse:
    """Generate narrative explanation of detection result.

    Args:
        request: Detection result and narration parameters

    Returns:
        NarrationResponse with human-readable explanation
    """
    try:
        engine = _get_narrative_engine()

        if engine is None:
            # Fallback narration
            detection = request.detection_result
            is_anomaly = detection.get("anomaly_detected", False)
            score = detection.get("anomaly_score", 0)
            confidence = detection.get("confidence", 0.5)

            if is_anomaly:
                summary = f"Anomaly detected with score {score:.2f}"
                explanation = (
                    f"The analysis indicates an anomaly with a score of {score:.2f}. "
                    f"This exceeds the detection threshold and warrants attention."
                )
                recommendations = ["Review the flagged data", "Check related systems"]
                severity = "high" if score > 0.8 else "medium"
            else:
                summary = "No anomaly detected"
                explanation = "The data appears normal within expected parameters."
                recommendations = []
                severity = "low"

            confidence_stmt = (
                f"Confidence: {confidence:.0%}. "
                f"{'High confidence in this assessment.' if confidence > 0.8 else 'Moderate confidence - additional verification recommended.'}"
            )

            return NarrationResponse(
                summary=summary,
                detailed_explanation=explanation,
                recommendations=recommendations,
                confidence_statement=confidence_stmt,
                severity=severity,
            )

        # Use narrative engine
        result = engine.synthesize(
            request.detection_result,
            domain=request.domain,
        )

        return NarrationResponse(
            summary=result.summary,
            detailed_explanation=result.detailed_explanation,
            recommendations=result.recommendations,
            confidence_statement=result.uncertainty_disclosure,
            severity=result.confidence_level.value,
        )

    except Exception as e:
        logger.error("Narration error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while generating narration.",
        )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get Voice Interface Status",
    description="Get the current status of Mercury's voice interface and related components.",
)
async def get_status() -> StatusResponse:
    """Get voice interface status.

    Returns:
        StatusResponse with component status and statistics
    """
    voice = _get_voice()

    components = {
        "voice_interface": "operational",
        "narrative_engine": "operational" if _get_narrative_engine() else "fallback",
        "conversation_history": (
            len(voice.get_conversation_history())
            if hasattr(voice, "get_conversation_history")
            else 0
        ),
    }

    statistics = {
        "conversation_turns": components.get("conversation_history", 0),
    }

    return StatusResponse(
        status="operational",
        message="Mercury voice interface is active and ready for conversation.",
        components=components,
        statistics=statistics,
    )


@router.get(
    "/greet",
    response_model=GreetingResponse,
    summary="Get Mercury Greeting",
    description="Get an initial greeting from Mercury with available capabilities.",
)
async def greet(domain: str | None = None) -> GreetingResponse:
    """Get Mercury's greeting.

    Args:
        domain: Optional domain context

    Returns:
        GreetingResponse with greeting and capabilities
    """
    voice = _get_voice()
    response = voice.greet(domain=domain)

    capabilities = [
        "Natural language conversation",
        "Anomaly detection analysis",
        "System status reporting",
        "Detection result explanation",
        "Historical event lookup",
    ]

    message = response.message if hasattr(response, "message") else response.get("message", "")

    return GreetingResponse(
        message=message,
        capabilities=capabilities,
    )


@router.get(
    "/history",
    summary="Get Conversation History",
    description="Get the recent conversation history for the current session.",
)
async def get_history(limit: int = 10) -> dict[str, Any]:
    """Get recent conversation history.

    Args:
        limit: Maximum number of turns to return

    Returns:
        Dictionary with conversation history
    """
    voice = _get_voice()

    if hasattr(voice, "get_conversation_history"):
        history = voice.get_conversation_history()
        return {
            "history": history[-limit:] if history else [],
            "total_turns": len(history) if history else 0,
        }

    return {"history": [], "total_turns": 0}


# Function to add voice routes to main app
def add_voice_routes(app: Any) -> None:
    """Add voice routes to FastAPI application.

    Args:
        app: FastAPI application instance
    """
    app.include_router(router)

    # Add voice tag to OpenAPI
    if hasattr(app, "openapi_tags"):
        if app.openapi_tags is None:
            app.openapi_tags = []
        app.openapi_tags.append(
            {
                "name": "Voice Interface",
                "description": "Mercury's conversational voice interface for natural language interaction.",
            }
        )

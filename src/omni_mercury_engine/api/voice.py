# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Voice Interface API - Mercury Agent Conversational Endpoints.

Provides REST API endpoints for Mercury's voice/conversational interface:
- Natural language queries
- Detection result narration
- Status inquiries
- Proactive alerts (server-sent events on ``GET /api/v1/voice/alerts``)

Usage:
    curl -X POST "http://localhost:8000/api/v1/voice/speak" \
        -H "Content-Type: application/json" \
        -d '{"message": "What is my system status?"}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from omni_mercury_engine.api.auth import APIKeyAuth, JWTAuth, User

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
    confidence: float | None = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the response; null when the voice reported none",
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
    # Everything below is what MercuryVoice produces beyond the message. The
    # route used to drop all of it (and read ``intent``/``sources`` from attribute
    # names the voice never had, so every live reply said intent "unknown" with
    # no sources). A client that only knows the five fields above is unaffected.
    reasoning_summary: str | None = Field(
        default=None, description="Brief explanation of how the answer was formed"
    )
    search_performed: bool = Field(
        default=False, description="Whether Mercury searched its memory for this reply"
    )
    search_results_count: int = Field(
        default=0, ge=0, description="Memory results the reply was built from"
    )
    suggested_follow_ups: list[str] = Field(
        default_factory=list, description="Questions Mercury suggests asking next"
    )
    uncertainty_note: str | None = Field(
        default=None, description="What Mercury is unsure about, when it says so"
    )
    response_time_ms: float | None = Field(
        default=None, ge=0.0, description="Time the voice took to answer"
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
                    "reasoning_summary": "Answered from the live system status.",
                    "search_performed": False,
                    "search_results_count": 0,
                    "suggested_follow_ups": ["What anomalies were detected today?"],
                    "uncertainty_note": None,
                    "response_time_ms": 4.2,
                }
            ]
        }
    }


def _speak(
    voice: Any, message: str, domain: str | None, principal: str, session_id: str | None = None
) -> Any:
    """Call the voice with the caller's principal and session; the fallback voice takes neither.

    The session id keeps "why?" and "run 2" tied to the browser that asked:
    every unauthenticated console is the same principal (``anonymous``), so
    without it two anonymous browsers would share one follow-up context.
    """
    try:
        return voice.speak(message, domain=domain, principal=principal, session_id=session_id)
    except TypeError:
        return voice.speak(message, domain=domain)


def _voice_payload(response: Any) -> dict[str, Any]:
    """Normalise whatever a voice returns into one dict the response model reads.

    ``MercuryVoice.speak`` returns a ``VoiceResponse`` whose fields are named
    ``intent_understood`` and ``sources_cited``; ``_FallbackVoice.speak`` returns
    a dict with ``intent`` and ``sources``. Both end up with the same keys here.
    """
    if hasattr(response, "to_dict"):
        payload = dict(response.to_dict())
    elif isinstance(response, dict):
        payload = dict(response)
    else:  # an object that exposes attributes only
        payload = {
            key: getattr(response, key)
            for key in (
                "message",
                "confidence",
                "intent",
                "sources",
                "intent_understood",
                "sources_cited",
                "reasoning_summary",
                "search_performed",
                "search_results_count",
                "suggested_follow_ups",
                "uncertainty_note",
                "response_time_ms",
            )
            if hasattr(response, key)
        }
    intent = payload.get("intent") or payload.get("intent_understood") or "unknown"
    payload["intent"] = getattr(intent, "value", intent)
    sources = payload.get("sources")
    if not sources:
        sources = payload.get("sources_cited") or []
    payload["sources"] = [str(item) for item in sources]
    return payload


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
                "confidence_basis": "keyword-rule match; not a calibrated NLU score",
                "intent": "status",
                # No status source is consulted on this fallback path; the
                # old ["system_status"] claimed one was.
                "sources": [],
            }
        elif any(kw in message_lower for kw in ["hello", "hi", "hey"]):
            return {
                "message": "Hello. Mercury Agent at your service. "
                "Currently running with limited conversational capability.",
                "confidence": 0.95,
                "confidence_basis": "keyword-rule match; not a calibrated NLU score",
                "intent": "greeting",
                "sources": [],
            }
        elif any(kw in message_lower for kw in ["help", "what can"]):
            return {
                "message": "I can help with anomaly detection and monitoring. "
                "Ask about system status, recent detections, or submit data for analysis.",
                "confidence": 0.9,
                "confidence_basis": "keyword-rule match; not a calibrated NLU score",
                "intent": "help",
                "sources": [],
            }
        else:
            return {
                "message": f"I received your query: '{message[:100]}'. "
                "For detailed analysis, please use the detection endpoints.",
                "confidence": 0.7,
                "confidence_basis": "keyword-rule match; not a calibrated NLU score",
                "intent": "unknown",
                "sources": [],
            }

    def greet(self, domain: str | None = None) -> dict[str, Any]:
        """Generate greeting."""
        return {
            "message": "Mercury Agent online and monitoring. How can I assist you?",
            "confidence": 1.0,
            "confidence_basis": "fixed greeting; not a calibrated NLU score",
            "intent": "greeting",
            "sources": [],
        }

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return []


# API Endpoints
def _get_optional_user(
    api_key_user: User | None = Depends(APIKeyAuth(auto_error=False)),
    jwt_user: User | None = Depends(JWTAuth(auto_error=False)),
) -> User | None:
    """The caller's account when a valid API key or JWT is presented, else ``None``."""
    return api_key_user or jwt_user


def _principal_of(user: User | None) -> str:
    """The principal the voice answers for: the account id, or ``anonymous``.

    ``isinstance`` rather than ``is not None``: tests and the warm-up call the
    handler directly, and then ``user`` is the unresolved ``Depends`` marker.
    """
    return user.id if isinstance(user, User) else "anonymous"


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
async def speak(
    request: SpeakRequest, user: User | None = Depends(_get_optional_user)
) -> VoiceResponseModel:
    """Send a message to Mercury and receive a response.

    Args:
        request: SpeakRequest with user message

    Returns:
        VoiceResponseModel with Mercury's response
    """
    try:
        voice = _get_voice()
        payload = _voice_payload(
            _speak(voice, request.message, request.domain, _principal_of(user), request.session_id)
        )
        raw_conf = payload.get("confidence")
        # A missing confidence is reported as null, not filled with 0.8.
        confidence = (
            min(1.0, max(0.0, float(raw_conf)))
            if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool)
            else None
        )
        return VoiceResponseModel(
            message=str(payload.get("message", "")),
            confidence=confidence,
            intent=str(payload["intent"]),
            sources=payload["sources"],
            timestamp=time.time(),
            metadata={
                "session_id": request.session_id,
                "domain": request.domain,
            },
            reasoning_summary=payload.get("reasoning_summary"),
            search_performed=bool(payload.get("search_performed", False)),
            search_results_count=max(0, int(payload.get("search_results_count") or 0)),
            suggested_follow_ups=[
                str(q) for q in (payload.get("suggested_follow_ups") or []) if str(q).strip()
            ],
            uncertainty_note=payload.get("uncertainty_note"),
            response_time_ms=(
                None
                if payload.get("response_time_ms") is None
                else max(0.0, float(payload["response_time_ms"]))
            ),
        )

    except Exception as e:
        logger.error("Voice processing error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the message.",
        ) from e


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
            raw_conf = detection.get("confidence")
            confidence = (
                float(raw_conf)
                if isinstance(raw_conf, (int, float)) and not isinstance(raw_conf, bool)
                else None
            )

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

            if confidence is None:
                confidence_stmt = (
                    "Confidence was not reported for this detection, so it has not been "
                    "assessed. Treat the finding as unverified."
                )
            else:
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
        ) from e


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Get Voice Interface Status",
    description="Get the current status of Mercury's voice interface and related components.",
)
async def get_status(user: User | None = Depends(_get_optional_user)) -> StatusResponse:
    """Get voice interface status.

    Conversation and detection figures are the caller's own: the voice is one
    object per process, so unscoped counts would describe every account.

    Returns:
        StatusResponse with component status and statistics
    """
    voice = _get_voice()
    principal = _principal_of(user)

    components = {
        "voice_interface": "operational",
        "narrative_engine": "operational" if _get_narrative_engine() else "fallback",
        "conversation_history": (
            len(voice.get_conversation_history(1000, principal=principal))
            if hasattr(voice, "get_conversation_history")
            else 0
        ),
        "alerts": "operational",
    }

    statistics: dict[str, Any] = {
        "conversation_turns": components.get("conversation_history", 0),
    }
    retriever = getattr(voice, "retriever", None)
    if retriever is not None and hasattr(retriever, "detections"):
        runs = retriever.detections(principal=principal)
        statistics["detection_runs"] = len(runs)
        statistics["flagged_runs"] = sum(
            1 for e in runs if e["detection"].get("anomaly_detected", False)
        )
    try:
        from omni_mercury_engine.narrative.alert_bus import get_alert_bus

        statistics["alert_listeners"] = get_alert_bus().subscriber_count
    except Exception:  # pragma: no cover - bus import failure
        pass

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
async def greet(
    domain: str | None = None, user: User | None = Depends(_get_optional_user)
) -> GreetingResponse:
    """Get Mercury's greeting.

    The capabilities are read from this installation (the detector manifest,
    the loader package, the caller's own detection memory) rather than a
    fixed list, so the console's header states what this server really has.

    Args:
        domain: Optional domain context
        user: The caller, when an API key or token is presented

    Returns:
        GreetingResponse with greeting and capabilities
    """
    voice = _get_voice()
    response = voice.greet(domain=domain)
    capabilities = _grounded_capabilities(voice, _principal_of(user))

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
async def get_history(
    limit: int = 10, user: User | None = Depends(_get_optional_user)
) -> dict[str, Any]:
    """Get recent conversation history.

    Only the caller's own turns are returned (plus turns recorded without an
    account, such as the greeting): the voice is shared by every account on
    the server, and the route used to hand each of them all the others' turns.

    Args:
        limit: Maximum number of turns to return
        user: The caller, when an API key or token is presented

    Returns:
        Dictionary with conversation history
    """
    voice = _get_voice()

    if hasattr(voice, "get_conversation_history"):
        history = voice.get_conversation_history(1000, principal=_principal_of(user))
        return {
            "history": history[-max(0, limit) :] if history else [],
            "total_turns": len(history) if history else 0,
        }

    return {"history": [], "total_turns": 0}


def _grounded_capabilities(voice: Any, principal: str) -> list[str]:
    """What this installation can do, as short facts read from the code and the record."""
    caps: list[str] = []
    try:
        from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

        families = {e.category.value for e in DETECTOR_MANIFEST}
        caps.append(f"{len(DETECTOR_MANIFEST)} detectors across {len(families)} families")
    except Exception:  # pragma: no cover - registry import failure
        caps.append("Anomaly detection across the detector manifest")
    try:
        from omni_mercury_engine.narrative.voice import MercuryVoice

        caps.append(f"{len(MercuryVoice._loader_domains())} domain data loaders")
    except Exception:  # pragma: no cover - loader package missing
        pass
    retriever = getattr(voice, "retriever", None)
    if retriever is not None and hasattr(retriever, "detections"):
        runs = retriever.detections(principal=principal)
        flagged = sum(1 for e in runs if e["detection"].get("anomaly_detected", False))
        caps.append(
            f"{len(runs)} of your runs in memory ({flagged} flagged)"
            if runs
            else "Your detection runs remembered as you make them"
        )
    caps.append("Narration through two hard ethical gates")
    caps.append("Proactive alerts as each run lands")
    return caps


#: Seconds between keep-alive frames when nothing was recorded.
ALERT_HEARTBEAT_SECONDS = 15.0
#: Seconds between polls of the durable store for runs recorded on other workers.
ALERT_STORE_POLL_SECONDS = 2.0


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n".encode()


async def alert_events(
    principal: str,
    *,
    heartbeat: float = ALERT_HEARTBEAT_SECONDS,
    store_poll: float = ALERT_STORE_POLL_SECONDS,
    max_events: int | None = None,
) -> AsyncIterator[bytes]:
    """Server-sent events for every flagged run ``principal`` may see.

    Runs recorded in this process arrive through the alert bus; runs recorded
    by another worker on the same ``MERCURY_AGENT_MEMORY_PATH`` are picked
    up from the store by id. Each detection is sent once (the store's copy of
    a bus delivery is recognised by its detection id), clean runs are not
    sent, and a keep-alive comment goes out every ``heartbeat`` seconds so
    proxies hold the connection open. ``max_events`` ends the stream after
    that many alerts (tests).
    """
    from omni_mercury_engine.narrative.alert_bus import get_alert_bus, to_alert

    bus = get_alert_bus()
    token, queue = bus.subscribe(principal)
    voice = _get_voice()
    retriever = getattr(voice, "retriever", None)
    store = getattr(retriever, "_detection_store", None)
    cursor = store.latest_id() if store is not None else 0
    seen: dict[str, float] = {}
    sent = 0
    yield b": connected\n\n"
    try:
        while max_events is None or sent < max_events:
            fresh: list[dict[str, Any]] = []
            try:
                fresh.append(await asyncio.wait_for(queue.get(), timeout=store_poll))
                while not queue.empty():
                    fresh.append(queue.get_nowait())
            except TimeoutError:
                pass
            if store is not None:
                try:
                    for row_id, entry in store.load_after(cursor):
                        cursor = max(cursor, row_id)
                        from omni_mercury_engine.narrative.alert_bus import principal_may_see

                        if principal_may_see(principal, entry.get("principal")):
                            fresh.append(entry)
                except Exception as exc:  # pragma: no cover - store fault must not end the stream
                    logger.debug("alert store poll skipped: %s", exc)
            now = time.time()
            for entry in fresh:
                record = entry.get("detection") or {}
                if not record.get("anomaly_detected", False):
                    continue
                key = str(record.get("detection_id") or id(entry))
                if key in seen:
                    continue
                seen[key] = now
                yield _sse("alert", to_alert(entry))
                sent += 1
                if max_events is not None and sent >= max_events:
                    break
            if len(seen) > 2048:
                seen = {k: t for k, t in seen.items() if now - t < 3600}
            if not fresh and heartbeat > 0 and (now - _last_beat.get(token, 0.0)) >= heartbeat:
                _last_beat[token] = now
                yield b": keep-alive\n\n"
    finally:
        _last_beat.pop(token, None)
        bus.unsubscribe(token)


_last_beat: dict[int, float] = {}


@router.get(
    "/alerts",
    summary="Proactive alert stream",
    description=(
        "Server-sent events: one `alert` event for each detection run that flags a sample, "
        "as it is recorded, bounded to the caller's own runs. Keep-alive comments every "
        "15 seconds. Consume with a streaming fetch; the console's agent dock does."
    ),
    response_class=StreamingResponse,
)
async def alerts(
    request: Request,
    limit: int | None = Query(
        None, ge=1, le=1000, description="End the stream after this many alerts (scripts, curl)."
    ),
    user: User | None = Depends(_get_optional_user),
) -> StreamingResponse:
    """Stream the caller's flagged runs as server-sent events."""
    principal = _principal_of(user)
    return StreamingResponse(
        alert_events(principal, max_events=limit),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The voice route carries what the voice says, not a thinned copy of it.

``MercuryVoice.speak`` returns a ``VoiceResponse`` whose fields are named
``intent_understood`` and ``sources_cited`` and which also carries a reasoning
summary, suggested follow-ups, an uncertainty note and the memory search
counters. ``POST /api/v1/voice/speak`` used to read ``intent`` and ``sources``
off that object -- attributes it never had -- so every live reply reported
intent "unknown" with no sources, and the rest was dropped on the floor. The
console then told the operator the engine was answering with "message +
confidence only". These tests pin the pass-through for the narrative voice,
for the fallback voice's dict, and end to end through a real voice.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.voice import SpeakRequest, _voice_payload, speak
from omni_mercury_engine.narrative.retriever import QueryIntent
from omni_mercury_engine.narrative.voice import VoiceResponse


class _StubVoice:
    """Returns one fixed response so the route's mapping can be checked exactly."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def speak(self, message: str, domain: str | None = None) -> Any:
        return self._response


def _speak(monkeypatch: pytest.MonkeyPatch, response: Any, message: str = "status") -> Any:
    monkeypatch.setattr(voice_module, "_get_voice", lambda: _StubVoice(response))
    return asyncio.run(speak(SpeakRequest(message=message, session_id="s1", domain=None)))


def test_narrative_voice_response_is_passed_through_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    intent = next(iter(QueryIntent))
    response = VoiceResponse(
        message="Two anomalies in the last run.",
        confidence=0.82,
        sources_cited=["detection_log", "system_status"],
        reasoning_summary="Matched the two most recent detection events.",
        intent_understood=intent,
        search_performed=True,
        search_results_count=2,
        response_time_ms=3.5,
        suggested_follow_ups=["What was the top score?", "  ", "Which detector flagged it?"],
        uncertainty_note="Scores are relative to this dataset only.",
    )
    out = _speak(monkeypatch, response)
    assert out.message == "Two anomalies in the last run."
    assert out.confidence == pytest.approx(0.82)
    assert out.intent == intent.value
    assert out.sources == ["detection_log", "system_status"]
    assert out.reasoning_summary == "Matched the two most recent detection events."
    assert out.search_performed is True
    assert out.search_results_count == 2
    assert out.suggested_follow_ups == ["What was the top score?", "Which detector flagged it?"]
    assert out.uncertainty_note == "Scores are relative to this dataset only."
    assert out.response_time_ms == pytest.approx(3.5)
    assert out.metadata["session_id"] == "s1"


def test_fallback_voice_dict_still_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    out = _speak(
        monkeypatch,
        {"message": "Fallback mode.", "confidence": 0.9, "intent": "status", "sources": []},
    )
    assert out.message == "Fallback mode."
    assert out.intent == "status"
    assert out.sources == []
    assert out.search_performed is False
    assert out.suggested_follow_ups == []
    assert out.reasoning_summary is None


def test_payload_normaliser_prefers_explicit_intent_and_sources() -> None:
    payload = _voice_payload(
        {"message": "m", "intent": "help", "sources": ["a"], "sources_cited": ["b"]}
    )
    assert payload["intent"] == "help"
    assert payload["sources"] == ["a"]
    payload = _voice_payload({"message": "m"})
    assert payload["intent"] == "unknown"
    assert payload["sources"] == []


def test_real_voice_reply_is_no_longer_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a real voice answering a status question reports its intent."""
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        pytest.skip("narrative voice unavailable; fallback voice has no retriever")
    out = asyncio.run(
        speak(SpeakRequest(message="What is your status?", session_id=None, domain=None))
    )
    assert out.message
    assert out.intent != "unknown"
    assert isinstance(out.suggested_follow_ups, list)

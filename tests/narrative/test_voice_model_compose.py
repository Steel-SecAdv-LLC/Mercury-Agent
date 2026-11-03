# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An attached language model composes search answers from the retrieved records only.

Before this change the voice stored an adapter and never generated with it:
"Language model: attached" changed nothing a user read. Now, when a question
falls to the free-text search and the retriever finds records, the adapter
is asked to answer from those records alone (bounded prompt, four sentences,
say so when they do not answer); the reply replaces the template text, the
response names the model among its sources and says how many records it
composed from. An adapter failure, an empty reply or a reply the adapter
marks as an error leaves the template answer in place. Grounded answers
(record, code, narrative) never go through the model.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from omni_mercury_engine.narrative.retriever import (
    QueryIntent,
    RetrievalResult,
    RetrievalSource,
    SearchResponse,
)
from omni_mercury_engine.narrative.voice import MercuryVoice

pytest.importorskip("fastapi", reason="the voice end-to-end tests need the [api] extra")


class _Adapter:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.prompts.append((prompt, system_prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def is_available(self) -> bool:
        return True


def _results() -> SearchResponse:
    now = time.time()
    rows = [
        RetrievalResult(
            source=RetrievalSource.DETECTION_LOG,
            content={
                "detection": {"anomaly_detected": True, "anomaly_score": 6.2},
                "domain": "energy",
            },
            relevance_score=0.9,
            confidence=0.9,
            timestamp=now,
        ),
        RetrievalResult(
            source=RetrievalSource.KNOWLEDGE_GRAPH,
            content={"fact": "the energy loader derives labels from Kp"},
            relevance_score=0.6,
            confidence=0.8,
            timestamp=now,
        ),
    ]
    return SearchResponse(
        query="what did the energy run find",
        intent=QueryIntent.SEARCH_EVENTS,
        results=rows,
        total_found=2,
        search_time_ms=1.0,
    )


@pytest.fixture
def voice(monkeypatch: pytest.MonkeyPatch) -> MercuryVoice:
    from omni_mercury_engine.api import voice as voice_module

    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    instance = voice_module._get_voice()
    if not isinstance(instance, MercuryVoice):
        pytest.skip("narrative voice unavailable")
    return instance


def _attach(voice: Any, adapter: _Adapter) -> None:
    voice._llm_adapter = adapter
    voice._llm_provider = "groq"
    voice._llm_model_name = "llama-3.3-70b"


def test_the_model_sees_only_the_records_and_a_bounded_instruction(voice: MercuryVoice) -> None:
    adapter = _Adapter("The energy run flagged one sample with a score of 6.2.")
    _attach(voice, adapter)
    composed = voice._compose_with_model("what did the energy run find", _results())
    assert composed == "The energy run flagged one sample with a score of 6.2."
    prompt, system = adapter.prompts[0]
    assert "[detection_log]" in prompt and "[knowledge_graph]" in prompt and "6.2" in prompt
    assert system and "only the records" in system and "four sentences" in system


@pytest.mark.parametrize(
    "reply", ["", "API error: rate limited", "Request failed: boom", RuntimeError("down")]
)
def test_a_failed_or_empty_reply_leaves_the_template_answer(
    voice: MercuryVoice, reply: Any
) -> None:
    _attach(voice, _Adapter(reply))
    assert voice._compose_with_model("q", _results()) is None


def test_no_adapter_or_no_records_means_no_composition(voice: MercuryVoice) -> None:
    voice._llm_adapter = None
    assert voice._compose_with_model("q", _results()) is None
    _attach(voice, _Adapter("x"))
    empty = SearchResponse(
        query="q", intent=QueryIntent.SEARCH_EVENTS, results=[], total_found=0, search_time_ms=0.0
    )
    assert voice._compose_with_model("q", empty) is None


def test_a_search_answer_names_the_model_and_the_record_count(
    voice: MercuryVoice, monkeypatch: pytest.MonkeyPatch
) -> None:
    _attach(voice, _Adapter("Composed from the records."))
    monkeypatch.setattr(voice.retriever, "search", lambda *a, **k: _results())
    reply = voice.speak(
        "zzyzx records please", domain="general", principal="compose", session_id="c1"
    )
    assert reply.answer_kind == "search"
    assert reply.message == "Composed from the records."
    assert "language_model:groq" in reply.sources_cited
    assert reply.reasoning_summary and "2 retrieved record(s)" in reply.reasoning_summary


def test_grounded_answers_never_go_through_the_model(voice: MercuryVoice) -> None:
    adapter = _Adapter("should never be used")
    _attach(voice, adapter)
    reply = voice.speak("Which detectors do you have?", domain="general", principal="compose")
    assert reply.answer_kind == "capability" and "should never be used" not in reply.message
    assert adapter.prompts == []

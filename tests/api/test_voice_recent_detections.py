# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent remembers what the engine detects.

``KnowledgeRetriever`` has carried a ``log_detection`` hook and a
detection-log search path since it was written, and ``record_detection`` has
recorded every API detection into the export ``DataStore`` -- but nothing
ever connected the two, so "have there been any anomalies?" was answered
from an empty log on every deployment while the voice route's documentation
advertised a ``detection`` intent. ``record_detection`` now mirrors each
record into the live voice's retriever; these tests pin that the mirror
carries the fields the retriever searches on, and that the voice then finds
the event instead of apologising.
"""

from __future__ import annotations

import asyncio

import pytest

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import record_detection


@pytest.fixture
def fresh_voice(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A voice built for this test only, so earlier tests' memory cannot leak in."""
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        pytest.skip("narrative voice unavailable; fallback voice has no retriever")
    return voice


def _record(anomalies: list[bool], scores: list[float], domain: str = "network_security") -> None:
    asyncio.run(
        record_detection(
            user_id="test",
            method="multivariate",
            data=[[float(i)] for i in range(len(anomalies))],
            results={"anomalies": anomalies, "scores": scores, "summary": {"domain": domain}},
            sensitivity=0.5,
            metadata={"domain": domain},
        )
    )


def test_recorded_detection_lands_in_the_retriever_with_searchable_fields(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    before = len(fresh_voice.retriever._detection_log)
    _record([False, True, False], [0.2, 8.1, 0.3])
    log = fresh_voice.retriever._detection_log
    assert len(log) == before + 1
    entry = log[-1]
    # Stored through the voice's own domain whitelist: "network_security" is
    # not an EnvironmentDomain, so it is filed where a query will look for it.
    assert entry["domain"] == "general"
    det = entry["detection"]
    assert det["anomaly_detected"] is True
    assert det["anomaly_count"] == 1
    assert det["anomaly_score"] == pytest.approx(8.1)
    assert det["method"] == "multivariate"


def test_voice_now_finds_the_anomaly_instead_of_apologising(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    _record([False, True, False, False], [0.1, 7.4, 0.2, 0.1])
    response = fresh_voice.speak("What anomaly detection events happened recently?")
    message = getattr(response, "message", None) or response.get("message")  # type: ignore[union-attr]
    count = getattr(response, "search_results_count", None)
    if count is None and isinstance(response, dict):
        count = response.get("search_results_count")
    assert "couldn't find" not in message.lower()
    assert count is None or count >= 1
    assert "event" in message.lower() or "anomal" in message.lower()


def test_normal_run_is_remembered_but_not_as_an_anomaly(fresh_voice) -> None:  # type: ignore[no-untyped-def]
    _record([False, False], [0.1, 0.2], domain="general")
    det = fresh_voice.retriever._detection_log[-1]["detection"]
    assert det["anomaly_detected"] is False
    assert det["anomaly_count"] == 0

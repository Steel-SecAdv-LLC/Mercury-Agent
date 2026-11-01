# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The voice API streams the caller's flagged runs and keeps accounts apart.

Before this, ``GET /api/v1/voice/alerts`` did not exist (the console's dock
read "not served by this engine"), ``/history`` returned every account's
turns to every caller, ``/status`` counted them all, and ``/greet`` listed
five fixed phrases whatever the server had.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import record_detection
from omni_mercury_engine.api.server import app
from omni_mercury_engine.narrative import alert_bus

if TYPE_CHECKING:
    from omni_mercury_engine.narrative.voice import MercuryVoice


@pytest.fixture
def fresh_voice(monkeypatch: pytest.MonkeyPatch) -> MercuryVoice:
    monkeypatch.delenv("MERCURY_AGENT_MEMORY_PATH", raising=False)
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    monkeypatch.setattr(alert_bus, "_bus", alert_bus.AlertBus())
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        pytest.skip("narrative voice unavailable")
    return voice


async def _arecord(
    user: str, anomalies: list[bool], scores: list[float], threshold: float = 3.5
) -> None:
    await record_detection(
        user_id=user,
        method="univariate",
        data=[[float(i)] for i in range(len(anomalies))],
        results={"anomalies": anomalies, "scores": scores, "threshold": threshold},
        sensitivity=0.5,
        metadata={"domain": "general"},
    )


def _record(user: str, anomalies: list[bool], scores: list[float], threshold: float = 3.5) -> None:
    asyncio.run(_arecord(user, anomalies, scores, threshold))


def _events(raw: bytes) -> list[dict[str, Any]]:
    out = []
    for block in raw.decode().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def test_stream_sends_the_callers_flagged_runs_once(fresh_voice: MercuryVoice) -> None:
    async def run() -> bytes:
        gen = voice_module.alert_events("alice", heartbeat=0, store_poll=0.05, max_events=2)
        chunks: list[bytes] = []
        chunks.append(await gen.__anext__())  # ": connected"
        await _arecord("alice", [False, True], [0.1, 9.0])  # flagged, alice's
        await _arecord("bob", [False, True], [0.1, 9.0])  # flagged, not alice's
        await _arecord("alice", [False, False], [0.1, 0.2])  # clean, not an alert
        await _arecord("alice", [True, False], [8.0, 0.1])  # flagged, alice's
        async for chunk in gen:
            chunks.append(chunk)
        return b"".join(chunks)

    raw = asyncio.run(run())
    events = _events(raw)
    assert len(events) == 2
    assert all(e["initiative_type"].endswith("anomaly") for e in events)
    assert all("flagged" in e["message"] for e in events)
    assert raw.startswith(b": connected")


def test_stream_picks_up_other_workers_runs_from_the_store(
    fresh_voice: MercuryVoice, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine.narrative.detection_memory import DetectionMemoryStore

    store = DetectionMemoryStore(tmp_path / "memory.sqlite")
    monkeypatch.setattr(fresh_voice.retriever, "_detection_store", store)

    async def run() -> list[dict[str, Any]]:
        gen = voice_module.alert_events("alice", heartbeat=0, store_poll=0.05, max_events=1)
        await gen.__anext__()
        # Another worker appended straight to the shared store: no bus delivery here.
        store.append(
            {
                "detection": {
                    "detection_id": "elsewhere",
                    "method": "multivariate",
                    "anomaly_detected": True,
                    "anomaly_count": 1,
                    "n_samples": 4,
                    "anomaly_score": 9.0,
                    "threshold": 3.5,
                    "severity_label": "high",
                },
                "domain": "general",
                "principal": "alice",
                "timestamp": 1.0,
            }
        )
        return _events(b"".join([c async for c in gen]))

    events = asyncio.run(run())
    assert len(events) == 1 and events[0]["event"]["detection_id"] == "elsewhere"
    store.close()


def test_alerts_route_streams_event_stream(fresh_voice: MercuryVoice) -> None:
    """``?limit=1`` ends the stream after one alert; a run recorded meanwhile is that alert."""
    import threading
    import time

    def later() -> None:
        time.sleep(0.4)
        _record("anonymous", [False, True], [0.1, 9.0])

    threading.Thread(target=later).start()
    with TestClient(app) as client:
        response = client.get("/api/v1/voice/alerts", params={"limit": 1})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.startswith(b": connected")
    events = _events(response.content)
    assert len(events) == 1 and events[0]["initiative_type"].endswith("anomaly")


def test_history_and_status_are_scoped_to_the_caller(fresh_voice: MercuryVoice) -> None:
    fresh_voice.speak("What is a z-score?", domain="general", principal="alice")
    fresh_voice.speak("How many anomalies today?", domain="general", principal="bob")
    with TestClient(app) as client:
        history = client.get("/api/v1/voice/history").json()
        status = client.get("/api/v1/voice/status").json()
    # The unauthenticated caller is "anonymous": neither account's turns are visible.
    assert all("z-score" not in t["content"] for t in history["history"])
    assert all("anomalies today" not in t["content"] for t in history["history"])
    assert status["components"]["conversation_history"] == history["total_turns"]
    assert status["components"]["alerts"] == "operational"
    assert "detection_runs" in status["statistics"]
    alice = fresh_voice.get_conversation_history(100, principal="alice")
    assert any("z-score" in t["content"] for t in alice)
    assert not any("anomalies today" in t["content"] for t in alice)


def test_greet_lists_what_this_server_has(fresh_voice: MercuryVoice) -> None:
    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

    with TestClient(app) as client:
        body = client.get("/api/v1/voice/greet").json()
    assert body["message"]
    assert any(c.startswith(f"{len(DETECTOR_MANIFEST)} detectors") for c in body["capabilities"])
    assert any(c.endswith("domain data loaders") for c in body["capabilities"])
    assert "Proactive alerts as each run lands" in body["capabilities"]
    assert "Historical event lookup" not in body["capabilities"]

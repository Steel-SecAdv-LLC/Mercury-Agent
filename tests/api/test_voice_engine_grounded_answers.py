# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent explains its newest run through the narrative engine and describes the engine from code.

"Explain the latest detection" is one of the examples the voice route's own
documentation gives; it used to fall into the free-text search and come back
as a generic result. It now narrates the newest recorded run (verdict, score,
severity fraction, the run's facts as reasoning steps) through
``NarrativeEngine`` behind the same ethical gate as any detection, without
re-logging it. "Which detectors do you have?" and "What are your
capabilities?" are answered from ``DETECTOR_MANIFEST`` and the probed
``self_model.CAPABILITIES`` rather than from a fixed paragraph.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import record_detection
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

if TYPE_CHECKING:
    import pytest


def _voice(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    import pytest as _pytest

    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    if not hasattr(voice, "retriever"):
        _pytest.skip("narrative voice unavailable")
    return voice


def _record(anomalies: list[bool], scores: list[float], threshold: float) -> None:
    asyncio.run(
        record_detection(
            user_id="test",
            method="multivariate",
            data=[[float(i)] for i in range(len(anomalies))],
            results={"anomalies": anomalies, "scores": scores, "threshold": threshold},
            sensitivity=0.5,
            metadata={"domain": "general"},
        )
    )


def test_explain_latest_narrates_the_newest_run(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _voice(monkeypatch)
    _record([False, True, False], [0.1, 7.0, 0.2], 3.5)
    before = len(voice.retriever.detections(principal="test"))
    response = voice.speak("Explain the latest detection", domain="general", principal="test")
    message = response.message
    assert message.startswith(
        "Explaining multivariate run just now: 1 of 3 samples flagged (sample 1)"
    )
    # The narrative engine's own wording follows, built from the run's facts:
    # the severity fraction the mirror stored (2.0x the threshold = 50 % of the
    # critical boundary), the step count, and no invented confidence.
    assert "Detected moderate anomaly (score: 7.00, severity: 50%)" in message, message
    assert "Reasoning: 6 inference steps applied." in message, message
    assert "Confidence: not reported." in message, message
    assert "2.0x the 3.50 threshold (medium)" in message
    assert "narrative_synthesis" in response.sources_cited
    # Explaining does not re-log the run.
    assert len(voice.retriever.detections(principal="test")) == before


def test_explain_with_nothing_recorded_is_honest(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _voice(monkeypatch)
    response = voice.speak(
        "why was the last anomaly flagged?", domain="general", principal="nobody"
    )
    assert "no recorded run to explain" in response.message
    assert response.confidence is None


def test_detector_question_is_answered_from_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _voice(monkeypatch)
    response = voice.speak("Which detectors do you have?", domain="general")
    assert f"The manifest registers {len(DETECTOR_MANIFEST)} detectors." in response.message
    assert "By family:" in response.message and "By input:" in response.message
    assert response.sources_cited == ["detector_manifest"]


def test_family_question_lists_the_matching_detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _voice(monkeypatch)
    geological = [e for e in DETECTOR_MANIFEST if e.category.value == "geological"]
    response = voice.speak("Can you detect earthquakes?", domain="general")
    assert (
        f"{len(geological)} of the {len(DETECTOR_MANIFEST)} manifest detectors cover the geological family"
        in response.message
    )
    assert geological[0].name in response.message
    assert response.search_results_count == len(geological)


def test_capability_question_is_answered_from_the_self_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omni_mercury_engine import self_model

    voice = _voice(monkeypatch)
    response = voice.speak("What are your capabilities?", domain="general")
    assert "Capabilities on this machine:" in response.message
    for spec in self_model.CAPABILITIES:
        assert spec.name.replace("_", " ") in response.message
    assert "self_model" in response.sources_cited


def test_help_names_the_grounded_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    voice = _voice(monkeypatch)
    response = voice.speak("what can you do?", domain="general")
    assert "What was the latest anomaly?" in response.message
    assert "Which detectors do you have?" in response.message

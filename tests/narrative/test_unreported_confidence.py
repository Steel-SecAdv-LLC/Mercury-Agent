# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A detection that reports no confidence is narrated as "not reported", never 50 %.

``NarrativeEngine.synthesize`` defaulted ``confidence`` to 0.5 and ``severity``
to 0.0, and ``POST /api/v1/voice/narrate``'s fallback did the same, so a
detection that carried neither was narrated as "Detection confidence: 50%",
"Reasonable confidence (50%)" and "severity rating of 0%": numbers no detector
produced.
"""

from __future__ import annotations

import re

import pytest

from omni_mercury_engine.narrative.engine import ConfidenceLevel, NarrativeEngine

# The route test below needs the optional ``[api]`` extra; the synthesis test
# does not, so the import is guarded and only the route test skips without it.
fastapi = pytest.importorskip("fastapi", reason="the narrate-route test needs the [api] extra")
from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app

_BARE = {"anomaly_detected": True, "anomaly_score": 6.7, "summary": "1 of 60 flagged"}


def _all_text(result: object) -> str:
    return " ".join(
        str(getattr(result, k))
        for k in ("summary", "detailed_explanation", "uncertainty_disclosure", "recommendations")
    )


def test_unreported_confidence_and_severity_are_not_invented() -> None:
    result = NarrativeEngine().synthesize(dict(_BARE))
    assert result.confidence_score is None
    assert result.confidence_level is ConfidenceLevel.UNREPORTED
    text = _all_text(result)
    assert "50%" not in text and "severity rating of 0%" not in text, text
    assert "not reported" in text
    assert result.urgency_level == "unassessed"


def test_single_reasoning_step_is_singular() -> None:
    result = NarrativeEngine().synthesize(dict(_BARE, reasoning_chain=["1 of 60 flagged"]))
    assert "1 reasoning step." in result.detailed_explanation


def test_reported_values_are_still_narrated() -> None:
    result = NarrativeEngine().synthesize(dict(_BARE, confidence=0.82, severity=0.6))
    assert result.confidence_score == 0.82
    assert result.confidence_level is ConfidenceLevel.HIGH
    assert "82%" in _all_text(result)
    assert result.urgency_level != "unassessed"


def test_narrate_route_says_not_reported() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/voice/narrate", json={"detection_result": dict(_BARE)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["severity"] == "unreported"  # this field carries the confidence level
    assert not re.search(r"\b50%", " ".join(map(str, body.values()))), body
    assert (
        "not reported" in body["confidence_statement"] or "not been" in body["confidence_statement"]
    )

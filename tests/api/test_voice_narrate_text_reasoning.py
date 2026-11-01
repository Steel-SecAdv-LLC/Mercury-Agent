# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``POST /api/v1/voice/narrate`` accepts plain-text reasoning steps.

The route takes a free-form ``detection_result``. The narrative engine called
``step.get(...)`` on every ``reasoning_chain`` entry, so a client that sent
``["1 of 60 flagged"]`` (the Mercury Console does, on every run) got a 500.
A string is now an observation step, reported without an invented confidence.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app
from omni_mercury_engine.narrative.engine import _normalize_reasoning_chain


def test_text_reasoning_steps_narrate_instead_of_500() -> None:
    body = {
        "detection_result": {
            "anomaly_detected": True,
            "anomaly_score": 6.7,
            "reasoning_chain": ["1 of 60 flagged", "max score 6.70 at sample 42"],
            "summary": "1 of 60 flagged",
        },
        "verbosity": "brief",
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/voice/narrate", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["summary"]


def test_normalization_keeps_rule_dicts_and_wraps_strings() -> None:
    rule = {"rule": "zscore", "conclusion": "outlier", "confidence": 0.9}
    out = _normalize_reasoning_chain({"reasoning_chain": [rule, "seen twice", None]})
    assert out["reasoning_chain"] == [rule, {"rule": "observation", "conclusion": "seen twice"}]
    untouched = {"reasoning_chain": [rule]}
    assert _normalize_reasoning_chain(untouched) is untouched


def test_unscored_steps_are_not_reported_as_fifty_percent() -> None:
    from omni_mercury_engine.narrative.engine import NarrativeEngine

    narrative = NarrativeEngine()._verbalize_reasoning_chain(
        [{"rule": "observation", "conclusion": "1 of 60 flagged"}], []
    )
    assert narrative is not None
    assert all("50%" not in step for step in narrative.steps)
    assert "1 of 60 flagged" in narrative.hypotheses_considered
    assert "no step-level confidence" in narrative.confidence_rationale

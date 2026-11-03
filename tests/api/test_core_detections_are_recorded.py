# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``/detect/univariate`` and ``/detect/multivariate`` record their detections.

Every route in ``routes/detection.py`` sends its result through
``record_detection`` -- export history, alert egress and the agent's memory.
The two endpoints defined in ``api/server.py`` did not, so the most used
detections (the console's default ``detect`` path) never reached history,
alerts or the agent: "what happened recently?" was answered from nothing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omni_mercury_engine.api import voice as voice_module
from omni_mercury_engine.api.routes.export import get_data_store
from omni_mercury_engine.api.server import app


@pytest.mark.parametrize(
    ("path", "body", "method"),
    [
        (
            "/api/v1/detect/univariate",
            {"data": [1.0, 1.1, 0.9, 1.0, 9.0, 1.0, 1.05, 0.95], "sensitivity": 0.5},
            "univariate",
        ),
        (
            "/api/v1/detect/multivariate",
            {"data": [[1, 2], [1.1, 2], [0.9, 2.1], [1, 1.9], [9, 9], [1, 2], [1.05, 2]]},
            "multivariate",
        ),
    ],
)
def test_detection_reaches_history_and_the_agent(
    monkeypatch: pytest.MonkeyPatch, path: str, body: dict[str, object], method: str
) -> None:
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    voice = voice_module._get_voice()
    store = get_data_store()
    before = len(store._detections)
    with TestClient(app) as client:
        response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert len(store._detections) == before + 1
    record = store._detections[-1]
    assert record.method == method
    assert record.anomaly_count == sum(response.json()["anomalies"])
    if hasattr(voice, "retriever"):
        entry = voice.retriever._detection_log[-1]["detection"]
        assert entry["method"] == method
        assert entry["severity_label"] in {"low", "medium", "high", "critical"}

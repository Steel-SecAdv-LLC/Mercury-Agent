# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every detection route that judges against a threshold remembers that threshold.

The fusion route computed a threshold and left it out of its recording payload,
and the detector tier recorded it only inside ``summary``, so the agent kept
those runs with no threshold and no severity band. These tests go through the
routes themselves and read the agent's memory afterwards.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app
from omni_mercury_engine.api.voice import _get_voice


def _newest(method: str) -> dict[str, Any]:
    entries = _get_voice().retriever.detections(principal="anonymous")
    for entry in entries:
        if entry["detection"].get("method") == method:
            return dict(entry["detection"])
    raise AssertionError(f"no remembered {method} run among {len(entries)} entries")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_fusion_route_remembers_its_threshold(client: TestClient) -> None:
    response = client.post(
        "/api/v1/detect/fusion",
        json={"data": [0.0] * 11 + [10.0], "sensitivity": 0.52},
    )
    assert response.status_code == 200, response.text
    remembered = _newest("fusion")
    assert remembered["threshold"] == pytest.approx(response.json()["threshold"])
    assert remembered["severity"] is not None and remembered["severity_label"] is not None


def test_tier_route_remembers_its_threshold(client: TestClient) -> None:
    response = client.post(
        "/api/v1/detect/tier",
        json={"data": [0.4, 0.5, 0.45, 10.0, 0.5, 0.48, 0.47, 0.5, 0.46, 0.49, 0.51, 0.44]},
    )
    assert response.status_code == 200, response.text
    remembered = _newest("detector_tier")
    assert remembered["threshold"] == pytest.approx(response.json()["threshold"])
    assert remembered["severity"] is not None


def test_rca_route_is_not_remembered_as_a_verdict(client: TestClient) -> None:
    before = sum(
        1
        for e in _get_voice().retriever.detections(principal="anonymous")
        if e["detection"].get("method") == "rca"
    )
    response = client.post(
        "/api/v1/detect/rca",
        json={"observations": [[float(i), float(i) * 2.0, 1.0] for i in range(10)], "top_k": 2},
    )
    assert response.status_code == 200, response.text
    after = sum(
        1
        for e in _get_voice().retriever.detections(principal="anonymous")
        if e["detection"].get("method") == "rca"
    )
    assert after == before

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``/health`` reports how long the service has been up.

``HealthResponse.uptime_seconds`` was declared but never filled, so a
load balancer or an operator watching for restarts always read ``null``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app


def test_health_reports_a_non_negative_uptime_that_advances() -> None:
    client = TestClient(app)
    first = client.get("/health").json()
    second = client.get("/health").json()

    assert first["status"] == "healthy"
    assert isinstance(first["uptime_seconds"], float) and first["uptime_seconds"] >= 0.0
    assert second["uptime_seconds"] >= first["uptime_seconds"]

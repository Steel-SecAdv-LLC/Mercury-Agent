# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The unauthenticated ``/health`` stays minimal; uptime lives on ``/metrics``.

``/health`` answers load balancers and uptime monitors without credentials, so
it reports only what a probe needs. Process uptime would tell any caller when
the service last restarted (a crash loop, or the window after a patch), and the
field was never populated before. Operators read it from the Prometheus
exposition instead, which the platform's TLS edge does not proxy.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import app


def test_health_reports_status_and_version_only() -> None:
    body = TestClient(app).get("/health").json()

    assert body["status"] == "healthy"
    assert isinstance(body["version"], str) and body["version"]
    assert body.get("uptime_seconds") is None
    assert set(body) <= {"status", "version", "uptime_seconds"}


def test_uptime_is_exported_as_a_metric() -> None:
    text = TestClient(app).get("/metrics").text

    assert "omni_mercury_uptime_seconds" in text

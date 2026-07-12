# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""REST API end-to-end integration.

Drives the assembled FastAPI application (router → request validation →
engine → detector → serialised response) through a real ``TestClient`` and
asserts that detection results are *correct* — the injected anomaly is
surfaced at the right position — not merely well-shaped. The unit-level
``tests/test_api.py`` already pins response structure and the ``/metrics``
content type; this lane adds the cross-stack correctness and error-handling
contracts, and (in CI) runs with ``AMA_REQUIRE_REAL_PQC=true`` so the whole
app is exercised on top of the real post-quantum backend.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import API_VERSION, app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A module-scoped client over the real app object."""
    return TestClient(app)


class TestOperationalEndpoints:
    """Liveness + scrape surface — the contract orchestrators depend on."""

    def test_health_reports_live_and_versioned(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        # Version flows from the single source of truth, not a literal.
        assert body["version"] == API_VERSION

    def test_metrics_scrape_path_serves_prometheus(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "omni_mercury_up" in response.text

    def test_metrics_expose_ethical_gate_series(self, client: TestClient) -> None:
        """The σ_Alignment / benevolence gauges the helm PrometheusRule alerts on
        must actually be emitted (otherwise the ethical-gate alerts sit in
        NoData). Both series appear with parseable float values."""
        text = client.get("/metrics").text
        assert "mercury_agent_sigma_alignment" in text
        assert "mercury_agent_benevolence_score" in text
        emitted = {}
        for line in text.splitlines():
            if line.startswith(
                ("mercury_agent_sigma_alignment ", "mercury_agent_benevolence_score ")
            ):
                name, _, value = line.partition(" ")
                emitted[name] = float(value)
        assert emitted["mercury_agent_benevolence_score"] > 0.0
        assert 0.0 <= emitted["mercury_agent_sigma_alignment"] <= 3.0


class TestDetectionThroughTheApi:
    """Detection correctness across the full request/response path."""

    def test_univariate_detection_pins_the_spike(self, client: TestClient) -> None:
        """A single spike in an otherwise flat series is flagged at its index
        when routed through the API."""
        # A flat baseline with one unmistakable spike at index 40. Anomaly
        # statistics need a real sample to be meaningful, so this is sized
        # well past the handful of points the structural unit test uses.
        series = [1.0] * 40 + [50.0] + [1.0] * 9
        response = client.post(
            "/api/v1/detect/univariate",
            json={"data": series},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == "univariate"

        anomalies = body["anomalies"]
        scores = body["scores"]
        assert len(anomalies) == len(series)
        assert len(scores) == len(series)
        # Index 40 is the spike: flagged, and the single dominant score.
        assert bool(anomalies[40]) is True
        assert scores.index(max(scores)) == 40

    def test_multivariate_detection_returns_full_contract(self, client: TestClient) -> None:
        """The multivariate path flags an outlier row and returns the richer
        per-feature contract."""
        rng = np.random.default_rng(0)
        rows = (rng.normal(0.0, 0.2, size=(30, 2)) + np.array([1.0, 2.0])).tolist()
        rows.append([50.0, -50.0])  # the outlier row at index 30
        response = client.post(
            "/api/v1/detect/multivariate",
            json={"data": rows, "sensitivity": 0.9},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == "multivariate"
        for key in ("anomalies", "scores", "features", "feature_contributions"):
            assert key in body
        assert len(body["anomalies"]) == 31
        # The [50, -50] row is the outlier, and the highest-scoring sample.
        assert bool(body["anomalies"][30]) is True
        assert body["scores"].index(max(body["scores"])) == 30


class TestRequestValidation:
    """Malformed requests are rejected at the boundary, not 500'd downstream."""

    def test_empty_data_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/detect/univariate", json={"data": []})
        assert response.status_code == 422

    def test_missing_field_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/detect/univariate", json={})
        assert response.status_code == 422

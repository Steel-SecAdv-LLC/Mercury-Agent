# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every detection route writes the detection series the dashboard and alerts read.

``omni_detection_requests_total``, ``omni_detection_success_total`` and
``omni_detection_duration_seconds`` were written only by the engine, and of
the eight detection routes only ``/api/v1/detect/flagship`` runs the engine.
Found with Prometheus scraping a live server: one successful call to each of
the other seven moved none of the three series, so the "Detections by Type"
and "Detection Processing Time" panels were empty and the
``OmniMercuryDetectionSuccessRateDrop`` and ``OmniMercurySlowDetection``
alerts had nothing to evaluate for the routes most clients call.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the API server needs the [api] extra")
prometheus_client = pytest.importorskip(
    "prometheus_client", reason="the detection series need prometheus-client"
)

from fastapi import HTTPException
from fastapi.testclient import TestClient

from omni_mercury_engine.api.detection_metrics import ROUTE_PREFIX, records_detection

_REQUESTS = "omni_detection_requests_total"
_SUCCESS = "omni_detection_success_total"
_DURATION_COUNT = "omni_detection_duration_seconds_count"


def _value(metric: str, name: str) -> float:
    sample = prometheus_client.REGISTRY.get_sample_value(
        metric, {"detector_type": ROUTE_PREFIX + name}
    )
    return float(sample or 0.0)


def _snapshot(name: str) -> tuple[float, float, float]:
    return _value(_REQUESTS, name), _value(_SUCCESS, name), _value(_DURATION_COUNT, name)


def _delta(name: str, before: tuple[float, float, float]) -> tuple[float, float, float]:
    after = _snapshot(name)
    return (after[0] - before[0], after[1] - before[1], after[2] - before[2])


class TestRecorder:
    """What one decorated handler writes, by outcome."""

    def test_a_returning_handler_counts_a_request_a_success_and_a_duration(self) -> None:
        @records_detection("unit_ok")
        async def handler(value: int) -> int:
            return value * 2

        before = _snapshot("unit_ok")
        assert asyncio.run(handler(21)) == 42
        assert _delta("unit_ok", before) == (1.0, 1.0, 1.0)

    def test_a_raising_handler_counts_a_request_without_a_success(self) -> None:
        @records_detection("unit_raise")
        async def handler() -> None:
            raise RuntimeError("detector failed")

        before = _snapshot("unit_raise")
        with pytest.raises(RuntimeError, match="detector failed"):
            asyncio.run(handler())
        assert _delta("unit_raise", before) == (1.0, 0.0, 1.0)

    def test_a_server_error_response_counts_as_a_failed_attempt(self) -> None:
        @records_detection("unit_500")
        async def handler() -> None:
            raise HTTPException(status_code=503, detail="model unavailable")

        before = _snapshot("unit_500")
        with pytest.raises(HTTPException):
            asyncio.run(handler())
        assert _delta("unit_500", before) == (1.0, 0.0, 1.0)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422, 429])
    def test_a_refused_request_is_not_a_detection_attempt(self, status: int) -> None:
        """Bad input or a missing credential must not lower the success rate."""

        @records_detection("unit_refused")
        async def handler() -> None:
            raise HTTPException(status_code=status, detail="refused")

        before = _snapshot("unit_refused")
        with pytest.raises(HTTPException):
            asyncio.run(handler())
        assert _delta("unit_refused", before) == (0.0, 0.0, 0.0)

    def test_skip_leaves_the_series_untouched(self) -> None:
        skipping = True

        @records_detection("unit_skip", skip=lambda: skipping)
        async def handler() -> str:
            return "done"

        before = _snapshot("unit_skip")
        assert asyncio.run(handler()) == "done"
        assert _delta("unit_skip", before) == (0.0, 0.0, 0.0)
        skipping = False
        assert asyncio.run(handler()) == "done"
        assert _delta("unit_skip", before) == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize("name", ["", "Univariate", "three-r", "a b", "1st", 'x"} 1'])
    def test_a_name_that_is_not_a_plain_label_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="detection route name"):
            records_detection(name)

    def test_the_wrapper_keeps_the_signature_fastapi_reads(self) -> None:
        async def handler(request: int, user: str = "anonymous") -> int:
            """Docstring kept."""
            return request

        wrapped = records_detection("unit_signature")(handler)
        assert inspect.signature(wrapped) == inspect.signature(handler)
        assert inspect.unwrap(wrapped) is handler
        assert wrapped.__doc__ == "Docstring kept."


def _series() -> list[float]:
    rng = random.Random(3)
    values = [round(1 + 0.1 * math.sin(i / 4) + rng.gauss(0, 0.03), 4) for i in range(120)]
    values[90] = 9.5
    return values


def _matrix(width: int = 4, rows: int = 80) -> list[list[float]]:
    rng = random.Random(5)
    data = [[round(rng.gauss(0, 1), 4) for _ in range(width)] for _ in range(rows)]
    data[40] = [8.0] * width
    return data


def _records() -> list[dict[str, float]]:
    rng = random.Random(7)
    rows = [
        {"cpu": rng.gauss(50, 5), "mem": rng.gauss(60, 4), "latency": rng.gauss(20, 2)}
        for _ in range(60)
    ]
    rows[30] = {"cpu": 99.0, "mem": 98.0, "latency": 400.0}
    return rows


#: route name -> (path, request body). Flagship is absent on purpose: it runs
#: the engine, which writes the series per detector.
_ROUTES: dict[str, tuple[str, dict[str, Any]]] = {
    "univariate": ("/api/v1/detect/univariate", {"data": _series()}),
    "multivariate": ("/api/v1/detect/multivariate", {"data": _matrix()}),
    "fusion": ("/api/v1/detect/fusion", {"data": _matrix()}),
    "tier": ("/api/v1/detect/tier", {"data": _series()}),
    "neurosymbolic": ("/api/v1/detect/neurosymbolic", {"data": _records()}),
    "three_r": ("/api/v1/detect/three-r", {"data": _series()}),
    "rca": ("/api/v1/detect/rca", {"observations": _matrix()}),
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """One client over the real application for the module."""
    from omni_mercury_engine.api.server import app

    return TestClient(app)


class TestServedRoutes:
    """The real application: one successful call per route moves its series by one."""

    @pytest.mark.parametrize("name", sorted(_ROUTES))
    def test_a_successful_detection_is_counted_once(self, client: TestClient, name: str) -> None:
        path, body = _ROUTES[name]
        before = _snapshot(name)
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text[:300]
        assert _delta(name, before) == (1.0, 1.0, 1.0)

    def test_invalid_input_is_not_counted(self, client: TestClient) -> None:
        before = _snapshot("univariate")
        response = client.post("/api/v1/detect/univariate", json={"data": [1.0]})
        assert 400 <= response.status_code < 500
        assert _delta("univariate", before) == (0.0, 0.0, 0.0)

    def test_every_non_engine_detection_route_is_recorded(self, client: TestClient) -> None:
        """A new detection route must be recorded, or declared engine-backed here."""
        engine_backed = {"/api/v1/detect/flagship"}
        served = {
            path
            for path, methods in client.app.openapi()["paths"].items()  # type: ignore[attr-defined]
            if path.startswith("/api/v1/detect/") and "post" in methods
        }
        assert served - engine_backed == {path for path, _body in _ROUTES.values()}

    def test_the_warm_up_is_not_counted_as_traffic(self) -> None:
        """The server calls two of its own handlers at startup; those are not requests."""
        from omni_mercury_engine.api import server

        before = (_snapshot("univariate"), _snapshot("multivariate"))
        asyncio.run(server._warmup(server.app))
        assert _delta("univariate", before[0]) == (0.0, 0.0, 0.0)
        assert _delta("multivariate", before[1]) == (0.0, 0.0, 0.0)

    def test_the_series_reach_the_metrics_endpoint(self, client: TestClient) -> None:
        client.post("/api/v1/detect/univariate", json=_ROUTES["univariate"][1])
        text = client.get("/metrics").text
        assert 'omni_detection_requests_total{detector_type="api_univariate"}' in text
        assert 'omni_detection_success_total{detector_type="api_univariate"}' in text

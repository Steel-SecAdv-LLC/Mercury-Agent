# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every finished API request leaves an audit entry; the export serves them.

``record_audit_log`` used to have no caller, so ``/export/audit-logs`` always
returned an empty trail. These tests drive real requests through the app and
read the trail back through the export endpoint and the tamper-evident sink.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from omni_mercury_engine.api import auth_audit
from omni_mercury_engine.api.request_audit import audit_target

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def api_user() -> tuple[str, dict[str, str]]:
    from omni_mercury_engine.api.auth import get_api_key_store

    user_id = f"audit-user-{uuid.uuid4().hex[:8]}"
    raw_key, _ = get_api_key_store().create_key(name="audit_test_key", user_id=user_id)
    return user_id, {"X-API-Key": raw_key}


def _store_entries(user_id: str) -> list[Any]:
    from omni_mercury_engine.api.routes.export import get_data_store

    return [r for r in get_data_store()._audit_logs if r.user_id == user_id]


class TestAuditTarget:
    class _Req:
        def __init__(self, template: str | None, params: dict[str, str] | None = None) -> None:
            route = type("R", (), {"path": template})() if template is not None else None
            self.scope = {"route": route, "path_params": params or {}}
            self.method = "GET"

    @pytest.mark.parametrize(
        ("template", "params", "expected"),
        [
            ("/api/v1/export/jobs/{job_id}", {"job_id": "j1"}, ("export", "job_id=j1")),
            ("/api/v1/detect/univariate", {}, ("detect", "")),
            ("/auth/login", {}, ("auth", "")),
            ("/{name}", {"name": "x"}, ("root", "name=x")),
        ],
    )
    def test_resource_from_template(
        self, template: str, params: dict[str, str], expected: tuple[str, str]
    ) -> None:
        action, resource_type, resource_id = audit_target(self._Req(template, params))  # type: ignore[misc]
        assert action == f"GET {template}"
        assert (resource_type, resource_id) == expected

    @pytest.mark.parametrize("template", [None, "/health", "/metrics", "/docs", "/openapi.json"])
    def test_probes_docs_and_unmatched_are_not_audited(self, template: str | None) -> None:
        assert audit_target(self._Req(template)) is None


class TestRequestsAreAudited:
    def test_authenticated_request_is_attributed_and_exported(
        self, client: TestClient, api_user: tuple[str, dict[str, str]]
    ) -> None:
        user_id, headers = api_user
        response = client.get(
            "/api/v1/export/detections", headers=headers, params={"format": "json", "limit": 5}
        )
        assert response.status_code == 200
        entries = _store_entries(user_id)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.action == "GET /api/v1/export/detections"
        assert entry.resource_type == "export"
        assert entry.response_status == 200
        assert entry.duration_ms > 0
        assert entry.metadata["auth_method"] == "api_key"
        assert entry.metadata["query_params"] == ["format", "limit"]
        assert entry.metadata["correlation_id"] == response.headers["X-Correlation-ID"]
        # Values of query parameters are never stored.
        assert "json" not in json.dumps(entry.metadata)

        exported = client.get(
            "/api/v1/export/audit-logs", headers=headers, params={"format": "json", "limit": 50}
        )
        assert exported.status_code == 200
        records = json.loads(exported.text)
        assert any(
            r["action"] == "GET /api/v1/export/detections"
            and r["user_id"] == user_id
            and r["tlp"] == "TLP:AMBER+STRICT"
            for r in records
        )

    def test_rejected_request_is_audited_as_anonymous(self, client: TestClient) -> None:
        from omni_mercury_engine.api.routes.export import get_data_store

        before = len(get_data_store()._audit_logs)
        response = client.get("/api/v1/export/detections")
        assert response.status_code in (401, 403)
        new = list(get_data_store()._audit_logs)[before:]
        mine = [r for r in new if r.action == "GET /api/v1/export/detections"]
        assert mine and mine[-1].user_id == "anonymous"
        assert mine[-1].response_status == response.status_code

    def test_rate_limited_rejections_are_not_audited(self) -> None:
        """A 429 flood must not be able to push real entries out of the window."""
        import asyncio

        from starlette.requests import Request

        from omni_mercury_engine.api.request_audit import audit_request
        from omni_mercury_engine.api.routes.export import get_data_store

        def _request() -> Request:
            return Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/v1/detect/univariate",
                    "headers": [],
                    "query_string": b"",
                    "client": ("203.0.113.9", 4000),
                    "route": type("R", (), {"path": "/api/v1/detect/univariate"})(),
                    "path_params": {},
                }
            )

        store = get_data_store()
        before = len(store._audit_logs)
        asyncio.run(audit_request(_request(), 429, 1.0))
        assert len(store._audit_logs) == before
        # Control: the same request with any other status is recorded.
        asyncio.run(audit_request(_request(), 200, 1.0))
        assert len(store._audit_logs) == before + 1
        assert store._audit_logs[-1].ip_address == "203.0.113.9"

    def test_probes_leave_no_entry(self, client: TestClient) -> None:
        from omni_mercury_engine.api.routes.export import get_data_store

        before = len(get_data_store()._audit_logs)
        client.get("/health")
        client.get("/metrics")
        client.get(f"/no-such-route-{uuid.uuid4().hex}")
        assert len(get_data_store()._audit_logs) == before


class TestTamperEvidentSink:
    def test_requests_and_auth_share_one_verifiable_chain(
        self,
        client: TestClient,
        api_user: tuple[str, dict[str, str]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(auth_audit.AUDIT_DIR_ENV, str(tmp_path))
        user_id, headers = api_user
        assert client.get("/api/v1/export/detections", headers=headers).status_code == 200
        # An auth event through the auth auditor lands in the same chain.
        auth_audit.build_auth_auditor().record("login", outcome="success", account_id=user_id)

        secure = auth_audit.shared_secure_logger()
        assert secure is not None
        assert secure is auth_audit.shared_secure_logger()  # one sink per directory
        secure.flush()
        lines = (tmp_path / "audit.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in lines]
        assert [e["actor"] for e in events][-2:] == [user_id, user_id]
        assert events[-2]["action"] == "GET /api/v1/export/detections"
        assert events[-2]["category"] == "data_access"
        assert events[-1]["category"] == "authentication"
        ok, message = secure.verify_log_integrity()
        assert ok, message


def test_unhandled_exception_is_metered_and_audited(client: TestClient) -> None:
    """A route that raises still yields a 500 sample and an audit entry."""
    prometheus_client = pytest.importorskip("prometheus_client")
    from omni_mercury_engine.api.routes.export import get_data_store
    from omni_mercury_engine.api.server import app

    path = f"/api/v1/test-boom-{uuid.uuid4().hex[:8]}"

    async def boom() -> None:
        raise RuntimeError("unhandled route failure")

    app.add_api_route(path, boom, methods=["GET"])
    route = app.router.routes[-1]
    try:
        labels = {"method": "GET", "endpoint": path, "status": "500"}
        before = prometheus_client.REGISTRY.get_sample_value("http_requests_total", labels) or 0.0
        assert client.get(path).status_code == 500
        after = prometheus_client.REGISTRY.get_sample_value("http_requests_total", labels)
        assert after == before + 1
        entries = [r for r in get_data_store()._audit_logs if r.action == f"GET {path}"]
        assert [e.response_status for e in entries] == [500]
    finally:
        app.router.routes.remove(route)

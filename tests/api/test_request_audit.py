# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
    from omni_mercury_engine.api.auth import Permission, get_api_key_store

    user_id = f"audit-user-{uuid.uuid4().hex[:8]}"
    raw_key, _ = get_api_key_store().create_key(
        name="audit_test_key",
        user_id=user_id,
        permissions={Permission.READ, Permission.EXPORT},
    )
    return user_id, {"X-API-Key": raw_key}


def _bare_request(
    *,
    status_path: str = "/api/v1/detect/univariate",
    path: str | None = None,
    params: dict[str, str] | None = None,
    query: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Any:
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path or status_path,
            "headers": headers or [],
            "query_string": query,
            "client": ("203.0.113.9", 4000),
            "route": type("R", (), {"path": status_path})(),
            "path_params": params or {},
        }
    )


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

    def test_anonymous_rejection_is_counted_not_audited(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unauthenticated 4xx never reach the shared hash chain or the window."""
        from omni_mercury_engine.api.request_audit import anonymous_rejection_counts
        from omni_mercury_engine.api.routes.export import get_data_store

        monkeypatch.setenv(auth_audit.AUDIT_DIR_ENV, str(tmp_path))
        key = "GET /api/v1/export/detections -> 401"
        counted = anonymous_rejection_counts().get(key, 0)
        before = len(get_data_store()._audit_logs)
        for _ in range(3):
            assert client.get("/api/v1/export/detections").status_code == 401
        new = [
            r
            for r in list(get_data_store()._audit_logs)[before:]
            if r.action == "GET /api/v1/export/detections"
        ]
        assert new == []
        assert anonymous_rejection_counts()[key] == counted + 3
        secure = auth_audit.shared_secure_logger()
        assert secure is not None
        secure.flush()
        log = tmp_path / "audit.jsonl"
        assert not log.exists() or "export/detections" not in log.read_text()

    def test_authenticated_denial_is_audited(self, client: TestClient) -> None:
        """A 403 for a known principal is attributable, so it is recorded."""
        from omni_mercury_engine.api.auth import Permission, get_api_key_store

        user_id = f"detect-only-{uuid.uuid4().hex[:8]}"
        raw_key, _ = get_api_key_store().create_key(
            name="detect_only", user_id=user_id, permissions={Permission.DETECT}
        )
        response = client.get("/api/v1/export/audit-logs", headers={"X-API-Key": raw_key})
        assert response.status_code == 403
        entries = _store_entries(user_id)
        assert [e.response_status for e in entries] == [403]

    def test_entry_fields_are_bounded_and_store_the_template(self) -> None:
        """Raw paths, huge user agents and query floods are clipped."""
        import asyncio

        from omni_mercury_engine.api.request_audit import (
            MAX_FIELD_CHARS,
            MAX_PARAM_CHARS,
            MAX_QUERY_NAMES,
            audit_request,
            set_audit_actor,
        )
        from omni_mercury_engine.api.routes.export import get_data_store

        actor = f"bounded-{uuid.uuid4().hex[:8]}"
        query = "&".join(f"{'q' * 100}{i}=v" for i in range(40)).encode()
        request = _bare_request(
            status_path="/api/v1/export/jobs/{job_id}",
            path="/api/v1/export/jobs/" + "x" * 5000,
            params={"job_id": "x" * 5000},
            query=query,
            headers=[(b"user-agent", b"A" * 10_000)],
        )
        set_audit_actor(request, actor, "session")
        asyncio.run(audit_request(request, 200, 1.0))
        entry = get_data_store()._audit_logs[-1]
        assert entry.user_id == actor
        assert entry.request_path == "/api/v1/export/jobs/{job_id}"
        assert len(entry.user_agent) == MAX_FIELD_CHARS
        assert len(entry.resource_id) <= MAX_FIELD_CHARS
        assert entry.resource_id.startswith("job_id=" + "x" * (MAX_PARAM_CHARS - 3) + "...")
        names = entry.metadata["query_params"]
        assert len(names) == MAX_QUERY_NAMES
        assert all(len(n) <= MAX_PARAM_CHARS for n in names)
        assert entry.metadata["query_params_dropped"] == 40 - MAX_QUERY_NAMES
        assert entry.metadata["auth_method"] == "session"

    def test_anonymous_rejection_tally_is_bounded(self) -> None:
        """Distinct anonymous 404 routes cannot grow the tally without bound."""
        import asyncio

        from omni_mercury_engine.api import request_audit

        for i in range(request_audit._MAX_REJECTION_KEYS + 50):
            request = _bare_request(status_path=f"/api/v1/probe-{uuid.uuid4().hex}-{i}")
            asyncio.run(request_audit.audit_request(request, 404, 1.0))
        assert len(request_audit.anonymous_rejection_counts()) <= (
            request_audit._MAX_REJECTION_KEYS + 1
        )

    def test_audit_window_has_its_own_smaller_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from omni_mercury_engine.api.routes import export as export_module

        monkeypatch.delenv("MAX_AUDIT_LOG_RECORDS", raising=False)
        store = export_module.DataStore()
        assert store._audit_logs.maxlen == export_module.DEFAULT_MAX_AUDIT_LOG_RECORDS
        assert store._audit_logs.maxlen < store._detections.maxlen  # type: ignore[operator]
        monkeypatch.setenv("MAX_AUDIT_LOG_RECORDS", "7")
        assert export_module.DataStore()._audit_logs.maxlen == 7
        monkeypatch.setenv("MAX_AUDIT_LOG_RECORDS", "garbage")
        assert (
            export_module.DataStore()._audit_logs.maxlen
            == export_module.DEFAULT_MAX_AUDIT_LOG_RECORDS
        )

    def test_rate_limited_rejections_are_not_audited(self) -> None:
        """A 429 flood must not be able to push real entries out of the window."""
        import asyncio

        from omni_mercury_engine.api.request_audit import audit_request
        from omni_mercury_engine.api.routes.export import get_data_store

        def _request() -> Any:
            request = _bare_request()
            request.state.audit_actor = ("rate-limit-actor", "api_key")
            return request

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


def test_session_cookie_actions_are_attributed_to_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key creation through a browser session is audited as the account."""
    from fastapi import FastAPI

    from omni_mercury_engine.api import auth
    from omni_mercury_engine.api.auth_service import AuthService
    from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
    from omni_mercury_engine.api.routes import accounts
    from omni_mercury_engine.api.routes.export import get_data_store
    from omni_mercury_engine.api.server import CorrelationIDMiddleware

    class _Mailer:
        def __init__(self) -> None:
            self.bodies: list[str] = []

        def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
            self.bodies.append(body)

    import re

    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setattr(auth, "_api_key_store", None)
    mailer = _Mailer()
    service = AuthService(InMemoryIdentityStore(), mailer)
    app = FastAPI()
    app.add_middleware(CorrelationIDMiddleware)
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    email = f"audit-{uuid.uuid4().hex[:6]}@b.com"
    with TestClient(app) as client:
        client.post("/api/v1/auth/register", json={"email": email, "password": "a-strong-pw"})
        match = re.search(r"token=([\w\-]+)", mailer.bodies[-1])
        assert match is not None
        client.post("/api/v1/auth/verify-email", json={"token": match.group(1)})
        login = client.post("/api/v1/auth/login", json={"email": email, "password": "a-strong-pw"})
        csrf = login.json()["csrf_token"]
        account_id = client.get("/api/v1/auth/me").json()["id"]
        created = client.post(
            "/api/v1/auth/api-keys", json={"name": "k"}, headers={"X-CSRF-Token": csrf}
        )
        assert created.status_code == 201
    entries = [r for r in get_data_store()._audit_logs if r.action == "POST /api/v1/auth/api-keys"]
    assert entries and entries[-1].user_id == account_id
    assert entries[-1].metadata["auth_method"] == "session"

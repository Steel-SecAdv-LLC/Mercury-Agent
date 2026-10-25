# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The export API requires the ``export`` permission; CSV cells are inert.

Every ``/api/v1/export/*`` route used to accept any authenticated credential,
so a detect-only self-service key (the default ``READ`` + ``DETECT`` set) could
bulk-export audit logs and detections. Exported CSV carried caller-controlled
strings verbatim, so a cell such as ``=HYPERLINK(...)`` ran when the file was
opened in a spreadsheet.
"""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth import Permission, User, get_api_key_store
from omni_mercury_engine.api.routes import export as export_module

_EXPORT_ROUTES = (
    "/api/v1/export/summary",
    "/api/v1/export/detections",
    "/api/v1/export/audit-logs",
    "/api/v1/export/metrics",
    "/api/v1/export/report?format=json",
)


@pytest.fixture
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


def _key(*permissions: Permission) -> dict[str, str]:
    raw_key, _ = get_api_key_store().create_key(
        name="export_perm_key",
        user_id=f"export-perm-{uuid.uuid4().hex[:8]}",
        permissions=set(permissions),
    )
    return {"X-API-Key": raw_key}


@pytest.mark.parametrize("route", _EXPORT_ROUTES)
def test_detect_only_key_is_forbidden(client: TestClient, route: str) -> None:
    response = client.get(route, headers=_key(Permission.READ, Permission.DETECT))
    assert response.status_code == 403
    assert "export" in response.json()["detail"]


@pytest.mark.parametrize("route", _EXPORT_ROUTES)
def test_export_and_admin_keys_are_allowed(client: TestClient, route: str) -> None:
    for headers in (_key(Permission.READ, Permission.EXPORT), _key(Permission.ADMIN)):
        assert client.get(route, headers=headers).status_code in (200, 429)


def test_missing_credential_is_still_401(client: TestClient) -> None:
    assert client.get("/api/v1/export/audit-logs").status_code == 401


@pytest.mark.parametrize(
    ("permissions", "allowed"),
    [
        ({Permission.READ}, False),  # JWT without a permissions claim
        ({Permission.READ, Permission.DETECT}, False),
        ({Permission.EXPORT}, True),
        ({Permission.ADMIN}, True),
    ],
)
def test_dependency_applies_to_jwt_principals(permissions: set[Permission], allowed: bool) -> None:
    """The JWT branch of the dependency is gated identically."""
    user = User(id="jwt-user", username="jwt", permissions=permissions)
    if allowed:
        assert export_module._get_current_user(api_key_user=None, jwt_user=user) is user
    else:
        with pytest.raises(HTTPException) as excinfo:
            export_module._get_current_user(api_key_user=None, jwt_user=user)
        assert excinfo.value.status_code == 403


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ('=HYPERLINK("http://x")', '\'=HYPERLINK("http://x")'),
        ("+1+1", "'+1+1"),
        ("-2+3", "'-2+3"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("\tcmd", "'\tcmd"),
        ("\rcmd", "'\rcmd"),
        ("plain", "plain"),
        (-0.5, -0.5),
        (None, None),
    ],
)
def test_csv_safe(cell: Any, expected: Any) -> None:
    assert export_module._csv_safe(cell) == expected


def test_csv_export_neutralises_formula_cells() -> None:
    record = export_module.AuditLogRecord(
        log_id="l1",
        timestamp=datetime(2026, 1, 1),
        user_id="u",
        action="GET /x",
        resource_type="x",
        resource_id="job_id==cmd|' /C calc'!A0",
        ip_address="203.0.113.1",
        user_agent='=HYPERLINK("http://evil","x")',
        request_method="GET",
        request_path="/x",
        response_status=200,
        duration_ms=1.0,
    )

    async def _collect() -> str:
        chunks = [
            chunk
            async for chunk in export_module._stream_csv([record], export_module._audit_log_to_dict)
        ]
        return b"".join(chunks).decode()

    rows = list(csv.DictReader(io.StringIO(asyncio.run(_collect()))))
    assert rows[0]["user_agent"] == '\'=HYPERLINK("http://evil","x")'
    assert rows[0]["resource_id"] == "job_id==cmd|' /C calc'!A0"  # starts with "j"
    assert rows[0]["ip_address"] == "203.0.113.1"

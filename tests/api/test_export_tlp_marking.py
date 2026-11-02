# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TLP 2.0 marking on the export API and the detection report."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from omni_mercury_engine.api.routes import export as export_module
from omni_mercury_engine.compliance.tlp_handler import TLPColor

if TYPE_CHECKING:
    from collections.abc import Iterator

_USER = "tlp_export_user"


@pytest.fixture
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    from omni_mercury_engine.api.auth import Permission, get_api_key_store

    raw_key, _ = get_api_key_store().create_key(
        name="tlp_export_key",
        user_id=_USER,
        permissions={Permission.READ, Permission.EXPORT},
    )
    return {"X-API-Key": raw_key}


@pytest.fixture
def fresh_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[export_module.DataStore]:
    store = export_module.DataStore()
    monkeypatch.setattr(export_module, "_data_store", store)
    yield store


def _record(score: float, method: str = "statistical", **metadata: Any) -> None:
    asyncio.run(
        export_module.record_detection(
            user_id=_USER,
            method=method,
            data=[[0.0]],
            results={"anomalies": [score > 0.5], "scores": [score], "summary": {}},
            metadata=metadata,
        )
    )


def test_detection_export_is_marked_with_most_restrictive_colour(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.1)
    _record(0.95)
    response = client.get("/api/v1/export/detections", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["X-TLP"] == "TLP:RED"
    assert response.headers["X-TLP-Rank"] == str(TLPColor.RED.rank)
    assert response.headers["X-TLP-Watermark"].startswith("TLP:RED")
    labels = sorted(r["tlp"] for r in response.json())
    assert labels == ["TLP:CLEAR", "TLP:RED"]


def test_pii_metadata_escalates_a_low_score_record(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.05, contains_pii=True)
    response = client.get("/api/v1/export/detections?format=jsonl", headers=auth_headers)
    assert response.headers["X-TLP"] == "TLP:AMBER+STRICT"
    assert json.loads(response.text.splitlines()[0])["tlp"] == "TLP:AMBER+STRICT"


def test_csv_carries_tlp_column_and_optional_envelope(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.4)
    plain = client.get("/api/v1/export/detections?format=csv", headers=auth_headers)
    rows = list(csv.DictReader(io.StringIO(plain.text)))
    assert rows[0]["tlp"] == "TLP:GREEN"
    enveloped = client.get(
        "/api/v1/export/detections?format=csv&tlp_envelope=true", headers=auth_headers
    )
    assert enveloped.text.startswith("# TLP:GREEN")


def test_json_envelope_wraps_records(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.7)
    body = client.get("/api/v1/export/detections?tlp_envelope=true", headers=auth_headers).json()
    assert body["tlp"]["tlp_label"] == "TLP:AMBER"
    assert body["tlp"]["sharing_guidelines"]
    assert len(body["records"]) == 1


def test_audit_logs_have_amber_strict_floor(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    asyncio.run(
        fresh_store.add_audit_log(
            export_module.AuditLogRecord(
                log_id="l1",
                timestamp=datetime.now(),
                user_id=_USER,
                action="detect",
                resource_type="detection",
                resource_id="r1",
                ip_address="203.0.113.7",
                user_agent="pytest",
                request_method="POST",
                request_path="/api/v1/detect",
                response_status=200,
                duration_ms=1.0,
            )
        )
    )
    response = client.get("/api/v1/export/audit-logs", headers=auth_headers)
    assert response.headers["X-TLP"] == "TLP:AMBER+STRICT"
    assert response.json()[0]["tlp"] == "TLP:AMBER+STRICT"


def test_empty_export_is_clear(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    response = client.get("/api/v1/export/detections", headers=auth_headers)
    assert response.headers["X-TLP"] == "TLP:CLEAR"


def test_metrics_export_is_marked(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.9)
    response = client.get("/api/v1/export/metrics", headers=auth_headers)
    assert response.headers["X-TLP"] == "TLP:RED"
    assert response.json()["tlp"] == "TLP:RED"


@pytest.mark.parametrize(
    ("fmt", "marker"),
    [("html", 'class="tlp-banner"'), ("markdown", "> **TLP:"), ("json", '"tlp"')],
)
def test_report_endpoint_renders_tlp_marked_report(
    client: TestClient,
    auth_headers: dict[str, str],
    fresh_store: Any,
    fmt: str,
    marker: str,
) -> None:
    _record(0.75, method="<script>alert(1)</script>")
    response = client.get(f"/api/v1/export/report?format={fmt}", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["X-TLP"] == "TLP:AMBER"
    assert marker in response.text
    if fmt == "html":
        assert "<script>" not in response.text


def test_report_endpoint_pdf(
    client: TestClient, auth_headers: dict[str, str], fresh_store: Any
) -> None:
    _record(0.2)
    response = client.get("/api/v1/export/report?format=pdf", headers=auth_headers)
    if importlib.util.find_spec("reportlab") is None:
        assert response.status_code == 501
    else:
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
        assert response.headers["X-TLP"] == "TLP:CLEAR"

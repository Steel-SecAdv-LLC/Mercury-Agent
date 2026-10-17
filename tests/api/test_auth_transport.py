# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Credentials travel in headers only, and POST bodies are the routes' own models.

Two defects in the auth dependencies' signatures, both found while wiring the
assessment routes:

* ``APIKeyAuth.__call__(api_key: str | None = None)`` is a FastAPI *query*
  parameter, so ``?api_key=<raw key>`` authenticated and put the key in
  access logs, proxy logs and browser history.
* ``JWTAuth.__call__(credentials: HTTPAuthorizationCredentials | None)`` is a
  pydantic model, so FastAPI read it as a request *body* field; every POST
  route depending on it then expected its body embedded under its parameter
  name (``{"request": {...}}``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth import Permission, get_api_key_store


@pytest.fixture
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app, client=("10.250.0.1", 50000))


def test_an_api_key_in_the_query_string_does_not_authenticate(client: TestClient) -> None:
    raw, _ = get_api_key_store().create_key(
        name="query-probe", user_id="query-probe", permissions={Permission.READ, Permission.EXPORT}
    )
    assert client.get("/api/v1/export/summary", params={"api_key": raw}).status_code == 401
    assert client.get("/api/v1/export/summary", headers={"X-API-Key": raw}).status_code == 200


def test_no_route_documents_a_credential_parameter() -> None:
    from omni_mercury_engine.api.server import app

    spec = app.openapi()
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            names = {p["name"] for p in operation.get("parameters", [])}
            assert "api_key" not in names, f"{method.upper()} {path} takes ?api_key="
    schemas = spec.get("components", {}).get("schemas", {})
    assert "HTTPAuthorizationCredentials" not in schemas


@pytest.mark.parametrize("path", ["/api/v1/hazard/visualize", "/api/v1/detect/neurosymbolic"])
def test_post_bodies_are_the_route_models_not_embedded(path: str) -> None:
    from omni_mercury_engine.api.server import app

    schema = app.openapi()["paths"][path]["post"]["requestBody"]["content"]["application/json"]
    assert not schema["schema"]["$ref"].rsplit("/", 1)[-1].startswith("Body_")

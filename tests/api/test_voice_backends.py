# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The backends route reports what the engine can talk to, as facts and without secrets.

``GET /api/v1/voice/backends`` lists every shipped provider adapter with
its key reported as present or absent (never its value), probes the local
Ollama runtime through the egress gate with a cache, and returns the MCP
command and a configuration block per client. These tests pin the
provider list to the code-grounded catalog, the secret boundary, the
probe's two outcomes, the cache, and the route's presence in the OpenAPI
document.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omni_mercury_engine.api import backends as backends_module
from omni_mercury_engine.models.foundation.llm_adapter import IMPLEMENTED_LLM_PROVIDERS
from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

#: The real probe, captured before the autouse fixture replaces it.
_REAL_PROBE = backends_module._probe_ollama


@pytest.fixture(scope="module")
def client() -> TestClient:
    from omni_mercury_engine.api.server import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with an empty Ollama cache and no probe on the wire."""
    monkeypatch.setitem(backends_module._ollama_cache, "status", None)
    monkeypatch.setitem(backends_module._ollama_cache, "at", 0.0)
    monkeypatch.setattr(
        backends_module,
        "_probe_ollama",
        lambda: {
            "host": "localhost",
            "port": 11434,
            "base_url": "http://localhost:11434",
            "model": None,
            "reachable": False,
            "models": [],
            "error": "ConnectionError",
        },
    )


def test_every_shipped_provider_is_reported_once(client: TestClient) -> None:
    body = client.get("/api/v1/voice/backends").json()
    providers = [row["provider"] for row in body["providers"]]
    assert providers == list(PROVIDER_CATALOG)
    assert set(providers) == {p.value for p in IMPLEMENTED_LLM_PROVIDERS}
    for row in body["providers"]:
        assert (
            row["name"] and row["wire_format"] and row["locality"] in {"cloud", "local", "builtin"}
        )
        assert isinstance(row["configured"], bool)


def test_a_key_is_reported_as_present_never_as_its_value(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-test-0123456789-never-leaks"
    monkeypatch.setenv("XAI_API_KEY", secret)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.get("/api/v1/voice/backends")
    rows = {row["provider"]: row for row in response.json()["providers"]}
    assert rows["xai"]["configured"] is True and rows["xai"]["key_env_var"] == "XAI_API_KEY"
    assert rows["openai"]["configured"] is False
    assert secret not in response.text


def test_local_and_builtin_providers_need_no_key(client: TestClient) -> None:
    rows = {
        row["provider"]: row for row in client.get("/api/v1/voice/backends").json()["providers"]
    }
    assert rows["ollama"]["key_env_var"] is None and rows["ollama"]["configured"] is True
    assert rows["template"]["locality"] == "builtin" and rows["template"]["configured"] is True
    assert rows["cursor"]["requires_explicit_base_url"] is True


def test_the_ollama_probe_reports_reachable_with_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        backends_module,
        "_probe_ollama",
        lambda: {
            "host": "127.0.0.1",
            "port": 11434,
            "base_url": "http://127.0.0.1:11434",
            "model": "llama3.2:3b",
            "reachable": True,
            "models": ["llama3.2:3b", "mistral:7b"],
            "error": None,
        },
    )
    body = client.get("/api/v1/voice/backends").json()["ollama"]
    assert body["reachable"] is True and body["models"] == ["llama3.2:3b", "mistral:7b"]


def test_the_ollama_probe_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def probe() -> dict[str, object]:
        calls["n"] += 1
        return {
            "host": "localhost",
            "port": 11434,
            "base_url": "x",
            "model": None,
            "reachable": False,
            "models": [],
            "error": "e",
        }

    monkeypatch.setattr(backends_module, "_probe_ollama", probe)
    backends_module.ollama_status()
    backends_module.ollama_status()
    assert calls["n"] == 1
    backends_module.ollama_status(max_age=0.0)
    assert calls["n"] == 2


def test_the_real_probe_never_raises_without_a_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERCURY_OLLAMA_HOST", "127.0.0.1")
    status = _REAL_PROBE()
    assert status["base_url"].startswith("http://127.0.0.1:")
    assert isinstance(status["reachable"], bool)
    assert status["reachable"] or status["error"]


def test_mcp_clients_carry_a_pasteable_block_naming_the_command(client: TestClient) -> None:
    mcp = client.get("/api/v1/voice/backends").json()["mcp"]
    assert mcp["command"] == "mercury-agent mcp" and mcp["transport"] == "stdio"
    assert mcp["tool_count"] >= 5 and "mercury_detect_anomaly" in mcp["tools"]
    ids = [c["id"] for c in mcp["clients"]]
    assert ids == ["claude", "cursor", "copilot", "generic"]
    for entry in mcp["clients"]:
        text = json.dumps(entry["snippet"])
        assert '"command": "mercury-agent"' in text and '"mcp"' in text
    copilot = next(c for c in mcp["clients"] if c["id"] == "copilot")
    assert copilot["snippet"]["servers"]["mercury"]["type"] == "stdio"


def test_the_served_voice_reports_no_model_by_default(client: TestClient) -> None:
    active = client.get("/api/v1/voice/backends").json()["active"]
    assert active["enabled"] is False and active["provider"] is None


def test_the_route_is_in_the_openapi_document(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/voice/backends" in paths
    assert "get" in paths["/api/v1/voice/backends"]

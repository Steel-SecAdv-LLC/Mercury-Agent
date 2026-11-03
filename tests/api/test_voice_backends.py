# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The backends routes report what the engine can talk to and let an administrator connect it.

``GET /api/v1/voice/backends`` lists every shipped provider adapter with
its key reported as present or absent (never its value) and where it came
from, probes the local Ollama runtime through the egress gate with a cache,
and returns the MCP command and a configuration block per client. The admin
routes store a provider's key sealed, test a connection, select the voice's
model and detach it. These tests pin the provider list to the code-grounded
catalog, the secret boundary (a stored or environment key never appears in
any response), the probe's outcomes, the cache, the admin gate (401 without
a credential, 403 without the permission), the store's refusal without a
stable at-rest key, a selection that attaches a provider and persists, and
the routes' presence in the OpenAPI document.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")

import uuid

from fastapi.testclient import TestClient

from omni_mercury_engine.api import (
    backends as backends_module,
    provider_store as store_module,
    voice as voice_module,
)
from omni_mercury_engine.api.auth import Permission, get_api_key_store
from omni_mercury_engine.models.foundation.llm_adapter import IMPLEMENTED_LLM_PROVIDERS
from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

if TYPE_CHECKING:
    from pathlib import Path

    from omni_mercury_engine.api.provider_store import ProviderStore

STABLE_KEY = "3c" * 32


def _key(*permissions: Permission) -> dict[str, str]:
    raw, _ = get_api_key_store().create_key(
        name="backends-test",
        user_id=f"backends-{uuid.uuid4().hex[:8]}",
        permissions=set(permissions),
    )
    return {"X-API-Key": raw}


@pytest.fixture
def fresh_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ProviderStore:
    """An empty provider store at a temporary path, sealed under a stable key."""
    monkeypatch.setenv(store_module.ENV_VAR, str(tmp_path / "providers.json"))
    monkeypatch.setenv("MERCURY_DATA_ENC_KEY", STABLE_KEY)
    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(voice_module, "_mercury_voice", None)
    return store_module.get_provider_store()


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


# --------------------------------------------------------------------------- #
# Connecting a provider from the console: the admin routes
# --------------------------------------------------------------------------- #


def test_storing_needs_an_administrator(client: TestClient, fresh_store: ProviderStore) -> None:
    anonymous = client.put("/api/v1/voice/backends/groq", json={"model": "llama-3.3-70b"})
    assert anonymous.status_code == 401
    reader = client.put(
        "/api/v1/voice/backends/groq",
        json={"model": "llama-3.3-70b"},
        headers=_key(Permission.READ),
    )
    assert reader.status_code == 403
    assert client.post("/api/v1/voice/backends/groq/test").status_code == 401
    assert (
        client.post("/api/v1/voice/backends/select", json={"provider": "groq"}).status_code == 401
    )
    assert client.delete("/api/v1/voice/backends/select").status_code == 401


def test_an_administrator_stores_a_key_that_never_comes_back(
    client: TestClient, fresh_store: ProviderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    secret = "gsk-test-never-echoed-0123456789"
    response = client.put(
        "/api/v1/voice/backends/groq",
        json={"api_key": secret, "model": "llama-3.3-70b-versatile"},
        headers=_key(Permission.ADMIN),
    )
    assert response.status_code == 200, response.text
    row = response.json()
    assert row["provider"] == "groq" and row["configured"] is True
    assert row["configured_via"] == "store" and row["model_source"] == "store"
    assert secret not in response.text
    report = client.get("/api/v1/voice/backends")
    assert secret not in report.text
    groq = next(r for r in report.json()["providers"] if r["provider"] == "groq")
    assert groq["configured"] is True and groq["model"] == "llama-3.3-70b-versatile"
    assert fresh_store.key("groq") == secret


def test_an_unknown_provider_an_empty_update_and_an_unsafe_endpoint_are_refused(
    client: TestClient, fresh_store: ProviderStore
) -> None:
    admin = _key(Permission.ADMIN)
    assert (
        client.put("/api/v1/voice/backends/nope", json={"model": "x"}, headers=admin).status_code
        == 404
    )
    assert client.put("/api/v1/voice/backends/groq", json={}, headers=admin).status_code == 422
    imds = client.put(
        "/api/v1/voice/backends/openai_compatible",
        json={"base_url": "http://169.254.169.254/latest"},
        headers=admin,
    )
    assert imds.status_code == 422 and "egress gate" in imds.text


def test_without_a_stable_key_the_store_refuses_a_secret_with_the_reason(
    client: TestClient, fresh_store: ProviderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MERCURY_DATA_ENC_KEY", raising=False)
    monkeypatch.delenv("AMA_MASTER_SEED", raising=False)
    monkeypatch.setattr(store_module, "_store", None)
    response = client.put(
        "/api/v1/voice/backends/openai", json={"api_key": "sk-test"}, headers=_key(Permission.ADMIN)
    )
    assert response.status_code == 422 and "MERCURY_DATA_ENC_KEY" in response.text
    report = client.get("/api/v1/voice/backends").json()["store"]
    assert report["can_store_keys"] is False and "MERCURY_DATA_ENC_KEY" in report["reason"]


def test_forgetting_a_provider_removes_its_stored_fields(
    client: TestClient, fresh_store: ProviderStore
) -> None:
    admin = _key(Permission.ADMIN)
    client.put("/api/v1/voice/backends/mistral", json={"api_key": "k", "model": "m"}, headers=admin)
    response = client.delete("/api/v1/voice/backends/mistral", headers=admin)
    assert response.status_code == 200 and response.json()["removed"] is True
    assert fresh_store.get("mistral") is None
    assert client.delete("/api/v1/voice/backends/mistral", headers=admin).json()["removed"] is False


def test_a_connection_test_reports_the_probe_without_the_key(
    client: TestClient, fresh_store: ProviderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_probe(provider: str, **kwargs: object) -> dict[str, object]:
        seen.update(provider=provider, **kwargs)
        return {
            "ok": True,
            "status": 200,
            "models": ["llama-3.3-70b-versatile"],
            "detail": "answered with 1 model",
        }

    monkeypatch.setattr(backends_module, "probe_provider", fake_probe)
    response = client.post(
        "/api/v1/voice/backends/groq/test",
        json={"api_key": "gsk-once-only", "model": "llama-3.3-70b-versatile"},
        headers=_key(Permission.ADMIN),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True and body["models"] == ["llama-3.3-70b-versatile"]
    assert "gsk-once-only" not in response.text
    assert seen["provider"] == "groq" and seen["api_key"] == "gsk-once-only"
    assert fresh_store.get("groq") is None  # a test never stores


def test_the_real_probe_refuses_a_cloud_provider_without_a_key(
    client: TestClient, fresh_store: ProviderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    body = client.post("/api/v1/voice/backends/openai/test", headers=_key(Permission.ADMIN)).json()
    assert body["ok"] is False and "OPENAI_API_KEY" in body["detail"]
    template = client.post(
        "/api/v1/voice/backends/template/test", headers=_key(Permission.ADMIN)
    ).json()
    assert template["ok"] is True


def test_selecting_an_unconfigured_provider_is_refused_and_changes_nothing(
    client: TestClient, fresh_store: ProviderStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MERCURY_OPENAI_MODEL", raising=False)
    response = client.post(
        "/api/v1/voice/backends/select", json={"provider": "openai"}, headers=_key(Permission.ADMIN)
    )
    assert response.status_code == 422 and "needs a model" in response.text
    named = client.post(
        "/api/v1/voice/backends/select",
        json={"provider": "openai", "model": "gpt-4o"},
        headers=_key(Permission.ADMIN),
    )
    assert named.status_code == 422 and "unavailable" in named.text
    assert fresh_store.selection() is None
    assert client.get("/api/v1/voice/backends").json()["active"]["enabled"] is False


def test_selecting_the_template_provider_attaches_it_persists_and_detaches(
    client: TestClient, fresh_store: ProviderStore
) -> None:
    admin = _key(Permission.ADMIN)
    response = client.post(
        "/api/v1/voice/backends/select",
        json={"provider": "template", "model": "builtin"},
        headers=admin,
    )
    assert response.status_code == 200, response.text
    active = response.json()
    assert (
        active["enabled"] is True and active["provider"] == "template" and active["adapter_loaded"]
    )
    assert fresh_store.selection() == {"provider": "template", "model": "builtin"}
    # A fresh voice (a restart) comes up with the selection attached.
    voice_module._mercury_voice = None
    assert client.get("/api/v1/voice/backends").json()["active"]["provider"] == "template"
    detached = client.delete("/api/v1/voice/backends/select", headers=admin)
    assert detached.status_code == 200 and detached.json()["enabled"] is False
    assert fresh_store.selection() is None


def test_restoring_or_forgetting_the_active_provider_reaches_the_running_voice(
    client: TestClient, fresh_store: ProviderStore
) -> None:
    admin = _key(Permission.ADMIN)
    client.put("/api/v1/voice/backends/groq", json={"api_key": "k1", "model": "m1"}, headers=admin)
    selected = client.post(
        "/api/v1/voice/backends/select", json={"provider": "groq"}, headers=admin
    )
    assert selected.status_code == 200, selected.text
    first = voice_module._get_voice()
    assert getattr(first._llm_adapter, "api_key", None) == "k1"
    # A new key for the active provider rebuilds the served voice at once.
    restored = client.put("/api/v1/voice/backends/groq", json={"api_key": "k2"}, headers=admin)
    assert restored.status_code == 200 and restored.json()["reattached"] is True
    second = voice_module._get_voice()
    assert second is not first
    assert getattr(second._llm_adapter, "api_key", None) == "k2"
    # A provider that is not active is stored without touching the voice.
    other = client.put("/api/v1/voice/backends/mistral", json={"model": "x"}, headers=admin)
    assert other.json()["reattached"] is None and voice_module._get_voice() is second
    # Clearing the key of the active provider detaches rather than keeping a dead model.
    cleared = client.put("/api/v1/voice/backends/groq", json={"api_key": ""}, headers=admin)
    assert cleared.json()["reattached"] is False
    assert fresh_store.selection() is None
    assert client.get("/api/v1/voice/backends").json()["active"]["enabled"] is False
    # Forgetting the active provider detaches too and says so.
    client.put("/api/v1/voice/backends/groq", json={"api_key": "k3"}, headers=admin)
    client.post("/api/v1/voice/backends/select", json={"provider": "groq"}, headers=admin)
    forgotten = client.delete("/api/v1/voice/backends/groq", headers=admin)
    assert forgotten.json()["selection_cleared"] is True and forgotten.json()["removed"] is True
    assert fresh_store.selection() is None
    report = client.get("/api/v1/voice/backends").json()
    assert report["active"]["enabled"] is False
    groq = next(r for r in report["providers"] if r["provider"] == "groq")
    assert groq["stored"] == [] and groq["configured"] is False
    assert (
        client.delete("/api/v1/voice/backends/mistral", headers=admin).json()["selection_cleared"]
        is False
    )


def test_a_provider_row_names_its_stored_fields(
    client: TestClient, fresh_store: ProviderStore
) -> None:
    admin = _key(Permission.ADMIN)
    client.put(
        "/api/v1/voice/backends/azure_openai",
        json={"api_key": "k", "base_url": "https://res.openai.azure.com", "model": "dep"},
        headers=admin,
    )
    row = next(
        r
        for r in client.get("/api/v1/voice/backends").json()["providers"]
        if r["provider"] == "azure_openai"
    )
    assert row["stored"] == ["has_key", "base_url", "model"]
    assert row["configured"] is True and row["base_url_source"] == "store"


def test_the_admin_routes_are_in_the_openapi_document(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"put", "delete"} <= set(paths["/api/v1/voice/backends/{provider}"])
    assert "post" in paths["/api/v1/voice/backends/{provider}/test"]
    assert {"post", "delete"} <= set(paths["/api/v1/voice/backends/select"])

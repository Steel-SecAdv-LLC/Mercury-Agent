# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every OpenAI-compatible provider adapter honours the same contract.

Eleven providers joined the shared OpenAI-compatible base on 2026-11: six
cloud vendors with a public endpoint and a key (Mistral, Groq, Together,
Fireworks, OpenRouter, Perplexity), Azure OpenAI (the operator's resource
endpoint, a deployment name, an ``api-key`` header and an API version), a
generic endpoint the operator names, and three local servers reached over
plain http on a loopback host without a key (LM Studio, vLLM, llama.cpp).
These tests pin, for each: the provider is in the enum, the implemented set
and the catalog; no default model ships; a cloud adapter is unavailable
without its key and available with key plus model; a local adapter needs no
key and is loopback-gated; the request goes to ``/chat/completions`` with a
bearer token (Azure: the deployment URL, the api-version and the ``api-key``
header); the endpoint variables are read; and the factory builds the right
class.
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.models.foundation.llm_adapter import (
    IMPLEMENTED_LLM_PROVIDERS,
    LLMConfig,
    LLMProvider,
    _create_non_hf_adapter,
)
from omni_mercury_engine.models.foundation.ollama_adapter import (
    OPENAI_COMPATIBLE_ADAPTERS,
    AzureOpenAIAdapter,
    LMStudioAdapter,
    OpenAICompatibleAdapter,
)
from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

CLOUD = ("mistral", "groq", "together", "fireworks", "openrouter", "perplexity")
LOCAL = ("lmstudio", "vllm", "llamacpp")
NEW = (*CLOUD, "azure_openai", "openai_compatible", *LOCAL)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider in NEW:
        facts = PROVIDER_CATALOG[provider]
        if facts.api_key_env_var:
            monkeypatch.delenv(facts.api_key_env_var, raising=False)
        adapter = OPENAI_COMPATIBLE_ADAPTERS[provider]
        if adapter._MODEL_ENV_VAR:
            monkeypatch.delenv(adapter._MODEL_ENV_VAR, raising=False)
        if adapter._BASE_URL_ENV_VAR:
            monkeypatch.delenv(adapter._BASE_URL_ENV_VAR, raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("MERCURY_OFFLINE_ALLOW_PRIVATE", raising=False)


class _Capture:
    """Stands in for ``SafeHTTPClient.post_json`` and records the call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"url": url, **kwargs})
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


@pytest.mark.parametrize("provider", NEW)
def test_every_new_provider_is_registered_everywhere(provider: str) -> None:
    assert LLMProvider(provider) in IMPLEMENTED_LLM_PROVIDERS
    assert provider in PROVIDER_CATALOG
    assert provider in OPENAI_COMPATIBLE_ADAPTERS
    built = _create_non_hf_adapter(
        LLMConfig(
            provider=LLMProvider(provider),
            api_key="test-not-real",
            model_name="m",
            base_url="https://example.invalid/v1",
        )
    )
    assert type(built) is OPENAI_COMPATIBLE_ADAPTERS[provider]


@pytest.mark.parametrize("provider", NEW)
def test_catalog_facts_match_the_adapter_constants(provider: str) -> None:
    facts = PROVIDER_CATALOG[provider]
    adapter = OPENAI_COMPATIBLE_ADAPTERS[provider]
    assert facts.api_key_env_var == adapter._PROVIDER_ENV_VAR
    assert facts.requires_explicit_base_url == adapter._REQUIRE_EXPLICIT_BASE_URL
    assert (facts.locality == "local") == adapter._LOCAL
    assert facts.reports_token_usage is True


@pytest.mark.parametrize("provider", CLOUD)
def test_a_cloud_provider_needs_its_key_and_a_model_and_ships_no_default(
    provider: str,
) -> None:
    adapter_cls = OPENAI_COMPATIBLE_ADAPTERS[provider]
    enum = LLMProvider(provider)
    no_key = adapter_cls(LLMConfig(provider=enum, model_name="m"))
    assert no_key.is_available() is False
    no_model = adapter_cls(LLMConfig(provider=enum, api_key="test-not-real"))
    assert no_model.model == "" and no_model.is_available() is False
    ready = adapter_cls(LLMConfig(provider=enum, api_key="test-not-real", model_name="m"))
    assert ready.is_available() is True
    assert ready.base_url == adapter_cls._DEFAULT_BASE_URL
    assert ready.base_url is not None and ready.base_url.startswith("https://")


@pytest.mark.parametrize("provider", CLOUD)
def test_a_cloud_request_is_a_bearer_chat_completion_behind_the_full_gate(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine.security import safe_http

    capture = _Capture()
    monkeypatch.setattr(safe_http.SafeHTTPClient, "post_json", staticmethod(capture))
    adapter = OPENAI_COMPATIBLE_ADAPTERS[provider](
        LLMConfig(provider=LLMProvider(provider), api_key="test-not-real", model_name="m")
    )
    assert adapter.generate("hello", system_prompt="be brief") == "ok"
    call = capture.calls[0]
    assert call["url"].endswith("/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer test-not-real"
    assert call["user_configured"] is True and call["allow_http"] is False
    assert "loopback_only" not in call
    assert call["json_body"]["model"] == "m"
    assert [m["role"] for m in call["json_body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize("provider", LOCAL)
def test_a_local_server_needs_no_key_and_stays_on_the_loopback(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine.security import safe_http

    adapter_cls = OPENAI_COMPATIBLE_ADAPTERS[provider]
    enum = LLMProvider(provider)
    assert adapter_cls(LLMConfig(provider=enum)).is_available() is False  # no model
    adapter = adapter_cls(LLMConfig(provider=enum, model_name="local-model"))
    assert adapter.is_available() is True
    assert adapter.base_url is not None and adapter.base_url.startswith("http://localhost:")
    assert "Authorization" not in adapter._auth_headers()
    assert adapter._egress_kwargs() == {"loopback_only": True}
    capture = _Capture()
    monkeypatch.setattr(safe_http.SafeHTTPClient, "post_json", staticmethod(capture))
    assert adapter.generate("hi") == "ok"
    call = capture.calls[0]
    assert call["allow_http"] is True and call["loopback_only"] is True
    with_key = adapter_cls(LLMConfig(provider=enum, model_name="m", api_key="k"))
    assert with_key._auth_headers()["Authorization"] == "Bearer k"


def test_a_local_server_endpoint_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:4321/v1")
    adapter = LMStudioAdapter(LLMConfig(provider=LLMProvider.LMSTUDIO, model_name="m"))
    assert adapter.base_url == "http://127.0.0.1:4321/v1"
    assert adapter._chat_url() == "http://127.0.0.1:4321/v1/chat/completions"


def test_a_private_local_host_needs_the_offline_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LMStudioAdapter(
        LLMConfig(provider=LLMProvider.LMSTUDIO, model_name="m", base_url="http://10.0.0.5:1234/v1")
    )
    assert adapter._egress_kwargs() == {"loopback_only": True}
    monkeypatch.setenv("MERCURY_OFFLINE", "1")
    monkeypatch.setenv("MERCURY_OFFLINE_ALLOW_PRIVATE", "1")
    from omni_mercury_engine.datasets.exceptions import offline_allow_private_active

    if not offline_allow_private_active():
        pytest.skip("the offline private-network opt-in is not read from these variables here")
    assert adapter._egress_kwargs() == {"loopback_only": False, "allow_private": True}


def test_azure_openai_uses_the_deployment_url_the_api_version_and_the_api_key_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omni_mercury_engine.security import safe_http

    missing = AzureOpenAIAdapter(
        LLMConfig(provider=LLMProvider.AZURE_OPENAI, api_key="k", model_name="dep")
    )
    assert missing.is_available() is False  # no endpoint
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    adapter = AzureOpenAIAdapter(
        LLMConfig(provider=LLMProvider.AZURE_OPENAI, api_key="k", model_name="gpt-4o dep")
    )
    assert adapter.is_available() is True
    assert adapter._chat_url() == (
        "https://res.openai.azure.com/openai/deployments/gpt-4o%20dep/chat/completions"
        "?api-version=2025-01-01-preview"
    )
    headers = adapter._auth_headers()
    assert headers["api-key"] == "k" and "Authorization" not in headers
    capture = _Capture()
    monkeypatch.setattr(safe_http.SafeHTTPClient, "post_json", staticmethod(capture))
    assert adapter.generate("hi") == "ok"
    assert capture.calls[0]["headers"]["api-key"] == "k"


def test_the_generic_endpoint_is_never_guessed_and_takes_an_optional_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unnamed = OpenAICompatibleAdapter(
        LLMConfig(provider=LLMProvider.OPENAI_COMPATIBLE, model_name="m")
    )
    assert unnamed.is_available() is False
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://gateway.example.com/v1")
    named = OpenAICompatibleAdapter(
        LLMConfig(provider=LLMProvider.OPENAI_COMPATIBLE, model_name="m")
    )
    assert named.is_available() is True
    assert "Authorization" not in named._auth_headers()
    keyed = OpenAICompatibleAdapter(
        LLMConfig(provider=LLMProvider.OPENAI_COMPATIBLE, model_name="m", api_key="k")
    )
    assert keyed._auth_headers()["Authorization"] == "Bearer k"


@pytest.mark.parametrize("provider", NEW)
def test_every_new_provider_reads_its_model_from_its_own_variable(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_cls = OPENAI_COMPATIBLE_ADAPTERS[provider]
    assert adapter_cls._MODEL_ENV_VAR.startswith("MERCURY_")
    monkeypatch.setenv(adapter_cls._MODEL_ENV_VAR, "env-model")
    adapter = adapter_cls(
        LLMConfig(
            provider=LLMProvider(provider),
            api_key="test-not-real",
            base_url="https://example.invalid/v1",
        )
    )
    assert adapter.model == "env-model"

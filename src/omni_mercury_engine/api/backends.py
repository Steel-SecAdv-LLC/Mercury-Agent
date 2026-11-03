# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the engine can talk to: language-model providers and MCP clients, as facts.

``GET /api/v1/voice/backends`` answers, for the console and for scripts, three
questions a user asks when they want to "link an AI" to Mercury:

* **Which language-model providers does this engine ship adapters for, and
  which are configured here?** The twenty-one providers come from
  :data:`omni_mercury_engine.models.llm_registry.PROVIDER_CATALOG` (the
  code-grounded catalog a drift gate pins to the adapters). For each, the
  route reports the key environment variable the adapter reads and whether
  that variable is set on this server, never its value.
* **Is the local Ollama runtime reachable, and with which models?** The
  route probes ``/api/tags`` on the loopback host the adapter would use,
  through the same egress gate the adapter uses, with a short timeout and a
  30-second cache so a console that polls never hammers it.
* **How does an MCP client (Claude Desktop, Claude Code, Cursor, VS Code
  with Copilot, any other MCP-speaking agent) attach to Mercury?** The route
  returns the ``mercury-agent mcp`` stdio command, the protocol version and
  tool count the server advertises, and the configuration block each
  client takes, ready to paste.

It also says which provider, if any, the served voice generates prose with.
The grounded answers the console receives are built from the detection
record and the code without a model; a provider only matters where an
operator enabled one.

Everything here is read-only and holds no secret: a key is reported as
present or absent, a host as reachable or not.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from omni_mercury_engine.models.llm_registry import PROVIDER_CATALOG

logger = logging.getLogger(__name__)

__all__ = [
    "MCP_CLIENTS",
    "PROVIDER_NAMES",
    "backends_report",
    "ollama_status",
    "probe_provider",
    "provider_rows",
    "resolve_provider_config",
]

#: Display names for the catalog's provider ids.
PROVIDER_NAMES: dict[str, str] = {
    "openai": "OpenAI (ChatGPT models)",
    "anthropic": "Anthropic (Claude)",
    "xai": "xAI (Grok)",
    "gemini": "Google Gemini",
    "cohere": "Cohere",
    "deepseek": "DeepSeek",
    "cursor": "Cursor (OpenAI-compatible endpoint)",
    "huggingface": "Hugging Face Inference API",
    "ollama": "Ollama (local runtime)",
    "template": "Built-in templates (no model)",
    "mistral": "Mistral AI",
    "groq": "Groq",
    "together": "Together AI",
    "fireworks": "Fireworks AI",
    "openrouter": "OpenRouter (one key, many vendors)",
    "perplexity": "Perplexity",
    "azure_openai": "Azure OpenAI",
    "openai_compatible": "Any OpenAI-compatible endpoint",
    "lmstudio": "LM Studio (local server)",
    "vllm": "vLLM (local server)",
    "llamacpp": "llama.cpp server (local)",
}

#: One sentence per provider on what configuring it gives Mercury.
PROVIDER_NOTES: dict[str, str] = {
    "openai": "Cloud chat completions; set the key and name a model with --model.",
    "anthropic": "Cloud Messages API; set the key and name a model with --model.",
    "xai": "Grok through the OpenAI-compatible api.x.ai endpoint.",
    "gemini": "Google's generateContent API.",
    "cohere": "Cohere Chat v2.",
    "deepseek": "DeepSeek through its OpenAI-compatible endpoint.",
    "cursor": "An OpenAI-compatible endpoint you name explicitly (base_url); the adapter never guesses it.",
    "huggingface": "The Inference API text-generation route; usage is unmetered there.",
    "ollama": "Open-weights models on this machine; loopback only unless MERCURY_OFFLINE_ALLOW_PRIVATE opts into a private host.",
    "template": "Deterministic templates; what every answer falls back to when no model is configured.",
    "mistral": "Mistral's own API through its OpenAI-compatible chat endpoint.",
    "groq": "Groq's hosted open-weights models through its OpenAI-compatible endpoint.",
    "together": "Together AI's hosted open-weights models.",
    "fireworks": "Fireworks AI's inference endpoint.",
    "openrouter": "OpenRouter routes one key to many vendors; name the model as vendor/model.",
    "perplexity": "Perplexity's Sonar models.",
    "azure_openai": "Your Azure resource endpoint and a deployment name; the key travels in the api-key header.",
    "openai_compatible": "A vendor or gateway not listed by name: set its base URL; a key is optional.",
    "lmstudio": "LM Studio's local server on localhost:1234; no key; loopback only.",
    "vllm": "A vLLM server on localhost:8000; no key; loopback only.",
    "llamacpp": "The llama.cpp server on localhost:8080; no key; loopback only.",
}

#: The MCP server's stdio command and how each client is told about it. The
#: generic ``mcpServers`` block is the one docs/INTERCONNECT.md documents;
#: the Copilot block is VS Code's ``servers`` form of the same thing.
_MERCURY_SERVER = {"command": "mercury-agent", "args": ["mcp"]}
MCP_CLIENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "claude",
        "name": "Claude Desktop and Claude Code",
        "config_file": "claude_desktop_config.json (Desktop) or .mcp.json / ~/.claude.json (Claude Code)",
        "snippet": {"mcpServers": {"mercury": dict(_MERCURY_SERVER)}},
        "note": "Paste the block into the client's MCP configuration; the client runs the command and lists Mercury's tools.",
    },
    {
        "id": "cursor",
        "name": "Cursor",
        "config_file": ".cursor/mcp.json (project) or ~/.cursor/mcp.json (user)",
        "snippet": {"mcpServers": {"mercury": dict(_MERCURY_SERVER)}},
        "note": "Cursor can also be a model provider for Mercury (the 'cursor' adapter, with an explicit base_url).",
    },
    {
        "id": "copilot",
        "name": "VS Code with GitHub Copilot",
        "config_file": ".vscode/mcp.json",
        "snippet": {"servers": {"mercury": {"type": "stdio", **_MERCURY_SERVER}}},
        "note": "Copilot's agent mode reads the servers block and offers Mercury's tools.",
    },
    {
        "id": "generic",
        "name": "Any other MCP client (stdio)",
        "config_file": "the client's MCP configuration",
        "snippet": {"mcpServers": {"mercury": dict(_MERCURY_SERVER)}},
        "note": (
            "Mercury serves MCP over stdio only; a hosted assistant that attaches to remote MCP servers "
            "(ChatGPT, Grok) needs an HTTP transport Mercury does not ship. Those vendors reach Mercury "
            "the other way round, as model providers."
        ),
    },
)

_OLLAMA_CACHE_SECONDS = 30.0
_ollama_cache: dict[str, Any] = {"at": 0.0, "status": None}
_ollama_lock = threading.Lock()


#: Providers whose adapter reads a key but works without one (local servers,
#: a self-hosted gateway): "configured" means reachable in principle, not keyed.
_KEY_OPTIONAL = frozenset(
    {"ollama", "template", "lmstudio", "vllm", "llamacpp", "openai_compatible"}
)


def _adapter_facts(provider: str) -> dict[str, Any]:
    """The endpoint and model variables the provider's adapter reads, from its class."""
    facts: dict[str, Any] = {
        "default_base_url": None,
        "base_url_env_var": None,
        "model_env_var": None,
    }
    from omni_mercury_engine.models.foundation.ollama_adapter import OPENAI_COMPATIBLE_ADAPTERS

    cls = OPENAI_COMPATIBLE_ADAPTERS.get(provider)
    if cls is not None:
        facts["default_base_url"] = cls._DEFAULT_BASE_URL
        facts["base_url_env_var"] = cls._BASE_URL_ENV_VAR or None
        facts["model_env_var"] = cls._MODEL_ENV_VAR or None
        return facts
    fixed = {
        "openai": ("https://api.openai.com/v1", None, "MERCURY_OPENAI_MODEL"),
        "anthropic": ("https://api.anthropic.com", None, "MERCURY_ANTHROPIC_MODEL"),
        "huggingface": ("https://api-inference.huggingface.co", None, "MERCURY_HUGGINGFACE_MODEL"),
        "cohere": ("https://api.cohere.com", None, "MERCURY_COHERE_MODEL"),
        "gemini": (
            "https://generativelanguage.googleapis.com/v1beta",
            None,
            "MERCURY_GEMINI_MODEL",
        ),
        "ollama": ("http://localhost:11434", "MERCURY_OLLAMA_HOST", "MERCURY_OLLAMA_MODEL"),
        "template": (None, None, None),
    }
    base, base_var, model_var = fixed.get(provider, (None, None, None))
    facts.update(
        {"default_base_url": base, "base_url_env_var": base_var, "model_env_var": model_var}
    )
    return facts


def _store() -> Any:
    from omni_mercury_engine.api.provider_store import get_provider_store

    return get_provider_store()


def resolve_provider_config(provider: str) -> dict[str, Any]:
    """The key, endpoint and model the engine would use for ``provider``, and where each came from.

    The provider store (set from the console) wins over the environment for a
    field it holds; the environment is the adapter's own fallback. The key
    itself is returned for the caller that builds an ``LLMConfig``; the report
    only ever publishes its presence.
    """
    facts = PROVIDER_CATALOG[provider]
    adapter = _adapter_facts(provider)
    stored: dict[str, Any] = {"api_key": None, "base_url": None, "model": None}
    stored_fields: dict[str, Any] = {}
    try:
        stored = _store().config_for(provider)
        stored_fields = dict(_store().get(provider) or {})
    except Exception as exc:
        logger.warning("provider store unavailable for %s: %s", provider, exc)
    env_key = os.environ.get(facts.api_key_env_var) if facts.api_key_env_var else None
    env_base = os.environ.get(adapter["base_url_env_var"]) if adapter["base_url_env_var"] else None
    env_model = os.environ.get(adapter["model_env_var"]) if adapter["model_env_var"] else None
    if provider == "ollama" and env_base and "://" not in env_base:
        env_base = f"http://{env_base}:11434"
    return {
        "api_key": stored["api_key"] or env_key or None,
        "key_source": "store" if stored["api_key"] else ("env" if env_key else None),
        "base_url": stored["base_url"] or env_base or adapter["default_base_url"],
        "base_url_source": (
            "store"
            if stored["base_url"]
            else ("env" if env_base else ("default" if adapter["default_base_url"] else None))
        ),
        "model": stored["model"] or env_model or None,
        "model_source": "store" if stored["model"] else ("env" if env_model else None),
        "stored": stored_fields,
    }


def provider_rows() -> list[dict[str, Any]]:
    """One row per shipped provider adapter, with its key reported as present or absent."""
    rows: list[dict[str, Any]] = []
    for provider, facts in PROVIDER_CATALOG.items():
        resolved = resolve_provider_config(provider)
        adapter = _adapter_facts(provider)
        has_key = resolved["api_key"] is not None
        needs_endpoint = facts.requires_explicit_base_url and not resolved["base_url"]
        rows.append(
            {
                "provider": provider,
                "name": PROVIDER_NAMES.get(provider, provider),
                "wire_format": facts.wire_format,
                "locality": facts.locality,
                "key_env_var": facts.api_key_env_var,
                "key_optional": provider in _KEY_OPTIONAL,
                "configured": (has_key or provider in _KEY_OPTIONAL) and not needs_endpoint,
                "configured_via": resolved["key_source"],
                "base_url": resolved["base_url"],
                "base_url_source": resolved["base_url_source"],
                "base_url_env_var": adapter["base_url_env_var"],
                "model": resolved["model"],
                "model_source": resolved["model_source"],
                "model_env_var": adapter["model_env_var"],
                "requires_explicit_base_url": facts.requires_explicit_base_url,
                "reports_token_usage": facts.reports_token_usage,
                "stored": [
                    field
                    for field in ("has_key", "base_url", "model")
                    if resolved["stored"].get(field)
                ],
                "note": PROVIDER_NOTES.get(provider, ""),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Connection probes: one cheap, read-only call per provider
# --------------------------------------------------------------------------- #

_PROBE_TIMEOUT = 8.0


def _openai_models(data: Any) -> list[str]:
    items = data.get("data", []) if isinstance(data, dict) else []
    return sorted(str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id"))


def _named_models(data: Any, key: str = "models", field: str = "name") -> list[str]:
    items = data.get(key, []) if isinstance(data, dict) else []
    return sorted(str(m.get(field)) for m in items if isinstance(m, dict) and m.get(field))


def _failure(detail: str, status: int | None = None) -> dict[str, Any]:
    return {"ok": False, "status": status, "models": [], "detail": detail}


def probe_provider(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Make one read-only call to ``provider`` and report whether it answered.

    Lists the provider's models where the API has a listing (OpenAI-style
    ``/models``, Anthropic ``/v1/models``, Gemini ``/models``, Cohere
    ``/v1/models``, Ollama ``/api/tags``, Azure ``/openai/models``); for
    Perplexity, which lists none, sends a one-token chat completion to the
    named model. Hugging Face is checked with ``whoami``. The template
    provider needs no call. Every call goes through the egress gate: cloud
    endpoints as operator-configured public hosts, local servers loopback-only
    over plain http. The key is used for the call and never returned.
    """
    from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

    if provider not in PROVIDER_CATALOG:
        return _failure(f"unknown provider {provider!r}")
    resolved = resolve_provider_config(provider)
    key = api_key if api_key else resolved["api_key"]
    base = (base_url or resolved["base_url"] or "").rstrip("/")
    chosen = model or resolved["model"]
    facts = PROVIDER_CATALOG[provider]
    local = facts.locality == "local"
    if provider == "template":
        return {
            "ok": True,
            "status": 200,
            "models": [],
            "detail": "built-in templates need no connection",
        }
    if not base:
        return _failure("no endpoint: set base_url")
    if not key and provider not in _KEY_OPTIONAL:
        return _failure(f"no key: set {facts.api_key_env_var} or store one")
    egress: dict[str, Any] = {
        "timeout": _PROBE_TIMEOUT,
        "user_configured": True,
        "allow_http": local,
    }
    if local:
        egress["loopback_only"] = True
    headers: dict[str, str] = {"Accept": "application/json"}
    try:
        if provider == "ollama":
            data = SafeHTTPClient.get_json(f"{base}/api/tags", headers=headers, **egress)
            models = _named_models(data)
        elif provider == "anthropic":
            headers.update({"x-api-key": str(key), "anthropic-version": "2023-06-01"})
            data = SafeHTTPClient.get_json(f"{base}/v1/models", headers=headers, **egress)
            models = _openai_models(data)
        elif provider == "gemini":
            headers["x-goog-api-key"] = str(key)
            data = SafeHTTPClient.get_json(f"{base}/models", headers=headers, **egress)
            models = [m.split("/")[-1] for m in _named_models(data)]
        elif provider == "cohere":
            headers["Authorization"] = f"Bearer {key}"
            data = SafeHTTPClient.get_json(f"{base}/v1/models", headers=headers, **egress)
            models = _named_models(data)
        elif provider == "huggingface":
            headers["Authorization"] = f"Bearer {key}"
            data = SafeHTTPClient.get_json(
                "https://huggingface.co/api/whoami-v2", headers=headers, **egress
            )
            who = data.get("name") if isinstance(data, dict) else None
            return {
                "ok": True,
                "status": 200,
                "models": [],
                "detail": f"token accepted for {who or 'an account'}",
            }
        elif provider == "azure_openai":
            from omni_mercury_engine.models.foundation.ollama_adapter import AzureOpenAIAdapter

            version = os.environ.get("AZURE_OPENAI_API_VERSION", "") or (
                AzureOpenAIAdapter._DEFAULT_API_VERSION
            )
            headers["api-key"] = str(key)
            data = SafeHTTPClient.get_json(
                f"{base}/openai/models?api-version={version}", headers=headers, **egress
            )
            models = _openai_models(data)
        elif provider == "perplexity":
            if not chosen:
                return _failure("Perplexity lists no models: name one to test")
            headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            data = SafeHTTPClient.post_json(
                f"{base}/chat/completions",
                json_body={
                    "model": chosen,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                headers=headers,
                **egress,
            )
            models = [chosen] if isinstance(data, dict) and data.get("choices") else []
        else:
            if key:
                headers["Authorization"] = f"Bearer {key}"
            data = SafeHTTPClient.get_json(f"{base}/models", headers=headers, **egress)
            models = _openai_models(data)
    except UnsafeURLError as exc:
        return _failure(f"endpoint refused by the egress gate: {exc}")
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = {
            401: "the key was rejected",
            403: "the key is not allowed to list models",
            404: "no such endpoint",
        }.get(status or 0, type(exc).__name__)
        return _failure(reason, status)
    detail = f"answered with {len(models)} model{'s' if len(models) != 1 else ''}"
    if chosen and models and chosen not in models:
        detail += f"; the chosen model {chosen!r} is not among them"
    return {"ok": True, "status": 200, "models": models[:50], "detail": detail}


def _probe_ollama() -> dict[str, Any]:
    """Ask the local Ollama runtime for its models through the egress gate; never raise."""
    from omni_mercury_engine.models.foundation.ollama_adapter import OllamaConfig
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    config = OllamaConfig()
    host = os.environ.get("MERCURY_OLLAMA_HOST") or config.host
    model = os.environ.get("MERCURY_OLLAMA_MODEL") or config.model
    base_url = f"http://{host}:{config.port}"
    status: dict[str, Any] = {
        "host": host,
        "port": config.port,
        "base_url": base_url,
        "model": model or None,
        "reachable": False,
        "models": [],
        "error": None,
    }
    try:
        data = SafeHTTPClient.get_json(
            f"{base_url}/api/tags",
            headers={"Accept": "application/json"},
            timeout=min(config.connect_timeout, 2.0),
            allow_http=True,
            user_configured=True,
            loopback_only=True,
        )
        names = [str(m.get("name")) for m in data.get("models", []) if isinstance(m, dict)]
        status["reachable"] = True
        status["models"] = sorted(n for n in names if n and n != "None")[:50]
    except Exception as exc:
        status["error"] = type(exc).__name__
        logger.debug("Ollama probe failed: %s", exc)
    return status


def ollama_status(*, max_age: float = _OLLAMA_CACHE_SECONDS) -> dict[str, Any]:
    """The cached Ollama probe, refreshed when older than ``max_age`` seconds."""
    with _ollama_lock:
        now = time.monotonic()
        cached = _ollama_cache["status"]
        if cached is None or now - _ollama_cache["at"] > max_age:
            cached = _probe_ollama()
            _ollama_cache["status"] = cached
            _ollama_cache["at"] = now
        return dict(cached)


def _mcp_facts() -> dict[str, Any]:
    """The MCP server's command, protocol and tool count, read from the server itself."""
    try:
        from omni_mercury_engine.mcp_server import MercuryMCPServer

        server = MercuryMCPServer()
        tools = sorted(getattr(server, "_tools", {}) or {})
        protocol = getattr(server, "PROTOCOL_VERSION", None) or "2024-11-05"
    except Exception as exc:
        logger.debug("MCP server unavailable for the backends report: %s", exc)
        tools, protocol = [], "2024-11-05"
    return {
        "command": "mercury-agent mcp",
        "transport": "stdio",
        "protocol": str(protocol),
        "tool_count": len(tools),
        "tools": tools,
        "clients": [dict(c) for c in MCP_CLIENTS],
    }


def backends_report(voice: Any) -> dict[str, Any]:
    """The whole report: the voice's active model, the providers, Ollama, the store and MCP."""
    enabled = bool(getattr(voice, "enable_llm", False))
    provider = getattr(voice, "_llm_provider", None)
    adapter = getattr(voice, "_llm_adapter", None)
    selection = None
    can_store, why_not = True, ""
    try:
        selection = _store().selection()
        can_store, why_not = _store().can_store_keys()
    except Exception as exc:
        logger.warning("provider store unavailable: %s", exc)
    return {
        "active": {
            "enabled": enabled,
            "provider": str(provider) if enabled and provider else None,
            "model": getattr(voice, "_llm_model_name", None) if enabled else None,
            "adapter_loaded": adapter is not None,
            "selection": selection,
            "note": (
                "Console answers are built from the detection record and the code; a selected "
                "provider composes the free-text search answers from the retrieved records and "
                "nothing else."
            ),
        },
        "store": {"can_store_keys": can_store, "reason": why_not},
        "providers": provider_rows(),
        "ollama": ollama_status(),
        "mcp": _mcp_facts(),
    }

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the engine can talk to: language-model providers and MCP clients, as facts.

``GET /api/v1/voice/backends`` answers, for the console and for scripts, three
questions a user asks when they want to "link an AI" to Mercury:

* **Which language-model providers does this engine ship adapters for, and
  which are configured here?** The ten providers come from
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
    "provider_rows",
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


def provider_rows() -> list[dict[str, Any]]:
    """One row per shipped provider adapter, with its key reported as present or absent."""
    rows: list[dict[str, Any]] = []
    for provider, facts in PROVIDER_CATALOG.items():
        env_var = facts.api_key_env_var
        rows.append(
            {
                "provider": provider,
                "name": PROVIDER_NAMES.get(provider, provider),
                "wire_format": facts.wire_format,
                "locality": facts.locality,
                "key_env_var": env_var,
                "configured": (
                    bool(os.environ.get(env_var)) if env_var else facts.locality != "cloud"
                ),
                "requires_explicit_base_url": facts.requires_explicit_base_url,
                "reports_token_usage": facts.reports_token_usage,
                "note": PROVIDER_NOTES.get(provider, ""),
            }
        )
    return rows


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
    """The whole report: the voice's active model, the providers, Ollama and MCP."""
    enabled = bool(getattr(voice, "enable_llm", False))
    provider = getattr(voice, "_llm_provider", None)
    adapter = getattr(voice, "_llm_adapter", None)
    return {
        "active": {
            "enabled": enabled,
            "provider": str(provider) if enabled and provider else None,
            "model": getattr(voice, "_llm_model_name", None) if enabled else None,
            "adapter_loaded": adapter is not None,
            "note": (
                "Console answers are built from the detection record and the code; a provider only "
                "generates prose where an operator enabled one (create_mercury_voice(enable_llm=True))."
            ),
        },
        "providers": provider_rows(),
        "ollama": ollama_status(),
        "mcp": _mcp_facts(),
    }

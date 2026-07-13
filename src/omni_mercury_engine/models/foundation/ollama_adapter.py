# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ollama LLM Adapter for Local Inference.

Provides offline-first LLM capability using Ollama for:
- Air-gapped deployments
- Privacy-sensitive environments
- Low-latency local inference
- Model flexibility (Llama, Mistral, Phi, etc.)

Architecture:
    Ollama (Primary) → Cloud (Optional) → Template (Fallback)

This ensures Mercury Agent maintains conversational capability
even when completely disconnected from external services.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.parse
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import requests

from omni_mercury_engine.models.foundation.llm_adapter import (
    BaseLLMAdapter,
    LLMConfig,
    LLMProvider,
)
from omni_mercury_engine.security.safe_http import SafeHTTPClient, UnsafeURLError

if TYPE_CHECKING:
    from omni_mercury_engine.models.foundation.llm_usage import UsageLedger

logger = logging.getLogger(__name__)


def _as_count(value: Any) -> int | None:
    """Coerce a provider-reported token count to a non-negative int.

    Providers occasionally serialize integer counts as floats (e.g. Cohere
    reports its token counts as ``42.0``); such integer-valued floats are
    accepted exactly. Anything that is not a non-negative, integer-valued
    number — a bool, a non-numeric type, a negative, or a genuinely fractional
    float — is treated as unreported (``None``) rather than guessed at or
    silently truncated, so a count is only booked when it is provider-truthful.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def _usage_from_openai(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract usage from an OpenAI-style Chat Completions response.

    Wire format: ``{"usage": {"prompt_tokens", "completion_tokens",
    "total_tokens"}}`` — shared by OpenAI, xAI, DeepSeek, and Cursor.
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    return (
        _as_count(usage.get("prompt_tokens")),
        _as_count(usage.get("completion_tokens")),
        _as_count(usage.get("total_tokens")),
    )


def _usage_from_anthropic(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract usage from an Anthropic Messages response.

    Wire format: ``{"usage": {"input_tokens", "output_tokens"}}`` (no total).
    """
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    return (
        _as_count(usage.get("input_tokens")),
        _as_count(usage.get("output_tokens")),
        None,
    )


def _usage_from_gemini(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract usage from a Gemini ``generateContent`` response.

    Wire format: ``{"usageMetadata": {"promptTokenCount",
    "candidatesTokenCount", "totalTokenCount"}}``.
    """
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None, None, None
    return (
        _as_count(usage.get("promptTokenCount")),
        _as_count(usage.get("candidatesTokenCount")),
        _as_count(usage.get("totalTokenCount")),
    )


def _usage_from_cohere(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract usage from a Cohere Chat v2 response.

    Wire format: ``{"usage": {"tokens": {"input_tokens", "output_tokens"}}}``
    (no total; ``billed_units`` is a billing view, not the raw token count).
    """
    usage = data.get("usage")
    tokens = usage.get("tokens") if isinstance(usage, dict) else None
    if not isinstance(tokens, dict):
        return None, None, None
    return (
        _as_count(tokens.get("input_tokens")),
        _as_count(tokens.get("output_tokens")),
        None,
    )


def _usage_from_ollama(data: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Extract usage from an Ollama ``/api/generate`` or ``/api/chat`` response.

    Wire format: top-level ``prompt_eval_count`` / ``eval_count`` (no total).
    """
    return (
        _as_count(data.get("prompt_eval_count")),
        _as_count(data.get("eval_count")),
        None,
    )


class OllamaModel(StrEnum):
    """Supported Ollama models with recommended use cases."""

    # Llama family - Best for general reasoning
    LLAMA_3_2_1B = "llama3.2:1b"  # Ultra-lightweight, fast
    LLAMA_3_2_3B = "llama3.2:3b"  # Balanced speed/quality
    LLAMA_3_1_8B = "llama3.1:8b"  # Strong reasoning
    LLAMA_3_1_70B = "llama3.1:70b"  # Maximum capability

    # Mistral family - Excellent instruction following
    MISTRAL_7B = "mistral:7b"
    MISTRAL_NEMO = "mistral-nemo:12b"
    MIXTRAL_8X7B = "mixtral:8x7b"

    # Phi family - Compact and efficient
    PHI_3_MINI = "phi3:mini"  # 3.8B, fast
    PHI_3_MEDIUM = "phi3:medium"  # 14B, accurate

    # Qwen family - Strong multilingual
    QWEN2_5_7B = "qwen2.5:7b"
    QWEN2_5_14B = "qwen2.5:14b"

    # Code-specialized
    CODELLAMA_7B = "codellama:7b"
    DEEPSEEK_CODER = "deepseek-coder:6.7b"

    # Small/Embedded
    TINYLLAMA = "tinyllama:1.1b"
    GEMMA2_2B = "gemma2:2b"


@dataclass
class OllamaConfig:
    """Configuration for Ollama adapter."""

    host: str = "localhost"
    port: int = 11434
    model: str = "llama3.2:3b"  # Default: balanced speed/quality
    timeout: float = 60.0
    temperature: float = 0.1  # Low temp for consistent anomaly detection
    num_ctx: int = 4096  # Context window
    num_predict: int = 512  # Max tokens to generate
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

    # Connection settings
    connect_timeout: float = 5.0  # Fast fail for availability check
    retry_attempts: int = 2

    @property
    def base_url(self) -> str:
        """Get the Ollama API base URL."""
        return f"http://{self.host}:{self.port}"


@dataclass
class ModelProfile:
    """Profile for a specific model's capabilities."""

    name: str
    provider: LLMProvider
    size_category: str  # tiny, small, medium, large, xl
    reasoning_strength: float  # 0-1
    speed_rating: float  # 0-1 (higher = faster)
    offline_capable: bool
    context_length: int
    recommended_domains: list[str] = field(default_factory=list)


# Model capability profiles
MODEL_PROFILES: dict[str, ModelProfile] = {
    "llama3.2:1b": ModelProfile(
        name="Llama 3.2 1B",
        provider=LLMProvider.OLLAMA,
        size_category="tiny",
        reasoning_strength=0.5,
        speed_rating=0.95,
        offline_capable=True,
        context_length=4096,
        recommended_domains=["simple_alerts", "status_queries"],
    ),
    "llama3.2:3b": ModelProfile(
        name="Llama 3.2 3B",
        provider=LLMProvider.OLLAMA,
        size_category="small",
        reasoning_strength=0.7,
        speed_rating=0.85,
        offline_capable=True,
        context_length=4096,
        recommended_domains=["anomaly_detection", "log_analysis", "general"],
    ),
    "llama3.1:8b": ModelProfile(
        name="Llama 3.1 8B",
        provider=LLMProvider.OLLAMA,
        size_category="medium",
        reasoning_strength=0.85,
        speed_rating=0.7,
        offline_capable=True,
        context_length=8192,
        recommended_domains=["complex_reasoning", "medical", "security"],
    ),
    "mistral:7b": ModelProfile(
        name="Mistral 7B",
        provider=LLMProvider.OLLAMA,
        size_category="medium",
        reasoning_strength=0.8,
        speed_rating=0.75,
        offline_capable=True,
        context_length=8192,
        recommended_domains=["instruction_following", "analysis"],
    ),
    "phi3:mini": ModelProfile(
        name="Phi-3 Mini",
        provider=LLMProvider.OLLAMA,
        size_category="small",
        reasoning_strength=0.75,
        speed_rating=0.85,
        offline_capable=True,
        context_length=4096,
        recommended_domains=["code", "math", "reasoning"],
    ),
}


class OllamaLLMAdapter(BaseLLMAdapter):
    """Ollama LLM adapter for local model inference.

    Provides offline-first LLM capability with support for multiple open-source models (Llama,
    Mistral, Phi, etc.)
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        ollama_config: OllamaConfig | None = None,
    ):
        """Initialize Ollama adapter.

        Args:
            config: Base LLM configuration
            ollama_config: Ollama-specific configuration
        """
        base_config = config or LLMConfig(provider=LLMProvider.OLLAMA)
        super().__init__(base_config)

        self.ollama_config = ollama_config or OllamaConfig()

        # Tier-0 canonical served-model endpoint (docs/INSTALLATION.md): a URL
        # like ``http://127.0.0.1:11434`` parsed into host+port. Loopback is still
        # enforced downstream by SafeHTTPClient(loopback_only=True), so this can
        # only point at a local server, never pivot to a remote host.
        env_endpoint = os.environ.get("MERCURY_MODEL_ENDPOINT", "").strip()
        if env_endpoint:
            parsed = urllib.parse.urlparse(
                env_endpoint if "://" in env_endpoint else f"http://{env_endpoint}"
            )
            try:
                # ``.port`` lazily validates and raises ValueError on a
                # non-numeric / out-of-range port; treat a malformed endpoint as
                # a no-op (log + keep defaults) rather than crashing construction
                # and taking the whole fallback chain down with it.
                host, port = parsed.hostname, parsed.port
            except ValueError:
                logger.warning(
                    "MERCURY_MODEL_ENDPOINT=%r is malformed (bad port); ignoring it",
                    env_endpoint,
                )
                host, port = None, None
            if host:
                self.ollama_config.host = host
            if port is not None:  # port 0 is valid; only skip when truly absent
                self.ollama_config.port = port

        # Override model from environment if set
        env_model = os.environ.get("MERCURY_OLLAMA_MODEL")
        if env_model:
            self.ollama_config.model = env_model

        # The more specific MERCURY_OLLAMA_HOST wins over the endpoint host, for
        # backward compatibility with existing deployments.
        env_host = os.environ.get("MERCURY_OLLAMA_HOST")
        if env_host:
            self.ollama_config.host = env_host

        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Ollama server is running and accessible."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.ollama_config.connect_timeout)
            result = sock.connect_ex((self.ollama_config.host, self.ollama_config.port))
            sock.close()

            if result == 0:
                self._is_available = self._verify_model_available()
            else:
                logger.info(
                    f"Ollama server not available at "
                    f"{self.ollama_config.host}:{self.ollama_config.port}"
                )
                self._is_available = False

        except OSError as e:
            logger.debug(f"Ollama availability check failed: {e}")
            self._is_available = False

    def _verify_model_available(self) -> bool:
        """Verify the configured model is available in Ollama.

        Returns ``False`` (and logs) when the configured model is not
        installed in the local Ollama server. We do NOT silently swap
        in the first available model -- that would route prompts
        through a completely different model than the operator
        configured, defeating the deliberate offline-deployment
        guarantee. The fallback chain handles model-unavailability
        explicitly (Ollama -> cloud -> template); this method's job
        is only to report the truth so that chain runs.
        """
        try:
            # Ollama runs on the local box; loopback_only enforces
            # that fact (an operator who points Ollama at a remote
            # host will hit the gate at config-time, not after a
            # silent SSRF pivot).
            data = SafeHTTPClient.get_json(
                f"{self.ollama_config.base_url}/api/tags",
                headers={"Accept": "application/json"},
                timeout=self.ollama_config.connect_timeout,
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )
            available_models = [m["name"] for m in data.get("models", [])]

            model_base = self.ollama_config.model.split(":")[0]

            # Check for exact match or base name match
            model_available = any(
                self.ollama_config.model in m or model_base in m for m in available_models
            )

            if not model_available:
                logger.warning(
                    "Configured Ollama model %r is not installed on the "
                    "server. Available: %s. Refusing to silently switch -- "
                    "marking adapter unavailable so the fallback chain can "
                    "route through cloud / template adapters explicitly. "
                    "Install the model with `ollama pull %s` or update the "
                    "configured model name.",
                    self.ollama_config.model,
                    available_models,
                    self.ollama_config.model,
                )
                return False

            return model_available

        except UnsafeURLError:
            # The configured Ollama URL was refused by the SafeHTTPClient
            # gate (e.g. operator pointed OLLAMA_HOST at a non-loopback
            # or RFC1918/IMDS address). This is an operator-actionable
            # config error, not a transient probe failure; let it
            # propagate so startup fails loudly instead of pretending
            # Ollama is available.
            raise
        except Exception as e:
            logger.debug(f"Model verification failed: {e}")
            # Network probe failed (Ollama server unreachable, JSON
            # malformed, etc.). Mark unavailable so the fallback chain
            # routes elsewhere -- silently claiming availability would
            # surface as a failed generate() call later, after which
            # the chain cannot recover the request.
            return False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        if not self._is_available:
            return self._unavailable_response()

        try:
            payload: dict[str, Any] = {
                "model": self.ollama_config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.ollama_config.temperature,
                    "num_ctx": self.ollama_config.num_ctx,
                    "num_predict": self.ollama_config.num_predict,
                    "top_p": self.ollama_config.top_p,
                    "top_k": self.ollama_config.top_k,
                    "repeat_penalty": self.ollama_config.repeat_penalty,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            result = SafeHTTPClient.post_json(
                f"{self.ollama_config.base_url}/api/generate",
                json_body=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.ollama_config.timeout,
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            content = str(result.get("response", ""))
            self._record_usage(self.ollama_config.model, *_usage_from_ollama(result))
            return content

        except UnsafeURLError:
            # SSRF / config refusal. Surface so the operator sees the
            # real misconfiguration instead of a fake "unavailable"
            # JSON stub.
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._unavailable_response()

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """Generate using chat API for multi-turn conversations.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        if not self._is_available:
            return self._unavailable_response()

        try:
            chat_messages: list[dict[str, str]] = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.extend(messages)

            payload: dict[str, Any] = {
                "model": self.ollama_config.model,
                "messages": chat_messages,
                "stream": False,
                "options": {
                    "temperature": self.ollama_config.temperature,
                    "num_ctx": self.ollama_config.num_ctx,
                    "num_predict": self.ollama_config.num_predict,
                },
            }

            result = SafeHTTPClient.post_json(
                f"{self.ollama_config.base_url}/api/chat",
                json_body=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.ollama_config.timeout,
                allow_http=True,
                user_configured=True,
                loopback_only=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            content = str(result.get("message", {}).get("content", ""))
            self._record_usage(self.ollama_config.model, *_usage_from_ollama(result))
            return content

        except UnsafeURLError:
            # SSRF / config refusal. Surface so the operator sees the
            # real misconfiguration instead of a fake "unavailable"
            # JSON stub.
            raise
        except Exception as e:
            logger.error(f"Ollama chat generation failed: {e}")
            return self._unavailable_response()

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._is_available

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        profile = MODEL_PROFILES.get(self.ollama_config.model)
        if profile:
            return {
                "name": profile.name,
                "provider": profile.provider.value,
                "size_category": profile.size_category,
                "reasoning_strength": profile.reasoning_strength,
                "speed_rating": profile.speed_rating,
                "offline_capable": profile.offline_capable,
                "context_length": profile.context_length,
            }
        return {
            "name": self.ollama_config.model,
            "provider": "ollama",
            "offline_capable": True,
        }

    def _unavailable_response(self) -> str:
        """Return response when Ollama is unavailable."""
        return json.dumps(
            {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "confidence": 0.0,
                "category": "unavailable",
                "explanation": "Ollama LLM not available - using fallback",
            }
        )


class TemplateLLMAdapter(BaseLLMAdapter):
    """Template-based fallback adapter for offline operation.

    Provides intelligent template responses when no LLM is available. Uses pattern matching and
    rule-based responses to maintain basic conversational capability.
    """

    def __init__(self, config: LLMConfig | None = None):
        """Initialize template adapter."""
        base_config = config or LLMConfig(provider=LLMProvider.TEMPLATE)
        super().__init__(base_config)
        self._is_available = True

        # Response templates for common scenarios
        self._templates = self._build_templates()

    def _build_templates(self) -> dict[str, Callable[[str], str]]:
        """Build template response functions."""
        return {
            "anomaly_detection": self._anomaly_template,
            "status_query": self._status_template,
            "greeting": self._greeting_template,
            "help": self._help_template,
            "unknown": self._unknown_template,
        }

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate template-based response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (used for context)

        Returns:
            Template-based response
        """
        prompt_lower = prompt.lower()

        # Detect prompt type
        if "anomal" in prompt_lower or "detect" in prompt_lower:
            return self._anomaly_template(prompt)
        elif "status" in prompt_lower or "health" in prompt_lower:
            return self._status_template(prompt)
        elif any(g in prompt_lower for g in ["hello", "hi", "hey", "greet"]):
            return self._greeting_template(prompt)
        elif "help" in prompt_lower:
            return self._help_template(prompt)
        else:
            return self._unknown_template(prompt)

    def _anomaly_template(self, prompt: str) -> str:
        """Template for anomaly detection responses."""
        # Try to extract any numeric data for basic analysis
        has_high_values = any(
            keyword in prompt.lower()
            for keyword in ["spike", "high", "elevated", "unusual", "error", "critical"]
        )

        score = 0.75 if has_high_values else 0.25
        is_anomaly = has_high_values

        return json.dumps(
            {
                "is_anomaly": is_anomaly,
                "anomaly_score": score,
                "confidence": 0.6,
                "category": "template_analysis",
                "explanation": (
                    "Template-based analysis detected potential anomaly indicators. "
                    "Full LLM analysis unavailable - using pattern matching fallback."
                    if is_anomaly
                    else "Template-based analysis found no obvious anomaly indicators. "
                    "Note: Operating in offline fallback mode."
                ),
            }
        )

    def _status_template(self, _prompt: str) -> str:
        """Template for status queries."""
        return json.dumps(
            {
                "status": "operational",
                "mode": "offline_fallback",
                "llm_available": False,
                "capabilities": [
                    "anomaly_detection",
                    "pattern_matching",
                    "basic_queries",
                ],
                "message": "Mercury Agent operating in offline template mode. "
                "Core detection capabilities remain fully functional.",
            }
        )

    def _greeting_template(self, _prompt: str) -> str:
        """Template for greetings."""
        return (
            "Mercury Agent online. Operating in offline template mode. "
            "Core anomaly detection and monitoring capabilities are fully "
            "functional. For enhanced natural language interaction, ensure "
            "Ollama is running with a local model."
        )

    def _help_template(self, _prompt: str) -> str:
        """Template for help requests."""
        return (
            "Mercury Agent - Anomaly Detection System\n\n"
            "Available Commands:\n"
            "- Status check: Query system health and capabilities\n"
            "- Anomaly detection: Analyze data for anomalies\n"
            "- Monitor: Start continuous monitoring\n\n"
            "Note: Currently in offline template mode. "
            "Start Ollama for enhanced interaction."
        )

    def _unknown_template(self, prompt: str) -> str:
        """Template for unknown prompts."""
        # Try to provide some useful response
        word_count = len(prompt.split())

        return (
            f"Received query ({word_count} words). "
            "Operating in offline template mode with limited response capability. "
            "Core detection and monitoring functions remain operational. "
            "For natural language interaction, ensure Ollama is running."
        )

    def is_available(self) -> bool:
        """Template adapter is always available as fallback."""
        return True


class OpenAICloudAdapter(BaseLLMAdapter):
    """OpenAI cloud adapter for GPT models.

    Provides integration with OpenAI's API for high-capability language model inference when local
    models are unavailable.
    """

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI adapter.

        Args:
            config: LLM configuration with API key
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.model = config.model_name or "gpt-4o-mini"

        if self.api_key:
            self._is_available = True
        else:
            logger.warning("OpenAI API key not found")
            self._is_available = False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        if not self._is_available:
            return "OpenAI adapter not available - API key required"

        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json_body={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            content = str(data["choices"][0]["message"]["content"])
            self._record_usage(self.model, *_usage_from_openai(data))
            return content

        except UnsafeURLError:
            # The configured ``base_url`` was refused (SSRF gate, scheme,
            # IMDS, etc.). Operator-actionable config error -- surface
            # it rather than returning a fake "API error" string.
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_msg = err_payload.get("error", {}).get("message", "Unknown error")
            logger.error(f"OpenAI API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if OpenAI adapter is available."""
        return self._is_available


class AnthropicCloudAdapter(BaseLLMAdapter):
    """Anthropic cloud adapter for Claude models.

    Provides integration with Anthropic's API for Claude model inference when local models are
    unavailable.
    """

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic adapter.

        Args:
            config: LLM configuration with API key
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = config.base_url or "https://api.anthropic.com"
        # Fallback model used only when the operator did not set ``model_name``.
        # It MUST be a currently-served Claude id: the previous default
        # ``claude-3-5-sonnet-20241022`` was retired on 2025-10-28 and now
        # returns ``404 not_found_error`` from ``/v1/messages``, so any operator
        # who selected the Anthropic provider without naming a model hit a hard
        # 404 on the very first call. ``claude-opus-4-8`` is the current
        # first-party default; operators still override it via ``model_name``.
        self.model = config.model_name or "claude-opus-4-8"

        if self.api_key:
            self._is_available = True
        else:
            logger.warning("Anthropic API key not found")
            self._is_available = False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        if not self._is_available:
            return "Anthropic adapter not available - API key required"

        try:
            body_dict: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                body_dict["system"] = system_prompt

            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/v1/messages",
                json_body=body_dict,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            content = data.get("content", [])
            if content and len(content) > 0:
                text = str(content[0].get("text", ""))
            else:
                text = ""
            self._record_usage(self.model, *_usage_from_anthropic(data))
            return text

        except UnsafeURLError:
            # See OpenAICloudAdapter.generate -- the same config-error
            # contract applies; refuse to mask an SSRF / config gate as
            # a transient "API error" string.
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_msg = err_payload.get("error", {}).get("message", "Unknown error")
            logger.error(f"Anthropic API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if Anthropic adapter is available."""
        return self._is_available


class HuggingFaceCloudAdapter(BaseLLMAdapter):
    """HuggingFace Inference API adapter.

    Provides integration with HuggingFace's hosted inference API for various open-source models.
    """

    def __init__(self, config: LLMConfig):
        """Initialize HuggingFace adapter.

        Args:
            config: LLM configuration with API key
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = config.api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.base_url = config.base_url or "https://api-inference.huggingface.co"
        self.model = config.model_name or "meta-llama/Llama-3.2-3B-Instruct"

        if self.api_key:
            self._is_available = True
        else:
            logger.warning("HuggingFace API key not found")
            self._is_available = False

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using HuggingFace Inference API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        if not self._is_available:
            return "HuggingFace adapter not available - API key required"

        try:
            # Combine prompts for text generation
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/models/{self.model}",
                json_body={
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": self.config.max_tokens,
                        "temperature": max(0.01, self.config.temperature),
                        "return_full_text": False,
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing). The Inference API text-generation route reports no
            # token usage, so the call is recorded as unmetered -- the spend
            # stays visible in the ledger instead of silently absent.
            if isinstance(data, list) and len(data) > 0:
                text = str(data[0].get("generated_text", ""))
            else:
                text = str(data)
            self._record_usage(self.model, None, None, None)
            return text

        except UnsafeURLError:
            # See OpenAICloudAdapter.generate -- the same config-error
            # contract applies; refuse to mask an SSRF / config gate as
            # a transient "API error" string.
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_msg = (
                err_payload.get("error", "Unknown error")
                if isinstance(err_payload, dict)
                else "Unknown error"
            )
            logger.error(f"HuggingFace API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"HuggingFace request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if HuggingFace adapter is available."""
        return self._is_available


class _OpenAICompatibleCloudAdapter(BaseLLMAdapter):
    """Shared base for OpenAI-compatible Chat Completions adapters.

    xAI (Grok), DeepSeek, and Cursor all expose the OpenAI Chat
    Completions wire format (``/chat/completions`` accepting
    ``{"model": ..., "messages": [...]}``); the only meaningful
    differences are the default ``base_url``, the env-var name for
    the API key, and the provider label that shows up in error
    messages and logs.  Centralising the request / response handling
    here keeps the SSRF gate behaviour (every call passes
    ``user_configured=True``) consistent across providers and
    prevents a future drift where one adapter forgets the gate.
    """

    # Subclasses override.
    _DEFAULT_BASE_URL: str | None = None
    _PROVIDER_ENV_VAR: str = ""
    _DEFAULT_MODEL: str = ""
    _PROVIDER_LABEL: str = ""
    # Some providers require operator-supplied base_url (no public
    # default endpoint).  When True and ``config.base_url`` is unset,
    # the adapter marks itself unavailable rather than guessing.
    _REQUIRE_EXPLICIT_BASE_URL: bool = False

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI-compatible cloud adapter."""
        super().__init__(config)

        self.api_key = config.api_key or os.environ.get(self._PROVIDER_ENV_VAR)
        self.base_url = config.base_url or self._DEFAULT_BASE_URL
        self.model = config.model_name or self._DEFAULT_MODEL

        if self._REQUIRE_EXPLICIT_BASE_URL and not config.base_url:
            logger.warning(
                "%s adapter requires an explicit base_url; none provided.",
                self._PROVIDER_LABEL,
            )
            self._is_available = False
        elif not self.api_key:
            logger.warning(
                "%s API key not found (set %s or LLMConfig.api_key).",
                self._PROVIDER_LABEL,
                self._PROVIDER_ENV_VAR,
            )
            self._is_available = False
        elif not self.base_url:
            logger.warning("%s base URL not configured.", self._PROVIDER_LABEL)
            self._is_available = False
        else:
            self._is_available = True

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text via the OpenAI-compatible Chat Completions API."""
        if not self._is_available:
            return f"{self._PROVIDER_LABEL} adapter not available - API key / base_url required"

        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            assert self.base_url is not None  # narrowed by _is_available
            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json_body={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            content = str(data["choices"][0]["message"]["content"])
            self._record_usage(self.model, *_usage_from_openai(data))
            return content

        except UnsafeURLError:
            # Operator-misconfigured base_url (SSRF gate, scheme, IMDS,
            # ...). Surface so config errors are loud and actionable.
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message", "Unknown error")
            elif isinstance(error_obj, str):
                error_msg = error_obj
            else:
                error_msg = "Unknown error"
            logger.error(f"{self._PROVIDER_LABEL} API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"{self._PROVIDER_LABEL} request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if adapter is available."""
        return self._is_available


class XAIGrokAdapter(_OpenAICompatibleCloudAdapter):
    """xAI Grok cloud adapter (api.x.ai, OpenAI-compatible)."""

    _DEFAULT_BASE_URL = "https://api.x.ai/v1"
    _PROVIDER_ENV_VAR = "XAI_API_KEY"
    _DEFAULT_MODEL = "grok-2-latest"
    _PROVIDER_LABEL = "xAI"


class DeepSeekAdapter(_OpenAICompatibleCloudAdapter):
    """DeepSeek cloud adapter (api.deepseek.com, OpenAI-compatible)."""

    _DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    _PROVIDER_ENV_VAR = "DEEPSEEK_API_KEY"
    _DEFAULT_MODEL = "deepseek-chat"
    _PROVIDER_LABEL = "DeepSeek"


class CursorAdapter(_OpenAICompatibleCloudAdapter):
    """Cursor cloud adapter (OpenAI-compatible).

    Cursor does not publish a single canonical public chat-completion
    base URL; operators supply ``LLMConfig.base_url`` (and the matching
    API key via ``CURSOR_API_KEY`` or ``LLMConfig.api_key``).  The
    adapter marks itself unavailable if no base URL is supplied so a
    silent default cannot land traffic at the wrong endpoint.
    """

    _DEFAULT_BASE_URL = None
    _PROVIDER_ENV_VAR = "CURSOR_API_KEY"
    _DEFAULT_MODEL = "cursor-small"
    _PROVIDER_LABEL = "Cursor"
    _REQUIRE_EXPLICIT_BASE_URL = True


class CohereCloudAdapter(BaseLLMAdapter):
    """Cohere Chat v2 cloud adapter (api.cohere.com)."""

    _DEFAULT_BASE_URL = "https://api.cohere.com"
    _PROVIDER_ENV_VAR = "COHERE_API_KEY"
    _DEFAULT_MODEL = "command-r-plus"

    def __init__(self, config: LLMConfig):
        """Initialize the instance."""
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get(self._PROVIDER_ENV_VAR)
        self.base_url = config.base_url or self._DEFAULT_BASE_URL
        self.model = config.model_name or self._DEFAULT_MODEL
        self._is_available = bool(self.api_key)
        if not self.api_key:
            logger.warning("Cohere API key not found")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text via Cohere Chat v2."""
        if not self._is_available:
            return "Cohere adapter not available - API key required"

        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/v2/chat",
                json_body={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing). Cohere v2 returns
            # {"message": {"content": [{"type": "text", "text": "..."}]}}.
            message = data.get("message", {})
            content_items = message.get("content", [])
            if isinstance(content_items, list):
                texts = [
                    item.get("text", "")
                    for item in content_items
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                text = "".join(texts)
            else:
                text = str(content_items)
            self._record_usage(self.model, *_usage_from_cohere(data))
            return text

        except UnsafeURLError:
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_msg = (
                err_payload.get("message", "Unknown error")
                if isinstance(err_payload, dict)
                else "Unknown error"
            )
            logger.error(f"Cohere API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"Cohere request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if Cohere adapter is available."""
        return self._is_available


class GeminiCloudAdapter(BaseLLMAdapter):
    """Google Gemini cloud adapter (generativelanguage.googleapis.com).

    Gemini's REST surface is ``POST .../v1beta/models/{model}:generateContent``
    with the API key passed as a query-string parameter.  The payload
    uses ``contents=[{"parts": [{"text": ...}], "role": "user"}, ...]``
    plus a top-level ``systemInstruction`` and a ``generationConfig``.
    """

    _DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _PROVIDER_ENV_VAR = "GEMINI_API_KEY"
    _DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, config: LLMConfig):
        """Initialize the instance."""
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get(self._PROVIDER_ENV_VAR)
        self.base_url = config.base_url or self._DEFAULT_BASE_URL
        self.model = config.model_name or self._DEFAULT_MODEL
        self._is_available = bool(self.api_key)
        if not self.api_key:
            logger.warning("Gemini API key not found")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text via Gemini ``generateContent``."""
        if not self._is_available:
            return "Gemini adapter not available - API key required"

        try:
            body: dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            }
            if system_prompt:
                body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            # Auth via ``x-goog-api-key`` header.  The legacy ``?key=``
            # query-string form is still accepted by the upstream but
            # would leak the key into request logs / proxies; the
            # header form is the documented production recipe.
            data = SafeHTTPClient.post_json(
                f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent",
                json_body=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key or "",
                },
                timeout=self.config.timeout,
                user_configured=True,
            )
            # Extract content first: record usage only after a successful
            # extraction, so a malformed payload books nothing (failed calls
            # book nothing).
            candidates = data.get("candidates", [])
            text = ""
            if candidates:
                first = candidates[0]
                parts = first.get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                text = "".join(texts)
            self._record_usage(self.model, *_usage_from_gemini(data))
            return text

        except UnsafeURLError:
            raise
        except requests.HTTPError as e:
            try:
                err_payload = e.response.json() if e.response is not None else {}
            except ValueError:
                err_payload = {}
            error_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
            error_msg = (
                error_obj.get("message", "Unknown error")
                if isinstance(error_obj, dict)
                else "Unknown error"
            )
            logger.error(f"Gemini API error: {error_msg}")
            return f"API error: {error_msg}"
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            return f"Request failed: {e}"

    def is_available(self) -> bool:
        """Check if Gemini adapter is available."""
        return self._is_available


class FallbackLLMChain:
    """Graceful fallback chain for LLM operations.

    Chain: Ollama (local) → Cloud (optional) → Template (always available)

    Ensures Mercury Agent maintains conversational capability
    regardless of network or service availability.
    """

    def __init__(
        self,
        ollama_config: OllamaConfig | None = None,
        enable_cloud: bool = False,
        cloud_config: LLMConfig | None = None,
        usage_ledger: UsageLedger | None = None,
    ):
        """Initialize fallback chain.

        Args:
            ollama_config: Ollama configuration
            enable_cloud: Whether to enable cloud fallback
            cloud_config: Cloud provider configuration (if enabled)
            usage_ledger: Optional shared usage ledger attached to every
                adapter in the chain, so provider-reported token usage is
                aggregated in one place regardless of which adapter served
                a given call.
        """
        self.ollama_config = ollama_config or OllamaConfig()
        self.enable_cloud = enable_cloud
        self.cloud_config = cloud_config
        self.usage_ledger = usage_ledger

        # Initialize adapters
        self._ollama: OllamaLLMAdapter | None = None
        self._cloud: BaseLLMAdapter | None = None
        self._template = TemplateLLMAdapter()
        self._template.attach_usage_ledger(usage_ledger)

        # Track which adapter is active
        self._active_adapter: BaseLLMAdapter | None = None
        self._active_name: str = "none"

        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize the fallback chain."""
        # Try Ollama first
        self._ollama = OllamaLLMAdapter(ollama_config=self.ollama_config)
        self._ollama.attach_usage_ledger(self.usage_ledger)

        if self._ollama.is_available():
            self._active_adapter = self._ollama
            self._active_name = f"ollama:{self.ollama_config.model}"
            logger.info(f"LLM chain using Ollama ({self.ollama_config.model})")
            return

        # Try cloud only if explicitly enabled AND not under the hard air-gap.
        # MERCURY_OFFLINE is the master switch (reused from the dataset layer):
        # when set, no cloud adapter is ever constructed — local + template only.
        from omni_mercury_engine.datasets.exceptions import offline_mode_active

        if self.enable_cloud and self.cloud_config and not offline_mode_active():
            self._cloud = self._create_cloud_adapter()
            if self._cloud is not None:
                self._cloud.attach_usage_ledger(self.usage_ledger)
            if self._cloud and self._cloud.is_available():
                self._active_adapter = self._cloud
                self._active_name = f"cloud:{self.cloud_config.provider.value}"
                logger.info(f"LLM chain using cloud ({self.cloud_config.provider})")
                return

        # Fall back to template
        self._active_adapter = self._template
        self._active_name = "template"
        logger.info("LLM chain using template fallback")

    @property
    def last_usage(self) -> Any:
        """Provider-reported usage of the active adapter's last generation."""
        return self._active_adapter.last_usage if self._active_adapter else None

    def _create_cloud_adapter(self) -> BaseLLMAdapter | None:
        """Create cloud adapter based on configuration.

        All cloud providers are interchangeable and equal here; none is
        privileged. The constructed adapter matches the operator-configured
        ``provider`` only. Each provider requires its own API key set via
        environment variable or configuration.

        Returns ``None`` under ``MERCURY_OFFLINE`` (defense in depth) so no
        cloud adapter is ever constructed in the hard air-gap, even if this is
        reached directly.
        """
        if not self.cloud_config:
            return None

        from omni_mercury_engine.datasets.exceptions import offline_mode_active

        if offline_mode_active():
            logger.info("MERCURY_OFFLINE set; refusing to construct a cloud LLM adapter")
            return None

        try:
            if self.cloud_config.provider == LLMProvider.OPENAI:
                return OpenAICloudAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.ANTHROPIC:
                return AnthropicCloudAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.HUGGINGFACE:
                return HuggingFaceCloudAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.XAI:
                return XAIGrokAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.DEEPSEEK:
                return DeepSeekAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.CURSOR:
                return CursorAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.COHERE:
                return CohereCloudAdapter(self.cloud_config)
            elif self.cloud_config.provider == LLMProvider.GEMINI:
                return GeminiCloudAdapter(self.cloud_config)
            else:
                logger.warning(f"Cloud provider {self.cloud_config.provider} not supported")
                return None
        except UnsafeURLError:
            # Cloud-adapter constructors do not currently call out to
            # the network, but the SSRF gate is the kind of error that
            # must propagate the moment it appears anywhere on this
            # path. Re-raise for symmetry with the generate() handlers
            # above so a future refactor that hits the gate at
            # construction time keeps the loud-failure contract.
            raise
        except Exception as e:
            logger.warning(f"Failed to create cloud adapter: {e}")
            return None

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using the best available adapter.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        if self._active_adapter:
            return self._active_adapter.generate(prompt, system_prompt)
        return self._template.generate(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if any adapter is available."""
        # Template is always available, so this is always True
        return True

    def get_active_adapter(self) -> str:
        """Get the name of the currently active adapter."""
        return self._active_name

    def get_chain_status(self) -> dict[str, Any]:
        """Get status of all adapters in the chain."""
        return {
            "active": self._active_name,
            "ollama": {
                "available": self._ollama.is_available() if self._ollama else False,
                "model": self.ollama_config.model,
                "host": f"{self.ollama_config.host}:{self.ollama_config.port}",
            },
            "cloud": {
                "enabled": self.enable_cloud,
                "available": self._cloud.is_available() if self._cloud else False,
                "provider": (self.cloud_config.provider.value if self.cloud_config else None),
            },
            "template": {
                "available": True,
                "mode": "always_available_fallback",
            },
        }

    def refresh(self) -> str:
        """Refresh the chain and return to best available adapter.

        Returns:
            Name of the newly active adapter
        """
        self._initialize_chain()
        return self._active_name


@dataclass
class ModelConfiguration:
    """Configuration for model selection and swapping.

    Allows easy switching between models based on:
    - Task requirements
    - Available resources
    - Domain specialization
    """

    # Model selection preferences
    preferred_models: list[str] = field(
        default_factory=lambda: [
            "llama3.2:3b",  # Default balanced option
            "mistral:7b",
            "phi3:mini",
        ]
    )

    # Resource constraints
    max_model_size: str = "medium"  # tiny, small, medium, large, xl
    require_offline: bool = True

    # Domain-specific model mapping
    domain_models: dict[str, str] = field(
        default_factory=lambda: {
            "medical": "llama3.1:8b",  # Stronger reasoning for medical
            "security": "llama3.1:8b",  # Security analysis needs strength
            "code": "deepseek-coder:6.7b",  # Code-specialized
            "simple": "llama3.2:1b",  # Fast for simple queries
        }
    )

    def get_model_for_domain(self, domain: str | None = None) -> str:
        """Get the best model for a given domain."""
        if domain and domain in self.domain_models:
            return self.domain_models[domain]
        return self.preferred_models[0] if self.preferred_models else "llama3.2:3b"

    def get_model_for_task(
        self,
        task_complexity: str = "medium",
        speed_priority: bool = False,
    ) -> str:
        """Get best model for task complexity and speed requirements.

        Args:
            task_complexity: low, medium, high
            speed_priority: If True, prefer faster models

        Returns:
            Model name
        """
        size_map = {
            "low": ["tiny", "small"],
            "medium": ["small", "medium"],
            "high": ["medium", "large", "xl"],
        }

        acceptable_sizes = size_map.get(task_complexity, ["small", "medium"])

        # Find matching model from profiles
        for model_name in self.preferred_models:
            profile = MODEL_PROFILES.get(model_name)
            if profile and profile.size_category in acceptable_sizes:
                if (speed_priority and profile.speed_rating >= 0.8) or not speed_priority:
                    return model_name

        # Default fallback
        return self.preferred_models[0] if self.preferred_models else "llama3.2:3b"


def create_ollama_adapter(
    model: str | None = None,
    host: str = "localhost",
    port: int = 11434,
    **kwargs: Any,
) -> OllamaLLMAdapter:
    """Factory function to create Ollama adapter.

    Args:
        model: Model name (default: llama3.2:3b)
        host: Ollama host
        port: Ollama port
        **kwargs: Additional OllamaConfig options

    Returns:
        Configured OllamaLLMAdapter
    """
    config = OllamaConfig(
        host=host,
        port=port,
        model=model or "llama3.2:3b",
        **kwargs,
    )
    return OllamaLLMAdapter(ollama_config=config)


def create_fallback_chain(
    ollama_model: str | None = None,
    enable_cloud: bool = False,
    cloud_provider: str | None = None,
    **kwargs: Any,
) -> FallbackLLMChain:
    """Factory function to create LLM fallback chain.

    Args:
        ollama_model: Preferred Ollama model
        enable_cloud: Whether to enable cloud fallback
        cloud_provider: Cloud provider name (if enabled)
        **kwargs: Additional configuration options

    Returns:
        Configured FallbackLLMChain
    """
    ollama_config = OllamaConfig(
        model=ollama_model or "llama3.2:3b",
        **{k: v for k, v in kwargs.items() if hasattr(OllamaConfig, k)},
    )

    cloud_config = None
    if enable_cloud and cloud_provider:
        try:
            provider_enum = LLMProvider(cloud_provider.lower())
            cloud_config = LLMConfig(provider=provider_enum)
        except ValueError:
            logger.warning(f"Unknown cloud provider: {cloud_provider}")

    return FallbackLLMChain(
        ollama_config=ollama_config,
        enable_cloud=enable_cloud,
        cloud_config=cloud_config,
    )

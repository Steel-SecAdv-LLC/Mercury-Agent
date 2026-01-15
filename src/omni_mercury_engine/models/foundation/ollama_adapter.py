"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Ollama LLM Adapter for Local Inference

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
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

from omni_mercury_engine.models.foundation.llm_adapter import (
    BaseLLMAdapter,
    LLMConfig,
    LLMProvider,
)


logger = logging.getLogger(__name__)

# Allowed URL schemes for Ollama API requests
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _validate_url_scheme(url: str) -> bool:
    """Validate URL has an allowed scheme (http/https only)."""
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in _ALLOWED_SCHEMES


class OllamaModel(str, Enum):
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
    """
    Ollama LLM adapter for local model inference.

    Provides offline-first LLM capability with support for
    multiple open-source models (Llama, Mistral, Phi, etc.)
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        ollama_config: OllamaConfig | None = None,
    ):
        """
        Initialize Ollama adapter.

        Args:
            config: Base LLM configuration
            ollama_config: Ollama-specific configuration
        """
        base_config = config or LLMConfig(provider=LLMProvider.OLLAMA)
        super().__init__(base_config)

        self.ollama_config = ollama_config or OllamaConfig()

        # Override model from environment if set
        env_model = os.environ.get("MERCURY_OLLAMA_MODEL")
        if env_model:
            self.ollama_config.model = env_model

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
        """Verify the configured model is available in Ollama."""
        try:
            import urllib.request

            url = f"{self.ollama_config.base_url}/api/tags"
            if not _validate_url_scheme(url):
                logger.warning(f"Invalid URL scheme for Ollama API: {url}")
                return False

            req = urllib.request.Request(  # noqa: S310 - URL scheme validated above
                url, method="GET"
            )
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(  # nosec B310 - URL scheme validated above
                req, timeout=self.ollama_config.connect_timeout
            ) as response:
                data = json.loads(response.read().decode())
                available_models = [m["name"] for m in data.get("models", [])]

                model_base = self.ollama_config.model.split(":")[0]

                # Check for exact match or base name match
                model_available = any(
                    self.ollama_config.model in m or model_base in m for m in available_models
                )

                if not model_available:
                    logger.warning(
                        f"Model '{self.ollama_config.model}' not found. "
                        f"Available: {available_models}"
                    )
                    # Try to fall back to first available model
                    if available_models:
                        self.ollama_config.model = available_models[0]
                        logger.info(f"Falling back to model: {available_models[0]}")
                        return True

                return model_available

        except Exception as e:
            logger.debug(f"Model verification failed: {e}")
            # Optimistically assume model is available if we can't verify
            return True

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        if not self._is_available:
            return self._unavailable_response()

        try:
            import urllib.request

            url = f"{self.ollama_config.base_url}/api/generate"
            if not _validate_url_scheme(url):
                logger.error(f"Invalid URL scheme for Ollama API: {url}")
                return self._unavailable_response()

            payload = {
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

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(  # nosec B310 - URL scheme validated above
                req, timeout=self.ollama_config.timeout
            ) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._unavailable_response()

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate using chat API for multi-turn conversations.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        if not self._is_available:
            return self._unavailable_response()

        try:
            import urllib.request

            url = f"{self.ollama_config.base_url}/api/chat"
            if not _validate_url_scheme(url):
                logger.error(f"Invalid URL scheme for Ollama API: {url}")
                return self._unavailable_response()

            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.extend(messages)

            payload = {
                "model": self.ollama_config.model,
                "messages": chat_messages,
                "stream": False,
                "options": {
                    "temperature": self.ollama_config.temperature,
                    "num_ctx": self.ollama_config.num_ctx,
                    "num_predict": self.ollama_config.num_predict,
                },
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")  # noqa: S310
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(  # nosec B310 - URL scheme validated above
                req, timeout=self.ollama_config.timeout
            ) as response:
                result = json.loads(response.read().decode())
                return result.get("message", {}).get("content", "")

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
    """
    Template-based fallback adapter for offline operation.

    Provides intelligent template responses when no LLM is available.
    Uses pattern matching and rule-based responses to maintain
    basic conversational capability.
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
        """
        Generate template-based response.

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


class FallbackLLMChain:
    """
    Graceful fallback chain for LLM operations.

    Chain: Ollama (local) → Cloud (optional) → Template (always available)

    Ensures Mercury Agent maintains conversational capability
    regardless of network or service availability.
    """

    def __init__(
        self,
        ollama_config: OllamaConfig | None = None,
        enable_cloud: bool = False,
        cloud_config: LLMConfig | None = None,
    ):
        """
        Initialize fallback chain.

        Args:
            ollama_config: Ollama configuration
            enable_cloud: Whether to enable cloud fallback
            cloud_config: Cloud provider configuration (if enabled)
        """
        self.ollama_config = ollama_config or OllamaConfig()
        self.enable_cloud = enable_cloud
        self.cloud_config = cloud_config

        # Initialize adapters
        self._ollama: OllamaLLMAdapter | None = None
        self._cloud: BaseLLMAdapter | None = None
        self._template = TemplateLLMAdapter()

        # Track which adapter is active
        self._active_adapter: BaseLLMAdapter | None = None
        self._active_name: str = "none"

        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize the fallback chain."""
        # Try Ollama first
        self._ollama = OllamaLLMAdapter(ollama_config=self.ollama_config)

        if self._ollama.is_available():
            self._active_adapter = self._ollama
            self._active_name = f"ollama:{self.ollama_config.model}"
            logger.info(f"LLM chain using Ollama ({self.ollama_config.model})")
            return

        # Try cloud if enabled
        if self.enable_cloud and self.cloud_config:
            self._cloud = self._create_cloud_adapter()
            if self._cloud and self._cloud.is_available():
                self._active_adapter = self._cloud
                self._active_name = f"cloud:{self.cloud_config.provider.value}"
                logger.info(f"LLM chain using cloud ({self.cloud_config.provider})")
                return

        # Fall back to template
        self._active_adapter = self._template
        self._active_name = "template"
        logger.info("LLM chain using template fallback")

    def _create_cloud_adapter(self) -> BaseLLMAdapter | None:
        """Create cloud adapter based on configuration."""
        if not self.cloud_config:
            return None

        # Import appropriate adapter based on provider
        # Currently returns None - cloud adapters can be added
        logger.info(f"Cloud adapter for {self.cloud_config.provider} not implemented")
        return None

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate text using the best available adapter.

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
        """
        Refresh the chain and return to best available adapter.

        Returns:
            Name of the newly active adapter
        """
        self._initialize_chain()
        return self._active_name


@dataclass
class ModelConfiguration:
    """
    Configuration for model selection and swapping.

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
        """
        Get best model for task complexity and speed requirements.

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
    """
    Factory function to create Ollama adapter.

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
    """
    Factory function to create LLM fallback chain.

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

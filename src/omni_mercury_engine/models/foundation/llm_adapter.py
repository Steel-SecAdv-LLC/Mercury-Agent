# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""LLM Adapter for Zero-Shot Anomaly Detection.

Provides integration with Large Language Models for:
- Zero-shot anomaly detection via prompting
- Text/log anomaly analysis
- Natural language anomaly explanations
- Multi-modal anomaly reasoning

Inspired by AnomalyGPT and similar LLM-based anomaly detection approaches.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

from omni_mercury_engine.models.foundation.llm_usage import LLMUsage, UsageLedger
from omni_mercury_engine.security.model_policy import SafeHFLoader, UnsafeModelError

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"  # Local Ollama inference
    LOCAL = "local"
    MOCK = "mock"  # For testing
    TEMPLATE = "template"  # Fallback template-based responses
    # Additional cloud providers (Omnidirectional LLM coverage):
    XAI = "xai"  # xAI Grok (api.x.ai, OpenAI-compatible)
    GEMINI = "gemini"  # Google Gemini (generativelanguage.googleapis.com)
    COHERE = "cohere"  # Cohere Chat v2 (api.cohere.com)
    DEEPSEEK = "deepseek"  # DeepSeek (api.deepseek.com, OpenAI-compatible)
    CURSOR = "cursor"  # Cursor (operator-supplied base_url, OpenAI-compatible)


IMPLEMENTED_LLM_PROVIDERS: frozenset[LLMProvider] = frozenset(
    {
        LLMProvider.OPENAI,
        LLMProvider.ANTHROPIC,
        LLMProvider.HUGGINGFACE,
        LLMProvider.OLLAMA,
        LLMProvider.TEMPLATE,
        LLMProvider.XAI,
        LLMProvider.GEMINI,
        LLMProvider.COHERE,
        LLMProvider.DEEPSEEK,
        LLMProvider.CURSOR,
    }
)


@dataclass
class LLMConfig:
    """Configuration for LLM adapter."""

    provider: LLMProvider = LLMProvider.MOCK
    # Vendor-neutral default: empty means "operator has not chosen a model", so
    # each adapter applies its own provider-appropriate default rather than a
    # hard-coded vendor model id leaking across providers (and the "template"
    # sentinel is never sent to a cloud API). No cloud provider is privileged;
    # operators set ``provider`` and ``model_name`` explicitly.
    model_name: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0

    # HuggingFace specific - revision pinning for supply chain security (CWE-494)
    # Set to a specific commit SHA for reproducible and secure model loading
    revision: str | None = None

    # Anomaly detection specific
    anomaly_prompt_template: str = ""
    include_context: bool = True
    return_explanation: bool = True


@dataclass
class AnomalyPrompt:
    """Structured anomaly detection prompt."""

    system_prompt: str
    user_prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMAnomalyResult:
    """Result from LLM-based anomaly detection."""

    is_anomaly: bool
    anomaly_score: float
    confidence: float
    explanation: str
    category: str
    raw_response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for fusion compatibility."""
        return {
            "anomaly_score": self.anomaly_score,
            "anomaly_prob": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "category": self.category,
            "severity": self.anomaly_score,
            "metadata": self.metadata,
        }


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self, config: LLMConfig):
        """Initialize LLM adapter.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._is_available = False
        # Usage accounting: ``last_usage`` always reflects the most recent
        # successful generation's provider-reported token counts;
        # ``usage_ledger`` aggregates across calls when a caller attaches
        # one (no ledger is created ambiently).
        self.usage_ledger: UsageLedger | None = None
        self.last_usage: LLMUsage | None = None

    def attach_usage_ledger(self, ledger: UsageLedger | None) -> None:
        """Attach (or detach, with ``None``) a shared usage ledger.

        Args:
            ledger: Ledger that subsequent generations record into.
        """
        self.usage_ledger = ledger

    def _record_usage(
        self,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None = None,
    ) -> None:
        """Record provider-reported usage for one successful generation.

        Counts must come from the provider's response payload — never a
        client-side estimate. When the provider reported nothing, pass all
        ``None``: the call is recorded as *unreported* so unmetered spend
        stays visible.

        Args:
            model: Model identifier the call was made with.
            prompt_tokens: Provider-reported input tokens, or ``None``.
            completion_tokens: Provider-reported output tokens, or ``None``.
            total_tokens: Provider-reported total, or ``None``.
        """
        reported = any(
            value is not None for value in (prompt_tokens, completion_tokens, total_tokens)
        )
        usage = LLMUsage(
            provider=str(self.config.provider.value),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reported=reported,
        )
        self.last_usage = usage
        if self.usage_ledger is not None:
            self.usage_ledger.record(usage)

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass

    def detect_anomaly(
        self,
        data: Any,
        context: dict[str, Any] | None = None,
    ) -> LLMAnomalyResult:
        """Detect anomaly using LLM.

        Args:
            data: Input data (text, structured data, etc.)
            context: Optional context about the data

        Returns:
            LLMAnomalyResult with detection outcome
        """
        prompt = self._build_anomaly_prompt(data, context)
        response = self.generate(prompt.user_prompt, prompt.system_prompt)
        return self._parse_anomaly_response(response)

    def _build_anomaly_prompt(
        self,
        data: Any,
        context: dict[str, Any] | None = None,
    ) -> AnomalyPrompt:
        """Build structured anomaly detection prompt."""
        system_prompt = """You are an expert anomaly detection system. Analyze the provided data and determine if it represents an anomaly.

Your response MUST be valid JSON with the following structure:
{
    "is_anomaly": true or false,
    "anomaly_score": float between 0.0 and 1.0,
    "confidence": float between 0.0 and 1.0,
    "category": string describing anomaly type,
    "explanation": string explaining your reasoning
}

Analyze carefully considering:
- Statistical patterns and deviations
- Contextual appropriateness
- Historical baselines if provided
- Domain-specific knowledge"""

        # Convert data to string representation
        if isinstance(data, np.ndarray):
            data_str = f"Numerical data: shape={data.shape}, mean={np.mean(data):.4f}, std={np.std(data):.4f}, min={np.min(data):.4f}, max={np.max(data):.4f}"
        elif isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
            data_str = f"Tensor data: shape={data_np.shape}, mean={np.mean(data_np):.4f}, std={np.std(data_np):.4f}"
        elif isinstance(data, dict):
            data_str = f"Structured data: {json.dumps(data, default=str, indent=2)}"
        elif isinstance(data, (list, tuple)):
            data_str = f"Sequence data: length={len(data)}, values={data[:10]}..."
        else:
            data_str = str(data)

        user_prompt = f"""Analyze the following data for anomalies:

{data_str}

"""
        if context:
            user_prompt += f"""
Context:
{json.dumps(context, default=str, indent=2)}

"""

        user_prompt += "Respond with valid JSON only."

        return AnomalyPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context or {},
        )

    def _parse_anomaly_response(self, response: str) -> LLMAnomalyResult:
        """Parse LLM response into structured result."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)

                return LLMAnomalyResult(
                    is_anomaly=bool(parsed.get("is_anomaly", False)),
                    anomaly_score=float(parsed.get("anomaly_score", 0.0)),
                    confidence=float(parsed.get("confidence", 0.5)),
                    explanation=str(parsed.get("explanation", "")),
                    category=str(parsed.get("category", "unknown")),
                    raw_response=response,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Fallback parsing
        is_anomaly = "anomaly" in response.lower() and "not" not in response.lower()[:50]
        return LLMAnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=0.5 if is_anomaly else 0.1,
            confidence=0.3,
            explanation="Failed to parse structured response",
            category="parse_error",
            raw_response=response,
        )


class MockLLMAdapter(BaseLLMAdapter):
    """Mock LLM adapter — hard-fails at construction.

    Phase 2 audit cure: silent mock degradation is not permitted in
    production.  Instantiating this class raises ``NotImplementedError``
    so operators are forced to configure a real LLM provider.
    """

    def __init__(self, config: LLMConfig | None = None):
        """Initialize the instance."""
        raise NotImplementedError(
            "MockLLMAdapter cannot be used in production. "
            "Configure a real LLM provider (e.g. LLMProvider.HUGGINGFACE)."
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:  # pragma: no cover
        """Generate."""
        raise NotImplementedError

    def is_available(self) -> bool:  # pragma: no cover
        """Is available."""
        return False


class HuggingFaceLLMAdapter(BaseLLMAdapter):
    """HuggingFace Transformers LLM adapter for local models."""

    def __init__(self, config: LLMConfig):
        """Initialize HuggingFace adapter.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if transformers is available and model can be loaded."""
        try:
            import importlib.util

            self._is_available = importlib.util.find_spec("transformers") is not None
        except Exception:
            logger.warning("transformers not installed. HuggingFace adapter unavailable.")
            self._is_available = False

    def _load_model(self) -> None:
        """Lazy load the model.

        Security Note: For supply chain security (CWE-494), we require revision
        pinning when loading models from HuggingFace Hub. Set config.revision to
        a specific commit SHA for reproducible and secure model loading.
        Local paths (starting with '/' or '.') are allowed without revision.
        """
        if self._model is not None:
            return

        # Local paths bypass revision pinning. Only absolute paths
        # qualify as local; relative paths would let resolution depend
        # on the current working directory.  The cross-platform check
        # mirrors ``security/model_policy._is_local_path`` (POSIX +
        # Windows + UNC) so a Hub id like ``Salesforce/blip`` is never
        # mistaken for a local path on either OS.
        from pathlib import PurePosixPath, PureWindowsPath

        model_name = self.config.model_name
        is_local_path = (
            PurePosixPath(model_name).is_absolute() or PureWindowsPath(model_name).is_absolute()
        )

        # Require revision for remote models (supply chain security).
        # Raise rather than silently degrade: ``generate()`` would
        # otherwise return a fake "unavailable" JSON stub on the very
        # first call, hiding the misconfiguration. Operators need to
        # see the actionable error so they can pin a SHA or switch to
        # a local path.
        if not is_local_path and not self.config.revision:
            raise UnsafeModelError(
                f"HuggingFace model '{self.config.model_name}' requires a "
                "revision pin (40-char commit SHA) for supply-chain "
                "security (CWE-494). Set config.revision to a verified SHA "
                "or use an absolute local path; mutable branch/tag refs "
                "are not accepted."
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Use revision for remote models, None for local paths.
            # SafeHFLoader.load_* enforces revision pinning and the
            # HuggingFace identifier shape; no allowlist here because
            # this adapter is the generic HF backend.
            revision = self.config.revision if not is_local_path else None

            self._tokenizer = SafeHFLoader.load_tokenizer(
                AutoTokenizer,
                self.config.model_name,
                revision=revision,
            )
            self._model = SafeHFLoader.load_model(
                AutoModelForCausalLM,
                self.config.model_name,
                revision=revision,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info(
                f"Loaded HuggingFace model: {self.config.model_name} (revision: {revision})"
            )
        except UnsafeModelError:
            # SafeHFLoader refusal -- propagate as-is so the operator
            # sees the actionable policy error rather than a generic
            # "model unavailable" stub.
            self._is_available = False
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_available = False
            raise

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate text using HuggingFace model."""
        if not self._is_available:
            return json.dumps(
                {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "confidence": 0.0,
                    "category": "unavailable",
                    "explanation": "HuggingFace model not available",
                }
            )

        self._load_model()

        # Check if model/tokenizer are available after loading attempt
        if self._model is None or self._tokenizer is None or not self._is_available:
            return json.dumps(
                {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "confidence": 0.0,
                    "category": "unavailable",
                    "explanation": "HuggingFace model could not be loaded",
                }
            )

        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"

        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            do_sample=self.config.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def is_available(self) -> bool:
        """Check if HuggingFace adapter is available."""
        return self._is_available


class ZeroShotAnomalyDetector:
    """Zero-shot anomaly detector using LLM prompting.

    Provides anomaly detection without training by leveraging LLM's world knowledge and reasoning
    capabilities.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        adapter: BaseLLMAdapter | None = None,
    ):
        """Initialize zero-shot detector.

        Args:
            config: LLM configuration
            adapter: Optional pre-configured adapter
        """
        self.config = config or LLMConfig()

        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = self._create_adapter()

    def _create_adapter(self) -> BaseLLMAdapter:
        """Create appropriate adapter based on config.

        Phase 2 audit cure: an unsupported / unimplemented provider must
        not silently fall back to ``MockLLMAdapter`` — that path masked
        misconfiguration and routed production traffic through a
        hard-fail mock.  ``MockLLMAdapter`` itself raises
        ``NotImplementedError`` at construction, so the explicit
        ``LLMProvider.MOCK`` branch below is preserved purely as a
        loud-fail signal (the operator opted into mock and gets the
        cure's error message immediately).  Any other unsupported
        provider raises ``NotImplementedError`` directly with a
        message that names every supported provider.
        """
        if self.config.provider == LLMProvider.MOCK:
            return MockLLMAdapter(self.config)
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceLLMAdapter(self.config)
        elif self.config.provider in {
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.OLLAMA,
            LLMProvider.TEMPLATE,
            LLMProvider.XAI,
            LLMProvider.GEMINI,
            LLMProvider.COHERE,
            LLMProvider.DEEPSEEK,
            LLMProvider.CURSOR,
        }:
            return _create_non_hf_adapter(self.config)
        else:
            supported = ", ".join(sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS))
            raise NotImplementedError(
                f"LLM provider {self.config.provider!r} has no adapter "
                "implementation in this build.  Configure a supported "
                f"provider (currently: {supported}).  Silent mock "
                "degradation is not permitted (Phase 2 audit cure)."
            )

    def detect(
        self,
        data: Any,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies using zero-shot LLM prompting.

        Args:
            data: Input data (text, numerical, structured)
            context: Optional context about expected patterns

        Returns:
            Detection result compatible with fusion pipeline
        """
        result = self.adapter.detect_anomaly(data, context)
        return result.to_dict()

    def detect_batch(
        self,
        data_batch: list[Any],
        contexts: list[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect anomalies in batch.

        Args:
            data_batch: List of data samples
            contexts: Optional list of contexts

        Returns:
            List of detection results
        """
        results = []
        contexts = contexts or [None] * len(data_batch)

        for data, ctx in zip(data_batch, contexts):
            results.append(self.detect(data, ctx))

        return results

    def explain_anomaly(
        self,
        data: Any,
        detection_result: dict[str, Any],
    ) -> str:
        """Generate natural language explanation for detected anomaly.

        Args:
            data: Original input data
            detection_result: Previous detection result

        Returns:
            Natural language explanation
        """
        prompt = f"""Given the following anomaly detection result:
Score: {detection_result.get('anomaly_score', 0)}
Category: {detection_result.get('category', 'unknown')}

And the original data:
{str(data)[:500]}

Provide a detailed, human-readable explanation of:
1. Why this was flagged as an anomaly (or not)
2. What specific patterns or features triggered the detection
3. Recommended actions or further investigation steps

Be concise but thorough."""

        system_prompt = "You are an expert anomaly analyst providing clear explanations to non-technical stakeholders."

        response = self.adapter.generate(prompt, system_prompt)
        return response


class TextLogAnomalyDetector:
    """Specialized detector for text and log anomalies.

    Uses LLM understanding of log patterns, error messages, and text semantics for anomaly
    detection.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        log_format: str | None = None,
        known_patterns: list[str] | None = None,
    ):
        """Initialize text/log anomaly detector.

        Args:
            config: LLM configuration
            log_format: Expected log format description
            known_patterns: List of known normal patterns
        """
        self.config = config or LLMConfig()
        self.log_format = log_format
        self.known_patterns = known_patterns or []
        self.detector = ZeroShotAnomalyDetector(self.config)

    def detect_log_anomaly(
        self,
        log_entry: str,
        surrounding_entries: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in log entries.

        Args:
            log_entry: Log entry to analyze
            surrounding_entries: Context from nearby log entries

        Returns:
            Detection result
        """
        context = {
            "type": "log_entry",
            "format": self.log_format,
            "known_normal_patterns": self.known_patterns[:5],  # Include sample patterns
        }

        if surrounding_entries:
            context["surrounding_context"] = surrounding_entries[:3]

        return self.detector.detect(log_entry, context)

    def detect_text_anomaly(
        self,
        text: str,
        expected_domain: str | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in text content.

        Args:
            text: Text to analyze
            expected_domain: Expected content domain

        Returns:
            Detection result
        """
        context = {
            "type": "text_content",
            "expected_domain": expected_domain,
        }

        return self.detector.detect(text, context)


def _parse_ollama_base_url(base_url: str | None) -> tuple[str | None, int | None]:
    """Return ``(host, port)`` from an Ollama base URL override."""
    if not base_url:
        return None, None
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    if not parsed.hostname:
        raise ValueError(f"Invalid Ollama base_url {base_url!r}: missing host")
    return parsed.hostname, parsed.port


def _create_non_hf_adapter(config: LLMConfig) -> BaseLLMAdapter:
    """Create implemented non-HuggingFace adapters."""
    from omni_mercury_engine.models.foundation.ollama_adapter import (
        AnthropicCloudAdapter,
        CohereCloudAdapter,
        CursorAdapter,
        DeepSeekAdapter,
        GeminiCloudAdapter,
        OllamaConfig,
        OllamaLLMAdapter,
        OpenAICloudAdapter,
        TemplateLLMAdapter,
        XAIGrokAdapter,
    )

    if config.provider == LLMProvider.OLLAMA:
        host, port = _parse_ollama_base_url(config.base_url)
        ollama_config = OllamaConfig(
            model=config.model_name or "llama3.2:3b",
            host=host or "localhost",
            port=port or 11434,
        )
        return OllamaLLMAdapter(config, ollama_config)
    if config.provider == LLMProvider.TEMPLATE:
        return TemplateLLMAdapter(config)
    if config.provider == LLMProvider.OPENAI:
        return OpenAICloudAdapter(config)
    if config.provider == LLMProvider.ANTHROPIC:
        return AnthropicCloudAdapter(config)
    if config.provider == LLMProvider.XAI:
        return XAIGrokAdapter(config)
    if config.provider == LLMProvider.DEEPSEEK:
        return DeepSeekAdapter(config)
    if config.provider == LLMProvider.CURSOR:
        return CursorAdapter(config)
    if config.provider == LLMProvider.COHERE:
        return CohereCloudAdapter(config)
    if config.provider == LLMProvider.GEMINI:
        return GeminiCloudAdapter(config)
    raise NotImplementedError(f"LLM provider {config.provider!r} has no adapter implementation")


def create_llm_detector(
    provider: str = "mock",
    model_name: str | None = None,
    **kwargs: Any,
) -> ZeroShotAnomalyDetector:
    """Factory function to create LLM-based anomaly detector.

    Args:
        provider: Implemented LLM provider name (ollama, huggingface,
            openai, anthropic, xai, gemini, cohere, deepseek, cursor,
            template). ``mock`` is a hard-fail sentinel.
        model_name: Model identifier
        **kwargs: Additional configuration

    Returns:
        Configured ZeroShotAnomalyDetector
    """
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError as exc:
        supported = ", ".join(sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS))
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported providers: {supported}."
        ) from exc

    if provider_enum == LLMProvider.MOCK:
        raise NotImplementedError(
            "MockLLMAdapter cannot be used in production. Configure a real "
            "provider or use the explicit template adapter for deterministic "
            "offline responses."
        )
    if provider_enum not in IMPLEMENTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(p.value for p in IMPLEMENTED_LLM_PROVIDERS))
        raise NotImplementedError(
            f"LLM provider {provider_enum.value!r} has no adapter implementation "
            f"in this build. Supported providers: {supported}."
        )

    config_kwargs = dict(kwargs)
    ollama_host = config_kwargs.pop("host", None)
    ollama_port = config_kwargs.pop("port", None)
    if provider_enum == LLMProvider.HUGGINGFACE:
        from pathlib import PurePosixPath, PureWindowsPath

        if not model_name:
            raise ValueError(
                "create_llm_detector(provider='huggingface') requires "
                "model_name=<HuggingFace model ID or absolute local path>."
            )
        is_local_path = (
            PurePosixPath(model_name).is_absolute() or PureWindowsPath(model_name).is_absolute()
        )
        if not is_local_path and not config_kwargs.get("revision"):
            raise ValueError(
                "HuggingFace remote model IDs require revision=<40-character "
                "commit SHA> so SafeHFLoader can enforce reproducible model loading."
            )

    if provider_enum == LLMProvider.TEMPLATE:
        # TemplateLLMAdapter is deterministic-offline and ignores model_name.
        resolved_model_name = model_name or "template"
    else:
        # Every real LLM provider needs an explicit model identifier --
        # silently substituting a cross-provider placeholder like
        # ``gpt-4o`` for an Ollama or Anthropic caller masks the
        # configuration error and (for Ollama) makes the per-adapter
        # default (``llama3.2:3b``) unreachable.
        if not model_name:
            raise ValueError(
                f"create_llm_detector(provider={provider_enum.value!r}) requires "
                "model_name=<provider-specific model identifier>."
            )
        resolved_model_name = model_name

    config = LLMConfig(
        provider=provider_enum,
        model_name=resolved_model_name,
        **config_kwargs,
    )

    # Handle Ollama specifically so host/port kwargs and base_url both
    # reach OllamaConfig instead of being ignored by LLMConfig.
    if provider_enum == LLMProvider.OLLAMA:
        from omni_mercury_engine.models.foundation.ollama_adapter import (
            OllamaConfig,
            OllamaLLMAdapter,
        )

        base_host, base_port = _parse_ollama_base_url(config.base_url)

        ollama_config = OllamaConfig(
            model=resolved_model_name,
            host=ollama_host or base_host or "localhost",
            port=ollama_port or base_port or 11434,
        )
        adapter = OllamaLLMAdapter(config, ollama_config)
        return ZeroShotAnomalyDetector(config, adapter=adapter)

    if provider_enum not in {LLMProvider.HUGGINGFACE, LLMProvider.MOCK}:
        return ZeroShotAnomalyDetector(config, adapter=_create_non_hf_adapter(config))

    return ZeroShotAnomalyDetector(config)

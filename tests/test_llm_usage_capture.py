# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for provider-reported usage capture in the LLM adapters.

Every adapter's success path must surface the token usage its provider
reported (``last_usage``) and aggregate it into an attached ledger; the
HuggingFace Inference route, whose responses carry no usage block, must
record the call as *unmetered* rather than silently dropping it. The HTTP
layer is mocked at the ``SafeHTTPClient`` boundary, exactly as in
``tests/test_cloud_llm_adapters.py`` — no network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation.llm_usage import UsageLedger
from omni_mercury_engine.models.foundation.ollama_adapter import (
    AnthropicCloudAdapter,
    CohereCloudAdapter,
    FallbackLLMChain,
    GeminiCloudAdapter,
    HuggingFaceCloudAdapter,
    OllamaConfig,
    OllamaLLMAdapter,
    OpenAICloudAdapter,
    XAIGrokAdapter,
    _as_count,
)

_POST_JSON = "omni_mercury_engine.models.foundation.ollama_adapter.SafeHTTPClient.post_json"


def _generate_with(adapter: Any, fake_response: dict[str, Any]) -> str:
    with patch(_POST_JSON, return_value=fake_response):
        return str(adapter.generate("hi"))


class TestProviderUsageParsing:
    """Each wire format's usage block lands in last_usage and the ledger."""

    def test_openai_usage_captured(self) -> None:
        ledger = UsageLedger()
        adapter = OpenAICloudAdapter(
            LLMConfig(provider=LLMProvider.OPENAI, api_key="sk-test", model_name="gpt-test")
        )
        adapter.attach_usage_ledger(ledger)
        text = _generate_with(
            adapter,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )
        assert text == "ok"
        assert adapter.last_usage is not None
        assert adapter.last_usage.provider == "openai"
        assert adapter.last_usage.model == "gpt-test"
        assert adapter.last_usage.total_tokens == 16
        assert ledger.totals()["total_tokens"] == 16
        assert ledger.totals()["unreported_calls"] == 0

    def test_openai_compatible_family_usage_captured(self) -> None:
        adapter = XAIGrokAdapter(
            LLMConfig(provider=LLMProvider.XAI, api_key="x-test", model_name="grok-test")
        )
        _generate_with(
            adapter,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )
        assert adapter.last_usage is not None
        assert adapter.last_usage.provider == "xai"
        assert (adapter.last_usage.prompt_tokens, adapter.last_usage.completion_tokens) == (7, 3)

    def test_anthropic_usage_captured_and_total_derived(self) -> None:
        """Anthropic reports input/output only; the total is their sum."""
        adapter = AnthropicCloudAdapter(
            LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="a-test", model_name="claude-test")
        )
        _generate_with(
            adapter,
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )
        assert adapter.last_usage is not None
        assert adapter.last_usage.prompt_tokens == 20
        assert adapter.last_usage.completion_tokens == 5
        assert adapter.last_usage.total_tokens == 25

    def test_gemini_usage_metadata_captured(self) -> None:
        adapter = GeminiCloudAdapter(
            LLMConfig(provider=LLMProvider.GEMINI, api_key="g-test", model_name="gemini-test")
        )
        _generate_with(
            adapter,
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 9,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 11,
                },
            },
        )
        assert adapter.last_usage is not None
        assert adapter.last_usage.total_tokens == 11

    def test_cohere_nested_token_block_captured(self) -> None:
        adapter = CohereCloudAdapter(
            LLMConfig(provider=LLMProvider.COHERE, api_key="c-test", model_name="command-test")
        )
        _generate_with(
            adapter,
            {
                "message": {"content": [{"type": "text", "text": "ok"}]},
                "usage": {"tokens": {"input_tokens": 6.0, "output_tokens": 2.0}},
            },
        )
        assert adapter.last_usage is not None
        # Cohere serializes counts as floats; they coerce to exact ints.
        assert adapter.last_usage.prompt_tokens == 6
        assert adapter.last_usage.completion_tokens == 2

    def test_ollama_eval_counts_captured(self) -> None:
        config = OllamaConfig()
        adapter = OllamaLLMAdapter(ollama_config=config)
        adapter._is_available = True  # bypass the socket probe; HTTP is mocked
        _generate_with(
            adapter,
            {"response": "ok", "prompt_eval_count": 15, "eval_count": 8},
        )
        assert adapter.last_usage is not None
        assert adapter.last_usage.provider == "ollama"
        assert adapter.last_usage.model == config.model
        assert adapter.last_usage.total_tokens == 23

    def test_huggingface_route_records_unmetered_call(self) -> None:
        """No usage block in the response -> visible unreported call."""
        ledger = UsageLedger()
        adapter = HuggingFaceCloudAdapter(
            LLMConfig(provider=LLMProvider.HUGGINGFACE, api_key="hf-test", model_name="m/test")
        )
        adapter.attach_usage_ledger(ledger)
        _generate_with(adapter, [{"generated_text": "ok"}])  # type: ignore[arg-type]
        assert adapter.last_usage is not None
        assert adapter.last_usage.reported is False
        assert ledger.totals() == {
            "calls": 1,
            "unreported_calls": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_missing_usage_block_is_unreported_not_zero(self) -> None:
        """A provider omitting usage must not be booked as zero tokens."""
        adapter = OpenAICloudAdapter(
            LLMConfig(provider=LLMProvider.OPENAI, api_key="sk-test", model_name="gpt-test")
        )
        _generate_with(adapter, {"choices": [{"message": {"content": "ok"}}]})
        assert adapter.last_usage is not None
        assert adapter.last_usage.reported is False
        assert adapter.last_usage.total_tokens is None

    def test_failed_call_records_no_usage(self) -> None:
        """HTTP errors return error strings and must not book usage."""
        import requests

        adapter = OpenAICloudAdapter(
            LLMConfig(provider=LLMProvider.OPENAI, api_key="sk-test", model_name="gpt-test")
        )
        with patch(_POST_JSON, side_effect=requests.HTTPError()):
            result = adapter.generate("hi")
        assert result.startswith("API error")
        assert adapter.last_usage is None


class TestAsCount:
    """Token-count coercion is provider-truthful: exact or unreported, never guessed."""

    def test_int_passes_through(self) -> None:
        assert _as_count(42) == 42
        assert _as_count(0) == 0

    def test_integer_valued_float_is_exact(self) -> None:
        """Cohere-style ``42.0`` coerces to the exact integer."""
        assert _as_count(42.0) == 42

    def test_fractional_float_is_unreported_not_truncated(self) -> None:
        """A genuinely fractional count must not be silently floored to an int."""
        assert _as_count(3.7) is None
        assert _as_count(0.5) is None

    def test_bool_is_unreported(self) -> None:
        """``bool`` is an ``int`` subclass; it is never a token count."""
        assert _as_count(True) is None
        assert _as_count(False) is None

    def test_negative_is_unreported(self) -> None:
        assert _as_count(-1) is None
        assert _as_count(-2.0) is None

    def test_non_numeric_is_unreported(self) -> None:
        assert _as_count("7") is None
        assert _as_count(None) is None


class TestFallbackChainLedger:
    """The chain threads one ledger through every adapter it builds."""

    def test_chain_attaches_ledger_to_all_adapters(self) -> None:
        ledger = UsageLedger()
        with patch.object(OllamaLLMAdapter, "_check_availability", lambda self: None):
            chain = FallbackLLMChain(
                enable_cloud=True,
                cloud_config=LLMConfig(
                    provider=LLMProvider.OPENAI, api_key="sk-test", model_name="gpt-test"
                ),
                usage_ledger=ledger,
            )
        assert chain._ollama is not None
        assert chain._ollama.usage_ledger is ledger
        assert chain._cloud is not None
        assert chain._cloud.usage_ledger is ledger
        assert chain._template.usage_ledger is ledger

    def test_chain_last_usage_delegates_to_active_adapter(self) -> None:
        ledger = UsageLedger()
        with patch.object(OllamaLLMAdapter, "_check_availability", lambda self: None):
            chain = FallbackLLMChain(
                enable_cloud=True,
                cloud_config=LLMConfig(
                    provider=LLMProvider.OPENAI, api_key="sk-test", model_name="gpt-test"
                ),
                usage_ledger=ledger,
            )
        # Ollama probe was bypassed (unavailable) -> the cloud adapter is active.
        assert chain._active_name == "cloud:openai"
        assert chain._cloud is not None
        with patch(
            _POST_JSON,
            return_value={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        ):
            chain._cloud.generate("hi")
        assert chain.last_usage is not None
        assert chain.last_usage.total_tokens == 4
        assert ledger.totals()["calls"] == 1

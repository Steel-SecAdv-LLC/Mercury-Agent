# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# ``omni_mercury_engine.models.foundation.llm_adapter`` transitively
# imports ``torch`` via :mod:`base_foundation`.  Skip the entire module
# cleanly at collection time when torch is absent so the rest of the
# suite stays discoverable in CI images without the optional ``ml`` extra.
pytest.importorskip("torch")

from omni_mercury_engine.models.foundation.llm_adapter import LLMConfig, LLMProvider
from omni_mercury_engine.models.foundation.ollama_adapter import (
    CohereCloudAdapter,
    CursorAdapter,
    DeepSeekAdapter,
    FallbackLLMChain,
    GeminiCloudAdapter,
    XAIGrokAdapter,
)
from omni_mercury_engine.security.safe_http import UnsafeURLError


class TestAdaptersConstructWithoutKey:
    """Each cloud adapter must mark itself unavailable when no key is set."""

    @pytest.mark.parametrize(
        ("cls", "provider", "env_var"),
        [
            (XAIGrokAdapter, LLMProvider.XAI, "XAI_API_KEY"),
            (DeepSeekAdapter, LLMProvider.DEEPSEEK, "DEEPSEEK_API_KEY"),
            (CohereCloudAdapter, LLMProvider.COHERE, "COHERE_API_KEY"),
            (GeminiCloudAdapter, LLMProvider.GEMINI, "GEMINI_API_KEY"),
        ],
    )
    def test_adapter_unavailable_without_key(
        self, cls: Any, provider: Any, env_var: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapters with public default base_url surface key absence as unavailable."""
        monkeypatch.delenv(env_var, raising=False)
        adapter = cls(LLMConfig(provider=provider, api_key=None))
        assert adapter.is_available() is False

    def test_cursor_unavailable_without_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cursor has no public default base_url; must refuse to default."""
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        adapter = CursorAdapter(
            LLMConfig(provider=LLMProvider.CURSOR, api_key="test-key", base_url=None)
        )
        assert adapter.is_available() is False

    def test_cursor_available_with_explicit_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cursor becomes available when operator supplies base_url AND key."""
        monkeypatch.setenv("CURSOR_API_KEY", "test-key")
        adapter = CursorAdapter(
            LLMConfig(
                provider=LLMProvider.CURSOR,
                api_key="test-key",
                base_url="https://cursor.example.com/v1",
            )
        )
        assert adapter.is_available() is True


class TestAdaptersCallSafeHTTPClient:
    """Cloud adapters MUST route through SafeHTTPClient with user_configured=True."""

    @pytest.mark.parametrize(
        ("cls", "provider", "expected_path_suffix"),
        [
            (XAIGrokAdapter, LLMProvider.XAI, "/chat/completions"),
            (DeepSeekAdapter, LLMProvider.DEEPSEEK, "/chat/completions"),
        ],
    )
    def test_openai_compatible_adapters_call_chat_completions(
        self, cls: Any, provider: Any, expected_path_suffix: Any
    ) -> None:
        """xAI / DeepSeek hit ``/chat/completions`` on their default base_url."""
        config = LLMConfig(provider=provider, api_key="sk-test", model_name="m")
        adapter = cls(config)
        assert adapter.is_available()
        fake_response = {
            "choices": [{"message": {"content": "hello world"}}],
        }
        with patch(
            "omni_mercury_engine.models.foundation.ollama_adapter.SafeHTTPClient.post_json",
            return_value=fake_response,
        ) as mock_post:
            result = adapter.generate("hi")
        assert result == "hello world"
        url_arg = mock_post.call_args.args[0]
        assert url_arg.endswith(expected_path_suffix)
        kwargs = mock_post.call_args.kwargs
        assert kwargs["user_configured"] is True
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test"

    def test_cohere_calls_v2_chat(self) -> None:
        """Cohere targets ``/v2/chat`` with Cohere v2 response shape."""
        adapter = CohereCloudAdapter(LLMConfig(provider=LLMProvider.COHERE, api_key="co-test"))
        fake_response = {
            "message": {"content": [{"type": "text", "text": "from cohere"}]},
        }
        with patch(
            "omni_mercury_engine.models.foundation.ollama_adapter.SafeHTTPClient.post_json",
            return_value=fake_response,
        ) as mock_post:
            result = adapter.generate("ping")
        assert result == "from cohere"
        assert mock_post.call_args.args[0].endswith("/v2/chat")
        assert mock_post.call_args.kwargs["user_configured"] is True

    def test_gemini_calls_generate_content(self) -> None:
        """Gemini targets ``/models/{model}:generateContent`` with x-goog-api-key."""
        adapter = GeminiCloudAdapter(
            LLMConfig(
                provider=LLMProvider.GEMINI,
                api_key="g-test",
                model_name="gemini-test",
            )
        )
        fake_response = {
            "candidates": [
                {"content": {"parts": [{"text": "from gemini"}]}},
            ],
        }
        with patch(
            "omni_mercury_engine.models.foundation.ollama_adapter.SafeHTTPClient.post_json",
            return_value=fake_response,
        ) as mock_post:
            result = adapter.generate("ping", system_prompt="be terse")
        assert result == "from gemini"
        kwargs = mock_post.call_args.kwargs
        assert kwargs["user_configured"] is True
        assert kwargs["headers"]["x-goog-api-key"] == "g-test"
        body = kwargs["json_body"]
        assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
        assert mock_post.call_args.args[0].endswith("/models/gemini-test:generateContent")


class TestAdaptersPropagateUnsafeURLError:
    """SSRF/config refusals MUST propagate -- not collapse to ``API error: ...``."""

    @pytest.mark.parametrize(
        ("cls", "provider"),
        [
            (XAIGrokAdapter, LLMProvider.XAI),
            (DeepSeekAdapter, LLMProvider.DEEPSEEK),
            (CohereCloudAdapter, LLMProvider.COHERE),
            (GeminiCloudAdapter, LLMProvider.GEMINI),
        ],
    )
    def test_unsafe_url_error_propagates(self, cls: Any, provider: Any) -> None:
        """Each adapter re-raises ``UnsafeURLError`` instead of returning a string."""
        adapter = cls(LLMConfig(provider=provider, api_key="k", model_name="m"))
        with (
            patch(
                "omni_mercury_engine.models.foundation.ollama_adapter.SafeHTTPClient.post_json",
                side_effect=UnsafeURLError("test-ssrf"),
            ),
            pytest.raises(UnsafeURLError),
        ):
            adapter.generate("hi")


class TestFallbackChainRoutesEveryProvider:
    """Every new LLMProvider value resolves to its adapter class."""

    @pytest.mark.parametrize(
        ("provider", "expected_cls"),
        [
            (LLMProvider.XAI, XAIGrokAdapter),
            (LLMProvider.DEEPSEEK, DeepSeekAdapter),
            (LLMProvider.CURSOR, CursorAdapter),
            (LLMProvider.COHERE, CohereCloudAdapter),
            (LLMProvider.GEMINI, GeminiCloudAdapter),
        ],
    )
    def test_create_cloud_adapter_routes(
        self, provider: Any, expected_cls: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``FallbackLLMChain._create_cloud_adapter`` returns the right type."""
        # Provide a key so the adapter constructs cleanly; Cursor also
        # needs a base_url, so we set one for that provider.
        for env_var in (
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "COHERE_API_KEY",
            "GEMINI_API_KEY",
            "CURSOR_API_KEY",
        ):
            monkeypatch.setenv(env_var, "test-key")

        base_url = "https://cursor.example.com/v1" if provider == LLMProvider.CURSOR else None
        cloud_config = LLMConfig(provider=provider, api_key="test-key", base_url=base_url)

        # Suppress the Ollama TCP probe before constructing the chain.
        # ``FallbackLLMChain.__init__`` calls ``_initialize_chain``
        # which instantiates ``OllamaLLMAdapter``; that adapter's own
        # ``__init__`` opens a TCP socket to the configured Ollama host.
        # On a workstation with a real Ollama daemon, the probe would
        # actually connect (and could trigger a follow-up HTTP request
        # via SafeHTTPClient), violating the "no real network" guarantee
        # the rest of this file relies on. Replacing
        # ``_check_availability`` with a no-op that marks the adapter
        # unavailable forces the chain to skip Ollama and reach the
        # cloud branch we actually want to exercise.
        import omni_mercury_engine.models.foundation.ollama_adapter as _ollama_mod

        def _noop_check_availability(self: Any) -> None:
            self._is_available = False

        monkeypatch.setattr(
            _ollama_mod.OllamaLLMAdapter,
            "_check_availability",
            _noop_check_availability,
        )

        chain = FallbackLLMChain(enable_cloud=True, cloud_config=cloud_config)
        created = chain._create_cloud_adapter()
        assert isinstance(created, expected_cls)

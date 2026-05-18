"""Tests for MercuryVoice LLM initialization behaviour.

Covers the post-1.7.0 contract documented in
``docs/MIGRATION-1.6-to-1.7.md``:

- ``enable_llm=False`` is a pure-template fast path with no LLM init.
- ``enable_llm=True`` without ``llm_provider`` warns + downgrades in
  development, raises ``MercuryProductionConfigError`` in production.
- ``llm_provider='mock'`` always raises — the Phase 2 mock cure made
  ``MockLLMAdapter`` hard-fail at construction.
- An unknown ``llm_provider`` raises a clean ``ValueError`` naming
  every supported provider.
- A configured provider whose optional dependency is missing degrades
  with a warning in dev and raises in production.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from omni_mercury_engine._env import (
    MERCURY_ENV_PRODUCTION,
    MERCURY_ENV_VAR,
    MercuryProductionConfigError,
)
from omni_mercury_engine.models.foundation.llm_adapter import create_llm_detector
from omni_mercury_engine.narrative.voice import MercuryVoice, create_mercury_voice

_HF_REVISION = "0123456789abcdef0123456789abcdef01234567"


class TestVoiceLLMDisabled:
    """Default path: ``enable_llm=False`` never touches the LLM stack."""

    def test_default_construction_does_not_init_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        voice = MercuryVoice()
        assert voice.enable_llm is False
        assert voice._llm_adapter is None
        stats = voice.get_statistics()
        assert stats["llm_enabled"] is False

    def test_disabled_in_production_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Production mode only restricts ``enable_llm=True``."""
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)
        voice = MercuryVoice(enable_llm=False)
        assert voice._llm_adapter is None


class TestVoiceLLMMissingProvider:
    """``enable_llm=True`` with no ``llm_provider`` configured."""

    def test_development_warns_and_disables(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        with caplog.at_level("WARNING", logger="omni_mercury_engine.narrative.voice"):
            voice = MercuryVoice(enable_llm=True)
        assert voice._llm_adapter is None
        assert any(
            "template-only narration" in r.message for r in caplog.records
        ), "Expected a warning naming the template-only fallback"

    def test_production_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)
        with pytest.raises(MercuryProductionConfigError) as exc_info:
            MercuryVoice(enable_llm=True)
        message = str(exc_info.value)
        assert "narrative LLM provider" in message
        assert "llm_provider=" in message


class TestVoiceLLMMockProvider:
    """``llm_provider='mock'`` is rejected in every mode."""

    @pytest.mark.parametrize("env", ["", MERCURY_ENV_PRODUCTION])
    def test_mock_provider_always_raises(self, monkeypatch: pytest.MonkeyPatch, env: str) -> None:
        if env:
            monkeypatch.setenv(MERCURY_ENV_VAR, env)
        else:
            monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        with pytest.raises(MercuryProductionConfigError) as exc_info:
            MercuryVoice(enable_llm=True, llm_provider="mock")
        assert "MockLLMAdapter" in str(exc_info.value)


class TestVoiceLLMUnknownProvider:
    """An unsupported provider name surfaces a clean ValueError."""

    def test_unknown_provider_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        with pytest.raises(ValueError) as exc_info:
            MercuryVoice(enable_llm=True, llm_provider="not-a-real-provider")
        message = str(exc_info.value)
        assert "not-a-real-provider" in message
        # Must list at least one real provider so the operator knows
        # what to switch to.
        assert "huggingface" in message


class TestVoiceLLMConfiguredProvider:
    """Behaviour when ``llm_provider`` is supported but may fail to load."""

    def test_provider_init_failure_degrades_in_development(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If the underlying adapter raises, dev mode warns and continues."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise ImportError("transformers not installed")

        with (
            patch(
                "omni_mercury_engine.models.foundation.llm_adapter.create_llm_detector",
                side_effect=_boom,
            ),
            caplog.at_level("WARNING", logger="omni_mercury_engine.narrative.voice"),
        ):
            voice = MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_model_name="facebook/bart-large-mnli",
                llm_revision=_HF_REVISION,
            )

        assert voice._llm_adapter is None
        assert any(
            "huggingface" in r.message and "template-only" in r.message for r in caplog.records
        )

    def test_huggingface_remote_model_requires_revision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Remote HuggingFace IDs need a SafeHFLoader revision pin."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        with caplog.at_level("WARNING", logger="omni_mercury_engine.narrative.voice"):
            voice = MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_model_name="facebook/bart-large-mnli",
            )

        assert voice._llm_adapter is None
        assert any("llm_revision" in r.message for r in caplog.records)

    def test_huggingface_provider_requires_model_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Missing HuggingFace model names get a direct, actionable error."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        with caplog.at_level("WARNING", logger="omni_mercury_engine.narrative.voice"):
            voice = MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_revision=_HF_REVISION,
            )

        assert voice._llm_adapter is None
        assert any("llm_model_name" in r.message for r in caplog.records)

    def test_provider_init_failure_fails_closed_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise ImportError("transformers not installed")

        with (
            patch(
                "omni_mercury_engine.models.foundation.llm_adapter.create_llm_detector",
                side_effect=_boom,
            ),
            pytest.raises(MercuryProductionConfigError) as exc_info,
        ):
            MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_model_name="facebook/bart-large-mnli",
                llm_revision=_HF_REVISION,
            )

        message = str(exc_info.value)
        assert "huggingface" in message
        assert "production" in message

    def test_successful_init_stores_adapter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A working provider populates ``_llm_adapter`` and ``llm_enabled``."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)

        class _FakeAdapter:
            pass

        class _FakeDetector:
            adapter = _FakeAdapter()

        with patch(
            "omni_mercury_engine.models.foundation.llm_adapter.create_llm_detector",
            return_value=_FakeDetector(),
        ):
            voice = MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_model_name="facebook/bart-large-mnli",
                llm_revision=_HF_REVISION,
            )

        assert isinstance(voice._llm_adapter, _FakeAdapter)
        stats = voice.get_statistics()
        assert stats["llm_enabled"] is True

    def test_unavailable_adapter_degrades_in_development(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A constructed-but-unavailable adapter is not accepted as enabled."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)

        class _UnavailableAdapter:
            def is_available(self) -> bool:
                return False

        class _FakeDetector:
            adapter = _UnavailableAdapter()

        with (
            patch(
                "omni_mercury_engine.models.foundation.llm_adapter.create_llm_detector",
                return_value=_FakeDetector(),
            ),
            caplog.at_level("WARNING", logger="omni_mercury_engine.narrative.voice"),
        ):
            voice = MercuryVoice(
                enable_llm=True,
                llm_provider="huggingface",
                llm_model_name="facebook/bart-large-mnli",
                llm_revision=_HF_REVISION,
            )

        assert voice._llm_adapter is None
        assert any("unavailable" in r.message for r in caplog.records)

    def test_ollama_base_url_reaches_adapter_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``llm_base_url`` configures the Ollama host/port, not a dead kwarg."""
        from omni_mercury_engine.models.foundation.ollama_adapter import OllamaLLMAdapter

        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)

        detector = create_llm_detector(
            provider="ollama",
            model_name="llama3.2:3b",
            base_url="http://ollama.internal:11435",
        )

        assert isinstance(detector.adapter, OllamaLLMAdapter)
        ollama_config = detector.adapter.ollama_config
        assert ollama_config.host == "ollama.internal"
        assert ollama_config.port == 11435

    def test_create_llm_detector_huggingface_requires_model_name(self) -> None:
        """The low-level factory does not substitute cross-provider placeholders."""
        with pytest.raises(ValueError, match="model_name"):
            create_llm_detector(
                provider="huggingface",
                revision=_HF_REVISION,
            )

    @pytest.mark.parametrize(
        "provider",
        ["ollama", "openai", "anthropic", "xai", "gemini", "cohere", "deepseek", "cursor"],
    )
    def test_create_llm_detector_real_provider_requires_model_name(self, provider: str) -> None:
        """Every real provider must demand an explicit model_name.

        The previous ``or "gpt-4o"`` fallback silently substituted a
        cross-provider placeholder, which (for Ollama) made the
        per-adapter ``llama3.2:3b`` default unreachable and (for cloud
        adapters) masked the missing-configuration error until adapter
        construction.
        """
        with pytest.raises(ValueError, match="model_name"):
            create_llm_detector(provider=provider)

    def test_create_llm_detector_template_allows_omitted_model_name(self) -> None:
        """TemplateLLMAdapter is deterministic-offline and ignores model_name."""
        detector = create_llm_detector(provider="template")
        # The factory routes through the implemented-provider switch and
        # returns a real detector wired to TemplateLLMAdapter.
        from omni_mercury_engine.models.foundation.ollama_adapter import (
            TemplateLLMAdapter,
        )

        assert isinstance(detector.adapter, TemplateLLMAdapter)


class TestCreateMercuryVoiceFactory:
    """Factory function forwards every LLM kwarg to the constructor."""

    def test_factory_forwards_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        voice = create_mercury_voice(
            enable_llm=True,
            llm_provider=None,  # explicit no-provider, dev warn path
        )
        assert voice._llm_adapter is None
        assert voice.enable_llm is True

    def test_factory_default_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        voice = create_mercury_voice()
        assert voice.enable_llm is False
        assert voice._llm_adapter is None

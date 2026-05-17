"""Tests for the Mercury production-mode env primitive.

Lives at ``tests/test_env.py`` rather than ``tests/security/`` because
:mod:`omni_mercury_engine._env` is a generic process-level primitive
shared by every "refuse to silently degrade" call site, not a
security-specific module.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine._env import (
    MERCURY_ENV_DEVELOPMENT,
    MERCURY_ENV_PRODUCTION,
    MERCURY_ENV_VAR,
    MercuryProductionConfigError,
    get_mercury_env,
    is_production,
    require_real_component,
)


class TestGetMercuryEnv:
    """Defaulting, normalisation, and validation of ``MERCURY_ENV``."""

    def test_unset_defaults_to_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        assert get_mercury_env() == MERCURY_ENV_DEVELOPMENT
        assert is_production() is False

    def test_empty_string_defaults_to_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, "")
        assert get_mercury_env() == MERCURY_ENV_DEVELOPMENT

    def test_whitespace_only_defaults_to_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, "   ")
        assert get_mercury_env() == MERCURY_ENV_DEVELOPMENT

    def test_production_is_recognised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)
        assert get_mercury_env() == MERCURY_ENV_PRODUCTION
        assert is_production() is True

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, "Production")
        assert get_mercury_env() == MERCURY_ENV_PRODUCTION

    def test_whitespace_tolerant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, "  production  ")
        assert get_mercury_env() == MERCURY_ENV_PRODUCTION

    def test_unknown_value_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo like ``MERCURY_ENV=prod`` must fail loudly, not fall through."""
        monkeypatch.setenv(MERCURY_ENV_VAR, "prod")
        with pytest.raises(MercuryProductionConfigError) as exc_info:
            get_mercury_env()
        message = str(exc_info.value)
        assert MERCURY_ENV_VAR in message
        assert "prod" in message
        assert MERCURY_ENV_PRODUCTION in message
        assert MERCURY_ENV_DEVELOPMENT in message

    def test_env_is_read_at_each_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests must be able to flip the env without re-importing."""
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        assert get_mercury_env() == MERCURY_ENV_DEVELOPMENT
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)
        assert get_mercury_env() == MERCURY_ENV_PRODUCTION


class TestRequireRealComponent:
    """``require_real_component`` is no-op in dev, fail-closed in prod."""

    def test_noop_in_development(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(MERCURY_ENV_VAR, raising=False)
        # No raise.
        require_real_component("test component", remediation="set llm_provider")

    def test_raises_in_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MERCURY_ENV_VAR, MERCURY_ENV_PRODUCTION)
        with pytest.raises(MercuryProductionConfigError) as exc_info:
            require_real_component(
                "narrative LLM provider",
                remediation="pass llm_provider= to MercuryVoice()",
            )
        message = str(exc_info.value)
        assert "narrative LLM provider" in message
        assert "pass llm_provider= to MercuryVoice()" in message
        assert MERCURY_ENV_PRODUCTION in message

    def test_production_config_error_is_runtime_error(self) -> None:
        """Existing fail-closed gates raise ``RuntimeError`` subclasses; match."""
        assert issubclass(MercuryProductionConfigError, RuntimeError)

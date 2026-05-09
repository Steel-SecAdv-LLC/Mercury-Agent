"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Regression tests asserting that mock fallbacks are gone — every mocked
component path that previously degraded silently must now raise
``NotImplementedError`` at construction instead.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# MockLLMAdapter
# ---------------------------------------------------------------------------


def test_mock_llm_adapter_raises() -> None:
    from omni_mercury_engine.models.foundation.llm_adapter import MockLLMAdapter

    with pytest.raises(NotImplementedError, match="MockLLMAdapter cannot be used"):
        MockLLMAdapter()


# ---------------------------------------------------------------------------
# MockLVLMBackend
# ---------------------------------------------------------------------------


def test_mock_lvlm_backend_initialize_raises() -> None:
    """MockLVLMBackend.initialize() raises NotImplementedError."""
    from omni_mercury_engine.detectors.vlm.lvlm_backends import MockLVLMBackend

    backend = object.__new__(MockLVLMBackend)
    with pytest.raises(NotImplementedError, match="MockLVLMBackend cannot be used"):
        backend.initialize()


# ---------------------------------------------------------------------------
# Financial stub — no silent fallback
# ---------------------------------------------------------------------------


def test_financial_service_alpha_vantage_no_key_raises() -> None:
    import os

    # Ensure no env key leaks in
    env_backup = os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
    try:
        from omni_mercury_engine.integrations.stubs.financial import (
            FinancialAPIProvider,
            FinancialService,
        )

        with pytest.raises(NotImplementedError, match="Alpha Vantage requires an API key"):
            FinancialService(provider=FinancialAPIProvider.ALPHA_VANTAGE)
    finally:
        if env_backup is not None:
            os.environ["ALPHA_VANTAGE_API_KEY"] = env_backup


def test_financial_service_no_fallback_to_stub_param() -> None:
    """The ``fallback_to_stub`` parameter was removed."""
    import inspect

    from omni_mercury_engine.integrations.stubs.financial import FinancialService

    sig = inspect.signature(FinancialService.__init__)
    assert "fallback_to_stub" not in sig.parameters


# ---------------------------------------------------------------------------
# Weather stub — no silent fallback
# ---------------------------------------------------------------------------


def test_weather_service_openweathermap_no_key_raises() -> None:
    import os

    env_backup = os.environ.pop("OPENWEATHERMAP_API_KEY", None)
    try:
        from omni_mercury_engine.integrations.stubs.weather import (
            WeatherAPIProvider,
            WeatherService,
        )

        with pytest.raises(NotImplementedError, match="OpenWeatherMap requires an API key"):
            WeatherService(provider=WeatherAPIProvider.OPENWEATHERMAP)
    finally:
        if env_backup is not None:
            os.environ["OPENWEATHERMAP_API_KEY"] = env_backup


def test_weather_service_no_fallback_to_stub_param() -> None:
    """The ``fallback_to_stub`` parameter was removed."""
    import inspect

    from omni_mercury_engine.integrations.stubs.weather import WeatherService

    sig = inspect.signature(WeatherService.__init__)
    assert "fallback_to_stub" not in sig.parameters


# ---------------------------------------------------------------------------
# GOSNN sliding-window cure
# ---------------------------------------------------------------------------


def test_gosnn_normalize_insufficient_samples_raises() -> None:
    """Normalize raises RuntimeError when not enough samples collected."""
    import numpy as np

    from omni_mercury_engine.core.gosnn_3r_integration import SlidingWindowNormalizer

    norm = SlidingWindowNormalizer()
    with pytest.raises(RuntimeError, match="Sliding-window normalization requires"):
        norm.normalize(np.array([1.0, 2.0, 3.0]))

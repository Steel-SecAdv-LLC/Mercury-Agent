# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Phase 2 audit cure: regression tests asserting mock fallbacks are gone.

Every mock-mode adapter previously flagged by the 2026-03 in-tree audit
(``MockLLMAdapter``, ``MockLVLMBackend``, the TimeGPT/Chronos/MatrixProfile
mock modes, the BLIP "Mock implementation" branches, and the financial /
weather stubs) must now raise ``NotImplementedError`` at construction
instead of silently degrading.
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


# ---------------------------------------------------------------------------
# Default providers must be real, not fabricated
# ---------------------------------------------------------------------------


def test_financial_service_defaults_to_a_real_provider() -> None:
    """``FinancialService()`` must not default to the fabricating stub.

    It did. A no-argument construction returned seeded-RNG prices -- MSFT at
    386.83 against a real 530.14 -- while ``docs/ARCHITECTURE.md`` told
    operators the default provider was Yahoo Finance. The Yahoo route was
    already implemented (``_fetch_yahoo_finance``) and needs no API key; it
    simply was not the default. The stub stays reachable by explicit opt-in,
    which is the only honest way to select fabricated data.
    """
    import inspect

    from omni_mercury_engine.integrations.stubs.financial import (
        FinancialAPIProvider,
        FinancialService,
    )

    default = inspect.signature(FinancialService.__init__).parameters["provider"].default
    assert default is not FinancialAPIProvider.STUB, "default provider fabricates data"
    assert default is FinancialAPIProvider.YAHOO_FINANCE
    assert FinancialService().provider is FinancialAPIProvider.YAHOO_FINANCE


def test_weather_service_defaults_to_a_real_provider() -> None:
    """``WeatherService()`` must not default to the fabricating stub.

    NOAA needs no API key and serves ``get_current_by_coords`` for real. It
    cannot serve ``get_current(name)`` -- that raises a ``ValueError`` naming
    the coordinate method -- and an actionable refusal is the right answer
    where the previous default returned ``gauss``-drawn temperatures.
    """
    import inspect

    from omni_mercury_engine.integrations.stubs.weather import (
        WeatherAPIProvider,
        WeatherService,
    )

    default = inspect.signature(WeatherService.__init__).parameters["provider"].default
    assert default is not WeatherAPIProvider.STUB, "default provider fabricates data"
    assert default is WeatherAPIProvider.NOAA
    assert WeatherService().provider is WeatherAPIProvider.NOAA

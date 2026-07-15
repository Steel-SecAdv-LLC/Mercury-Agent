# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Ollama LLM Adapter and Fallback Chain.

Verifies offline-first operation, graceful degradation,
and air-gapped functionality.

Note: These tests import directly from the ollama_adapter module
to avoid pulling in torch dependencies through __init__.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is on path for direct imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Check for torch/numpy - skip all tests if not available
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

pytestmark = [
    pytest.mark.foundation,
    pytest.mark.skipif(
        not HAS_TORCH or not HAS_NUMPY,
        reason="Ollama adapter tests require torch and numpy",
    ),
]


# Import modules directly - torch/numpy are required (see skipif above)
@pytest.fixture(scope="module")
def ollama_module():
    """Import ollama_adapter module."""
    llm_module_name = "omni_mercury_engine.models.foundation.llm_adapter"
    ollama_module_name = "omni_mercury_engine.models.foundation.ollama_adapter"

    # Load llm_adapter first (dependency)
    if llm_module_name not in sys.modules:
        llm_path = src_path / "omni_mercury_engine" / "models" / "foundation" / "llm_adapter.py"
        if not llm_path.exists():
            pytest.skip(f"llm_adapter.py not found at {llm_path}")

        llm_spec = importlib.util.spec_from_file_location(llm_module_name, llm_path)
        if llm_spec is None or llm_spec.loader is None:
            pytest.skip("Could not load llm_adapter module spec")
        assert llm_spec is not None and llm_spec.loader is not None  # narrowing for mypy

        llm_adapter = importlib.util.module_from_spec(llm_spec)
        sys.modules[llm_module_name] = llm_adapter
        try:
            llm_spec.loader.exec_module(llm_adapter)
        except Exception as e:
            del sys.modules[llm_module_name]
            pytest.skip(f"Could not execute llm_adapter module: {e}")

    # Load ollama_adapter
    if ollama_module_name in sys.modules:
        return sys.modules[ollama_module_name]

    ollama_path = src_path / "omni_mercury_engine" / "models" / "foundation" / "ollama_adapter.py"
    if not ollama_path.exists():
        pytest.skip(f"ollama_adapter.py not found at {ollama_path}")

    ollama_spec = importlib.util.spec_from_file_location(ollama_module_name, ollama_path)
    if ollama_spec is None or ollama_spec.loader is None:
        pytest.skip("Could not load ollama_adapter module spec")
    assert ollama_spec is not None and ollama_spec.loader is not None  # narrowing for mypy

    ollama_adapter = importlib.util.module_from_spec(ollama_spec)
    sys.modules[ollama_module_name] = ollama_adapter
    try:
        ollama_spec.loader.exec_module(ollama_adapter)
    except Exception as e:
        del sys.modules[ollama_module_name]
        pytest.skip(f"Could not execute ollama_adapter module: {e}")

    return ollama_adapter


@pytest.fixture(scope="module")
def llm_module():
    """Import llm_adapter module."""
    module_name = "omni_mercury_engine.models.foundation.llm_adapter"

    # Check if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    llm_path = src_path / "omni_mercury_engine" / "models" / "foundation" / "llm_adapter.py"
    if not llm_path.exists():
        pytest.skip(f"llm_adapter.py not found at {llm_path}")

    llm_spec = importlib.util.spec_from_file_location(module_name, llm_path)
    if llm_spec is None or llm_spec.loader is None:
        pytest.skip("Could not load llm_adapter module spec")
    assert llm_spec is not None and llm_spec.loader is not None  # narrowing for mypy

    llm_adapter = importlib.util.module_from_spec(llm_spec)
    # Register module in sys.modules BEFORE executing to allow dataclass decorator to work
    sys.modules[module_name] = llm_adapter
    try:
        llm_spec.loader.exec_module(llm_adapter)
    except Exception as e:
        # Clean up on failure
        del sys.modules[module_name]
        pytest.skip(f"Could not execute llm_adapter module: {e}")
    return llm_adapter


class TestOllamaConfig:
    """Tests for Ollama configuration."""

    def test_ollama_config_defaults(self, ollama_module: Any) -> None:
        """OllamaConfig defaults are vendor-neutral: no model id ships.

        Regression: the config used to default to a specific vendor's model
        ("llama3.2:3b"); Mercury now ships no default model for any provider,
        so an unset model is empty and the adapter reports itself
        unavailable until the operator names an installed model.
        """
        config = ollama_module.OllamaConfig()

        assert config.host == "localhost"
        assert config.port == 11434
        assert config.model == ""
        assert config.temperature == 0.1
        assert config.timeout == 60.0

    def test_ollama_config_custom(self, ollama_module: Any) -> None:
        """Test OllamaConfig with custom values."""
        config = ollama_module.OllamaConfig(
            host="192.168.1.100",
            port=8080,
            model="mistral:7b",
            temperature=0.3,
        )

        assert config.host == "192.168.1.100"
        assert config.port == 8080
        assert config.model == "mistral:7b"
        assert config.base_url == "http://192.168.1.100:8080"

    def test_ollama_config_base_url(self, ollama_module: Any) -> None:
        """Test base_url property generation."""
        config = ollama_module.OllamaConfig(host="myserver", port=12345)
        assert config.base_url == "http://myserver:12345"


class TestOllamaModel:
    """Tests for OllamaModel enum."""

    def test_model_enum_values(self, ollama_module: Any) -> None:
        """Test OllamaModel enum has expected models."""
        OllamaModel = ollama_module.OllamaModel

        # Check key models exist
        assert OllamaModel.LLAMA_3_2_3B == "llama3.2:3b"
        assert OllamaModel.MISTRAL_7B == "mistral:7b"
        assert OllamaModel.PHI_3_MINI == "phi3:mini"

    def test_all_models_are_strings(self, ollama_module: Any) -> None:
        """Test all model values are valid strings."""
        OllamaModel = ollama_module.OllamaModel

        for model in OllamaModel:
            assert isinstance(model.value, str)
            assert ":" in model.value or model.value.isalnum()


class TestModelProfile:
    """Tests for model profiles and capabilities."""

    def test_model_profiles_exist(self, ollama_module: Any) -> None:
        """Test MODEL_PROFILES dictionary exists."""
        MODEL_PROFILES = ollama_module.MODEL_PROFILES

        assert "llama3.2:3b" in MODEL_PROFILES
        assert "mistral:7b" in MODEL_PROFILES

    def test_model_profile_structure(self, ollama_module: Any) -> None:
        """Test model profile has correct structure."""
        MODEL_PROFILES = ollama_module.MODEL_PROFILES
        profile = MODEL_PROFILES["llama3.2:3b"]

        assert hasattr(profile, "name")
        assert hasattr(profile, "provider")
        assert hasattr(profile, "size_category")
        assert hasattr(profile, "reasoning_strength")
        assert hasattr(profile, "speed_rating")
        assert hasattr(profile, "offline_capable")
        assert hasattr(profile, "context_length")

    def test_all_profiles_offline_capable(self, ollama_module: Any) -> None:
        """Test all Ollama models are marked offline capable."""
        MODEL_PROFILES = ollama_module.MODEL_PROFILES

        for name, profile in MODEL_PROFILES.items():
            assert profile.offline_capable is True, f"{name} should be offline capable"


class TestOllamaLLMAdapter:
    """Tests for OllamaLLMAdapter."""

    def test_adapter_initialization(self, ollama_module: Any) -> None:
        """Test adapter can be initialized."""
        config = ollama_module.OllamaConfig()
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=config)

        assert adapter is not None
        assert adapter.ollama_config == config

    def test_adapter_unavailable_response(self, ollama_module: Any) -> None:
        """Test adapter returns valid JSON when unavailable."""
        # Use non-existent host to ensure unavailable
        config = ollama_module.OllamaConfig(host="nonexistent.invalid", port=99999)
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=config)

        assert adapter.is_available() is False

        response = adapter.generate("Test prompt")
        parsed = json.loads(response)

        assert "is_anomaly" in parsed
        assert "explanation" in parsed
        assert "fallback" in parsed["explanation"].lower()

    def test_adapter_get_model_info_known(self, ollama_module: Any) -> None:
        """Test get_model_info for known model."""
        config = ollama_module.OllamaConfig(model="llama3.2:3b")
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=config)

        info = adapter.get_model_info()

        assert info["name"] == "Llama 3.2 3B"
        assert info["offline_capable"] is True

    def test_adapter_get_model_info_unknown(self, ollama_module: Any) -> None:
        """Test get_model_info for unknown model."""
        config = ollama_module.OllamaConfig(model="custom:model")
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=config)

        info = adapter.get_model_info()

        assert info["name"] == "custom:model"
        assert info["offline_capable"] is True

    @patch("socket.socket")
    def test_adapter_availability_check(self, mock_socket: Any, ollama_module: Any) -> None:
        """Test availability check uses socket connection."""
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 1  # Connection failed
        mock_socket.return_value = mock_sock

        config = ollama_module.OllamaConfig()
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=config)

        assert adapter.is_available() is False


class TestModelEndpointEnv:
    """The Tier-0 ``MERCURY_MODEL_ENDPOINT`` env var configures the served backend."""

    def test_endpoint_sets_host_and_port(
        self, ollama_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_MODEL_ENDPOINT", "http://127.0.0.1:11500")
        monkeypatch.delenv("MERCURY_OLLAMA_HOST", raising=False)
        adapter = ollama_module.OllamaLLMAdapter()
        assert adapter.ollama_config.host == "127.0.0.1"
        assert adapter.ollama_config.port == 11500

    def test_bare_host_port_endpoint_parsed(
        self, ollama_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_MODEL_ENDPOINT", "127.0.0.1:11502")
        monkeypatch.delenv("MERCURY_OLLAMA_HOST", raising=False)
        adapter = ollama_module.OllamaLLMAdapter()
        assert adapter.ollama_config.host == "127.0.0.1"
        assert adapter.ollama_config.port == 11502

    def test_specific_ollama_host_wins_over_endpoint(
        self, ollama_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MERCURY_MODEL_ENDPOINT", "http://127.0.0.1:11500")
        monkeypatch.setenv("MERCURY_OLLAMA_HOST", "localhost")
        adapter = ollama_module.OllamaLLMAdapter()
        assert adapter.ollama_config.host == "localhost"  # specific var wins
        assert adapter.ollama_config.port == 11500  # port still from the endpoint

    def test_malformed_endpoint_port_is_ignored_not_fatal(
        self, ollama_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A bad port must NOT crash construction (and take the fallback chain
        # down) -- it is logged and ignored, keeping the config defaults.
        monkeypatch.setenv("MERCURY_MODEL_ENDPOINT", "localhost:99999")
        monkeypatch.delenv("MERCURY_OLLAMA_HOST", raising=False)
        adapter = ollama_module.OllamaLLMAdapter()  # must not raise
        assert adapter.ollama_config.port == 11434  # default retained


class TestTemplateLLMAdapter:
    """Tests for template-based fallback adapter."""

    def test_template_adapter_initialization(self, ollama_module: Any) -> None:
        """Test template adapter can be initialized."""
        adapter = ollama_module.TemplateLLMAdapter()

        assert adapter is not None
        assert adapter.is_available() is True

    def test_template_adapter_always_available(self, ollama_module: Any) -> None:
        """Test template adapter is always available."""
        adapter = ollama_module.TemplateLLMAdapter()
        assert adapter.is_available() is True

    def test_template_anomaly_detection(self, ollama_module: Any) -> None:
        """Test template adapter detects anomaly keywords."""
        adapter = ollama_module.TemplateLLMAdapter()

        # Test with anomaly indicators (must include "anomal" or "detect" to trigger anomaly path)
        response = adapter.generate("Detect anomaly in this spike in CPU usage")
        parsed = json.loads(response)

        assert parsed["is_anomaly"] is True
        assert parsed["anomaly_score"] > 0.5

        # Test without anomaly indicators
        response = adapter.generate("Detect anomaly in this normal data")
        parsed = json.loads(response)

        assert parsed["is_anomaly"] is False
        assert parsed["anomaly_score"] < 0.5

    def test_template_status_query(self, ollama_module: Any) -> None:
        """Test template adapter handles status queries."""
        adapter = ollama_module.TemplateLLMAdapter()
        response = adapter.generate("What is the system status?")
        parsed = json.loads(response)

        assert "status" in parsed
        assert parsed["mode"] == "offline_fallback"

    def test_template_greeting(self, ollama_module: Any) -> None:
        """Test template adapter handles greetings."""
        adapter = ollama_module.TemplateLLMAdapter()
        response = adapter.generate("Hello Mercury")

        assert "Mercury Agent" in response
        assert "offline" in response.lower()

    def test_template_help(self, ollama_module: Any) -> None:
        """Test template adapter handles help requests."""
        adapter = ollama_module.TemplateLLMAdapter()
        response = adapter.generate("Help me understand this system")

        assert "Mercury Agent" in response
        assert "Commands" in response or "anomaly" in response.lower()

    def test_template_unknown_query(self, ollama_module: Any) -> None:
        """Test template adapter handles unknown queries."""
        adapter = ollama_module.TemplateLLMAdapter()
        response = adapter.generate("Random query about nothing specific")

        assert "offline" in response.lower() or "template" in response.lower()


class TestFallbackLLMChain:
    """Tests for graceful fallback chain."""

    def test_chain_initialization(self, ollama_module: Any) -> None:
        """Test fallback chain can be initialized."""
        chain = ollama_module.FallbackLLMChain()

        assert chain is not None
        assert chain.is_available() is True

    def test_chain_always_available(self, ollama_module: Any) -> None:
        """Test chain is always available due to template fallback."""
        # Even with invalid Ollama config, chain should be available
        config = ollama_module.OllamaConfig(host="invalid.invalid", port=99999)
        chain = ollama_module.FallbackLLMChain(ollama_config=config)

        assert chain.is_available() is True

    def test_chain_status(self, ollama_module: Any) -> None:
        """Test chain provides status information."""
        chain = ollama_module.FallbackLLMChain()
        status = chain.get_chain_status()

        assert "active" in status
        assert "ollama" in status
        assert "cloud" in status
        assert "template" in status
        assert status["template"]["available"] is True

    def test_chain_falls_back_to_template(self, ollama_module: Any) -> None:
        """Test chain falls back to template when Ollama unavailable."""
        config = ollama_module.OllamaConfig(host="invalid.invalid", port=99999)
        chain = ollama_module.FallbackLLMChain(ollama_config=config)

        # Should fall back to template
        active = chain.get_active_adapter()
        assert active == "template"

    def test_chain_generate(self, ollama_module: Any) -> None:
        """Test chain can generate responses."""
        chain = ollama_module.FallbackLLMChain()
        response = chain.generate("What is my system status?")

        assert response is not None
        assert len(response) > 0

    def test_chain_refresh(self, ollama_module: Any) -> None:
        """Test chain can be refreshed."""
        chain = ollama_module.FallbackLLMChain()
        new_active = chain.refresh()

        assert new_active is not None


class TestModelConfiguration:
    """Tests for model configuration and selection."""

    def test_model_config_defaults_are_vendor_neutral(self, ollama_module: Any) -> None:
        """ModelConfiguration ships empty: Mercury has no default model.

        Regression: preferred/domain models used to default to a vendor list;
        they are now operator-populated (the documented EXAMPLE_* constants
        are reference material the operator opts into explicitly).
        """
        config = ollama_module.ModelConfiguration()

        assert config.preferred_models == []
        assert config.domain_models == {}
        assert config.require_offline is True
        assert config.get_model_for_domain("medical") == ""
        assert config.get_model_for_task("low") == ""

    def test_model_for_domain(self, ollama_module: Any) -> None:
        """Domain lookups serve the operator-populated mapping."""
        config = ollama_module.ModelConfiguration(
            preferred_models=list(ollama_module.EXAMPLE_OLLAMA_PREFERRED_MODELS),
            domain_models=dict(ollama_module.EXAMPLE_OLLAMA_DOMAIN_MODELS),
        )

        medical_model = config.get_model_for_domain("medical")
        assert medical_model == "llama3.1:8b"

        code_model = config.get_model_for_domain("code")
        assert "code" in code_model.lower() or "deepseek" in code_model.lower()

        # Unknown domain falls to the operator's first preference.
        unknown = config.get_model_for_domain("unknown_domain")
        assert unknown == config.preferred_models[0]

    def test_model_for_task_complexity(self, ollama_module: Any) -> None:
        """Task-complexity selection serves the operator-populated preferences."""
        config = ollama_module.ModelConfiguration(
            preferred_models=list(ollama_module.EXAMPLE_OLLAMA_PREFERRED_MODELS),
        )

        low_model = config.get_model_for_task("low")
        high_model = config.get_model_for_task("high")

        assert low_model
        assert high_model


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_ollama_adapter(self, ollama_module: Any) -> None:
        """Test create_ollama_adapter factory."""
        adapter = ollama_module.create_ollama_adapter(model="llama3.2:1b")

        assert adapter is not None
        assert adapter.ollama_config.model == "llama3.2:1b"

    def test_create_fallback_chain(self, ollama_module: Any) -> None:
        """Test create_fallback_chain factory."""
        chain = ollama_module.create_fallback_chain(ollama_model="mistral:7b")

        assert chain is not None
        assert chain.ollama_config.model == "mistral:7b"


class TestOfflineOperation:
    """Tests specifically for offline/air-gapped operation."""

    def test_offline_anomaly_detection(self, ollama_module: Any) -> None:
        """Test anomaly detection works completely offline."""
        # Force template fallback (simulating offline)
        config = ollama_module.OllamaConfig(host="offline.invalid", port=99999)
        chain = ollama_module.FallbackLLMChain(ollama_config=config, enable_cloud=False)

        # Should still be able to detect anomalies (must include "anomal" or "detect" to trigger anomaly path)
        prompt = "Detect anomaly in this data: {score: 0.95, type: spike, severity: high}"
        response = chain.generate(prompt)

        parsed = json.loads(response)
        assert parsed["is_anomaly"] is True
        assert (
            "template" in parsed.get("category", "").lower()
            or "template" in chain.get_active_adapter()
        )

    def test_offline_status_query(self, ollama_module: Any) -> None:
        """Test status queries work offline."""
        config = ollama_module.OllamaConfig(host="offline.invalid", port=99999)
        chain = ollama_module.FallbackLLMChain(ollama_config=config, enable_cloud=False)

        response = chain.generate("What is system status?")
        parsed = json.loads(response)

        assert parsed["status"] == "operational"
        assert parsed["mode"] == "offline_fallback"

    def test_no_network_required(self, ollama_module: Any) -> None:
        """Test that template mode requires no network."""
        adapter = ollama_module.TemplateLLMAdapter()

        # Should work without any network
        responses = [
            adapter.generate("Detect anomalies in this error log"),
            adapter.generate("What is the health status?"),
            adapter.generate("Hello"),
            adapter.generate("Help"),
        ]

        for response in responses:
            assert response is not None
            assert len(response) > 0

    def test_chain_graceful_degradation(self, ollama_module: Any) -> None:
        """Test chain degrades gracefully through levels."""
        # Level 1: No Ollama
        config = ollama_module.OllamaConfig(host="no-ollama.invalid", port=99999)
        chain = ollama_module.FallbackLLMChain(ollama_config=config, enable_cloud=False)

        status = chain.get_chain_status()
        assert status["ollama"]["available"] is False
        assert status["cloud"]["enabled"] is False
        assert status["template"]["available"] is True
        assert status["active"] == "template"

        # Chain should still generate responses
        response = chain.generate("Test query")
        assert response is not None


class TestLLMAdapterIntegration:
    """Integration tests with main LLM adapter system."""

    def test_llm_provider_enum_includes_ollama(self, llm_module: Any) -> None:
        """Test LLMProvider enum includes OLLAMA."""
        LLMProvider = llm_module.LLMProvider

        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.TEMPLATE == "template"

    def test_create_llm_detector_ollama(self, llm_module: Any) -> None:
        """Test create_llm_detector with ollama provider."""
        detector = llm_module.create_llm_detector(provider="ollama", model_name="llama3.2:1b")

        assert detector is not None

    def test_zero_shot_with_ollama(self, llm_module: Any, ollama_module: Any) -> None:
        """Test ZeroShotAnomalyDetector can use Ollama adapter."""
        ollama_config = ollama_module.OllamaConfig(host="test.invalid", port=99999)
        adapter = ollama_module.OllamaLLMAdapter(ollama_config=ollama_config)

        detector = llm_module.ZeroShotAnomalyDetector(
            config=llm_module.LLMConfig(provider=llm_module.LLMProvider.OLLAMA),
            adapter=adapter,
        )

        # Should still work (will use fallback response)
        result = detector.detect("Test data")

        assert "anomaly_score" in result
        assert "is_anomaly" in result

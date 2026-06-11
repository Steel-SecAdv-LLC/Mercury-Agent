# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the LLM model registry and the provider catalog drift gate.

The registry layer is pure configuration (importable without torch); only
the drift gate that pins the catalog to the shipped adapters imports the
adapter module, and skips cleanly when torch is absent.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.models.llm_registry import (
    KNOWN_CAPABILITIES,
    PROVIDER_CATALOG,
    LLMModelRegistry,
    LLMModelSpec,
)


def _spec(**overrides: object) -> LLMModelSpec:
    base: dict[str, object] = {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        "context_window": 200_000,
        "capabilities": frozenset({"chat", "tool_use"}),
    }
    base.update(overrides)
    return LLMModelSpec(**base)  # type: ignore[arg-type]


class TestLLMModelSpecValidation:
    """Specs must be internally consistent and provenance-carrying."""

    def test_valid_spec_constructs(self) -> None:
        spec = _spec()
        assert spec.key == "anthropic:claude-sonnet-4-5"

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            _spec(provider="not-a-provider")

    def test_unknown_capability_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown capabilities"):
            _spec(capabilities=frozenset({"chat", "tooluse"}))

    def test_nonpositive_context_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_window"):
            _spec(context_window=0)

    def test_price_without_provenance_rejected(self) -> None:
        """A declared price must carry the date it was checked."""
        with pytest.raises(ValueError, match="pricing_as_of"):
            _spec(input_cost_per_mtok=3.0)

    def test_price_with_malformed_date_rejected(self) -> None:
        with pytest.raises(ValueError):
            _spec(input_cost_per_mtok=3.0, pricing_as_of="June 2026")

    def test_price_with_provenance_accepted(self) -> None:
        spec = _spec(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0, pricing_as_of="2026-06-01")
        assert spec.input_cost_per_mtok == 3.0

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="input_cost_per_mtok"):
            _spec(input_cost_per_mtok=-1.0, pricing_as_of="2026-06-01")

    def test_to_dict_is_json_safe(self) -> None:
        payload = _spec().to_dict()
        assert payload["capabilities"] == ["chat", "tool_use"]


class TestLLMModelRegistry:
    """Registration and capability/cost-aware selection."""

    def test_register_get_list(self) -> None:
        registry = LLMModelRegistry()
        registry.register(_spec())
        assert registry.list_models() == ["anthropic:claude-sonnet-4-5"]
        assert registry.get("anthropic", "claude-sonnet-4-5").context_window == 200_000
        assert len(registry) == 1

    def test_duplicate_registration_rejected_unless_replace(self) -> None:
        registry = LLMModelRegistry()
        registry.register(_spec())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_spec())
        registry.register(_spec(context_window=100_000), replace=True)
        assert registry.get("anthropic", "claude-sonnet-4-5").context_window == 100_000

    def test_get_unregistered_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            LLMModelRegistry().get("openai", "missing")

    def _populated(self) -> LLMModelRegistry:
        registry = LLMModelRegistry()
        registry.register(
            _spec(
                provider="openai",
                model_id="cheap",
                context_window=128_000,
                capabilities=frozenset({"chat", "tool_use", "json_mode"}),
                input_cost_per_mtok=0.5,
                output_cost_per_mtok=1.5,
                pricing_as_of="2026-06-01",
            )
        )
        registry.register(
            _spec(
                provider="anthropic",
                model_id="strong",
                context_window=200_000,
                capabilities=frozenset({"chat", "tool_use", "vision"}),
                input_cost_per_mtok=3.0,
                output_cost_per_mtok=15.0,
                pricing_as_of="2026-06-01",
            )
        )
        registry.register(
            _spec(
                provider="ollama",
                model_id="local-unpriced",
                context_window=32_000,
                capabilities=frozenset({"chat"}),
            )
        )
        return registry

    def test_select_filters_capabilities_and_context(self) -> None:
        registry = self._populated()
        matches = registry.select(required_capabilities=("chat", "tool_use"), min_context=150_000)
        assert [s.key for s in matches] == ["anthropic:strong"]

    def test_select_orders_priced_cheapest_first_then_unpriced(self) -> None:
        registry = self._populated()
        matches = registry.select(required_capabilities=("chat",))
        assert [s.key for s in matches] == [
            "openai:cheap",
            "anthropic:strong",
            "ollama:local-unpriced",
        ]

    def test_cost_budget_excludes_unpriced_specs(self) -> None:
        """An unpriced spec cannot honestly satisfy a budget — exclude it."""
        registry = self._populated()
        matches = registry.select(max_input_cost_per_mtok=1.0)
        assert [s.key for s in matches] == ["openai:cheap"]

    def test_select_rejects_unknown_capability_loudly(self) -> None:
        """A typo'd capability must raise, not silently return nothing."""
        with pytest.raises(ValueError, match="unknown required capabilities"):
            self._populated().select(required_capabilities=("tooluse",))

    def test_select_one_returns_best_or_raises(self) -> None:
        registry = self._populated()
        assert registry.select_one(required_capabilities=("chat",)).key == "openai:cheap"
        with pytest.raises(LookupError, match="no registered model satisfies"):
            registry.select_one(min_context=10_000_000)


class TestProviderCatalogDriftGate:
    """The catalog must mirror the adapters Mercury actually ships."""

    def test_catalog_matches_implemented_providers(self) -> None:
        """Catalog keys == IMPLEMENTED_LLM_PROVIDERS values, exactly."""
        pytest.importorskip("torch")
        from omni_mercury_engine.models.foundation.llm_adapter import (
            IMPLEMENTED_LLM_PROVIDERS,
        )

        implemented = {provider.value for provider in IMPLEMENTED_LLM_PROVIDERS}
        assert set(PROVIDER_CATALOG) == implemented

    def test_catalog_facts_match_adapter_code(self) -> None:
        """Spot-check code-grounded facts against the adapter constants."""
        pytest.importorskip("torch")
        from omni_mercury_engine.models.foundation.ollama_adapter import (
            CursorAdapter,
            DeepSeekAdapter,
            XAIGrokAdapter,
        )

        assert PROVIDER_CATALOG["xai"].api_key_env_var == XAIGrokAdapter._PROVIDER_ENV_VAR
        assert PROVIDER_CATALOG["deepseek"].api_key_env_var == DeepSeekAdapter._PROVIDER_ENV_VAR
        assert (
            PROVIDER_CATALOG["cursor"].requires_explicit_base_url
            == CursorAdapter._REQUIRE_EXPLICIT_BASE_URL
        )
        # The one provider whose response carries no usage block.
        assert PROVIDER_CATALOG["huggingface"].reports_token_usage is False

    def test_known_capabilities_is_nonempty_vocabulary(self) -> None:
        assert "chat" in KNOWN_CAPABILITIES and "tool_use" in KNOWN_CAPABILITIES

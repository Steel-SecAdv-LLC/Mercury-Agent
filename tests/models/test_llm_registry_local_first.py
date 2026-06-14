# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""'Free and local first' ordering in LLMModelRegistry.select().

Local/builtin models are genuinely free (no per-token charge), so they sort
ahead of any paid cloud model, satisfy any budget, and win ties — while an
*undeclared cloud* price stays unknown and is excluded from budget queries
rather than assumed free.
"""

from __future__ import annotations

from omni_mercury_engine.models.llm_registry import LLMModelRegistry, LLMModelSpec


def _local_spec() -> LLMModelSpec:
    """An unpriced local (Ollama) model — treated as free."""
    return LLMModelSpec(
        provider="ollama",  # locality == "local" in PROVIDER_CATALOG
        model_id="llama3.2:3b",
        context_window=8_192,
        capabilities=frozenset({"chat"}),
    )


def _cloud_spec(model_id: str = "x", price: float = 3.0) -> LLMModelSpec:
    """A priced cloud model."""
    return LLMModelSpec(
        provider="openai",
        model_id=model_id,
        context_window=128_000,
        capabilities=frozenset({"chat"}),
        input_cost_per_mtok=price,
        output_cost_per_mtok=price * 2,
        pricing_as_of="2026-06-01",
    )


class TestLocalFirstOrdering:
    """select()/select_one() prefer free/local regardless of the offline flag."""

    def test_local_free_beats_paid_cloud(self) -> None:
        reg = LLMModelRegistry()
        reg.register(_cloud_spec())  # registered first
        reg.register(_local_spec())
        best = reg.select_one(required_capabilities=("chat",))
        assert best.provider == "ollama"  # free local wins despite registration order

    def test_local_satisfies_any_budget(self) -> None:
        reg = LLMModelRegistry()
        reg.register(_local_spec())  # unpriced local
        # A zero budget excludes any paid cloud, but local is free -> qualifies.
        out = reg.select(required_capabilities=("chat",), max_input_cost_per_mtok=0.0)
        assert [s.provider for s in out] == ["ollama"]

    def test_undeclared_cloud_excluded_from_budget_and_sorted_last(self) -> None:
        reg = LLMModelRegistry()
        reg.register(_local_spec())
        reg.register(
            LLMModelSpec(
                provider="anthropic",  # undeclared price -> unknown cost
                model_id="unpriced",
                context_window=200_000,
                capabilities=frozenset({"chat"}),
            )
        )
        # Budget query: only the free local qualifies (unknown cloud excluded).
        budgeted = reg.select(required_capabilities=("chat",), max_input_cost_per_mtok=10.0)
        assert [s.provider for s in budgeted] == ["ollama"]
        # No budget: local first, unknown-cost cloud last.
        ordered = reg.select(required_capabilities=("chat",))
        assert ordered[0].provider == "ollama"
        assert ordered[-1].provider == "anthropic"

    def test_cheapest_paid_before_costlier_paid(self) -> None:
        reg = LLMModelRegistry()
        reg.register(_cloud_spec(model_id="pricey", price=15.0))
        reg.register(_cloud_spec(model_id="cheap", price=1.0))
        ordered = reg.select(required_capabilities=("chat",))
        assert [s.model_id for s in ordered] == ["cheap", "pricey"]

    def test_local_wins_tie_against_zero_priced_cloud(self) -> None:
        # Two options at cost 0: a local model and a cloud model explicitly
        # priced 0.0. Local/builtin wins the tie.
        reg = LLMModelRegistry()
        reg.register(
            LLMModelSpec(
                provider="openai",
                model_id="free-promo",
                context_window=128_000,
                capabilities=frozenset({"chat"}),
                input_cost_per_mtok=0.0,
                output_cost_per_mtok=0.0,
                pricing_as_of="2026-06-01",
            )
        )
        reg.register(_local_spec())
        best = reg.select_one(required_capabilities=("chat",))
        assert best.provider == "ollama"

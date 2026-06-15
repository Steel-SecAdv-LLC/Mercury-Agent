# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for provider-reported LLM usage records and the usage ledger.

The accounting contract: counts come from provider payloads only (never
client-side estimates), unmetered calls are visible (``reported=False``),
aggregate totals are exact for the ledger's lifetime, and the per-call
history ring being bounded must not bend the totals.
"""

from __future__ import annotations

import threading

import pytest

from omni_mercury_engine.models.foundation.llm_usage import LLMUsage, UsageLedger


class TestLLMUsage:
    """Validation and derivation rules of a single usage record."""

    def test_total_derived_from_provider_reported_sides(self) -> None:
        """Prompt+completion without a total derives the provider-side sum."""
        usage = LLMUsage(provider="openai", model="m", prompt_tokens=11, completion_tokens=7)
        assert usage.total_tokens == 18

    def test_provider_reported_total_is_preserved(self) -> None:
        """An explicit provider total is never overwritten by the sum."""
        usage = LLMUsage(
            provider="gemini", model="m", prompt_tokens=10, completion_tokens=5, total_tokens=16
        )
        assert usage.total_tokens == 16

    def test_unreported_call_carries_no_counts(self) -> None:
        usage = LLMUsage(provider="huggingface", model="m", reported=False)
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
            None,
            None,
            None,
        )

    def test_unreported_with_counts_rejected(self) -> None:
        """reported=False with counts would be a contradiction — refuse it."""
        with pytest.raises(ValueError, match="reported=False"):
            LLMUsage(provider="openai", model="m", prompt_tokens=3, reported=False)

    @pytest.mark.parametrize("bad", [-1, 1.5, "7"])
    def test_invalid_counts_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            LLMUsage(provider="openai", model="m", prompt_tokens=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
    def test_bool_counts_rejected(self, field: str) -> None:
        """``bool`` is an ``int`` subclass; a stray True/False is not 1/0 tokens."""
        with pytest.raises(ValueError, match=field):
            LLMUsage(provider="openai", model="m", **{field: True})

    def test_empty_provider_or_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            LLMUsage(provider="", model="m")
        with pytest.raises(ValueError, match="model"):
            LLMUsage(provider="openai", model="")

    def test_to_dict_is_json_safe(self) -> None:
        payload = LLMUsage(provider="openai", model="m", prompt_tokens=1).to_dict()
        assert payload["provider"] == "openai"
        assert payload["reported"] is True


class TestUsageLedger:
    """Aggregation, bounded history, and thread-safety."""

    def test_totals_aggregate_across_models_and_providers(self) -> None:
        ledger = UsageLedger()
        ledger.record(LLMUsage(provider="openai", model="a", prompt_tokens=10, completion_tokens=2))
        ledger.record(LLMUsage(provider="openai", model="b", prompt_tokens=5, completion_tokens=5))
        ledger.record(
            LLMUsage(provider="anthropic", model="c", prompt_tokens=1, completion_tokens=1)
        )
        ledger.record(LLMUsage(provider="huggingface", model="d", reported=False))

        totals = ledger.totals()
        assert totals["calls"] == 4
        assert totals["unreported_calls"] == 1
        assert totals["prompt_tokens"] == 16
        assert totals["completion_tokens"] == 8
        assert totals["total_tokens"] == 24

        by_model = ledger.totals_by_model()
        assert by_model[("openai", "a")]["total_tokens"] == 12
        assert by_model[("huggingface", "d")] == {
            "calls": 1,
            "unreported_calls": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_global_totals_match_sum_of_per_model_breakdown(self) -> None:
        """The O(1) running global must equal the per-key breakdown summed.

        Guards the running-counter optimization against drift: ``totals()`` is
        served from a single global aggregate, so it must agree field-for-field
        with the independently maintained ``totals_by_model`` across many keys.
        """
        ledger = UsageLedger()
        for i in range(50):
            ledger.record(
                LLMUsage(
                    provider=f"p{i % 7}",
                    model=f"m{i % 5}",
                    prompt_tokens=i,
                    completion_tokens=2 * i,
                )
            )
        ledger.record(LLMUsage(provider="p0", model="unmetered", reported=False))

        summed = {
            "calls": 0,
            "unreported_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for agg in ledger.totals_by_model().values():
            for key in summed:
                summed[key] += agg[key]
        assert ledger.totals() == summed

    def test_recent_ring_is_bounded_but_totals_are_not(self) -> None:
        """Evicting old records from the ring must not change the totals."""
        ledger = UsageLedger(max_recent=3)
        for i in range(10):
            ledger.record(LLMUsage(provider="openai", model="m", prompt_tokens=1))
        assert len(ledger.recent()) == 3
        assert len(ledger) == 10
        assert ledger.totals()["prompt_tokens"] == 10
        assert ledger.totals()["calls"] == 10

    def test_recent_returns_newest_last(self) -> None:
        ledger = UsageLedger()
        ledger.record(LLMUsage(provider="openai", model="first", prompt_tokens=1))
        ledger.record(LLMUsage(provider="openai", model="second", prompt_tokens=1))
        assert [u.model for u in ledger.recent(1)] == ["second"]

    def test_invalid_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_recent"):
            UsageLedger(max_recent=0)

    def test_concurrent_recording_keeps_exact_totals(self) -> None:
        """N threads x M records: totals must be exact, not approximate."""
        ledger = UsageLedger(max_recent=8)
        n_threads, per_thread = 8, 200

        def work() -> None:
            for _ in range(per_thread):
                ledger.record(
                    LLMUsage(provider="openai", model="m", prompt_tokens=2, completion_tokens=3)
                )

        threads = [threading.Thread(target=work) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        totals = ledger.totals()
        assert totals["calls"] == n_threads * per_thread
        assert totals["prompt_tokens"] == 2 * n_threads * per_thread
        assert totals["completion_tokens"] == 3 * n_threads * per_thread
        assert len(ledger) == n_threads * per_thread

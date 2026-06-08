# Copyright (C) 2025 Steel Security Advisors LLC
"""Regression suite for ``CachedBenevolenceScorer``.

Acceptance criteria from the punch list (item 6):

(a) Ruleset-version bump invalidates the cache.
(b) Identical input hits the cache.
(c) Violations are never cached — positive cases always recompute.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from omni_mercury_engine.cognitive.benevolence_cache import (
    DEFAULT_CACHE_CAPACITY,
    CachedBenevolenceScorer,
)
from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalConstraintViolationError,
    EthicalScore,
)
from omni_mercury_engine.core import centralized_constants

# ---------------------------------------------------------------------------
# Test doubles — let us count how often the underlying scorer is invoked
# and force violation outcomes deterministically.
# ---------------------------------------------------------------------------


def _make_permissible_score(action: str, score_id_suffix: str = "ok") -> EthicalScore:
    """Build a stand-alone permissible EthicalScore for cache-only tests."""
    return EthicalScore(
        score_id=f"test_{score_id_suffix}",
        action=action,
        benevolence_score=0.995,
        harm_score=0.0,
        benefit_score=0.95,
        equity_score=0.95,
        long_term_score=0.95,
        is_permissible=True,
        principle_scores={},
        harm_breakdown={},
        benefit_breakdown={},
        explanation="synthetic permissible result for cache test",
        recommendations=[],
    )


class _CountingScorer(BenevolenceScorer):
    """Always-permissible synthetic scorer that counts enforce() invocations.

    Bypasses the real scoring pipeline because cache behaviour is independent
    of scoring math — the wrapper only branches on raise vs return.
    """

    def __init__(self) -> None:
        super().__init__(benevolence_threshold=0.99)
        self.enforce_calls = 0

    def enforce(self, action: str, context: dict[str, Any]) -> EthicalScore:
        self.enforce_calls += 1
        return _make_permissible_score(action, score_id_suffix=str(self.enforce_calls))


class _AlwaysViolatingScorer(BenevolenceScorer):
    """Synthetic scorer whose enforce() always raises."""

    def __init__(self) -> None:
        super().__init__(benevolence_threshold=0.99)
        self.enforce_calls = 0

    def enforce(self, action: str, context: dict[str, Any]) -> EthicalScore:
        self.enforce_calls += 1
        raise EthicalConstraintViolationError(
            action=action,
            score=0.10,
            threshold=self.benevolence_threshold,
        )


def _permissible_action() -> tuple[str, dict[str, object]]:
    """Stable action + context payload used across the cache tests."""
    return (
        "deliver humanitarian aid to affected population",
        {
            "sustainable": True,
            "stakeholders": ["civilians", "responders"],
            "consent": True,
            "evidence_based": True,
            "harm_potential": 0.0,
        },
    )


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_cache_default_capacity_constant() -> None:
    cache = CachedBenevolenceScorer(scorer=BenevolenceScorer())
    assert cache.capacity == DEFAULT_CACHE_CAPACITY


def test_cache_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        CachedBenevolenceScorer(scorer=BenevolenceScorer(), capacity=0)


def test_cache_threshold_passthrough() -> None:
    scorer = BenevolenceScorer(benevolence_threshold=0.99)
    cache = CachedBenevolenceScorer(scorer=scorer)
    assert cache.benevolence_threshold == scorer.benevolence_threshold


# ---------------------------------------------------------------------------
# (b) Identical input hits the cache
# ---------------------------------------------------------------------------


def test_identical_input_hits_cache_and_underlying_runs_once() -> None:
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    first = cache.enforce(action, ctx)
    second = cache.enforce(action, ctx)
    third = cache.enforce(action, dict(ctx))  # logically identical dict

    # The underlying scorer should have been hit exactly once across three calls.
    assert scorer.enforce_calls == 1
    assert isinstance(first, EthicalScore)
    assert second is first  # exact-object reuse from the cache
    assert third is first

    stats = cache.stats
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["size"] == 1
    assert stats["violations_uncached"] == 0


def test_different_input_misses_cache_independently() -> None:
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    cache.enforce(action, ctx)
    cache.enforce(action, {**ctx, "stakeholders": ["children", "elderly"]})

    assert scorer.enforce_calls == 2
    stats = cache.stats
    assert stats["misses"] == 2
    assert stats["hits"] == 0
    assert stats["size"] == 2


def test_canonicalisation_is_order_invariant_for_dict_keys() -> None:
    """A dict with the same content but different key insertion order is a hit."""
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, _ = _permissible_action()

    ctx_a = {"a": 1, "b": 2, "c": 3, "sustainable": True}
    ctx_b = {"sustainable": True, "c": 3, "b": 2, "a": 1}

    cache.enforce(action, ctx_a)
    cache.enforce(action, ctx_b)

    assert scorer.enforce_calls == 1
    assert cache.stats["hits"] == 1


# ---------------------------------------------------------------------------
# (a) Ruleset-version bump invalidates the cache
# ---------------------------------------------------------------------------


def test_ruleset_version_bump_invalidates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    original = centralized_constants.ETHICAL
    cache.enforce(action, ctx)
    assert scorer.enforce_calls == 1
    assert cache.stats["size"] == 1

    # Bump the ruleset version. Because EthicalConstants is a frozen dataclass,
    # we must replace the module-level singleton, which is exactly what an
    # operator deploying a new ruleset would do.
    bumped = replace(original, RULESET_VERSION=original.RULESET_VERSION + 1)
    monkeypatch.setattr(centralized_constants, "ETHICAL", bumped)
    try:
        # A repeat call with identical inputs must be a miss because the key
        # prefix changed; the stale entry must also be purged.
        cache.enforce(action, ctx)
        assert scorer.enforce_calls == 2, "ruleset-version bump did not invalidate the cache"
        stats = cache.stats
        # After invalidation + new insert, exactly one entry remains, all under
        # the new version.
        assert stats["size"] == 1
        assert stats["ruleset_version"] == bumped.RULESET_VERSION
    finally:
        monkeypatch.setattr(centralized_constants, "ETHICAL", original)


def test_ruleset_version_unchanged_does_not_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated lookups under the same version must not silently drop entries."""
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    cache.enforce(action, ctx)
    cache.enforce(action, ctx)
    cache.enforce(action, ctx)
    assert scorer.enforce_calls == 1
    assert cache.stats["size"] == 1


# ---------------------------------------------------------------------------
# (c) Violations are never cached
# ---------------------------------------------------------------------------


def test_violations_are_never_cached_and_always_recompute() -> None:
    scorer = _AlwaysViolatingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    for _ in range(5):
        with pytest.raises(EthicalConstraintViolationError):
            cache.enforce(action, ctx)

    # Every single call hit the underlying scorer; nothing was cached.
    assert scorer.enforce_calls == 5
    stats = cache.stats
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["violations_uncached"] == 5


def test_violation_then_recovery_is_cached_correctly() -> None:
    """If a previously-violating input later starts passing, the new positive
    is cached; subsequent calls hit. No stale violation lingers."""

    class _FlippingScorer(BenevolenceScorer):
        def __init__(self) -> None:
            super().__init__(benevolence_threshold=0.99)
            self.calls = 0
            self.permit_after = 2  # Calls 1 and 2 violate; calls ≥ 3 pass.

        def enforce(self, action: str, context: dict[str, Any]) -> EthicalScore:
            self.calls += 1
            if self.calls <= self.permit_after:
                raise EthicalConstraintViolationError(
                    action=action,
                    score=0.10,
                    threshold=self.benevolence_threshold,
                )
            return _make_permissible_score(action, score_id_suffix=f"flip_{self.calls}")

    scorer = _FlippingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    with pytest.raises(EthicalConstraintViolationError):
        cache.enforce(action, ctx)
    with pytest.raises(EthicalConstraintViolationError):
        cache.enforce(action, ctx)
    # Third call passes through and the result is now cached.
    first_pass = cache.enforce(action, ctx)
    second_pass = cache.enforce(action, ctx)

    assert second_pass is first_pass
    assert scorer.calls == 3, "permitting call must run; cached result must be served"
    assert cache.stats["violations_uncached"] == 2
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1


# ---------------------------------------------------------------------------
# Capacity enforcement (LRU)
# ---------------------------------------------------------------------------


def test_capacity_enforced_via_lru() -> None:
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer, capacity=3)
    action, base_ctx = _permissible_action()

    # Fill four distinct keys; the oldest must be evicted.
    cache.enforce(action, {**base_ctx, "id": 1})
    cache.enforce(action, {**base_ctx, "id": 2})
    cache.enforce(action, {**base_ctx, "id": 3})
    cache.enforce(action, {**base_ctx, "id": 4})

    assert cache.stats["size"] == 3

    # Re-querying id=1 (the evicted one) should miss; id=4 (still in cache) hits.
    cache.enforce(action, {**base_ctx, "id": 1})
    cache.enforce(action, {**base_ctx, "id": 4})

    assert cache.stats["hits"] == 1
    # 5 cold inserts (id=1..4 + id=1 again after eviction) plus 1 hit.
    assert scorer.enforce_calls == 5


def test_clear_drops_all_entries() -> None:
    scorer = _CountingScorer()
    cache = CachedBenevolenceScorer(scorer=scorer)
    action, ctx = _permissible_action()

    cache.enforce(action, ctx)
    cache.clear()
    cache.enforce(action, ctx)

    assert scorer.enforce_calls == 2
    assert cache.stats["size"] == 1

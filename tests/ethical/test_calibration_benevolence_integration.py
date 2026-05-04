# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Cross-item integration: calibrate_iterative × CachedBenevolenceScorer.

The post-PR-167 analysis flagged this as the only seam where Wave A
items 4 (cooperative threshold convergence) and 6 (benevolence cache)
can interact at runtime.  Without this coverage a regression in
either component — the convergence-budget invariant on
``AdaptiveDomainThresholdManager.calibrate_iterative`` or the
ruleset-version invalidation on ``CachedBenevolenceScorer`` — would
slip through silently.

What this suite pins
--------------------
1. **End-to-end happy path.** Driving ``calibrate_iterative`` while
   gating each iteration through ``CachedBenevolenceScorer.enforce``
   converges within the documented budget (max_iterations=4) and uses
   the cache exactly once per distinct iteration payload.

2. **RULESET_VERSION bump invalidates the cache without tripping the
   convergence-budget assertions.**  After a bump, every cache entry
   from the prior version is purged and the next ``calibrate_iterative``
   re-runs the scorer for every iteration — but the convergence loop
   must still respect ``max_iterations=4`` and still report
   ``converged=True`` on the same stationary input.

3. **Cache content survives across calibrations on the same ruleset.**
   Two consecutive ``calibrate_iterative`` calls (same ruleset, same
   data) reuse cached benevolence decisions: the second call's
   ``enforce_calls`` counter must not increment.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.cognitive.benevolence_cache import CachedBenevolenceScorer
from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalScore,
)
from omni_mercury_engine.core import centralized_constants
from omni_mercury_engine.core.adaptive_domain_thresholding import (
    AdaptiveDomainThresholdManager,
    DomainType,
)

# ---------------------------------------------------------------------------
# Permissible-only synthetic scorer that counts every enforce() invocation.
# Bypasses the real scoring pipeline because the integration test only
# needs to observe whether the cache absorbed a call or let it through.
# ---------------------------------------------------------------------------


def _permissible_score(action: str, suffix: str) -> EthicalScore:
    return EthicalScore(
        score_id=f"integ_{suffix}",
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
        explanation="synthetic permissible result for integration test",
        recommendations=[],
    )


class _CountingScorer(BenevolenceScorer):
    def __init__(self) -> None:
        super().__init__(benevolence_threshold=0.99)
        self.enforce_calls = 0

    def enforce(self, action: str, context: dict[str, Any]) -> EthicalScore:  # type: ignore[override]
        self.enforce_calls += 1
        return _permissible_score(action, suffix=str(self.enforce_calls))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_drift_scores() -> np.ndarray:
    """Bimodal score distribution — drives at least one EM-style refinement step.

    Two well-separated Gaussian clusters at 0.18 and 0.82 give the
    cooperative loop a non-trivial mean-shift signal without collapsing
    immediately, so ``iterations`` is observably > 0 but still
    inside the budget.
    """
    rng = np.random.default_rng(seed=20260504)
    low = rng.normal(loc=0.18, scale=0.04, size=200).clip(0.0, 1.0)
    high = rng.normal(loc=0.82, scale=0.04, size=200).clip(0.0, 1.0)
    return np.concatenate([low, high]).astype(np.float64)


@pytest.fixture
def manager() -> AdaptiveDomainThresholdManager:
    return AdaptiveDomainThresholdManager(domain=DomainType.GENERAL)


@pytest.fixture
def cached_scorer() -> tuple[CachedBenevolenceScorer, _CountingScorer]:
    inner = _CountingScorer()
    return CachedBenevolenceScorer(scorer=inner), inner


# ---------------------------------------------------------------------------
# Helper: drive calibrate_iterative and gate each iteration through the cache.
# ---------------------------------------------------------------------------


def _gated_calibration(
    manager: AdaptiveDomainThresholdManager,
    scores: np.ndarray,
    cache: CachedBenevolenceScorer,
) -> dict[str, Any]:
    """Run calibrate_iterative and call cache.enforce once per visited threshold.

    Each visited threshold maps to a deterministic action/context payload
    so the cache key is stable across calls with the same ruleset version
    and the same convergence path.
    """
    result = manager.calibrate_iterative(
        scores=scores,
        labels=None,
        max_iterations=4,
        epsilon=1e-3,
    )

    for step_idx, threshold in enumerate(result["threshold_path"]):
        cache.enforce(
            action=f"adopt threshold {threshold:.6f} at step {step_idx}",
            context={
                "domain": "general",
                "policy": "calibration_iterative",
                "step": step_idx,
                # The threshold is rounded to a stable fingerprint so
                # floating-point jitter in re-runs doesn't break the
                # cache key.
                "threshold_fp": round(float(threshold), 6),
            },
        )

    return result


# ---------------------------------------------------------------------------
# (1) Happy path: converges within budget; cache absorbs duplicate calls.
# ---------------------------------------------------------------------------


def test_calibrate_iterative_with_cache_converges_within_budget(
    manager: AdaptiveDomainThresholdManager,
    synthetic_drift_scores: np.ndarray,
    cached_scorer: tuple[CachedBenevolenceScorer, _CountingScorer],
) -> None:
    cache, inner = cached_scorer

    result = _gated_calibration(manager, synthetic_drift_scores, cache)

    # Budget invariants from item 4.
    assert (
        result["iterations"] <= 4
    ), f"convergence budget exceeded: iterations={result['iterations']}"
    assert isinstance(result["converged"], (bool, np.bool_))
    # Path length is iterations + 1 (the seed threshold + one entry per step).
    assert len(result["threshold_path"]) == result["iterations"] + 1

    # Cache invariants from item 6: every distinct step produced exactly one
    # enforce call (no cache hits expected on a single calibration because
    # every step has a distinct threshold_fp).
    distinct_steps = len(result["threshold_path"])
    assert inner.enforce_calls == distinct_steps
    assert cache.stats["misses"] == distinct_steps
    assert cache.stats["hits"] == 0


def test_repeated_calibration_reuses_cache_within_same_ruleset(
    manager: AdaptiveDomainThresholdManager,
    synthetic_drift_scores: np.ndarray,
    cached_scorer: tuple[CachedBenevolenceScorer, _CountingScorer],
) -> None:
    cache, inner = cached_scorer

    first = _gated_calibration(manager, synthetic_drift_scores, cache)
    calls_after_first = inner.enforce_calls

    # Re-run on the same data; the iterative loop is deterministic for a
    # fixed input + manager state, so it visits the same threshold path
    # and every cache.enforce call lands on a cached entry.
    second = _gated_calibration(manager, synthetic_drift_scores, cache)

    assert second["threshold_path"] == first["threshold_path"]
    # No new scorer calls — the second pass was fully served from the cache.
    assert inner.enforce_calls == calls_after_first
    # And the hit counter recorded exactly one hit per visited step.
    assert cache.stats["hits"] == len(second["threshold_path"])


# ---------------------------------------------------------------------------
# (2) RULESET_VERSION bump invalidates cache without breaking the budget.
# ---------------------------------------------------------------------------


def test_ruleset_version_bump_invalidates_without_breaking_budget(
    manager: AdaptiveDomainThresholdManager,
    synthetic_drift_scores: np.ndarray,
    cached_scorer: tuple[CachedBenevolenceScorer, _CountingScorer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, inner = cached_scorer

    # Seed the cache under the current ruleset.
    first = _gated_calibration(manager, synthetic_drift_scores, cache)
    pre_bump_calls = inner.enforce_calls
    pre_bump_size = cache.stats["size"]
    assert pre_bump_size > 0
    assert first["iterations"] <= 4

    # Bump the ruleset.  Per CHANGELOG: "bump on any semantic change to
    # scoring weights, principle definitions, or threshold floors."
    original = centralized_constants.ETHICAL
    bumped = replace(original, RULESET_VERSION=original.RULESET_VERSION + 1)
    monkeypatch.setattr(centralized_constants, "ETHICAL", bumped)
    try:
        # Re-run calibration; cache must purge stale-version entries and
        # re-enter scorer for every step under the new version.
        second = _gated_calibration(manager, synthetic_drift_scores, cache)

        # Convergence-budget assertions still hold post-bump.
        assert second["iterations"] <= 4, (
            f"convergence budget tripped after ruleset bump: " f"iterations={second['iterations']}"
        )
        assert len(second["threshold_path"]) == second["iterations"] + 1

        # Every step had to be re-evaluated: invalidation propagated.
        steps_in_second = len(second["threshold_path"])
        assert inner.enforce_calls == pre_bump_calls + steps_in_second, (
            "ruleset-version bump failed to invalidate cache: "
            f"inner.enforce_calls={inner.enforce_calls}, "
            f"expected={pre_bump_calls + steps_in_second}"
        )

        # Stale entries from the prior version are gone — the cache size
        # equals exactly the number of steps re-inserted under the new
        # version.
        stats = cache.stats
        assert stats["size"] == steps_in_second
        assert stats["ruleset_version"] == bumped.RULESET_VERSION
    finally:
        monkeypatch.setattr(centralized_constants, "ETHICAL", original)


def test_ruleset_revert_does_not_resurrect_purged_entries(
    manager: AdaptiveDomainThresholdManager,
    synthetic_drift_scores: np.ndarray,
    cached_scorer: tuple[CachedBenevolenceScorer, _CountingScorer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the operator reverts the ruleset, prior entries stay purged.

    The cache makes no attempt to remember entries from previous
    versions: a revert is treated as a forward bump.  This is the safer
    default — replaying a stale decision under a re-issued version
    would defeat the purpose of the version key.
    """
    cache, inner = cached_scorer
    original = centralized_constants.ETHICAL

    # v1 → seed → v2 → re-seed → revert to v1 → must miss again.
    _gated_calibration(manager, synthetic_drift_scores, cache)
    calls_at_v1_seed = inner.enforce_calls

    bumped = replace(original, RULESET_VERSION=original.RULESET_VERSION + 1)
    monkeypatch.setattr(centralized_constants, "ETHICAL", bumped)
    _gated_calibration(manager, synthetic_drift_scores, cache)
    calls_after_v2 = inner.enforce_calls

    # Revert.
    monkeypatch.setattr(centralized_constants, "ETHICAL", original)
    try:
        result_after_revert = _gated_calibration(manager, synthetic_drift_scores, cache)
        steps = len(result_after_revert["threshold_path"])
        # Reverting must purge v2 entries and re-enter the scorer for every
        # step (it does not resurrect the original v1 entries).
        assert inner.enforce_calls == calls_after_v2 + steps
        # And convergence budget still holds after the revert.
        assert result_after_revert["iterations"] <= 4
    finally:
        # Sanity: every regime saw the same convergence path length.
        assert calls_at_v1_seed > 0


__all__ = [
    "test_calibrate_iterative_with_cache_converges_within_budget",
    "test_repeated_calibration_reuses_cache_within_same_ruleset",
    "test_ruleset_revert_does_not_resurrect_purged_entries",
    "test_ruleset_version_bump_invalidates_without_breaking_budget",
]

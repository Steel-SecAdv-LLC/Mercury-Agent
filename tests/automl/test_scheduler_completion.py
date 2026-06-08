# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

from omni_mercury_engine.automl.schedulers import ASHAScheduler, HyperbandBracket


def test_hyperband_bracket_mark_complete() -> None:
    bracket = HyperbandBracket(
        s=3,
        n_configs=8,
        budget=1.0,
        eta=3.0,
        max_budget=81.0,
    )
    bracket.add_result("trial_1", metric=0.5, budget=1.0)
    bracket.mark_complete("trial_1")
    assert not bracket.should_promote("trial_1")


def test_hyperband_bracket_completed_set_tracks() -> None:
    bracket = HyperbandBracket(
        s=3,
        n_configs=8,
        budget=1.0,
        eta=3.0,
        max_budget=81.0,
    )
    bracket.mark_complete("trial_x")
    assert "trial_x" in bracket._completed


def test_asha_on_trial_complete() -> None:
    scheduler = ASHAScheduler(max_budget=81, min_budget=1, reduction_factor=3)
    scheduler._trial_budgets["trial_1"] = 1.0
    scheduler.on_trial_complete("trial_1")
    assert "trial_1" not in scheduler._trial_budgets


def test_asha_on_trial_complete_missing_id() -> None:
    scheduler = ASHAScheduler(max_budget=81, min_budget=1, reduction_factor=3)
    # Should not raise when trial_id is not tracked
    scheduler.on_trial_complete("nonexistent")

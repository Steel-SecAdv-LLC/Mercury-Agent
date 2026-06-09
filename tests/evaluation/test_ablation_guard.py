# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the reusable ablation confound guard (WS-B follow-on).

Guards the exact failure that produced a spurious +0.48 "KEEP" in PR #262: a
paired ROC-AUC delta where one arm collapsed to an inverted ranking (AUC < 0.5).
"""

from __future__ import annotations

from omni_mercury_engine.evaluation.ablation_guard import (
    INVERSION_FLOOR,
    check_ablation_confound,
    confound_free_or_quarantine,
)


def test_clean_comparison_is_not_confounded() -> None:
    report = check_ablation_confound([0.80, 0.82, 0.79], [0.81, 0.83, 0.80])
    assert report.confounded is False
    assert report.reasons == []
    assert report.n_pairs == 3


def test_inverted_baseline_is_confounded() -> None:
    """The historical confound: baseline collapses (AUC ~0.05), treatment ~0.5,
    delta ~+0.45 -- must be flagged, not celebrated."""
    report = check_ablation_confound([0.05, 0.02, 0.10], [0.50, 0.52, 0.48])
    assert report.confounded is True
    assert report.n_degenerate_baseline == 3
    assert any("baseline arm inverted" in r for r in report.reasons)


def test_inverted_treatment_is_confounded() -> None:
    report = check_ablation_confound([0.85, 0.86, 0.84], [0.40, 0.30, 0.45])
    assert report.confounded is True
    assert report.n_degenerate_treatment >= 2


def test_single_inverted_seed_confounds_under_strict_default() -> None:
    # Default max_degenerate_fraction=0.0 -> any inverted seed confounds.
    report = check_ablation_confound([0.9, 0.9, 0.2], [0.9, 0.9, 0.9])
    assert report.confounded is True


def test_tolerant_fraction_allows_one_noisy_seed() -> None:
    report = check_ablation_confound([0.9, 0.9, 0.2], [0.9, 0.9, 0.9], max_degenerate_fraction=0.34)
    assert report.confounded is False


def test_empty_input_is_confounded() -> None:
    report = check_ablation_confound([], [])
    assert report.confounded is True


def test_confound_downgrades_keep_to_quarantine() -> None:
    report = check_ablation_confound([0.05, 0.05, 0.05], [0.50, 0.50, 0.50])
    final, note = confound_free_or_quarantine(cleared=True, report=report)
    assert final is False
    assert "FORCED QUARANTINE" in note


def test_clean_keep_is_preserved() -> None:
    report = check_ablation_confound([0.80, 0.80, 0.80], [0.84, 0.84, 0.84])
    final, note = confound_free_or_quarantine(cleared=True, report=report)
    assert final is True
    assert "KEEP" in note


def test_clean_quarantine_is_preserved() -> None:
    report = check_ablation_confound([0.80, 0.80, 0.80], [0.80, 0.80, 0.80])
    final, note = confound_free_or_quarantine(cleared=False, report=report)
    assert final is False
    assert "does not clear the bar" in note


def test_inversion_floor_is_sane() -> None:
    assert 0.0 < INVERSION_FLOOR < 0.5

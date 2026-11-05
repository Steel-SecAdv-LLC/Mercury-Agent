# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every published PPO margin figure, recomputed from the per-seed data.

The streaming-alarm-policy checkpoint ships a provenance sidecar whose
``baseline_comparator`` block states how thin the PPO policy's advantage over
the best single-feature baseline is. Those figures were hand-written. This
module recomputes each one from ``benchmarks/streaming_alarm_policy_results.json``
so the record cannot drift from the data it describes, and so a future retrain
that changes the margins fails here instead of leaving a stale claim shipping
inside the checkpoint.

Verified when this module was added: ``per_fold_seed_cells_lost_to_baseline``
(4 of 15, 4 of 12 excluding the dev fold), ``thinnest_per_seed_margin_without_dev_fold``
(+1.50), ``thinnest_per_seed_margin_mean_of_per_fold`` (+0.08) and
``margin_without_dev_fold`` (+3.86) all reproduce exactly.

``thinnest_per_fold_margin`` (0.55) did **not** mean what its name suggests. It
is the thinnest margin per fold *after averaging the three seeds*, which is a
real quantity but not a floor on any run: the worst single fold-seed cell is
**-7.49**. The sidecar now says so and carries the worst cell beside it, and
the distinction is pinned below, because "the thinnest margin is +0.55" invites
exactly the reading the data does not support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "benchmarks" / "streaming_alarm_policy_results.json"
_PROVENANCE = (
    _ROOT
    / "src"
    / "omni_mercury_engine"
    / "models"
    / "checkpoints"
    / "streaming_alarm_policy.provenance.json"
)


@pytest.fixture(scope="module")
def evidence() -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], float]]:
    """``(results, comparator_block, per-cell margins)``."""
    results = json.loads(_RESULTS.read_text(encoding="utf-8"))
    comparator = json.loads(_PROVENANCE.read_text(encoding="utf-8"))["baseline_comparator"]
    baseline = results["baselines"][results["best_single_feature_chosen_on_held_out_folds"]]
    cells = {
        (seed, fold): results["ppo_by_seed"][seed]["per_fold"][fold] - baseline["per_fold"][fold]
        for seed in results["ppo_by_seed"]
        for fold in sorted(baseline["per_fold"])
    }
    return results, comparator, cells


def test_the_published_margin_is_the_clean_mean_minus_the_baseline(
    evidence: tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], float]],
) -> None:
    """``margin_without_dev_fold`` is arithmetic on two published scores."""
    results, comparator, _ = evidence
    baseline = results["baselines"][results["best_single_feature_chosen_on_held_out_folds"]]
    expected = results["ppo_mean_score_without_dev_fold"] - baseline["score_without_dev_fold"]
    assert comparator["margin_without_dev_fold"] == pytest.approx(expected, abs=0.01)


def test_the_lost_cell_count_matches_the_per_seed_data(
    evidence: tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], float]],
) -> None:
    """``4 of 15 (4 of 12 excluding the dev fold)`` recomputed.

    This is the figure that makes the margin's thinness legible, so it is the
    one most worth holding to the data.
    """
    results, comparator, cells = evidence
    dev = results["dev_fold"]
    lost = sum(1 for margin in cells.values() if margin <= 0)
    lost_excluding_dev = sum(
        1 for (_, fold), margin in cells.items() if margin <= 0 and fold != dev
    )
    total = len(cells)
    total_excluding_dev = sum(1 for (_, fold) in cells if fold != dev)
    assert comparator["per_fold_seed_cells_lost_to_baseline"] == (
        f"{lost} of {total} ({lost_excluding_dev} of {total_excluding_dev} "
        "excluding the dev fold)"
    )


def test_the_thinnest_per_seed_margins_match_both_readings(
    evidence: tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], float]],
) -> None:
    """Both the pooled and the mean-of-per-fold readings are published, and both check.

    The sidecar states both deliberately: pooling the NAB score across folds
    gives seed 0 a +1.50 margin, while averaging that seed's per-fold margins
    gives +0.08. Publishing only the first would read as more robust than the
    data supports.
    """
    results, comparator, cells = evidence
    baseline = results["baselines"][results["best_single_feature_chosen_on_held_out_folds"]]
    dev = results["dev_fold"]

    pooled = [
        payload["score_without_dev_fold"] - baseline["score_without_dev_fold"]
        for payload in results["ppo_by_seed"].values()
    ]
    mean_of_per_fold = [
        float(np.mean([margin for (s, fold), margin in cells.items() if s == seed and fold != dev]))
        for seed in results["ppo_by_seed"]
    ]
    assert comparator["thinnest_per_seed_margin_without_dev_fold"] == pytest.approx(
        min(pooled), abs=0.01
    )
    assert comparator["thinnest_per_seed_margin_mean_of_per_fold"] == pytest.approx(
        min(mean_of_per_fold), abs=0.01
    )


def test_the_thinnest_per_fold_margin_is_seed_averaged_and_is_not_a_floor(
    evidence: tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], float]],
) -> None:
    """The correction this module was written for.

    ``thinnest_per_fold_margin`` is the minimum over folds of the seed-averaged
    margin. It is **not** the worst case: averaging three seeds turns a -7.49
    into a +0.55. Both are now published, and the assertion that the two differ
    in sign is what stops the friendlier number being quoted as a floor.
    """
    results, comparator, cells = evidence
    folds = sorted({fold for _, fold in cells})
    per_fold_mean = {
        fold: float(np.mean([margin for (_, f), margin in cells.items() if f == fold]))
        for fold in folds
    }
    worst_cell = min(cells.items(), key=lambda item: item[1])

    assert comparator["thinnest_per_fold_margin"] == pytest.approx(
        min(per_fold_mean.values()), abs=0.01
    )
    assert comparator["worst_fold_seed_cell_margin"] == pytest.approx(worst_cell[1], abs=0.01)
    assert comparator["per_fold_mean_margins"] == {
        fold: pytest.approx(value, abs=0.01) for fold, value in per_fold_mean.items()
    }
    # The whole point: the seed-averaged figure is positive and the worst real
    # cell is a loss. If a retrain ever makes both positive, this becomes a
    # genuinely robust result and the note should be rewritten to say so.
    assert comparator["worst_fold_seed_cell_margin"] < 0 < comparator["thinnest_per_fold_margin"], (
        "the seed-averaged thinnest margin and the worst single cell no longer "
        "straddle zero; re-derive the robustness note, which exists because "
        "they did"
    )
    assert "not a floor" in comparator["thinnest_per_fold_margin_note"]

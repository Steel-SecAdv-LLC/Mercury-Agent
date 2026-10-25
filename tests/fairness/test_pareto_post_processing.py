# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST_PROCESSING: NSGA-II per-group thresholds (error vs fairness gap)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.ml.fairness import (
    BiasmitigationProcessor,
    FairnessMetric,
    MitigationStrategy,
)


def _biased(seed: int, n: int = 3000) -> tuple[np.ndarray[Any, Any], ...]:
    rng = np.random.default_rng(seed)
    groups = rng.choice(np.array(["A", "B"]), n, p=[0.6, 0.4])
    labels = rng.binomial(1, np.where(groups == "A", 0.35, 0.25))
    logits = 1.6 * labels - 0.8 + rng.normal(0, 1, n) + np.where(groups == "A", 0.5, -0.3)
    return 1.0 / (1.0 + np.exp(-logits)), labels, groups


def _gaps(
    decisions: np.ndarray[Any, Any], labels: np.ndarray[Any, Any], groups: np.ndarray[Any, Any]
) -> tuple[float, float, float]:
    rates = [float(decisions[groups == g].mean()) for g in ("A", "B")]
    tpr = [float(decisions[(groups == g) & (labels == 1)].mean()) for g in ("A", "B")]
    fpr = [float(decisions[(groups == g) & (labels == 0)].mean()) for g in ("A", "B")]
    error = float(np.mean(decisions != labels))
    return error, abs(rates[0] - rates[1]), max(abs(tpr[0] - tpr[1]), abs(fpr[0] - fpr[1]))


def test_equalized_odds_post_processing_generalises() -> None:
    scores, labels, groups = _biased(0)
    test_scores, test_labels, test_groups = _biased(1)
    baseline = _gaps((test_scores >= 0.5).astype(float), test_labels, test_groups)
    proc = BiasmitigationProcessor(
        MitigationStrategy.POST_PROCESSING, FairnessMetric.EQUALIZED_ODDS
    ).fit(scores, labels, groups)
    error, _, eo_gap = _gaps(proc.transform(test_scores, test_groups), test_labels, test_groups)
    assert eo_gap < baseline[2] / 4
    assert error <= baseline[0]
    assert proc.pareto_front is not None and len(proc.pareto_front.solutions) > 1


def test_tolerance_picks_most_accurate_feasible_point() -> None:
    scores, labels, groups = _biased(2)
    proc = BiasmitigationProcessor(
        MitigationStrategy.POST_PROCESSING,
        FairnessMetric.DEMOGRAPHIC_PARITY,
        max_fairness_gap=0.03,
    ).fit(scores, labels, groups)
    feasible = [s for s in proc.pareto_front.solutions if s.objectives[1] <= 0.03]
    assert feasible
    best_error = min(float(s.objectives[0]) for s in feasible)
    error, dp_gap, _ = _gaps(proc.transform(scores, groups), labels, groups)
    assert error == pytest.approx(best_error, abs=1e-9)
    assert dp_gap <= 0.03 + 1e-9


def test_post_processing_is_deterministic() -> None:
    scores, labels, groups = _biased(3)
    a = BiasmitigationProcessor(MitigationStrategy.POST_PROCESSING, seed=5).fit(
        scores, labels, groups
    )
    b = BiasmitigationProcessor(MitigationStrategy.POST_PROCESSING, seed=5).fit(
        scores, labels, groups
    )
    assert a.group_thresholds == b.group_thresholds


def test_single_group_is_rejected() -> None:
    scores, labels, _ = _biased(4, n=200)
    with pytest.raises(ValueError, match="two groups"):
        BiasmitigationProcessor(MitigationStrategy.POST_PROCESSING).fit(
            scores, labels, np.full(len(scores), "A")
        )


def test_adversarial_debiasing_refuses_fixed_predictions() -> None:
    scores, labels, groups = _biased(5, n=200)
    with pytest.raises(ValueError, match="cannot be applied to fixed predictions"):
        BiasmitigationProcessor(MitigationStrategy.ADVERSARIAL_DEBIASING).fit(
            scores, labels, groups
        )

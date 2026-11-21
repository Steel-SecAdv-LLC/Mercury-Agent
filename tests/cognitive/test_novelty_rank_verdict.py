# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``is_novel`` is a rank test with a stated error rate, and unmeasured until it can be one.

The novelty score was measured to discriminate; its verdict was a cut at 0.7
that nothing had measured.  On the recorded benchmark tables that cut called
14.5% of ordinary held-out rows novel.  The verdict is now the rank of the
observation's distance among the reference's own distances, novel in the top
``alpha``: 3.9% measured, at most 4.9% on any table
(``artifacts/novelty_discovery.json``).

Pinned here: the recorded measurement and its bar; the rule itself; that the
verdict is undetermined (and carries no evidence) until there are enough
reference distances to rank among; and that a measured verdict carries the
numbers it can be recomputed from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.cognitive.cognitive_evolution_engine import (
    MIN_CALIBRATION,
    CuriosityEngine,
)
from omni_mercury_engine.discovery import rank_p_value, rms_standardized_distance

_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "novelty_discovery.json"


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    return loaded


class TestRecordedMeasurement:
    def test_the_rank_verdict_meets_its_bar(self, recorded: dict[str, Any]) -> None:
        verdict, mean = recorded["verdict"], recorded["mean"]
        assert verdict["bar_met"] is True
        assert mean["rank_normal_called_novel"] <= 0.07
        assert verdict["rank_normal_called_novel_worst_table"] <= 0.10
        assert verdict["alpha"] == 0.05

    def test_the_cut_it_replaced_is_recorded_as_it_was(self, recorded: dict[str, Any]) -> None:
        mean = recorded["mean"]
        assert mean["score_cut_normal_called_novel"] > 0.10  # 14.5%: not a 5% verdict
        assert mean["rank_normal_called_novel"] < mean["score_cut_normal_called_novel"] / 2

    def test_the_cost_is_recorded_too(self, recorded: dict[str, Any]) -> None:
        """Fewer anomalies are called novel. That is the price, and it is stated."""
        mean = recorded["mean"]
        assert mean["rank_anomalies_called_novel"] < mean["score_cut_anomalies_called_novel"]
        assert mean["rank_anomalies_called_novel"] >= 0.6

    def test_the_score_is_unchanged(self, recorded: dict[str, Any]) -> None:
        assert recorded["mean"]["read_only_auroc"] >= 0.92


def _warmed(rng: np.random.Generator, rows: int = 120, width: int = 4) -> CuriosityEngine:
    engine = CuriosityEngine()
    for row in rng.normal(size=(rows, width)):
        engine.explore("fit", row)
    return engine


class TestTheRule:
    def test_the_verdict_is_undetermined_until_it_can_be_ranked(self) -> None:
        """Fails on the prior code, which called a point novel against a reference of two."""
        rng = np.random.default_rng(0)
        engine = CuriosityEngine()
        results = [engine.explore("t", row) for row in rng.normal(size=(MIN_CALIBRATION + 1, 3))]
        far = engine.novelty("t", np.full(3, 50.0))
        # Two rows give a variance, so a distance exists from the third on; but
        # twenty distances are needed before one more can be ranked among them.
        assert all(not r.measured and not r.is_novel for r in results)
        assert all(r.evidence is None for r in results)
        assert far.measured is False and far.is_novel is False  # 19 distances: not yet
        engine.explore("t", rng.normal(size=3))
        ranked = engine.novelty("t", np.full(3, 50.0))
        assert ranked.measured is True and ranked.is_novel is True
        assert ranked.evidence is not None and ranked.evidence.n == MIN_CALIBRATION

    def test_a_measured_verdict_carries_what_it_can_be_recomputed_from(self) -> None:
        engine = _warmed(np.random.default_rng(1))
        result = engine.novelty("t", np.array([3.0, -3.0, 3.0, -3.0]))
        assert result.evidence is not None
        assert result.evidence.method == "conformal_rank"
        assert (result.evidence.statistic, result.evidence.threshold) == (result.p_value, 0.05)
        # The distance and the p-value, recomputed from the result's own numbers.
        distance = rms_standardized_distance(
            result.observation, result.reference_mean, result.reference_std
        )
        assert distance is not None and result.p_value is not None
        assert distance == pytest.approx(result.standardized_distance)
        assert rank_p_value(distance, result.calibration) == pytest.approx(result.p_value)
        assert result.is_novel is (result.p_value <= 0.05)

    def test_ordinary_observations_are_called_novel_about_alpha_of_the_time(self) -> None:
        rng = np.random.default_rng(2)
        engine = _warmed(rng, rows=300)
        fresh = rng.normal(size=(2000, 4))
        share = float(np.mean([engine.novelty("t", row).is_novel for row in fresh]))
        assert share <= 0.09  # 0.05 by construction, plus the reference's own noise
        far = rng.normal(size=(200, 4)) + 4.0
        assert float(np.mean([engine.novelty("t", row).is_novel for row in far])) >= 0.95

    def test_reading_a_score_does_not_move_the_reference(self) -> None:
        engine = _warmed(np.random.default_rng(3))
        before = (engine.observations_seen, engine.get_statistics()["calibration_distances"])
        for _ in range(50):
            engine.novelty("t", np.full(4, 9.0))
        assert (
            engine.observations_seen,
            engine.get_statistics()["calibration_distances"],
        ) == before

    def test_a_new_feature_space_starts_over(self) -> None:
        engine = _warmed(np.random.default_rng(4))
        result = engine.explore("t", np.zeros(7))
        assert result.measured is False
        assert engine.get_statistics()["calibration_distances"] == 0

    @pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.0}, {"calibration_window": 5}])
    def test_a_level_or_window_that_cannot_work_is_refused(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"alpha|calibration_window"):
            CuriosityEngine(**kwargs)

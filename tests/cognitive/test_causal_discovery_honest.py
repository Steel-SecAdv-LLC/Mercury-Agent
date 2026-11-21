# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The causal instrument reports edges that are there, and says when it cannot tell.

Two layers.  The committed harness artifact
(``artifacts/causal_discovery.json``, written by
``research/cognitive/harness_causal_discovery.py``) is pinned: what the
shipped procedure achieves, what the procedure it replaced did, which
alternatives were refused, and which bar is **not** met.  And a small live
run on fresh windows checks the instrument still behaves that way, so the
artifact cannot go stale behind a passing test.

The live checks fail on the prior procedure, which reported an edge on
independent noise in 76-94% of analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.discovery import causal_artifact

_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "causal_discovery.json"


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    return loaded


class TestRecordedMeasurement:
    def test_the_shipped_procedure_meets_every_truth_bar(self, recorded: dict[str, Any]) -> None:
        verdict = recorded["verdicts"]["shipped"]
        assert recorded["adopted"] == "shipped"
        for bar in (
            "false_edge_plain",
            "false_edge_disturbed",
            "planted_found_n200",
            "planted_other_edge",
        ):
            assert verdict[bar]["met"] is True, bar
        assert verdict["false_edge_plain"]["worst"] <= 0.05
        assert verdict["false_edge_disturbed"]["worst"] <= 0.05
        assert verdict["planted_found_n200"]["worst"] >= 0.90

    def test_the_bar_it_does_not_meet_is_recorded_as_unmet(self, recorded: dict[str, Any]) -> None:
        """Coverage: most strongly autocorrelated windows are declined. Stated, not hidden."""
        coverage = recorded["verdicts"]["shipped"]["stable_windows_declined"]
        assert coverage["met"] is False
        assert coverage["worst"] > recorded["bars"]["stable_windows_declined_max"]
        assert "autocorrelat" in recorded["open_problem"]

    def test_the_prior_procedure_is_recorded_as_it_was(self, recorded: dict[str, Any]) -> None:
        prior = recorded["verdicts"]["prior"]
        assert prior["false_edge_plain"]["met"] is False
        assert prior["false_edge_plain"]["worst"] >= 0.75  # an edge on noise, most of the time
        assert recorded["results"]["noise_n200_d5"]["prior"]["false_edge"] >= 0.75

    @pytest.mark.parametrize("refused", ["first_difference", "residual_guard"])
    def test_each_refused_alternative_fails_a_truth_bar(
        self, recorded: dict[str, Any], refused: str
    ) -> None:
        assert refused in recorded["refused"]
        verdict = recorded["verdicts"][refused]
        assert not all(
            verdict[bar]["met"]
            for bar in ("false_edge_plain", "false_edge_disturbed", "planted_other_edge")
        )

    def test_a_declined_window_never_carries_an_edge(self, recorded: dict[str, Any]) -> None:
        """Where the shipped procedure declines everything, it claims nothing."""
        for name, row in recorded["results"].items():
            shipped = row["shipped"]
            if shipped["not_assessed"] == 1.0:
                assert shipped["false_edge"] == 0.0, name


def _edges(window: np.ndarray) -> set[tuple[str, str]]:
    artifact = causal_artifact(window, max_lag=5, alpha=0.05)
    return {(e["source"], e["target"]) for e in artifact["claim"]["edges"]}


class TestLive:
    """Sixty fresh windows per case; the bounds leave room for sampling noise."""

    TRIALS = 60

    def test_independent_noise_yields_almost_no_edge(self) -> None:
        rng = np.random.default_rng(101)
        with_edge = sum(bool(_edges(rng.normal(size=(200, 5)))) for _ in range(self.TRIALS))
        assert with_edge <= 6  # 2-3% expected; the prior procedure: about 56 of 60

    def test_a_common_shock_yields_almost_no_edge(self) -> None:
        rng = np.random.default_rng(102)
        with_edge = 0
        for _ in range(self.TRIALS):
            window = rng.normal(size=(200, 5))
            start = int(rng.integers(20, 170))
            window[start : start + 6] += 9.0
            with_edge += bool(_edges(window))
        assert with_edge <= 6  # the prior procedure: 60 of 60

    def test_a_sustained_shift_is_declined_not_guessed(self) -> None:
        rng = np.random.default_rng(103)
        for _ in range(20):
            window = rng.normal(size=(200, 5))
            start = int(rng.integers(20, 130))
            window[start : start + 40] += 3.0
            artifact = causal_artifact(window, max_lag=5, alpha=0.05)
            assert artifact["claim"]["assessed"] is False
            assert artifact["claim"]["edges"] == []

    def test_a_planted_edge_is_found_and_stands_alone(self) -> None:
        rng = np.random.default_rng(104)
        found = stray = 0
        for _ in range(self.TRIALS):
            window = rng.normal(size=(200, 4))
            window[1:, 1] += 0.5 * window[:-1, 0]
            edges = _edges(window)
            found += ("X0", "X1") in edges
            stray += bool(edges - {("X0", "X1")})
        assert found >= 52  # 95-97% expected
        assert stray <= 6

    def test_the_engine_reports_what_the_artifact_does(self) -> None:
        """One implementation: the engine's graph and the artifact agree."""
        from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

        rng = np.random.default_rng(105)
        window = rng.normal(size=(200, 4))
        window[1:, 1] += 0.5 * window[:-1, 0]
        graph = CausalDiscoveryEngine().discover_temporal_causation(window)
        assert {(e.source, e.target) for e in graph.edges} == _edges(window)
        edge = graph.edges[0]
        assert 0.0 < edge.strength < 1.0  # a partial R-squared; this field once held 1 - p
        assert edge.p_value < 0.05
        summary = graph.to_dict()
        assert summary["assessed"] is True and summary["pairs_tested"] == 12
        assert summary["corrections"] == ["bonferroni_over_lags", "holm_over_pairs"]

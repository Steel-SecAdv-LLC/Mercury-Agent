# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The live-contributor ranking/calibration decision is pinned to its artifact.

The Phase 2 proposal — rank probes only among live contributors, calibrate
each probe locally — was measured on the recorded corpora
(``research/math_arrest/harness_live_contributor_ranking.py``) and every
candidate was refused: each degrades the shipped uniform fusion.  These tests
pin the artifact's existence, the pre-registered bar, and the refusals, so
the decision cannot silently rot or be re-asserted without re-measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = _ROOT / "artifacts" / "math_arrest_live_ranking.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert _ARTIFACT.is_file(), "run research/math_arrest/harness_live_contributor_ranking.py"
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


class TestLiveContributorRankingDecision:
    def test_protocol_is_the_preregistered_bar(self, artifact: dict) -> None:
        protocol = artifact["protocol"]
        assert protocol["adoption_band"] == 0.003
        assert protocol["regression_floor"] == 0.002
        assert protocol["live_floor"] == 0.75
        assert protocol["n_evaluations"] == 19

    def test_every_candidate_was_refused(self, artifact: dict) -> None:
        verdicts = artifact["verdicts"]
        assert set(verdicts) == {
            "phi_affinity_all",
            "phi_affinity_live",
            "uniform_live",
            "local_rank",
        }
        for name, verdict in verdicts.items():
            assert verdict["adopted"] is False, f"{name} was recorded as adopted"
            # A refusal here must be a real measurement, not a missing one.
            assert isinstance(verdict["mean_delta_vs_shipped"], float)

    def test_refusals_follow_from_the_bar(self, artifact: dict) -> None:
        protocol = artifact["protocol"]
        for verdict in artifact["verdicts"].values():
            clears = (
                verdict["mean_delta_vs_shipped"] > protocol["adoption_band"]
                and verdict["worst_delta"] >= -protocol["regression_floor"]
            )
            assert verdict["adopted"] == clears

    def test_the_ledger_decision_matches_the_artifact(self, artifact: dict) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d
            for d in data["decisions"]
            if d["component"] == "detectors.math_arrest.live_contributor_ranking"
        )
        assert entry["verdict"] == "refused"
        assert entry["impact"]["shipped_uniform_mean_auc"] == pytest.approx(
            artifact["shipped_uniform_mean_auc"], abs=1e-4
        )
        assert entry["impact"]["phi_affinity_live_mean_delta"] == pytest.approx(
            artifact["verdicts"]["phi_affinity_live"]["mean_delta_vs_shipped"], abs=1e-4
        )

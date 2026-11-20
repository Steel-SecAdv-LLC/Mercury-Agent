# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Meta-Fusion Controller decision is pinned to its artifact.

The controller (select per dataset among the shipped average, the
historically best single probe, a historical-AUC-weighted sum, and the
decorrelated subset, by leave-one-dataset-out historical performance within
the dataset's corpus) was measured on the 19 recorded evaluations.  Its LODO
prior is the first label-free selector signal that raises the mean
out-of-fold (+0.0413) — and it is still refused, because every
mean-improving candidate regresses individual datasets far past the standing
keep bar.  These tests pin the artifact, the bar, and the refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = _ROOT / "artifacts" / "meta_fusion_controller.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert _ARTIFACT.is_file(), "run research/ensembles/harness_meta_fusion.py"
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


class TestMetaFusionDecision:
    def test_protocol_is_lodo_with_the_standing_bar(self, artifact: dict) -> None:
        protocol = artifact["protocol"]
        assert "leave-one-dataset-out" in protocol["selection"]
        assert protocol["adoption_band"] == 0.003
        assert protocol["regression_floor"] == 0.002
        assert protocol["n_evaluations"] == 19

    def test_every_candidate_was_refused(self, artifact: dict) -> None:
        for name, verdict in artifact["verdicts"].items():
            assert verdict["adopted"] is False, f"{name} recorded as adopted"

    def test_refusals_follow_from_the_bar(self, artifact: dict) -> None:
        protocol = artifact["protocol"]
        for verdict in artifact["verdicts"].values():
            clears = (
                verdict["mean_delta_vs_shipped"] > protocol["adoption_band"]
                and verdict["worst_delta"] >= -protocol["regression_floor"]
            )
            assert verdict["adopted"] == clears

    def test_the_controller_gain_is_real_and_its_cost_is_real(self, artifact: dict) -> None:
        """Both halves of the finding are pinned: +mean AND a deep regression."""
        controller = artifact["verdicts"]["meta_controller"]
        assert controller["mean_delta_vs_shipped"] > 0.03
        assert controller["worst_delta"] < -0.05

    def test_the_ledger_decision_matches_the_artifact(self, artifact: dict) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d for d in data["decisions"] if d["component"] == "ensembles.meta_fusion_controller"
        )
        assert entry["verdict"] == "refused"
        controller = artifact["verdicts"]["meta_controller"]
        assert entry["impact"]["meta_controller_mean_delta"] == pytest.approx(
            controller["mean_delta_vs_shipped"], abs=1e-3
        )
        assert entry["impact"]["meta_controller_worst_delta"] == pytest.approx(
            controller["worst_delta"], abs=1e-3
        )


class TestGuardedControllerDecision:
    def test_guarded_controller_measured_and_refused(self, artifact: dict) -> None:
        guarded = artifact["verdicts"]["meta_guarded"]
        assert guarded["adopted"] is False
        # Both halves of the sharpened finding are pinned: the guard mostly
        # abstains, and where history cleared the bar the pick still
        # regressed out-of-fold.
        from collections import Counter

        choices = Counter(artifact["guarded_controller_choices"].values())
        assert choices["uniform"] >= 15
        assert guarded["mean_delta_vs_shipped"] < 0.003

    def test_ledger_records_the_guarded_refusal(self, artifact: dict) -> None:
        from omni_mercury_engine.governance.improvement_ledger import published

        data = published()
        assert data is not None
        entry = next(
            d
            for d in data["decisions"]
            if d["component"] == "ensembles.meta_fusion_controller.guarded"
        )
        assert entry["verdict"] == "refused"
        assert entry["impact"]["guarded_mean_delta"] == pytest.approx(
            artifact["verdicts"]["meta_guarded"]["mean_delta_vs_shipped"], abs=1e-3
        )


class TestCrossModalConsistencyDecision:
    def test_refused_with_its_artifact(self) -> None:
        import json
        from pathlib import Path

        from omni_mercury_engine.governance.improvement_ledger import published

        artifact = json.loads(
            (
                Path(__file__).resolve().parents[2] / "artifacts/cross_modal_consistency.json"
            ).read_text(encoding="utf-8")
        )
        assert artifact["adopted"] is False
        # The refusal's ground is pinned: both classes agree near-perfectly.
        assert artifact["mean_agreement_false_positive"] > 0.9
        assert artifact["mean_agreement_true_positive"] > 0.9

        data = published()
        assert data is not None
        entry = next(
            d for d in data["decisions"] if d["component"] == "decision.cross_modal_consistency"
        )
        assert entry["verdict"] == "refused"
        assert entry["impact"]["mean_agreement_false_positive"] == pytest.approx(
            artifact["mean_agreement_false_positive"], abs=1e-4
        )

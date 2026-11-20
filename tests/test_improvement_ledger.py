# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Improvement Ledger records every adaptive change, with its rollback.

Before this, an adaptive change left at most a log line: the online
recalibrator moving the live operating threshold, a conformal refit, the
label-scarcity schedule resolving the symbolic weight, and the math-arrest
ensemble's decorrelation/refinement were all unreconstructable after the
fact, and the measured adopt/refuse decisions (entropy-constrained weight
re-selection, locally-adaptive recursion depth, lambda adaptation) lived
only in commit messages.  These tests pin the instrument and its hooks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.governance.improvement_ledger import (
    ImprovementEntry,
    ImprovementLedger,
    default_improvement_ledger,
    published,
    record_adaptive_change,
)


class TestEntryContract:
    def test_entry_requires_what_why_and_rollback(self) -> None:
        with pytest.raises(ValueError, match="change"):
            ImprovementEntry(component="x", change="", reason="r", rollback="undo")
        with pytest.raises(ValueError, match="reason"):
            ImprovementEntry(component="x", change="c", reason=" ", rollback="undo")
        with pytest.raises(ValueError, match="rollback"):
            ImprovementEntry(component="x", change="c", reason="r", rollback="")
        with pytest.raises(ValueError, match="component"):
            ImprovementEntry(component="", change="c", reason="r", rollback="undo")

    def test_entry_round_trips(self) -> None:
        entry = ImprovementEntry(
            component="engine.test",
            change="moved a dial",
            reason="measured drift",
            before=0.5,
            after=0.61,
            impact={"auc_delta": 0.002},
            rollback="set the dial back to 0.5",
        )
        rebuilt = ImprovementEntry.from_dict(json.loads(json.dumps(entry.to_dict())))
        assert rebuilt == entry

    def test_unmeasured_impact_is_an_explicit_none(self) -> None:
        entry = ImprovementEntry(component="engine.test", change="c", reason="r", rollback="undo")
        assert entry.to_dict()["impact"] is None


class TestLedger:
    def test_bounded_ring_buffer_keeps_aggregates_in_step(self) -> None:
        ledger = ImprovementLedger(maxlen=2)
        for i in range(3):
            ledger.record(
                ImprovementEntry(
                    component=f"c{i}",
                    change="c",
                    reason="r",
                    rollback="u",
                    impact={"n": i} if i == 0 else None,
                )
            )
        assert len(ledger) == 2
        summary = ledger.summary()
        assert summary["total"] == 2
        assert summary["by_component"] == {"c1": 1, "c2": 1}
        # The evicted measured entry no longer counts.
        assert summary["measured"] == 0

    def test_json_round_trip(self) -> None:
        ledger = ImprovementLedger()
        ledger.record(ImprovementEntry(component="a", change="c", reason="r", rollback="u"))
        rebuilt = ImprovementLedger.from_json(ledger.to_json())
        assert [e.to_dict() for e in rebuilt.entries()] == [e.to_dict() for e in ledger.entries()]

    def test_default_ledger_is_process_wide(self) -> None:
        first = default_improvement_ledger()
        assert default_improvement_ledger() is first
        before = len(first)
        record_adaptive_change("test.component", "c", "r", rollback="u", after=1.0)
        assert len(first) == before + 1


class TestCommittedDecisions:
    """The shipped artifact is complete: what, why, impact, verdict, rollback."""

    def test_artifact_exists_and_parses(self) -> None:
        data = published()
        assert data is not None, "artifacts/improvement_ledger.json is missing"
        assert data["schema"].startswith("mercury.governance.improvement_ledger/")
        assert data["decisions"], "the committed ledger must not be empty"

    def test_every_decision_is_complete(self) -> None:
        data = published()
        assert data is not None
        for decision in data["decisions"]:
            for key in ("component", "change", "reason", "verdict", "impact", "rollback"):
                assert decision.get(key), f"decision missing {key}: {decision}"
            assert decision["verdict"] in (
                "adopted",
                "adopted_opt_in",
                "refused",
                "recorded_not_adopted",
            )
            impact = decision["impact"]
            assert impact.get("harness"), "impact must name its harness"
            assert impact.get("artifact"), "impact must name its artifact"
            # A refusal must say why; so must a recorded-not-adopted.
            if decision["verdict"] in ("refused", "recorded_not_adopted"):
                assert decision.get("why_refused") or decision.get("why_not_adopted")

    def test_named_artifacts_exist(self) -> None:
        data = published()
        assert data is not None
        root = Path(__file__).resolve().parents[1]
        for decision in data["decisions"]:
            artifact = decision["impact"]["artifact"]
            assert (root / artifact).is_file(), f"{artifact} not committed"

    def test_adaptive_depth_refusal_matches_its_artifact(self) -> None:
        """The ledger's depth-refusal numbers are the artifact's, not prose."""
        data = published()
        assert data is not None
        depth = next(
            d for d in data["decisions"] if d["component"] == "core.three_r.recursion_depth"
        )
        root = Path(__file__).resolve().parents[1]
        measured = json.loads(
            (root / "artifacts/oae_adaptive_depth.json").read_text(encoding="utf-8")
        )
        assert depth["verdict"] == "refused"
        assert depth["impact"]["fixed_depth_3_mean_rank"] == pytest.approx(
            measured["results"]["depth_3"]["mean_rank"]
        )
        assert depth["impact"]["served_depth_5_mean_rank"] == pytest.approx(
            measured["results"]["depth_5"]["mean_rank"]
        )

    def test_entropy_reselection_matches_its_artifact(self) -> None:
        data = published()
        assert data is not None
        entry = next(
            d
            for d in data["decisions"]
            if d["component"] == "detectors.statistical.adaptive_weights"
        )
        root = Path(__file__).resolve().parents[1]
        measured = json.loads(
            (root / "artifacts/fusion_weight_entropy.json").read_text(encoding="utf-8")
        )
        selected = measured["selected_in_band"]
        assert entry["impact"]["winner_mean_auroc"] == pytest.approx(selected["mean_auc"])
        assert entry["impact"]["winner_mean_rank"] == pytest.approx(selected["mean_rank"])
        assert entry["verdict"] == "recorded_not_adopted"


class TestRuntimeHooks:
    def test_decorrelator_calibration_records_an_entry(self) -> None:
        from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

        ledger = default_improvement_ledger()
        rng = np.random.default_rng(0)
        series = rng.normal(size=400)
        arrest = AnomalyMathArrest(probes="minimal")
        # fit() itself may calibrate once; the baseline is taken after it so
        # the assertion isolates the explicit calibration call.
        arrest.fit(series)
        before = ledger.summary()["by_component"].get("math_arrest.decorrelator", 0)
        multipliers = arrest.calibrate_decorrelator(series)
        after = ledger.summary()["by_component"].get("math_arrest.decorrelator", 0)
        if multipliers and any(abs(m - 1.0) > 1e-12 for m in multipliers.values()):
            assert after == before + 1
            entry = next(
                e for e in reversed(ledger.entries()) if e.component == "math_arrest.decorrelator"
            )
            assert entry.rollback
            assert entry.before is not None and entry.after is not None
        else:
            # Nothing was down-weighted: an identity calibration is not an
            # adaptive change and must not pollute the ledger.
            assert after == before

    def test_online_recalibration_records_takeover_not_every_call(self) -> None:
        pytest.importorskip("torch", reason="the engine needs the [ml] extra")
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
        if not engine.load_default_fusion_checkpoint():
            pytest.skip("no shipped default fusion checkpoint in this install")
        engine.enable_online_recalibration(warmup=5)

        ledger = default_improvement_ledger()
        component = "engine.online_recalibration"
        start = ledger.summary()["by_component"].get(component, 0)

        rng = np.random.default_rng(1)
        for _ in range(4):
            engine.detect_with_fusion(rng.normal(size=(4, 16)))

        count = ledger.summary()["by_component"].get(component, 0)
        assert count >= start + 1, "the warmup crossing must be recorded"
        # Sixteen scored rows, one takeover plus at most a few >= 0.01 moves:
        # the hook must not write one entry per call.
        assert count - start <= 3
        entry = next(e for e in reversed(ledger.entries()) if e.component == component)
        assert "disable_online_recalibration" in entry.rollback
        assert entry.after is not None

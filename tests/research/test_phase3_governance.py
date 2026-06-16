# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 3 governed Reflexion, drift, and dormant-revival tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from research.governed_fusion.phase3_governance import (
    Phase3Action,
    Phase3DecisionStore,
    evaluate_phase3_report,
    main,
    route_dormant_revival_candidate,
    route_drift_recalibration,
    route_reflexion_executor,
    route_reflexion_threshold,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict[str, object]:
    return {
        "provenance_summary": {
            "transparent_fitness_bucket": "external_label",
            "real": {"external_label": {"n_events": 2, "n_rows": 100, "n_pos": 20}},
        }
    }


def _ledger() -> dict[str, object]:
    return {
        "schema_version": 1,
        "transparent_fitness_bucket": "external_label",
        "runs": [
            {
                "status": "ok",
                "full": {"auroc": 0.770, "auprc": 0.550, "f1": 0.180, "n_events": 2},
            }
        ],
    }


def _results(auroc: float, auprc: float, f1: float, *, n_events: int = 2) -> dict[str, object]:
    return {
        "per_provenance": {
            "external_label": {
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "precision": 0.50,
                "recall": 0.50,
                "n_events": n_events,
            },
            "self_label": {"auroc": 0.99, "auprc": 0.99, "f1": 0.99, "n_events": 21},
        },
        "overall": {"auroc": 0.90, "auprc": 0.90, "f1": 0.90, "n_events": 23},
    }


def _candidate_record(candidate_id: str = "phase3-candidate") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "baseline_id": "main",
        "evaluation_mode": "held_out_replay",
        "optimization_bucket": "external_label",
        "shadow_replay": {
            "baseline": _results(0.770, 0.550, 0.180),
            "candidate": _results(0.783, 0.560, 0.195),
        },
        "ablation": {"full": {"auroc": 0.783, "auprc": 0.560, "f1": 0.195, "n_events": 2}},
        "safety": {
            "sigma_immutable": 0.94,
            "benevolence": 0.995,
            "conformal_coverage": 0.91,
            "lyapunov_lambda": 0.001,
        },
        "capability_regression": {"passed": True, "failed": []},
    }


class _ReflexionProbe:
    def __init__(self, recommendation: dict[str, object]) -> None:
        self._recommendation = recommendation

    def get_threshold_recommendation(self) -> dict[str, object]:
        return self._recommendation


def test_reflexion_executor_maintains_without_gate_when_recommendation_is_maintain() -> None:
    decision = route_reflexion_threshold(
        {"recommendation": "maintain", "current_threshold": 0.5, "suggested_threshold": 0.5},
        candidate_record=None,
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.MAINTAIN.value
    assert decision.gate_decision is None
    assert decision.gate_result is None
    assert decision.reasons == ["Reflexion recommendation is maintain; no candidate routed."]


def test_reflexion_executor_routes_threshold_change_through_promotion_gate() -> None:
    decision = route_reflexion_executor(
        _ReflexionProbe(
            {"recommendation": "increase", "current_threshold": 0.5, "suggested_threshold": 0.63}
        ),
        candidate_record=_candidate_record("reflexion-threshold-063"),
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.QUEUE_REFLEXION_CANDIDATE.value
    assert decision.candidate_id == "reflexion-threshold-063"
    assert decision.gate_decision == "promote"
    assert decision.gate_result is not None
    assert decision.gate_result["requires_human_approval"] is True


def test_drift_recalibration_requires_high_or_critical_drift_trigger() -> None:
    decision = route_drift_recalibration(
        [{"is_drift": True, "severity": "medium", "message": "minor PSI movement"}],
        candidate_record=_candidate_record("strict-isotonic-recalibration"),
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.MAINTAIN.value
    assert decision.gate_decision is None
    assert "No high/critical drift trigger" in decision.reasons[0]


def test_drift_recalibration_fails_closed_without_candidate_evidence() -> None:
    decision = route_drift_recalibration(
        [{"is_drift": True, "severity": "high", "message": "external-label AUROC drift"}],
        candidate_record=None,
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.REJECT.value
    assert decision.candidate_id is None
    assert decision.reasons == ["High/critical drift requires recalibration candidate evidence."]


def test_drift_recalibration_rejected_when_gate_fails_external_label_metrics() -> None:
    record = _candidate_record("unsafe-recalibration")
    record["shadow_replay"] = {
        "baseline": _results(0.770, 0.550, 0.180),
        "candidate": _results(0.760, 0.540, 0.170),
    }

    decision = route_drift_recalibration(
        [{"is_drift": True, "severity": "critical", "message": "coverage collapse"}],
        candidate_record=record,
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.REJECT.value
    assert decision.gate_decision == "reject"
    assert any("auroc regressed" in reason for reason in decision.reasons)
    assert any("f1 regressed" in reason for reason in decision.reasons)


def test_dormant_candidate_archives_when_signal_bar_not_cleared() -> None:
    decision = route_dormant_revival_candidate(
        "predictive_coding",
        {"mean_auc": 0.61, "carries_signal": False},
        candidate_record=_candidate_record("predictive-coding-revival"),
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.MAINTAIN.value
    assert decision.candidate_id == "predictive_coding"
    assert decision.gate_decision is None
    assert "pre-registered signal bar" in decision.reasons[0]


def test_dormant_candidate_revival_is_gate_routed_when_signal_bar_clears() -> None:
    decision = route_dormant_revival_candidate(
        "case_based_knn",
        {"mean_auc": 0.74, "carries_signal": True},
        candidate_record=_candidate_record("case-based-knn-revival"),
        manifest=_manifest(),
        ledger=_ledger(),
    )

    assert decision.action == Phase3Action.QUEUE_DORMANT_REVIVAL_CANDIDATE.value
    assert decision.candidate_id == "case-based-knn-revival"
    assert decision.gate_decision == "promote"
    assert decision.reasons == []


def test_phase3_composite_report_and_append_only_store(tmp_path: Path) -> None:
    report = {
        "reflexion_threshold": {
            "recommendation": {
                "recommendation": "decrease",
                "current_threshold": 0.5,
                "suggested_threshold": 0.42,
            },
            "candidate_record": _candidate_record("reflexion-threshold-042"),
        },
        "drift_recalibration": {
            "drift_results": [{"is_drift": False, "severity": "none"}],
            "candidate_record": _candidate_record("unused-recalibration"),
        },
        "dormant_revival": {
            "candidates": {
                "kmeans_distance": {"mean_auc": 0.72, "carries_signal": True},
                "predictive_coding": {"mean_auc": 0.62, "carries_signal": False},
            },
            "candidate_records": {"kmeans_distance": _candidate_record("kmeans-revival")},
        },
    }

    decisions = evaluate_phase3_report(report, manifest=_manifest(), ledger=_ledger())
    record = Phase3DecisionStore(tmp_path).write_many(decisions)
    index_rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()

    assert [decision.action for decision in decisions] == [
        Phase3Action.QUEUE_REFLEXION_CANDIDATE.value,
        Phase3Action.MAINTAIN.value,
        Phase3Action.QUEUE_DORMANT_REVIVAL_CANDIDATE.value,
        Phase3Action.MAINTAIN.value,
    ]
    assert record.exists()
    assert len(index_rows) == 4
    assert json.loads(index_rows[0])["candidate_id"] == "reflexion-threshold-042"


def test_phase3_cli_check_exits_nonzero_on_rejection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.json"
    report = tmp_path / "report.json"
    out = tmp_path / "decisions.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    ledger.write_text(json.dumps(_ledger()), encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "drift_recalibration": {
                    "drift_results": [{"is_drift": True, "severity": "high"}],
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--report",
            str(report),
            "--manifest",
            str(manifest),
            "--ledger",
            str(ledger),
            "--out",
            str(out),
            "--check",
        ]
    )
    decisions = json.loads(out.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert decisions[0]["action"] == Phase3Action.REJECT.value
    maintain_decision = evaluate_phase3_report(
        {"drift_recalibration": {"drift_results": []}},
        manifest=_manifest(),
        ledger=_ledger(),
    )
    assert maintain_decision[0].action == Phase3Action.MAINTAIN.value

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 2 governed-promotion gate tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from research.governed_fusion.promotion_gate import (
    ExperimentStore,
    GateDecision,
    evaluate_candidate,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict[str, object]:
    return {
        "provenance_summary": {
            "transparent_fitness_bucket": "external_label",
            "real": {
                "external_label": {
                    "n_events": 2,
                    "n_rows": 100,
                    "n_pos": 20,
                }
            },
        }
    }


def _ledger_with_ok_baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "transparent_fitness_bucket": "external_label",
        "runs": [
            {
                "status": "ok",
                "full": {
                    "auroc": 0.77,
                    "auprc": 0.55,
                    "f1": 0.18,
                    "n_events": 2,
                },
            }
        ],
    }


def _empty_ledger() -> dict[str, object]:
    return {"schema_version": 1, "transparent_fitness_bucket": "external_label", "runs": []}


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
            "self_label": {
                "auroc": 0.99,
                "auprc": 0.99,
                "f1": 0.99,
                "n_events": 21,
            },
        },
        "overall": {"auroc": 0.90, "auprc": 0.90, "f1": 0.90, "n_events": 23},
    }


def _candidate_record() -> dict[str, object]:
    return {
        "candidate_id": "fusion-weight-candidate-a",
        "baseline_id": "main",
        "evaluation_mode": "held_out_replay",
        "optimization_bucket": "external_label",
        "shadow_replay": {
            "baseline": _results(0.770, 0.550, 0.180),
            "candidate": _results(0.783, 0.560, 0.195),
        },
        "ablation": {
            "full": {
                "auroc": 0.781,
                "auprc": 0.559,
                "f1": 0.192,
                "n_events": 2,
            }
        },
        "safety": {
            "sigma_immutable": 0.94,
            "benevolence": 0.995,
            "conformal_coverage": 0.91,
            "lyapunov_lambda": 0.001,
        },
        "capability_regression": {"passed": True, "failed": []},
    }


def test_gate_promotes_external_label_improvement_with_human_approval() -> None:
    result = evaluate_candidate(
        _candidate_record(),
        manifest=_manifest(),
        ledger=_ledger_with_ok_baseline(),
    )

    assert result.decision == GateDecision.PROMOTE.value
    assert result.requires_human_approval is True
    assert result.fitness_bucket == "external_label"
    assert result.manifest_external_label_events == 2
    assert result.metric_deltas["auroc"] > 0
    assert result.metric_deltas["f1"] > 0
    assert result.reasons == []


def test_gate_rejects_when_only_non_fitness_buckets_improve() -> None:
    record = _candidate_record()
    record["shadow_replay"] = {
        "baseline": _results(0.770, 0.550, 0.180),
        "candidate": _results(0.760, 0.560, 0.170),
    }

    result = evaluate_candidate(record, manifest=_manifest(), ledger=_empty_ledger())

    assert result.decision == GateDecision.REJECT.value
    assert result.requires_human_approval is False
    assert any("auroc regressed" in reason for reason in result.reasons)
    assert any("f1 regressed" in reason for reason in result.reasons)


def test_gate_rejects_if_candidate_broadens_optimization_bucket() -> None:
    record = _candidate_record()
    record["optimization_bucket"] = "self_label"

    result = evaluate_candidate(record, manifest=_manifest(), ledger=_empty_ledger())

    assert result.decision == GateDecision.REJECT.value
    assert any("optimization_bucket" in reason for reason in result.reasons)


def test_gate_requires_capability_regression_evidence() -> None:
    record = _candidate_record()
    del record["capability_regression"]

    result = evaluate_candidate(record, manifest=_manifest(), ledger=_empty_ledger())

    assert result.decision == GateDecision.REJECT.value
    assert "capability_regression block is required" in result.reasons


def test_gate_rejects_latest_ablation_baseline_regression() -> None:
    record = _candidate_record()
    record["ablation"] = {
        "full": {
            "auroc": 0.760,
            "auprc": 0.540,
            "f1": 0.170,
            "n_events": 2,
        }
    }

    result = evaluate_candidate(
        record,
        manifest=_manifest(),
        ledger=_ledger_with_ok_baseline(),
    )

    assert result.decision == GateDecision.REJECT.value
    assert any("ablation auroc regressed" in reason for reason in result.reasons)
    assert result.evidence["ledger_baseline_status"] == "checked_latest_ok"


def test_failed_canary_emits_rollback_decision() -> None:
    record = _candidate_record()
    record["evaluation_mode"] = "canary"
    record["safety"] = {
        "sigma_immutable": 0.92,
        "benevolence": 0.995,
        "conformal_coverage": 0.91,
        "lyapunov_lambda": 0.001,
    }

    result = evaluate_candidate(record, manifest=_manifest(), ledger=_empty_ledger())

    assert result.decision == GateDecision.ROLLBACK.value
    assert result.rollback["enabled"] is True
    assert result.rollback["triggered"] is True
    assert result.rollback["target"] == "main"


def test_gate_rejects_manifest_event_count_mismatch() -> None:
    record = _candidate_record()
    record["shadow_replay"] = {
        "baseline": _results(0.770, 0.550, 0.180, n_events=1),
        "candidate": _results(0.783, 0.560, 0.195, n_events=1),
    }

    result = evaluate_candidate(record, manifest=_manifest(), ledger=_empty_ledger())

    assert result.decision == GateDecision.REJECT.value
    assert sum("does not match manifest" in reason for reason in result.reasons) == 2


def test_experiment_store_is_append_only(tmp_path: Path) -> None:
    result = evaluate_candidate(
        _candidate_record(),
        manifest=_manifest(),
        ledger=_ledger_with_ok_baseline(),
    )
    store = ExperimentStore(tmp_path)

    record_path = store.write(result)

    assert record_path.exists()
    with record_path.open(encoding="utf-8") as fh:
        assert json.load(fh)["candidate_id"] == result.candidate_id
    index_path = tmp_path / "index.jsonl"
    assert index_path.exists()
    rows = index_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["record"] == str(record_path)
    assert asdict(result)["gate"] == "governed-promotion"

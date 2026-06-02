"""Tests for :mod:`scripts.run_equation_research_protocol`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_equation_research_protocol


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _minimal_benchmark() -> dict[str, Any]:
    def domain(mean_auc: float, mean_recall: float, n: int = 1) -> dict[str, Any]:
        return {"n_measured": n, "stats": {"mean_auc": mean_auc, "mean_recall": mean_recall}}

    return {
        "domain_summary": {
            "adbench": domain(0.82, 0.80),
            "timeseries": domain(0.79, 0.74),
            "industrial": domain(0.80, 0.72),
            "academic": domain(0.81, 0.76),
            "disaster": domain(0.84, 0.78),
            "environmental": domain(0.85, 0.77),
            "air_quality": domain(0.83, 0.75),
            "climate": domain(0.82, 0.76),
            "security": domain(0.91, 0.80),
        }
    }


def _minimal_ablation(include_theorem_fields: bool) -> dict[str, Any]:
    advance: dict[str, Any] = {
        "id": "advance_recursion_v1",
        "empirical_gain": True,
        "ethical_gates_preserved": True,
        "stability_preserved": True,
        "originality_preserved": True,
    }
    if include_theorem_fields:
        advance.update(
            {
                "claim": "Recursion sweep improves unknown discovery.",
                "assumptions": "No hard-gate regressions and fixed protocol domains.",
                "proof_sketch": "Paired significance over seeded deltas.",
                "empirical_evidence": "AUC uplift and p<=0.05 across seeds.",
                "risk_limits": "Reject on any ethical/stability regression.",
                "deployment_constraints": "Promotion requires lane consistency.",
            }
        )

    return {
        "component_deltas": {
            "recursion": {"delta_auc": 0.01, "p_value": 0.01},
            "resonance": {"delta_auc": 0.01, "p_value": 0.01},
            "optimization": {"delta_oracle_f1": 0.02, "p_value": 0.01},
        },
        "hard_gates": {
            "benevolence_violations": 0,
            "ethical_regressions": 0,
            "sigma_immutable_violations": 0,
            "gosnn_unavailable_violations": 0,
        },
        "advances": [advance],
    }


def _minimal_gates() -> dict[str, Any]:
    return {
        "gates": {
            "uncertainty": {"coverage_ok": True},
            "calibration": {"ece": 0.02},
            "boundary_stress": {"pass": True},
        },
        "lanes": {
            "humanitarian_first": {"mean_auc": 0.86, "mean_recall": 0.79},
            "security_first": {"mean_auc": 0.87, "mean_recall": 0.80},
        },
    }


def test_protocol_passes_and_emits_theorem_artifact(tmp_path: Path) -> None:
    cfg = _REPO_ROOT / "configs" / "equation_research_protocol.yaml"
    benchmark = tmp_path / "benchmark.json"
    ablation = tmp_path / "ablation.json"
    gates = tmp_path / "gates.json"
    out = tmp_path / "result.json"
    theorem_dir = tmp_path / "theorems"

    _write_json(benchmark, _minimal_benchmark())
    _write_json(ablation, _minimal_ablation(include_theorem_fields=True))
    _write_json(gates, _minimal_gates())

    rc = run_equation_research_protocol.main(
        [
            "--config",
            str(cfg),
            "--benchmark",
            str(benchmark),
            "--ablation",
            str(ablation),
            "--gates",
            str(gates),
            "--out",
            str(out),
            "--theorem-out",
            str(theorem_dir),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["protocol_passed"] is True
    assert payload["promotion"]["promoted_count"] == 1
    assert payload["theorem_artifacts"]
    assert Path(payload["theorem_artifacts"][0]).exists()
    assert payload["ai_equation_library"]["passed"] is True
    assert payload["ai_equation_library"]["sections"]["known_reference_equations"]["count"] > 0
    assert payload["ai_equation_library"]["sections"]["in_house_equations"]["count"] > 0
    assert payload["decision_justification"]


def test_missing_theorem_fields_blocks_promotion(tmp_path: Path) -> None:
    cfg = _REPO_ROOT / "configs" / "equation_research_protocol.yaml"
    benchmark = tmp_path / "benchmark.json"
    ablation = tmp_path / "ablation.json"
    gates = tmp_path / "gates.json"
    out = tmp_path / "result.json"

    _write_json(benchmark, _minimal_benchmark())
    _write_json(ablation, _minimal_ablation(include_theorem_fields=False))
    _write_json(gates, _minimal_gates())

    rc = run_equation_research_protocol.main(
        [
            "--config",
            str(cfg),
            "--benchmark",
            str(benchmark),
            "--ablation",
            str(ablation),
            "--gates",
            str(gates),
            "--out",
            str(out),
        ]
    )

    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["protocol_passed"] is False
    assert payload["promotion"]["missing_theorem_fields"]


def test_missing_config_or_benchmark_exits_2(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    rc = run_equation_research_protocol.main(
        [
            "--config",
            str(tmp_path / "nope.yaml"),
            "--benchmark",
            str(tmp_path / "none.json"),
            "--out",
            str(out),
        ]
    )
    assert rc == 2

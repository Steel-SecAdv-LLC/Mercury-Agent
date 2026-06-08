# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test equation optimizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.equation_optimizer import _cli, _run_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_emits_expected_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    summary = _run_pipeline(
        math_spec=_REPO_ROOT / "docs" / "MATH_SPEC.md",
        dataset_path=None,
        output_dir=out,
        iterations=40,
        seed=11,
    )
    assert summary["ok"] is True
    assert summary["preserve_original_equations"] is True

    expected = [
        "equation_inventory.json",
        "baseline_profile.json",
        "ai_equation_library.json",
        "search_space.json",
        "candidate_ranking.json",
        "winner.json",
        "equation_profiles_v1.json",
        "rollback_switch.json",
        "continuous_revalidation.json",
        "decision_ledger.json",
        "summary.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"missing {name}"

    winner = json.loads((out / "winner.json").read_text(encoding="utf-8"))
    assert "candidate_id" in winner
    assert "metrics" in winner

    baseline = json.loads((out / "baseline_profile.json").read_text(encoding="utf-8"))
    assert baseline["preserve_original_equations"] is True

    ai_library = json.loads((out / "ai_equation_library.json").read_text(encoding="utf-8"))
    assert ai_library["known_reference_equations"]
    assert ai_library["in_house_equations"]
    assert "separation_rule" in ai_library


def test_cli_runs_and_reports_json(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.equation_optimizer",
            "--math-spec",
            str(_REPO_ROOT / "docs" / "MATH_SPEC.md"),
            "--output-dir",
            str(out),
            "--iterations",
            "25",
            "--seed",
            "5",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["winner_id"]
    assert payload["inventory_count"] >= 8


def test_pipeline_and_cli_fail_closed_when_hard_constraints_fail(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "r": 0.7,
                    "h": 0.6,
                    "o": 0.5,
                    "eta": 0.95,
                    "label": 0.8,
                    "alpha": 1.2,
                    "lyapunov_lambda": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "hard_gate_artifacts"
    summary = _run_pipeline(
        math_spec=_REPO_ROOT / "docs" / "MATH_SPEC.md",
        dataset_path=dataset,
        output_dir=out,
        iterations=5,
        seed=17,
    )

    assert summary["ok"] is False
    assert summary["winner_constraints_ok"] is False
    assert (
        _cli(
            [
                "--math-spec",
                str(_REPO_ROOT / "docs" / "MATH_SPEC.md"),
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "cli_hard_gate_artifacts"),
                "--iterations",
                "5",
                "--seed",
                "17",
            ]
        )
        == 1
    )

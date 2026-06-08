# Copyright (C) 2025 Steel Security Advisors LLC
"""Test compare runtime equation profiles."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import compare_runtime_equation_profiles

if TYPE_CHECKING:
    from pathlib import Path


def test_compare_runtime_equation_profiles_emits_report(tmp_path: Path) -> None:
    out = tmp_path / "runtime_equation_compare.json"
    rc = compare_runtime_equation_profiles.main(
        [
            "--seed",
            "3",
            "--n",
            "120",
            "--out",
            str(out),
        ]
    )

    assert rc in (0, 1)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["baseline_profile"] == "baseline_original_v1"
    assert payload["candidate_profile"] == "quiet_horizon_v1"
    assert "auc" in payload["baseline"]
    assert "neuro_symbolic_satisfaction" in payload["candidate"]
    assert "latency_ms_per_1k" in payload["delta"]
    assert "security" in payload["candidate"]["domains"]
    assert isinstance(payload["hard_gates_preserved"], bool)

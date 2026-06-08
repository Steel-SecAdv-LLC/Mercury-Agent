# Copyright (C) 2025 Steel Security Advisors LLC
"""Unit tests for the fusion regression guard's *gate logic* (WS5/WS6)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks import fusion_regression_guard as guard

if TYPE_CHECKING:
    from pathlib import Path


def _baseline() -> dict[str, Any]:
    return {
        "auc": 0.99,
        "f1": 0.95,
        "conformal": {"target_coverage": 0.9, "empirical_coverage": 0.9, "average_set_size": 1.2},
    }


def test_floors_are_measured_minus_margin() -> None:
    floors = guard._floors_from(_baseline())
    assert floors["auc_floor"] == pytest.approx(0.99 - guard.AUC_MARGIN)
    assert floors["f1_floor"] == pytest.approx(0.95 - guard.F1_MARGIN)
    # Coverage floor is target - margin (a coverage guarantee floor).
    assert floors["coverage_floor"] == pytest.approx(0.9 - guard.COVERAGE_MARGIN)


def test_check_passes_when_metrics_above_floors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(_baseline()))
    monkeypatch.setattr(guard, "BASELINE_PATH", bpath)
    measured = {
        "auc": 0.99,
        "f1": 0.95,
        "conformal": {"target_coverage": 0.9, "empirical_coverage": 0.9, "average_set_size": 1.2},
    }
    assert guard.check(measured) == []


def test_check_fails_on_auc_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(_baseline()))
    monkeypatch.setattr(guard, "BASELINE_PATH", bpath)
    measured = {
        "auc": 0.80,  # well below 0.99 - 0.05
        "f1": 0.95,
        "conformal": {"target_coverage": 0.9, "empirical_coverage": 0.9, "average_set_size": 1.2},
    }
    violations = guard.check(measured)
    assert any("AUC" in v for v in violations)


def test_check_fails_on_coverage_collapse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(_baseline()))
    monkeypatch.setattr(guard, "BASELINE_PATH", bpath)
    measured = {
        "auc": 0.99,
        "f1": 0.95,
        "conformal": {"target_coverage": 0.9, "empirical_coverage": 0.70, "average_set_size": 1.0},
    }
    violations = guard.check(measured)
    assert any("coverage" in v for v in violations)


def test_check_reports_missing_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "BASELINE_PATH", tmp_path / "absent.json")
    violations = guard.check(
        {
            "auc": 1.0,
            "f1": 1.0,
            "conformal": {
                "target_coverage": 0.9,
                "empirical_coverage": 0.9,
                "average_set_size": 1.0,
            },
        }
    )
    assert violations and "baseline missing" in violations[0]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for the fusion regression guard's *gate logic* (WS5/WS6).

These exercise floor computation and the pass/fail decision with synthetic
metrics — fast, no training. The real train+eval measurement (``evaluate``) is
run by the ``fusion-regression`` CI workflow and pinned in
``benchmarks/fusion_capacity/fusion_gate_baseline.json``.
"""

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


def _fit(fit_seed: int, auc: float, f1: float, coverage: float = 0.92) -> dict[str, Any]:
    return {
        "fit_seed": fit_seed,
        "n_train": 10,
        "n_cal": 5,
        "n_test": 5,
        "auc": auc,
        "f1": f1,
        "conformal": {
            "target_coverage": 0.9,
            "empirical_coverage": coverage,
            "average_set_size": 1.0,
        },
    }


def test_the_gated_metrics_are_medians_over_the_seeded_fits() -> None:
    """One fit is one draw; the gate reads the middle of several."""
    fits = [_fit(0, 0.99, 0.87), _fit(1, 0.98, 0.83), _fit(2, 1.0, 0.99)]
    fits += [_fit(3, 0.995, 0.97), _fit(4, 0.999, 0.985)]
    told = guard.summarize(fits, epochs=8, seed=1)

    assert (told["auc"], told["f1"]) == (0.995, 0.97)
    assert told["metadata"]["fit_seeds"] == [0, 1, 2, 3, 4]
    assert [one["f1"] for one in told["realizations"]] == [0.87, 0.83, 0.99, 0.97, 0.985]


def test_one_unlucky_fit_passes_and_a_regressed_majority_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bpath = tmp_path / "baseline.json"
    bpath.write_text(json.dumps(_baseline()))
    monkeypatch.setattr(guard, "BASELINE_PATH", bpath)
    good = [_fit(seed, 0.99, 0.95) for seed in range(4)]

    assert guard.check(guard.summarize([*good, _fit(4, 0.99, 0.5)], 8, 1)) == []
    regressed = [*good[:2], *(_fit(seed, 0.99, 0.5) for seed in (2, 3, 4))]
    assert any("F1" in line for line in guard.check(guard.summarize(regressed, 8, 1)))


def test_evaluate_fits_at_the_gates_thread_count_and_restores_the_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fitted model must not depend on how many cores the runner has."""
    torch = pytest.importorskip("torch")
    seen: list[tuple[int, int]] = []

    def one(fit_seed: int, epochs: int, seed: int) -> dict[str, Any]:
        seen.append((fit_seed, torch.get_num_threads()))
        return _fit(fit_seed, 0.99, 0.95)

    monkeypatch.setattr(guard, "_evaluate_one", one)
    before = torch.get_num_threads()
    told = guard.evaluate(fit_seeds=(3, 1, 2))

    assert seen == [(seed, guard.GATE_TORCH_THREADS) for seed in (3, 1, 2)]
    assert torch.get_num_threads() == before
    assert told["metadata"]["torch_threads"] == guard.GATE_TORCH_THREADS
    assert len(guard.GATE_FIT_SEEDS) % 2 == 1

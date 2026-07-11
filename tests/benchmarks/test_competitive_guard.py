# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the deterministic competitive-position regression guard.

Two tiers, mirroring ``test_hazard_regression_guard`` /
``test_anomaly_regression_guard`` style:

* A fast, no-network tier validates the committed baseline's structure and
  provenance, the derived floors' non-vacuity (strictly below the pinned
  measurements AND strictly above a seeded random-scores strawman computed
  from the baseline's own recorded test-set label counts), and drives
  ``check()`` with synthetic degraded measurements to prove every gate --
  absolute floors and the competitive-position gap ceiling -- can actually
  fail.
* A ``network`` + ``slow`` tier runs the live measurement and asserts it
  clears every pinned bound (auto-skipped unless ``MERCURY_NETWORK_TESTS=1``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_GUARD = _REPO / "benchmarks" / "competitive_regression_guard.py"
_BASELINE = _REPO / "benchmarks" / "competitive_baseline.json"


def _load_guard() -> Any:
    spec = importlib.util.spec_from_file_location("competitive_regression_guard", _GUARD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_BASELINE.read_text())
    return data


# ---------------------------------------------------------------------------
# Fast tier: baseline structure + provenance
# ---------------------------------------------------------------------------


def test_baseline_exists_and_parses() -> None:
    assert _BASELINE.exists(), "competitive baseline must be committed (run --update)"
    data = _baseline()
    assert set(data) >= {"metadata", "datasets", "aggregate"}


def test_baseline_has_full_provenance() -> None:
    meta = _baseline()["metadata"]
    for key in (
        "seed",
        "mercury_method",
        "eval_path",
        "pyod_methods",
        "commit",
        "dataset_source",
        "dataset_license",
        "margins",
        "pyod_version",
    ):
        assert key in meta, f"missing provenance field: {key}"
    assert meta["seed"] == 42
    assert meta["margins"]["justification"], "margins must be justified, not bare numbers"
    for name, d in _baseline()["datasets"].items():
        assert len(d["npz_sha256"]) == 64, f"{name}: dataset content hash required"
        assert 0.0 <= d["mercury_tier_auc"] <= 1.0
        assert d["n_test"] > 0 and 0 < d["n_test_anomalies"] < d["n_test"]
        for algo, auc in d["pyod_auc"].items():
            assert 0.0 <= auc <= 1.0, f"{name}/{algo}"


def test_guard_set_is_fixed_and_covered() -> None:
    mod = _load_guard()
    data = _baseline()
    assert set(mod.GUARD_DATASETS) == set(data["datasets"]), "baseline must cover the fixed set"
    # Below-random / circular datasets must not creep in.
    assert "vertebral" not in mod.GUARD_DATASETS


def test_baseline_never_stores_floors() -> None:
    """Floors are derived (measured - margin), never persisted."""
    raw = _BASELINE.read_text()
    assert "floor" not in raw and "ceiling" not in raw, (
        "baseline must store measurements + margins only; floors are derived by "
        "_floors_from at check time"
    )


# ---------------------------------------------------------------------------
# Fast tier: floors are non-vacuous
# ---------------------------------------------------------------------------


def test_floors_are_below_measured_baseline() -> None:
    """Floors strictly below pinned values; gap ceiling strictly above."""
    mod = _load_guard()
    data = _baseline()
    floors = mod._floors_from(data)
    for name, d in data["datasets"].items():
        assert floors["datasets"][name]["mercury_auc_floor"] < d["mercury_tier_auc"], name
    assert floors["mercury_mean_auc_floor"] < data["aggregate"]["mercury_tier_mean_auc"]
    assert floors["competitive_gap_ceiling"] > data["aggregate"]["competitive_gap"]


def _random_strawman_auc(n_test: int, n_pos: int, seeds: range) -> float:
    """Best ROC-AUC a seeded random scorer achieves on (n_test, n_pos) labels."""
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_test - n_pos, dtype=int)])
    best = 0.0
    for seed in seeds:
        scores = np.random.RandomState(seed).uniform(size=n_test)
        best = max(best, float(roc_auc_score(labels, scores)))
    return best


def test_floors_beat_random_scores_strawman() -> None:
    """Every AUC floor must certify skill a random scorer cannot fake.

    The strawman is computed from the baseline's own recorded test-set label
    counts (AUC is invariant to which rows are positive, only counts matter),
    maximised over 25 seeds so the bound is not a fluke of one draw.
    """
    mod = _load_guard()
    data = _baseline()
    floors = mod._floors_from(data)
    for name, d in data["datasets"].items():
        strawman = _random_strawman_auc(d["n_test"], d["n_test_anomalies"], range(25))
        floor = floors["datasets"][name]["mercury_auc_floor"]
        assert floor > strawman, (
            f"{name}: floor {floor:.4f} does not beat the random strawman "
            f"{strawman:.4f} -- the gate would certify noise"
        )
    # Mean floor beats the random scorer's mean (~0.5) by a real margin too.
    assert floors["mercury_mean_auc_floor"] > 0.6


def test_gap_ceiling_is_a_real_competitive_bound() -> None:
    """The pinned gap ceiling must be small: within margin of the measurement.

    If a future re-pin ever records Mercury far behind the best PyOD baseline,
    this makes the decision visible instead of silently normalising it.
    """
    mod = _load_guard()
    agg = _baseline()["aggregate"]
    ceiling = mod._floors_from(_baseline())["competitive_gap_ceiling"]
    assert ceiling == pytest.approx(agg["competitive_gap"] + mod.GAP_MARGIN, abs=1e-4)
    # The measured position itself: best-PyOD-minus-Mercury stays under 5
    # AUC points on the guard subset. A re-pin that violates this is a real
    # competitive regression and must be a deliberate, visible decision.
    assert agg["competitive_gap"] < 0.05


# ---------------------------------------------------------------------------
# Gate-logic tier: check() can actually fail
# ---------------------------------------------------------------------------


def test_check_passes_on_identical_measurement() -> None:
    mod = _load_guard()
    assert mod.check(_baseline()) == []


def test_check_fails_on_degraded_dataset_auc() -> None:
    mod = _load_guard()
    measured = copy.deepcopy(_baseline())
    name = next(iter(measured["datasets"]))
    measured["datasets"][name]["mercury_tier_auc"] = 0.5
    violations = mod.check(measured)
    assert any(name in v and "mercury AUC" in v for v in violations), violations


def test_check_fails_on_degraded_mean_auc() -> None:
    mod = _load_guard()
    measured = copy.deepcopy(_baseline())
    measured["aggregate"]["mercury_tier_mean_auc"] = 0.5
    violations = mod.check(measured)
    assert any("mercury mean AUC" in v for v in violations), violations


def test_check_fails_when_competitive_gap_widens() -> None:
    """PyOD pulling ahead (or Mercury slipping) must trip the position gate."""
    mod = _load_guard()
    measured = copy.deepcopy(_baseline())
    agg = measured["aggregate"]
    agg["competitive_gap"] = float(agg["competitive_gap"]) + mod.GAP_MARGIN + 0.01
    agg["best_pyod_mean_auc"] = agg["mercury_tier_mean_auc"] + agg["competitive_gap"]
    violations = mod.check(measured)
    assert any("competitive gap" in v for v in violations), violations


def test_check_reports_dataset_missing_from_measurement() -> None:
    mod = _load_guard()
    measured = copy.deepcopy(_baseline())
    name = next(iter(measured["datasets"]))
    del measured["datasets"][name]
    violations = mod.check(measured)
    assert any(name in v and "missing" in v for v in violations), violations


def test_check_reports_missing_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_guard()
    monkeypatch.setattr(mod, "BASELINE_PATH", tmp_path / "absent.json")
    violations = mod.check(_baseline())
    assert violations and "baseline missing" in violations[0]


def test_check_fails_against_mutated_stricter_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tightening a pinned value beyond reality must trip the gate."""
    mod = _load_guard()
    mutated = copy.deepcopy(_baseline())
    name = next(iter(mutated["datasets"]))
    mutated["datasets"][name]["mercury_tier_auc"] = 1.0
    bpath = tmp_path / "mutated_baseline.json"
    bpath.write_text(json.dumps(mutated))
    monkeypatch.setattr(mod, "BASELINE_PATH", bpath)
    violations = mod.check(_baseline())  # the *real* measurement
    assert any(name in v for v in violations), violations


# ---------------------------------------------------------------------------
# Network + slow tier: live measurement must clear the pinned bounds
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_live_guard_clears_floors() -> None:
    mod = _load_guard()
    violations = mod.check()
    assert not violations, "competitive position regressed below pinned bounds:\n" + "\n".join(
        violations
    )

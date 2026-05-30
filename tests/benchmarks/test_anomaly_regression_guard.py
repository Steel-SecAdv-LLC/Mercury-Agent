"""Tests for the deterministic anomaly-detector regression guard (WS-A).

Two tiers:

* A fast, no-network tier validates the committed baseline's structure,
  provenance, and floor derivation -- runs in every lane.
* A ``network`` + ``slow`` tier downloads the fixed ADBench subset and asserts
  the live detector clears every pinned floor -- auto-skipped unless
  ``MERCURY_NETWORK_TESTS=1`` (see ``tests/conftest.py``).

Context: issue #261's apparent regression (AUC 0.8466->0.8259) was the PR #255
eval-honesty de-leak, not a detector regression (see
``docs/ANOMALY_REGRESSION_WS_A.md``).  This guard pins the *real* per-dataset
metric floor so a genuine future regression cannot land silently.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_GUARD = _REPO / "benchmarks" / "anomaly_regression_guard.py"
_BASELINE = _REPO / "benchmarks" / "anomaly_regression_baseline.json"


def _load_guard():
    spec = importlib.util.spec_from_file_location("anomaly_regression_guard", _GUARD)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fast, no-network tier
# ---------------------------------------------------------------------------


def test_baseline_exists_and_parses():
    assert _BASELINE.exists(), "regression baseline must be committed"
    data = json.loads(_BASELINE.read_text())
    assert set(data) >= {"metadata", "datasets", "aggregate"}


def test_baseline_has_full_provenance():
    data = json.loads(_BASELINE.read_text())
    meta = data["metadata"]
    # Determinism + provenance contract required by the project's operating rules.
    for key in ("seed", "detector", "eval_path", "commit", "dataset_source", "dataset_license"):
        assert key in meta, f"missing provenance field: {key}"
    assert meta["seed"] == 42
    assert meta["dataset_license"] == "MIT"
    for name, d in data["datasets"].items():
        assert len(d["npz_sha256"]) == 64, f"{name}: dataset content hash required"
        assert 0.0 <= d["auc"] <= 1.0
        assert 0.0 <= d["f1"] <= 1.0


def test_floors_are_below_measured_baseline():
    """Floors must sit *below* the pinned values (guard, not a tautology)."""
    mod = _load_guard()
    data = json.loads(_BASELINE.read_text())
    floors = mod._floors_from(data)
    for name, d in data["datasets"].items():
        assert floors["datasets"][name]["auc_floor"] <= d["auc"]
        assert floors["datasets"][name]["f1_floor"] <= d["f1"]
    assert floors["mean_auc_floor"] <= data["aggregate"]["mean_auc"]
    assert floors["mean_f1_floor"] <= data["aggregate"]["mean_f1"]


def test_guard_set_is_fixed_and_genuine():
    """The guard set is a documented, fixed list of genuine-label datasets."""
    mod = _load_guard()
    data = json.loads(_BASELINE.read_text())
    assert set(mod.GUARD_DATASETS) == set(data["datasets"]), "baseline must cover the fixed set"
    # Below-random / circular datasets must not creep in.
    assert "vertebral" not in mod.GUARD_DATASETS


# ---------------------------------------------------------------------------
# Network + slow tier: live detector must clear the floors
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_live_detector_clears_floors():
    mod = _load_guard()
    violations = mod.check()
    assert not violations, "anomaly detector regressed below pinned floors:\n" + "\n".join(
        violations
    )

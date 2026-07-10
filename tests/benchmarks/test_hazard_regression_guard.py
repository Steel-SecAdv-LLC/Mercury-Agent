# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the deterministic hazard-detector regression guard.

Three tiers, all offline (the scenario sets are committed and hash-pinned):

* A fast tier validates the committed baseline's structure and provenance,
  the scenario manifest integrity, and -- critically -- that every floor is
  **non-vacuous**: strictly beyond the pinned measurement AND strictly better
  than the degenerate predictors (always-alarm / never-alarm / majority-class
  / climatology-mean) computed in-test from the same committed scenario sets.
* A gate-logic tier drives ``check()`` with synthetic degraded measurements
  and mutated baselines to prove the guard can actually fail.
* A ``slow`` tier runs the full live measurement twice (determinism) and
  asserts it clears every pinned bound, plus regenerates the constructed
  scenario sets from their seeds and confirms the content matches the
  committed sets (exact for integers/strings, last-ulp float tolerance --
  numpy's CPU-dispatched transcendental kernels are not bit-stable across
  hardware). No network marker is needed anywhere.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks import hazard_regression_guard as guard
from benchmarks.hazard_scenarios import scenario_io

_REPO = Path(__file__).resolve().parents[2]
_BASELINE = _REPO / "benchmarks" / "hazard_domain_baseline.json"
_SCENARIOS = _REPO / "benchmarks" / "hazard_scenarios"

EXPECTED_DOMAINS = {"tornado", "flood", "hurricane", "earthquake", "tsunami", "volcano", "solar"}


def _baseline() -> dict[str, Any]:
    baseline: dict[str, Any] = json.loads(_BASELINE.read_text())
    return baseline


# ---------------------------------------------------------------------------
# Fast tier: baseline structure, provenance, manifest integrity
# ---------------------------------------------------------------------------


def test_baseline_exists_and_parses() -> None:
    assert _BASELINE.exists(), "hazard regression baseline must be committed"
    data = _baseline()
    assert set(data) >= {"metadata", "domains"}
    assert set(data["domains"]) == EXPECTED_DOMAINS


def test_baseline_has_full_provenance() -> None:
    meta = _baseline()["metadata"]
    for key in ("commit", "python", "numpy", "margins", "honesty_tripwires", "scenario_seeds"):
        assert key in meta, f"missing provenance field: {key}"
    assert meta["margins"]["justification"], "margins must be justified, not bare numbers"
    for domain, entry in _baseline()["domains"].items():
        assert entry["label_source"] in ("constructed", "measured"), domain
        for name, sha in entry["scenario_files"].items():
            assert len(sha) == 64, f"{domain}/{name}: content hash required"


def test_manifest_provenance_complete() -> None:
    manifest = scenario_io.load_manifest()
    for name, entry in manifest["files"].items():
        assert len(entry["sha256"]) == 64, name
        if entry["label_source"] == "measured":
            prov = entry["provenance"]
            assert prov["fetched_at"].startswith("2026-"), name
            for source in prov["sources"]:
                assert source["url"].startswith("https://services.swpc.noaa.gov/"), name
                assert len(source["sha256"]) == 64, f"{name}: raw payload hash required"
                assert source["rows"] > 0, name
        else:
            assert entry["label_source"] == "constructed", name
            assert entry["construction"], f"{name}: construction must be documented"


def test_scenario_files_match_manifest() -> None:
    """Committed scenario bytes must match their pinned manifest hashes."""
    for spec in guard.HAZARD_METRICS.values():
        for name in spec["scenario_files"]:
            scenario_io.verify_file_against_manifest(name)


def test_guard_set_is_fixed_and_exclusions_documented() -> None:
    assert set(guard.HAZARD_METRICS) == EXPECTED_DOMAINS
    # Hurricane track error must stay an explicit, documented exclusion (no
    # track model exists); same for the untrained magnitude estimate.
    assert "track_error_km" in guard.HAZARD_METRICS["hurricane"]["exclusions"]
    assert "magnitude_mae" in guard.HAZARD_METRICS["earthquake"]["exclusions"]
    for domain, spec in guard.HAZARD_METRICS.items():
        assert spec["metrics"], f"{domain}: guarded domain without gated metrics"


# ---------------------------------------------------------------------------
# Fast tier: floors are non-vacuous
# ---------------------------------------------------------------------------


def test_floors_are_beyond_measured_baseline() -> None:
    """Floors strictly below (ceilings strictly above) the pinned values."""
    floors = guard._floors_from(_baseline())
    for domain, bounds in floors["domains"].items():
        measured = _baseline()["domains"][domain]["metrics"]
        for metric, bound in bounds.items():
            if "floor" in bound:
                assert bound["floor"] < measured[metric], f"{domain}.{metric} floor tautology"
            else:
                assert bound["ceiling"] > measured[metric], f"{domain}.{metric} ceiling tautology"


def _detection_labels(domain: str) -> np.ndarray:
    name = guard.HAZARD_METRICS[domain]["scenario_files"][0]
    path = _SCENARIOS / name
    if name.endswith(".npz"):
        with np.load(path) as data:
            return np.asarray(data["labels"])
    payload = json.loads(path.read_text())
    return np.array([s["label"] for s in payload["scenarios"]])


@pytest.mark.parametrize("domain", ["tornado", "flood", "hurricane", "earthquake", "tsunami"])
def test_detection_floors_beat_degenerate_forecasters(domain: str) -> None:
    """POD/CSI floors beat always-alarm; FAR ceiling forbids it; POD floor > 0.

    Computed in-test from the same committed scenario labels the guard runs
    on, so these floors are genuinely non-vacuous: an always-alarm detector
    scores CSI == base rate and FAR == null fraction, and both must violate
    the pinned bounds.
    """
    labels = _detection_labels(domain)
    n_events, n_nulls = int(labels.sum()), int((labels == 0).sum())
    assert n_events >= 10 and n_nulls >= 10, "scenario set too thin to be meaningful"
    always_alarm_csi = n_events / (n_events + n_nulls)
    always_alarm_far = n_nulls / (n_events + n_nulls)
    bounds = guard._floors_from(_baseline())["domains"][domain]
    assert bounds["pod"]["floor"] > 0.0
    assert bounds["csi"]["floor"] > always_alarm_csi, "CSI floor must beat always-alarm"
    assert bounds["far"]["ceiling"] < always_alarm_far, "FAR ceiling must forbid always-alarm"


def test_volcano_floor_beats_majority_class() -> None:
    payload = json.loads((_SCENARIOS / "volcano_scenarios.json").read_text())
    labels = [s["alert_level"] for s in payload["scenarios"]]
    levels = guard.VOLCANO_ALERT_LEVELS
    best_exact = max(labels.count(level) / len(labels) for level in levels)
    idx = {level: i for i, level in enumerate(levels)}
    best_within1 = max(
        sum(abs(idx[label] - idx[level]) <= 1 for label in labels) / len(labels) for level in levels
    )
    bounds = guard._floors_from(_baseline())["domains"]["volcano"]
    assert bounds["alert_exact_accuracy"]["floor"] > best_exact
    assert bounds["alert_within_one_accuracy"]["floor"] > best_within1


def test_flare_floor_beats_majority_class() -> None:
    payload = json.loads((_SCENARIOS / "solar_flare_windows.json").read_text())
    labels = [w["flare_class"] for w in payload["windows"]]
    best_exact = max(labels.count(c) / len(labels) for c in guard.FLARE_CLASSES)
    bounds = guard._floors_from(_baseline())["domains"]["solar"]
    assert bounds["flare_class_exact_accuracy"]["floor"] > best_exact
    # within-one is deliberately ungated: always-M scores 27/28 within-one on
    # this real week (C/M dominance), so a floor there would be near-vacuous.
    assert "flare_class_within_one_accuracy" not in bounds


def test_kp_ceiling_beats_climatology_and_zero_predictors() -> None:
    """The Kp MAE ceiling must certify skill over trivial predictors.

    Computed from the committed real windows: predicting 0 always, or the
    week's own mean Kp (climatology), must both violate the pinned ceiling --
    i.e. the gate certifies the Boyle physics genuinely beats them.
    """
    payload = json.loads((_SCENARIOS / "solar_kp_windows.json").read_text())
    kp = np.array([w["kp_observed"] for w in payload["windows"]])
    mae_always_zero = float(np.mean(np.abs(kp)))
    mae_climatology = float(np.mean(np.abs(kp - kp.mean())))
    ceiling = guard._floors_from(_baseline())["domains"]["solar"]["kp_mae"]["ceiling"]
    assert ceiling < mae_always_zero
    assert ceiling < mae_climatology, (
        "kp_mae ceiling must sit below the climatology-mean predictor's MAE; "
        "if a re-pin broke this, the Boyle path has lost real skill"
    )


def test_kp_g_bucket_accuracy_is_reported_but_not_gated() -> None:
    """The G-bucket rate is deliberately ungated (vacuous vs. degenerates).

    On the committed quiet-dominated week the always-G0 predictor scores
    49/55 = 0.891 exact -- above the physics path's own 0.873 -- and on the
    six Kp>=5 storm windows the Boyle path's exact-bucket skill (1/6) sits
    below even always-G1 (3/6). A floor here could only certify quietness.
    kp_mae carries the solar skill gate instead (see
    ``test_kp_ceiling_beats_climatology_and_zero_predictors``).
    """
    payload = json.loads((_SCENARIOS / "solar_kp_windows.json").read_text())
    kp = [w["kp_observed"] for w in payload["windows"]]
    always_g0_accuracy = sum(k < 5.0 for k in kp) / len(kp)
    measured = float(_baseline()["domains"]["solar"]["metrics"]["kp_g_bucket_accuracy"])
    # The premise that makes gating vacuous, pinned so a future window
    # re-record that changes it forces this decision to be revisited:
    assert always_g0_accuracy > measured
    bounds = guard._floors_from(_baseline())["domains"]["solar"]
    assert "kp_g_bucket_accuracy" not in bounds


def test_check_reports_domain_missing_from_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A baseline missing a guarded domain must be a loud violation.

    Regression: ``_floors_from`` used to KeyError first, making check()'s
    friendly "missing from baseline" branch unreachable.
    """
    baseline = _baseline()
    del baseline["domains"]["solar"]
    reduced = tmp_path / "reduced_baseline.json"
    reduced.write_text(json.dumps(baseline))
    monkeypatch.setattr(guard, "BASELINE_PATH", reduced)

    floors = guard._floors_from(baseline)  # must not raise
    assert "solar" not in floors["domains"]

    violations = guard.check(_baseline())  # full measurement stands in
    assert any("solar: missing from baseline" in v for v in violations)


def test_kp_windows_span_storm_and_quiet_conditions() -> None:
    """The real week must include both quiet and G3+ storm windows -- a flat
    week could not pin a meaningful Kp baseline."""
    payload = json.loads((_SCENARIOS / "solar_kp_windows.json").read_text())
    kp = [w["kp_observed"] for w in payload["windows"]]
    assert min(kp) < 2.0 and max(kp) >= 7.0
    assert len(kp) >= 40


# ---------------------------------------------------------------------------
# Gate-logic tier: check() can actually fail
# ---------------------------------------------------------------------------


def test_check_passes_on_identical_measurement() -> None:
    assert guard.check(_baseline()) == []


@pytest.mark.parametrize(
    ("domain", "metric", "bad_value", "needle"),
    [
        ("tornado", "pod", 0.75, "pod"),
        ("tornado", "far", 0.25, "far"),
        ("flood", "mean_lead_time_hours", 3.0, "mean_lead_time_hours"),
        ("earthquake", "sp_distance_mae_km", 5.0, "sp_distance_mae_km"),
        ("volcano", "alert_exact_accuracy", 0.5, "alert_exact_accuracy"),
        ("solar", "kp_mae", 2.5, "kp_mae"),
        ("solar", "flare_class_exact_accuracy", 0.6, "flare_class_exact_accuracy"),
    ],
)
def test_check_fails_on_degraded_metric(
    domain: str, metric: str, bad_value: float, needle: str
) -> None:
    measured = copy.deepcopy(_baseline())
    measured["domains"][domain]["metrics"][metric] = bad_value
    violations = guard.check(measured)
    assert any(domain in v and needle in v for v in violations), violations


def test_check_fails_on_scenario_set_drift() -> None:
    measured = copy.deepcopy(_baseline())
    files = measured["domains"]["tsunami"]["scenario_files"]
    files["tsunami_scenarios.npz"] = "0" * 64
    violations = guard.check(measured)
    assert any("tsunami" in v and "changed since the baseline" in v for v in violations)


def test_check_fails_against_mutated_stricter_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tightening a pinned value beyond reality must trip the gate."""
    mutated = copy.deepcopy(_baseline())
    mutated["domains"]["solar"]["metrics"]["kp_mae"] = 0.05
    bpath = tmp_path / "mutated_baseline.json"
    bpath.write_text(json.dumps(mutated))
    monkeypatch.setattr(guard, "BASELINE_PATH", bpath)
    violations = guard.check(_baseline())  # the *real* measurement
    assert any("kp_mae" in v for v in violations)


def test_check_reports_missing_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "BASELINE_PATH", tmp_path / "absent.json")
    violations = guard.check(_baseline())
    assert violations and "baseline missing" in violations[0]


def test_tampered_scenario_file_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scenario file whose bytes drift from the manifest must raise."""
    manifest = scenario_io.load_manifest()
    tampered = copy.deepcopy(manifest)
    tampered["files"]["volcano_scenarios.json"]["sha256"] = "f" * 64
    fake_manifest = tmp_path / "manifest.json"
    fake_manifest.write_text(json.dumps(tampered))
    monkeypatch.setattr(scenario_io, "MANIFEST_PATH", fake_manifest)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        scenario_io.verify_file_against_manifest("volcano_scenarios.json")


# ---------------------------------------------------------------------------
# Slow tier: live measurement + scenario reproducibility (offline)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_live_guard_clears_floors_and_is_deterministic() -> None:
    """The real detectors must clear every pinned bound, twice identically."""
    first = guard.evaluate()
    violations = guard.check(first)
    assert not violations, "hazard detectors regressed below pinned floors:\n" + "\n".join(
        violations
    )
    second = guard.evaluate()
    for domain in EXPECTED_DOMAINS:
        assert (
            first["domains"][domain]["metrics"] == second["domains"][domain]["metrics"]
        ), f"{domain}: evaluation is not deterministic"


def _assert_json_equal_with_float_tolerance(committed: Any, rebuilt: Any, ctx: str) -> None:
    """Structural equality with last-ulp float tolerance (see regen test)."""
    if isinstance(committed, dict):
        assert isinstance(rebuilt, dict), f"{ctx}: type diverged"
        assert set(committed) == set(rebuilt), f"{ctx}: keys diverged"
        for k in committed:
            _assert_json_equal_with_float_tolerance(committed[k], rebuilt[k], f"{ctx}.{k}")
    elif isinstance(committed, list):
        assert isinstance(rebuilt, list), f"{ctx}: type diverged"
        assert len(committed) == len(rebuilt), f"{ctx}: length diverged"
        for i, (c, r) in enumerate(zip(committed, rebuilt, strict=True)):
            _assert_json_equal_with_float_tolerance(c, r, f"{ctx}[{i}]")
    elif isinstance(committed, float) and not isinstance(committed, bool):
        assert isinstance(rebuilt, (int, float)), f"{ctx}: type diverged"
        assert rebuilt == pytest.approx(committed, rel=1e-9, abs=1e-12), f"{ctx}: value diverged"
    else:
        assert committed == rebuilt, f"{ctx}: value diverged"


@pytest.mark.slow
def test_constructed_sets_regenerate_from_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running the committed generator reproduces the committed content.

    Integer/string/structural content must match **exactly**; floating-point
    content must match to within last-ulp tolerance (rel 1e-9). Bit-exactness
    is deliberately NOT asserted here: numpy dispatches transcendental kernels
    (``exp``/``sin``) by runtime CPU capability, so the same seed can produce
    last-ulp float differences on different hardware (observed: the committed
    earthquake set regenerated on a CI runner differed only in float bits).
    The committed files themselves stay byte/content-pinned in the manifest --
    that is what makes the *gate* deterministic and tamper-evident (see
    ``test_scenario_files_match_manifest``); this test proves those
    committed files are what the seeded generator actually produces.
    """
    from benchmarks.hazard_scenarios import generate_scenarios as gen

    monkeypatch.setattr(gen, "SCENARIO_DIR", tmp_path)
    rebuilt_names = {
        "tornado_scenarios.npz": gen.build_tornado,
        "earthquake_scenarios.npz": gen.build_earthquake,
        "tsunami_scenarios.npz": gen.build_tsunami,
        "hurricane_scenarios.npz": gen.build_hurricane,
        "flood_scenarios.json": gen.build_flood,
        "volcano_scenarios.json": gen.build_volcano,
    }
    for name, build in rebuilt_names.items():
        build()
        committed_path = _SCENARIOS / name
        rebuilt_path = tmp_path / name
        if name.endswith(".npz"):
            with np.load(committed_path) as committed, np.load(rebuilt_path) as regen:
                assert sorted(committed.files) == sorted(regen.files), f"{name}: keys diverged"
                for key in committed.files:
                    c, r = committed[key], regen[key]
                    assert c.dtype == r.dtype, f"{name}[{key}]: dtype diverged"
                    assert c.shape == r.shape, f"{name}[{key}]: shape diverged"
                    if np.issubdtype(c.dtype, np.floating):
                        np.testing.assert_allclose(
                            r,
                            c,
                            rtol=1e-9,
                            atol=1e-12,
                            err_msg=f"{name}[{key}]: regeneration diverged beyond float ulp",
                        )
                    else:
                        assert np.array_equal(c, r), f"{name}[{key}]: exact content diverged"
        else:
            committed_payload = json.loads(committed_path.read_text())
            rebuilt_payload = json.loads(rebuilt_path.read_text())
            _assert_json_equal_with_float_tolerance(committed_payload, rebuilt_payload, name)

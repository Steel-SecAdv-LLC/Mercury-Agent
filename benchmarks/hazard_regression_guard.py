# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic hazard-detector regression guard.

Why this exists
---------------
The hazard honesty wave (volcanic 4a3ba33, space 30a5582, tsunami/earthquake
1afd151, meteorological 56903a5) replaced untrained-network theater with
deterministic physics paths in the hazard detectors -- but nothing pinned
their *skill*. A future change could silently break the Doppler couplet
maths, the STA/LTA picker, the Boyle-index coupling, or the alert-level
logic, and no CI lane would notice. This guard closes that: it runs every
guarded detector over committed, hash-pinned scenario sets (real recorded
SWPC data where an allow-listed feed provides labelled series; seeded
physics scenarios built against the documented input contracts elsewhere --
see ``benchmarks/hazard_scenarios/generate_scenarios.py``), computes the
standard skill scores from ``omni_mercury_engine.evaluation.hazard_metrics``,
and fails non-zero if any metric crosses its pinned floor/ceiling.

Fully offline and deterministic: scenario files are committed and verified
against ``hazard_scenarios/manifest.json`` before use; detectors run their
untrained-physics paths, which are RNG-free by the honesty-wave contract.

Honesty tripwires (fail loud, not a metric): the untrained
``EarthquakeDetector`` must keep ``estimated_magnitude is None`` and the
``HurricanePredictionResult`` must not regrow the removed track-forecast
fields. Hurricane track error is EXCLUDED from the registry -- see
``HAZARD_METRICS["hurricane"]["exclusions"]``.

Usage::

    python benchmarks/hazard_regression_guard.py --check    # CI gate
    python benchmarks/hazard_regression_guard.py --update   # re-pin baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from hazard_scenarios.scenario_io import verify_file_against_manifest

from omni_mercury_engine.evaluation import hazard_metrics as hm

BASELINE_PATH = _HERE / "hazard_domain_baseline.json"

VOLCANO_ALERT_LEVELS = ("normal", "advisory", "watch", "warning")
FLARE_CLASSES = ("A", "B", "C", "M", "X")

# Margins (floor = measured - margin for higher-is-better; ceiling = measured
# + margin for lower-is-better). The evaluation is bit-deterministic, so the
# margins do not absorb numerical drift -- they define how much degradation
# is tolerated before the gate trips:
#   * Rate metrics (POD/FAR/CSI/HSS/accuracies): 0.05 absolute. One flipped
#     scenario moves any of these by >= 1/n_scenarios (>= 0.056 on the
#     largest guarded set), so a SINGLE genuine flip already trips the gate
#     while an intentional, reviewed scenario-set change re-pins cleanly.
#   * Lead times: 15% relative. Warning lead is measured in whole
#     frames/hours; a one-step pick delay on any scenario exceeds 15% of the
#     pinned means (~6.8 units).
#   * MAE-type metrics: 15% relative with a small absolute slack
#     (min-abs) so a near-zero measured MAE cannot pin an unachievable
#     zero-width ceiling. Kp min-abs 0.10 (a third of one Kp step); S-P
#     distance min-abs 0.25 km (a quarter of the 8.4 km/s rule's resolution
#     at the 100 Hz pick grid).
RATE_MARGIN = 0.05
LEAD_MARGIN_REL = 0.15
MAE_MARGIN_REL = 0.15
KP_MAE_MIN_ABS = 0.10
SP_DISTANCE_MAE_MIN_ABS = 0.25

# Registry: every guarded domain, the scenario sets it runs on, the gated
# metrics with directions/margins and aspirational targets, and documented
# exclusions where gating a metric would require fabricating a capability.
HAZARD_METRICS: dict[str, dict[str, Any]] = {
    "tornado": {
        "detector": "TornadoDetector.predict_tornado (Doppler velocity-couplet physics)",
        "scenario_files": ("tornado_scenarios.npz",),
        "metrics": {
            "pod": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "far": {"direction": "lower", "margin_abs": RATE_MARGIN, "aspirational": 0.0},
            "csi": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "hss": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "mean_lead_time_min": {
                "direction": "higher",
                "margin_rel": LEAD_MARGIN_REL,
                "aspirational": 10.0,
            },
        },
    },
    "flood": {
        "detector": "FloodDetector.predict_flood (precip/gauge/soil physics)",
        "scenario_files": ("flood_scenarios.json",),
        "metrics": {
            "pod": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "far": {"direction": "lower", "margin_abs": RATE_MARGIN, "aspirational": 0.0},
            "csi": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "hss": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "mean_lead_time_hours": {
                "direction": "higher",
                "margin_rel": LEAD_MARGIN_REL,
                "aspirational": 8.0,
            },
        },
    },
    "hurricane": {
        "detector": "HurricaneDetector.predict_hurricane (pressure + wind-field kinematics)",
        "scenario_files": ("hurricane_scenarios.npz",),
        "metrics": {
            "pod": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "far": {"direction": "lower", "margin_abs": RATE_MARGIN, "aspirational": 0.0},
            "csi": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "hss": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
        },
        "exclusions": {
            # NOT a placeholder: this documents why the metric is absent.
            "track_error_km": (
                "No track model exists. The honesty wave (56903a5) deliberately "
                "REMOVED track_forecast/landfall_probability/time_to_landfall_hours "
                "from HurricanePredictionResult because they were declared but never "
                "computed; an honest track forecast needs steering-flow data and a "
                "track model this detector does not have. Gating a track metric "
                "would require fabricating that capability. A tripwire below fails "
                "the guard if the dead fields ever regrow."
            ),
        },
    },
    "earthquake": {
        "detector": "EarthquakeDetector.predict_earthquake (STA/LTA + S-P physics)",
        "scenario_files": ("earthquake_scenarios.npz",),
        "metrics": {
            "pod": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "far": {"direction": "lower", "margin_abs": RATE_MARGIN, "aspirational": 0.0},
            "csi": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "sp_distance_mae_km": {
                "direction": "lower",
                "margin_rel": MAE_MARGIN_REL,
                "margin_min_abs": SP_DISTANCE_MAE_MIN_ABS,
                "aspirational": 0.5,
            },
        },
        "exclusions": {
            "magnitude_mae": (
                "The untrained detector honestly emits estimated_magnitude=None "
                "(magnitude_class 'undetermined') -- an uncalibrated single station "
                "has no honest Richter estimate (1afd151). Gate what it DOES "
                "produce: detection skill and S-P distance error. A tripwire fails "
                "the guard if a magnitude is ever fabricated while untrained."
            ),
            "location_error_km": (
                "A single station yields an epicentral DISTANCE, not a location; "
                "hazard_metrics.location_error_km stays available for the day a "
                "multi-station location path exists."
            ),
        },
    },
    "tsunami": {
        "detector": "TsunamiDetector.predict_tsunami (DART amplitude + resonance physics)",
        "scenario_files": ("tsunami_scenarios.npz",),
        "metrics": {
            "pod": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
            "far": {"direction": "lower", "margin_abs": RATE_MARGIN, "aspirational": 0.0},
            "csi": {"direction": "higher", "margin_abs": RATE_MARGIN, "aspirational": 1.0},
        },
    },
    "volcano": {
        "detector": "VolcanicEruptionDetector.predict_eruption (multi-precursor physics)",
        "scenario_files": ("volcano_scenarios.json",),
        "metrics": {
            "alert_exact_accuracy": {
                "direction": "higher",
                "margin_abs": RATE_MARGIN,
                "aspirational": 1.0,
            },
            "alert_within_one_accuracy": {
                "direction": "higher",
                "margin_abs": RATE_MARGIN,
                "aspirational": 1.0,
            },
        },
        "exclusions": {
            "vei_accuracy": (
                "The physics-path VEI is documented as a coarse precursor-magnitude "
                "proxy, not a physical VEI prediction (volcanic.py). Gating it "
                "would dress a proxy up as skill; hazard_metrics.vei_accuracy is "
                "implemented and unit-tested for the day a trained forecast model "
                "provides a real VEI."
            ),
        },
    },
    "solar": {
        "detector": "SolarStormDetector.predict_solar_storm (NOAA flare chain + Boyle-index Kp)",
        "scenario_files": ("solar_flare_windows.json", "solar_kp_windows.json"),
        "metrics": {
            "flare_class_exact_accuracy": {
                "direction": "higher",
                "margin_abs": RATE_MARGIN,
                "aspirational": 1.0,
            },
            # flare_class_within_one_accuracy is REPORTED in the baseline but
            # deliberately NOT gated: C/M windows dominate the recorded week,
            # so a degenerate always-M predictor scores 27/28 within-one --
            # a floor there would be near-vacuous. Exact accuracy carries the
            # gate (always-M scores only 0.5 exact).
            "kp_mae": {
                "direction": "lower",
                "margin_rel": MAE_MARGIN_REL,
                "margin_min_abs": KP_MAE_MIN_ABS,
                "aspirational": 1.0,
            },
            "kp_g_bucket_accuracy": {
                "direction": "higher",
                "margin_abs": RATE_MARGIN,
                "aspirational": 0.95,
            },
        },
        "notes": {
            "kp_offline": (
                "Kp is NOT None offline: the honesty wave (30a5582) wired a "
                "deterministic Boyle-index physics path, so Kp MAE is gated against "
                "real measured Kp. Genuine forecast skill: on the recorded week the "
                "Boyle MAE beats the climatology-mean predictor (asserted by the "
                "non-vacuity tests)."
            ),
            "flare_labels_definitional": (
                "Flare-class labels are definitionally the NOAA class of the "
                "window's measured peak flux, so that metric gates the "
                "classification chain's correctness on real GOES data, not "
                "forecast skill."
            ),
            "g_bucket_distribution": (
                "The recorded week is quiet-dominated (49/55 windows G0), so an "
                "always-G0 predictor scores ~0.89 bucket accuracy; the bucket "
                "metric is a regression tripwire on the storm windows, while "
                "skill-over-climatology is carried by kp_mae."
            ),
            "flare_brier": (
                "No per-class probability surface exists on the flare path "
                "(predict_solar_storm emits a class, not class probabilities), so "
                "no Brier score is gated; hazard_metrics.brier_score is implemented "
                "and unit-tested for when a probabilistic path exists."
            ),
        },
    },
}

TORNADO_WINDOW_FRAMES = 12


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _detection_rates(y_true: list[int], y_pred: list[int], domain: str) -> dict[str, float]:
    """Compute POD/FAR/CSI (+HSS) from per-scenario binary outcomes.

    Args:
        y_true: Scenario labels (1 = event).
        y_pred: Detector outcomes (1 = alerted appropriately).
        domain: Domain name for fail-loud messages.

    Returns:
        Mapping with ``pod``, ``far``, ``csi``, ``hss``.

    Raises:
        RuntimeError: If the detector emitted no alerts at all (FAR undefined
            -- a collapsed detector must fail loudly, not silently score).
    """
    hits, misses, false_alarms, correct_negatives = hm.contingency_table(y_true, y_pred)
    if hits + false_alarms == 0:
        raise RuntimeError(f"{domain}: detector emitted no alerts at all; guard cannot score")
    return {
        "pod": hm.probability_of_detection(hits, misses),
        "far": hm.false_alarm_ratio(hits, false_alarms),
        "csi": hm.critical_success_index(hits, misses, false_alarms),
        "hss": hm.heidke_skill_score(hits, misses, false_alarms, correct_negatives),
    }


# ---------------------------------------------------------------------------
# Domain runners (each: load pinned scenarios, run the real detector, score)
# ---------------------------------------------------------------------------


def _run_tornado(path: Path) -> dict[str, float]:
    """Sliding-window mesocyclone detection + warning lead time."""
    from omni_mercury_engine.detectors.geological.tornado_detector import TornadoDetector

    data = np.load(path)
    frames, labels, events = data["frames"], data["labels"], data["event_frame"]
    y_true, y_pred, leads = [], [], []
    for i in range(len(labels)):
        detector = TornadoDetector()  # fresh instance per scenario
        alert_frame: int | None = None
        for start in range(frames.shape[1] - TORNADO_WINDOW_FRAMES + 1):
            window = frames[i, start : start + TORNADO_WINDOW_FRAMES]
            if detector.predict_tornado({"radar_sequence": window}).tornado_likely:
                alert_frame = start + TORNADO_WINDOW_FRAMES - 1
                break
        if labels[i] == 1:
            hit = alert_frame is not None and alert_frame <= events[i]
            y_true.append(1)
            y_pred.append(int(hit))
            if hit:
                leads.append(float(events[i] - alert_frame))
        else:
            y_true.append(0)
            y_pred.append(int(alert_frame is not None))
    metrics = _detection_rates(y_true, y_pred, "tornado")
    if not leads:
        raise RuntimeError("tornado: no event was alerted before touchdown; lead undefined")
    metrics["mean_lead_time_min"] = float(np.mean(leads))
    return metrics


def _run_flood(path: Path) -> dict[str, float]:
    """Static flood detection + rain-before-crest warning lead time."""
    from omni_mercury_engine.detectors.geological.flood_detector import FloodDetector

    payload = json.loads(path.read_text())
    y_true, y_pred = [], []
    for scenario in payload["scenarios"]:
        detector = FloodDetector()
        result = detector.predict_flood(scenario["data"])
        y_true.append(int(scenario["label"]))
        y_pred.append(int(result.flood_likely))
    metrics = _detection_rates(y_true, y_pred, "flood")

    leads = []
    for series in payload["series"]:
        detector = FloodDetector()
        alert_hour: int | None = None
        for step in series["steps"]:
            step_data = {k: v for k, v in step.items() if k != "hour"}
            if detector.predict_flood(step_data).flood_likely:
                alert_hour = step["hour"]
                break
        if alert_hour is None:
            raise RuntimeError(f"flood: series {series['kind']} never alerted; lead undefined")
        leads.append(float(series["event_hour"] - alert_hour))
    metrics["mean_lead_time_hours"] = float(np.mean(leads))
    return metrics


def _run_hurricane(path: Path) -> dict[str, float]:
    """Cyclone detection from pressure deficit + wind-field kinematics."""
    from omni_mercury_engine.detectors.geological.hurricane_detector import (
        HurricaneDetector,
        HurricanePredictionResult,
    )

    # Honesty tripwire: the removed (never-computed) track fields must not
    # regrow without a real track model (see registry exclusion).
    dead_fields = {"track_forecast", "landfall_probability", "time_to_landfall_hours"}
    regrown = dead_fields & set(HurricanePredictionResult.__dataclass_fields__)
    if regrown:
        raise RuntimeError(
            f"hurricane: dead track fields regrew without a track model: {sorted(regrown)}"
        )

    data = np.load(path)
    spacing = float(data["grid_spacing_m"])
    y_true, y_pred = [], []
    for i in range(len(data["labels"])):
        detector = HurricaneDetector()
        result = detector.predict_hurricane(
            {
                "pressure_data": {
                    "central_pressure_mb": float(data["central_pressure_mb"][i]),
                    "environmental_pressure_mb": float(data["environmental_pressure_mb"][i]),
                },
                "wind_field": {"u": data["u"][i], "v": data["v"][i], "grid_spacing_m": spacing},
            }
        )
        y_true.append(int(data["labels"][i]))
        y_pred.append(int(result.cyclone_detected))
    return _detection_rates(y_true, y_pred, "hurricane")


def _run_earthquake(path: Path) -> dict[str, float]:
    """STA/LTA detection skill + S-P epicentral-distance error."""
    from omni_mercury_engine.detectors.geological.disaster_detectors import EarthquakeDetector

    data = np.load(path)
    y_true, y_pred, distance_errors = [], [], []
    n_events_detected = 0
    for i in range(len(data["labels"])):
        detector = EarthquakeDetector(sampling_rate=100.0)
        result = detector.predict_earthquake(data["traces"][i])
        # Honesty tripwire: an untrained station must never fabricate a
        # magnitude (1afd151); a regrown estimate fails the guard loudly.
        if result.estimated_magnitude is not None:
            raise RuntimeError(
                "earthquake: untrained detector fabricated a magnitude "
                f"({result.estimated_magnitude}) on scenario {i}"
            )
        label = int(data["labels"][i])
        y_true.append(label)
        y_pred.append(int(result.earthquake_detected))
        if label == 1 and result.earthquake_detected:
            n_events_detected += 1
            if result.epicenter_distance_km is not None:
                true_km = float(data["sp_distance_km"][i])
                distance_errors.append(abs(result.epicenter_distance_km - true_km))
    metrics = _detection_rates(y_true, y_pred, "earthquake")
    if len(distance_errors) < max(1, n_events_detected // 2):
        raise RuntimeError(
            f"earthquake: only {len(distance_errors)}/{n_events_detected} detected events "
            "yielded an S-P distance; the picker chain is broken"
        )
    metrics["sp_distance_mae_km"] = float(np.mean(distance_errors))
    metrics["sp_distance_n"] = float(len(distance_errors))
    return metrics


def _run_tsunami(path: Path) -> dict[str, float]:
    """DART-record tsunami detection skill."""
    from omni_mercury_engine.detectors.geological.disaster_detectors import TsunamiDetector

    data = np.load(path)
    y_true, y_pred = [], []
    for i in range(len(data["labels"])):
        detector = TsunamiDetector(sampling_rate=1.0)
        result = detector.predict_tsunami(data["records"][i])
        y_true.append(int(data["labels"][i]))
        y_pred.append(int(result.tsunami_detected))
    return _detection_rates(y_true, y_pred, "tsunami")


def _run_volcano(path: Path) -> dict[str, float]:
    """USGS alert-level ordinal accuracy over multi-precursor scenarios."""
    from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

    payload = json.loads(path.read_text())
    intended, predicted = [], []
    for scenario in payload["scenarios"]:
        detector = VolcanicEruptionDetector()  # fresh: HMM/optimizer state per scenario
        data = dict(scenario["data"])
        data["seismic_sequence"] = np.asarray(data["seismic_sequence"], dtype=float)
        thermal = dict(data["thermal_data"])
        thermal["brightness_temperature_k"] = np.asarray(
            thermal["brightness_temperature_k"], dtype=float
        )
        data["thermal_data"] = thermal
        result = detector.predict_eruption(data)
        intended.append(scenario["alert_level"])
        predicted.append(result.alert_level)
    exact, within_one = hm.ordinal_accuracy(intended, predicted, VOLCANO_ALERT_LEVELS)
    return {"alert_exact_accuracy": exact, "alert_within_one_accuracy": within_one}


def _run_solar(flare_path: Path, kp_path: Path) -> dict[str, float]:
    """Flare-class chain accuracy + Boyle-index Kp skill on real SWPC data."""
    from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

    detector = SolarStormDetector()

    flare_payload = json.loads(flare_path.read_text())
    flare_true, flare_pred = [], []
    for window in flare_payload["windows"]:
        result = detector.predict_solar_storm(
            {
                "xray_data": {
                    "flux_short_wm2": window["peak_flux_short_wm2"],
                    "flux_long_wm2": window["peak_flux_long_wm2"],
                }
            }
        )
        flare_true.append(window["flare_class"])
        flare_pred.append(result.flare_class)
    flare_exact, flare_within_one = hm.ordinal_accuracy(flare_true, flare_pred, FLARE_CLASSES)

    kp_payload = json.loads(kp_path.read_text())
    kp_true, kp_pred = [], []
    for window in kp_payload["windows"]:
        result = detector.predict_solar_storm(
            {
                "magnetosphere_data": {
                    "solar_wind_speed_km_s": window["solar_wind_speed_km_s"],
                    "bz_imf_nt": window["bz_imf_nt"],
                    "by_imf_nt": window["by_imf_nt"],
                }
            }
        )
        if result.kp_index is None:
            raise RuntimeError("solar: kp_index is None -- the offline Boyle physics path is gone")
        kp_true.append(float(window["kp_observed"]))
        kp_pred.append(float(result.kp_index))

    return {
        "flare_class_exact_accuracy": flare_exact,
        "flare_class_within_one_accuracy": flare_within_one,
        "kp_mae": hm.kp_mae(kp_true, kp_pred),
        "kp_g_bucket_accuracy": hm.g_bucket_accuracy(kp_true, kp_pred),
    }


# ---------------------------------------------------------------------------
# Guard mechanics (mirrors anomaly_regression_guard)
# ---------------------------------------------------------------------------


def evaluate() -> dict[str, Any]:
    """Deterministically evaluate every guarded hazard detector.

    Verifies each committed scenario file against the manifest hash, runs the
    real detector physics paths, and returns per-domain skill metrics with
    full provenance metadata.

    Returns:
        ``{"metadata": ..., "domains": {domain: {..., "metrics": ...}}}``.

    Raises:
        RuntimeError: On any honesty-tripwire breach or undefined metric
            (collapsed detector) -- the guard fails loud, never scores quietly.
    """
    # The guard deliberately exercises the untrained-physics paths; each fresh
    # detector instance would otherwise log the same (correct, expected)
    # untrained warning hundreds of times and bury real CI output.
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    runners = {
        "tornado": _run_tornado,
        "flood": _run_flood,
        "hurricane": _run_hurricane,
        "earthquake": _run_earthquake,
        "tsunami": _run_tsunami,
        "volcano": _run_volcano,
    }

    from hazard_scenarios.scenario_io import load_manifest

    manifest = load_manifest()
    domains: dict[str, Any] = {}
    for domain, spec in HAZARD_METRICS.items():
        paths = [verify_file_against_manifest(name) for name in spec["scenario_files"]]
        if domain == "solar":
            metrics = _run_solar(*paths)
        else:
            metrics = runners[domain](paths[0])
        file_hashes = {name: manifest["files"][name]["sha256"] for name in spec["scenario_files"]}
        label_sources = sorted(
            {manifest["files"][name]["label_source"] for name in spec["scenario_files"]}
        )
        domains[domain] = {
            "scenario_files": file_hashes,
            "label_source": label_sources[0] if len(label_sources) == 1 else label_sources,
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        }

    return {
        "metadata": {
            "purpose": (
                "Deterministic regression guard for the hazard detectors' physics "
                "paths (post honesty-wave). Offline: committed, hash-pinned "
                "scenario sets only. Real recorded SWPC data for solar; seeded "
                "constructed physics scenarios elsewhere (see "
                "hazard_scenarios/manifest.json for per-set provenance)."
            ),
            "guard": "benchmarks/hazard_regression_guard.py",
            "scenario_manifest": "benchmarks/hazard_scenarios/manifest.json",
            "scenario_seeds": manifest.get("seeds"),
            "commit": _git_commit(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "margins": {
                "rate_abs": RATE_MARGIN,
                "lead_rel": LEAD_MARGIN_REL,
                "mae_rel": MAE_MARGIN_REL,
                "kp_mae_min_abs": KP_MAE_MIN_ABS,
                "sp_distance_mae_min_abs": SP_DISTANCE_MAE_MIN_ABS,
                "justification": (
                    "Evaluation is bit-deterministic, so margins define tolerated "
                    "degradation, not numerical drift: 0.05 absolute on rate "
                    "metrics (< one scenario flip on every guarded set); 15% "
                    "relative on lead times (< one pick-step delay); 15% relative "
                    "+ small min-abs on MAE metrics (min-abs prevents a near-zero "
                    "measurement pinning a zero-width ceiling)."
                ),
            },
            "honesty_tripwires": [
                "earthquake: estimated_magnitude must stay None while untrained",
                "hurricane: removed track fields must not regrow without a track model",
                "solar: kp_index must not be None on the offline Boyle physics path",
            ],
        },
        "domains": domains,
    }


def _floors_from(baseline: dict[str, Any]) -> dict[str, Any]:
    """Derive per-domain floors/ceilings from a pinned baseline.

    Args:
        baseline: Parsed ``hazard_domain_baseline.json``.

    Returns:
        ``{"domains": {domain: {metric: {"floor": x} | {"ceiling": x}}}}``.
    """
    out: dict[str, Any] = {"domains": {}}
    for domain, spec in HAZARD_METRICS.items():
        base_metrics = baseline["domains"][domain]["metrics"]
        bounds: dict[str, dict[str, float]] = {}
        for metric, mspec in spec["metrics"].items():
            measured = float(base_metrics[metric])
            if "margin_abs" in mspec:
                margin = float(mspec["margin_abs"])
            else:
                margin = abs(measured) * float(mspec["margin_rel"])
                margin = max(margin, float(mspec.get("margin_min_abs", 0.0)))
            if mspec["direction"] == "higher":
                bounds[metric] = {"floor": round(measured - margin, 6)}
            else:
                bounds[metric] = {"ceiling": round(measured + margin, 6)}
        out["domains"][domain] = bounds
    return out


def check(measured: dict[str, Any] | None = None) -> list[str]:
    """Return a list of human-readable violations (empty == pass).

    Args:
        measured: Pre-computed :func:`evaluate` result (measured live when
            omitted).

    Returns:
        Violation strings; empty means every metric respects its bound and
        the scenario sets match the ones the baseline was pinned on.
    """
    if not BASELINE_PATH.exists():
        return [f"baseline missing: {BASELINE_PATH} (run with --update)"]
    baseline = json.loads(BASELINE_PATH.read_text())
    floors = _floors_from(baseline)
    if measured is None:
        measured = evaluate()

    violations: list[str] = []
    for domain, spec in HAZARD_METRICS.items():
        base_domain = baseline["domains"].get(domain)
        meas_domain = measured["domains"].get(domain)
        if base_domain is None:
            violations.append(f"{domain}: missing from baseline (re-pin with --update)")
            continue
        if meas_domain is None:
            violations.append(f"{domain}: missing from measured run")
            continue
        for name, pinned_sha in base_domain["scenario_files"].items():
            actual_sha = meas_domain["scenario_files"].get(name)
            if actual_sha != pinned_sha:
                violations.append(
                    f"{domain}: scenario set {name} changed since the baseline was "
                    f"pinned ({str(actual_sha)[:12]} != {pinned_sha[:12]}); review and "
                    "re-pin with --update"
                )
        for metric, bound in floors["domains"][domain].items():
            value = meas_domain["metrics"].get(metric)
            if value is None:
                violations.append(f"{domain}: metric {metric} missing from measured run")
                continue
            if "floor" in bound and value < bound["floor"]:
                violations.append(f"{domain}: {metric} {value:.4f} < floor {bound['floor']:.4f}")
            if "ceiling" in bound and value > bound["ceiling"]:
                violations.append(
                    f"{domain}: {metric} {value:.4f} > ceiling {bound['ceiling']:.4f}"
                )
        _ = spec  # registry consulted via floors; kept for symmetry/clarity
    return violations


def main() -> int:
    """CLI entry point: ``--update`` re-pins the baseline; ``--check`` gates."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-measure and re-pin the baseline")
    ap.add_argument("--check", action="store_true", help="fail if any metric crosses its bound")
    args = ap.parse_args()

    if args.update:
        baseline = evaluate()
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {BASELINE_PATH}")
        for domain, entry in baseline["domains"].items():
            summary = "  ".join(f"{k}={v:.4f}" for k, v in entry["metrics"].items())
            print(f"  {domain}: {summary}")
        return 0

    if args.check:
        violations = check()
        if violations:
            print("HAZARD REGRESSION GUARD: FAIL")
            for violation in violations:
                print(f"  - {violation}")
            return 1
        print("HAZARD REGRESSION GUARD: PASS (all hazard metrics within pinned bounds)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

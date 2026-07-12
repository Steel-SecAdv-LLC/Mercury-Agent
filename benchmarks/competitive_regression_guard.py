# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic competitive-position regression guard (Mercury vs PyOD).

Why this exists
---------------
``benchmarks/anomaly_regression_guard.py`` pins Mercury's *absolute* metric
floor, but says nothing about Mercury's position **relative to the
competition**. A silent regression could leave Mercury above its absolute
floor while the PyOD baselines (same data, same protocol) pull ahead -- the
exact failure mode the competitive benchmark exists to expose. This guard
pins both:

* **Absolute floors** -- Mercury tier per-dataset AUC and mean AUC must stay
  above ``measured - margin`` (the exact ``_floors_from`` measured-minus-
  margin pattern of the anomaly/hazard guards; margins live in metadata,
  floors are always derived, never stored).
* **Competitive-position gate** -- the gap ``best PyOD baseline mean AUC -
  Mercury tier mean AUC`` (positive when PyOD leads) must not grow beyond
  ``measured_gap + gap_margin``. Both sides are re-measured in the same run,
  so a PyOD version bump that pulls the baselines ahead trips the gate just
  like a Mercury regression does.

What this measures
------------------
The Mercury *tier* (:class:`MercuryAnomalyDetector`, unsupervised) and the
default PyOD baselines on the fixed 8-dataset genuine-label ADBench guard
subset (same sets as ``anomaly_regression_guard.GUARD_DATASETS``: small, fast,
cached, deterministic with seed 42), under the identical shared protocol of
``benchmarks/competitive_benchmark.py`` (normal-only train, de-leak shuffle,
train-fitted scaler). The fusion engine is deliberately NOT gated here: its
consensus-label training path is torch-stochastic across environments, and a
deterministic gate must not flap.

This is a *guard subset*, not the headline: the committed 57-dataset position
lives in ``benchmarks/competitive_results.json`` /
``benchmarks/COMPETITIVE_BENCHMARK.md``.

Usage::

    python benchmarks/competitive_regression_guard.py --check    # CI gate
    python benchmarks/competitive_regression_guard.py --update   # re-pin baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import numpy as np

import competitive_benchmark as cb

BASELINE_PATH = _HERE / "competitive_baseline.json"

#: Fixed guard set -- identical to ``anomaly_regression_guard.GUARD_DATASETS``
#: (small, fast, genuinely ground-truth-labelled ADBench Classical sets).
GUARD_DATASETS: tuple[str, ...] = (
    "breastw",
    "cardio",
    "Ionosphere",
    "WBC",
    "Lymphography",
    "Pima",
    "glass",
    "pendigits",
)

# Floors are measured-minus-margin (mirrors anomaly_regression_guard exactly).
# Margins absorb cross-environment numerical drift (BLAS/scipy/sklearn) while
# still catching real regressions, which are typically >> these margins.
AUC_MARGIN = 0.03
MEAN_AUC_MARGIN = 0.02
# Competitive-position margin: the measured Mercury-to-best-PyOD mean-AUC gap
# may widen by at most this much before the gate fires.
GAP_MARGIN = 0.02


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def evaluate() -> dict[str, Any]:
    """Deterministically measure Mercury tier + PyOD baselines on the guard set."""
    from omni_mercury_engine.comparison.pyod_integration import (
        pyod_version,
        run_pyod_baselines,
    )

    datasets: dict[str, Any] = {}
    mercury_aucs: list[float] = []
    pyod_aucs: dict[str, list[float]] = {m: [] for m in cb.PYOD_METHODS}
    for name in GUARD_DATASETS:
        X, y, sha, _url = cb._load_classical(name)
        split = cb.prepare_split(X, y, seed=cb.SEED)
        if isinstance(split, str):
            raise RuntimeError(f"guard dataset {name} not splittable: {split}")
        X_train, X_test, y_test = split

        mercury = cb._run_mercury_tier(X_train, X_test, y_test)
        if np.isnan(mercury["roc_auc"]):
            raise RuntimeError(f"guard dataset {name}: mercury AUC not measurable")

        per_pyod: dict[str, float] = {}
        for algo, run in run_pyod_baselines(X_train, X_test, seed=cb.SEED).items():
            if "error" in run:
                raise RuntimeError(f"guard dataset {name}: {algo} failed: {run['error']}")
            metrics = cb._metrics(y_test, run["scores"])
            per_pyod[algo] = round(metrics["roc_auc"], 6)
            pyod_aucs[algo].append(metrics["roc_auc"])

        best_algo = max(per_pyod, key=per_pyod.__getitem__)
        datasets[name] = {
            "npz_sha256": sha,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_test_anomalies": int(y_test.sum()),
            "mercury_tier_auc": round(float(mercury["roc_auc"]), 6),
            "pyod_auc": per_pyod,
            "best_pyod_method": best_algo,
            "best_pyod_auc": per_pyod[best_algo],
        }
        mercury_aucs.append(float(mercury["roc_auc"]))

    mercury_mean = float(np.mean(mercury_aucs))
    pyod_means = {m: round(float(np.mean(v)), 6) for m, v in pyod_aucs.items()}
    best_method = max(pyod_means, key=pyod_means.__getitem__)
    best_mean = pyod_means[best_method]
    return {
        "metadata": {
            "purpose": (
                "Deterministic competitive-position guard: Mercury tier vs PyOD baselines "
                "on a fixed genuine-label ADBench subset. NOT the headline competitive "
                "benchmark (full 57-dataset run in competitive_benchmark.py)."
            ),
            "mercury_method": "mercury_tier (MercuryAnomalyDetector, unsupervised fit)",
            "eval_path": "competitive_benchmark.prepare_split + _run_mercury_tier",
            "pyod_methods": list(cb.PYOD_METHODS),
            "seed": cb.SEED,
            "dataset_source": "https://github.com/Minqi824/ADBench",
            "dataset_license": "MIT",
            "commit": _git_commit(),
            "python": sys.version.split()[0],
            "pyod_version": pyod_version(),
            "margins": {
                "auc": AUC_MARGIN,
                "mean_auc": MEAN_AUC_MARGIN,
                "competitive_gap": GAP_MARGIN,
                "justification": (
                    "measured-minus-margin floors; margins absorb cross-environment "
                    "numerical drift (BLAS/scipy/sklearn versions) while real "
                    "regressions (broken component, protocol leak) move AUC by >> 0.03"
                ),
            },
        },
        "datasets": datasets,
        "aggregate": {
            "mercury_tier_mean_auc": round(mercury_mean, 6),
            "pyod_mean_auc": pyod_means,
            "best_pyod_method": best_method,
            "best_pyod_mean_auc": best_mean,
            # Positive when the best PyOD baseline leads Mercury on this subset.
            "competitive_gap": round(best_mean - mercury_mean, 6),
        },
    }


def _floors_from(baseline: dict[str, Any]) -> dict[str, Any]:
    """Derive floors/ceilings from a measured baseline (never stored)."""
    out: dict[str, Any] = {"datasets": {}}
    for name, d in baseline["datasets"].items():
        out["datasets"][name] = {
            "mercury_auc_floor": round(d["mercury_tier_auc"] - AUC_MARGIN, 4),
        }
    agg = baseline["aggregate"]
    out["mercury_mean_auc_floor"] = round(agg["mercury_tier_mean_auc"] - MEAN_AUC_MARGIN, 4)
    out["competitive_gap_ceiling"] = round(agg["competitive_gap"] + GAP_MARGIN, 4)
    return out


def check(measured: dict[str, Any] | None = None) -> list[str]:
    """Return a list of human-readable violations (empty == pass)."""
    if not BASELINE_PATH.exists():
        return [f"baseline missing: {BASELINE_PATH} (run with --update)"]
    baseline = json.loads(BASELINE_PATH.read_text())
    floors = _floors_from(baseline)
    if measured is None:
        measured = evaluate()

    violations: list[str] = []
    for name, fl in floors["datasets"].items():
        m = measured["datasets"].get(name)
        if m is None:
            violations.append(f"{name}: missing from measured run")
            continue
        if m["mercury_tier_auc"] < fl["mercury_auc_floor"]:
            violations.append(
                f"{name}: mercury AUC {m['mercury_tier_auc']:.4f} < "
                f"floor {fl['mercury_auc_floor']:.4f}"
            )
    agg = measured["aggregate"]
    if agg["mercury_tier_mean_auc"] < floors["mercury_mean_auc_floor"]:
        violations.append(
            f"mercury mean AUC {agg['mercury_tier_mean_auc']:.4f} < "
            f"floor {floors['mercury_mean_auc_floor']:.4f}"
        )
    if agg["competitive_gap"] > floors["competitive_gap_ceiling"]:
        violations.append(
            f"competitive gap {agg['competitive_gap']:.4f} "
            f"(best PyOD {agg['best_pyod_method']} mean {agg['best_pyod_mean_auc']:.4f} - "
            f"mercury mean {agg['mercury_tier_mean_auc']:.4f}) > "
            f"ceiling {floors['competitive_gap_ceiling']:.4f}"
        )
    return violations


def main() -> int:
    """CLI entry point (mirrors anomaly_regression_guard)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-measure and re-pin the baseline")
    ap.add_argument("--check", action="store_true", help="fail if any gate is violated")
    args = ap.parse_args()

    if args.update:
        baseline = evaluate()
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        agg = baseline["aggregate"]
        print(f"baseline written: {BASELINE_PATH}")
        print(
            f"  mercury mean AUC={agg['mercury_tier_mean_auc']:.4f}  "
            f"best PyOD ({agg['best_pyod_method']}) mean AUC={agg['best_pyod_mean_auc']:.4f}  "
            f"gap={agg['competitive_gap']:+.4f}"
        )
        return 0

    if args.check:
        violations = check()
        if violations:
            print("COMPETITIVE REGRESSION GUARD: FAIL")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("COMPETITIVE REGRESSION GUARD: PASS (absolute floors + competitive gap held)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

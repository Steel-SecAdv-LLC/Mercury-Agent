# Copyright (C) 2025 Steel Security Advisors LLC
"""Deterministic anomaly-detector regression guard.

Why this exists
---------------
Issue #261 reported the headline benchmark dropping from AUC 0.8466 / F1 0.6428
to AUC 0.8259 / F1 0.6046 after PR #255.  Investigation (see
``docs/ANOMALY_REGRESSION_WS_A.md``) showed this was **not** a detector
regression: PR #255 made the headline *honest* by excluding 13 circular
manufactured-label datasets (mean AUC 0.9479) from the supervised headline.
On an apples-to-apples all-datasets basis the detector was flat-to-up
(0.8466 -> 0.8495 AUC; 0.6428 -> 0.6422 F1) and the only ``statistical.py``
change across the merge was type-annotation-only.

The lesson: the *real* metric floor was never pinned deterministically, so a
genuine future regression in :class:`MercuryAnomalyDetector` could land
silently while the headline number moved for unrelated (dataset-set) reasons.

What this guards
----------------
A fast, fully deterministic subset evaluation of the **unchanged** detector
eval path (``mercury_benchmark._benchmark_single``) on a fixed list of
genuinely-labelled (ground-truth) ADBench datasets, with seed 42.  Per-dataset
AUC/F1 floors (measured value minus a small margin) are pinned in
``anomaly_regression_baseline.json``.  ``--check`` fails non-zero if any metric
falls below its floor, so CI catches a real detector regression deterministically.

This is intentionally a *guard subset*, not the headline benchmark: the
headline remains the full 75-dataset run in ``mercury_benchmark.py``.  The guard
trades coverage for speed + determinism so it can gate every PR.

Usage::

    python benchmarks/anomaly_regression_guard.py --check    # CI gate
    python benchmarks/anomaly_regression_guard.py --update   # re-pin baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import mercury_benchmark as mb

from omni_mercury_engine.datasets.adbench import (
    ADBENCH_CATALOG,
    ADBenchLoader,
)
from omni_mercury_engine.datasets.base import DatasetConfig

BASELINE_PATH = _HERE / "anomaly_regression_baseline.json"

# Fixed guard set: small, fast, *genuinely* labelled (ground-truth) ADBench
# datasets spanning strong-signal and weak-signal regimes so a real regression
# in any component shows up.  Below-random datasets (e.g. vertebral) are
# excluded because a "drop" there carries no regression signal.
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

# Floors are measured-minus-margin.  Margins absorb cross-environment numerical
# drift (BLAS/scipy) while still catching real regressions, which are typically
# >> these margins (a broken component drops AUC by 0.05+).
AUC_MARGIN = 0.03
F1_MARGIN = 0.05
MEAN_AUC_MARGIN = 0.02
MEAN_F1_MARGIN = 0.03

_NAME_TO_INDEX = {n.lower(): i for i, n in ADBENCH_CATALOG.items()}


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_with_hash(name: str) -> tuple[Any, Any, str]:
    """Load an ADBench dataset and return (X, y, npz_sha256) for provenance."""
    cfg = DatasetConfig(name=f"adbench-{name}", preprocessing={"dataset": name})
    loader = ADBenchLoader(cfg)
    loader.download()
    npz_path = loader.data_path / loader.npz_filename
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    X, y = loader._load_raw()
    y = (y > 0).astype(int)
    return X, y, sha


def evaluate() -> dict[str, Any]:
    """Deterministically evaluate the detector on the fixed guard set."""
    datasets: dict[str, Any] = {}
    aucs: list[float] = []
    f1s: list[float] = []
    for name in GUARD_DATASETS:
        X, y, sha = _load_with_hash(name)
        res = mb._benchmark_single(
            {"name": name, "category": "adbench", "X": X, "y": y, "label_source": "ground_truth"}
        )
        if res.get("error"):
            raise RuntimeError(f"guard dataset {name} failed: {res['error']}")
        auc = float(res["ensemble_auc"])
        f1 = float(res["oracle_f1"])
        datasets[name] = {
            "adbench_index": _NAME_TO_INDEX.get(name.lower()),
            "n_samples": len(y),
            "n_features": int(X.shape[1]),
            "npz_sha256": sha,
            "auc": auc,
            "f1": f1,
        }
        aucs.append(auc)
        f1s.append(f1)
    mean_auc = sum(aucs) / len(aucs)
    mean_f1 = sum(f1s) / len(f1s)
    return {
        "metadata": {
            "purpose": (
                "Deterministic regression guard for MercuryAnomalyDetector on a fixed "
                "genuine-label ADBench subset. NOT the headline benchmark (full 75-dataset "
                "run in mercury_benchmark.py)."
            ),
            "detector": "MercuryAnomalyDetector",
            "eval_path": "mercury_benchmark._benchmark_single",
            "seed": 42,
            "metric_definitions": {
                "auc": "ROC-AUC of ensemble anomaly scores vs ground-truth labels",
                "f1": "oracle (best-threshold) F1 via _oracle_threshold_f1 multi-strategy sweep",
            },
            "dataset_source": "https://github.com/Minqi824/ADBench",
            "dataset_license": "MIT",
            "commit": _git_commit(),
            "python": sys.version.split()[0],
            "margins": {
                "auc": AUC_MARGIN,
                "f1": F1_MARGIN,
                "mean_auc": MEAN_AUC_MARGIN,
                "mean_f1": MEAN_F1_MARGIN,
            },
        },
        "datasets": datasets,
        "aggregate": {"mean_auc": mean_auc, "mean_f1": mean_f1},
    }


def _floors_from(baseline: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"datasets": {}}
    for name, d in baseline["datasets"].items():
        out["datasets"][name] = {
            "auc_floor": round(d["auc"] - AUC_MARGIN, 4),
            "f1_floor": round(d["f1"] - F1_MARGIN, 4),
        }
    out["mean_auc_floor"] = round(baseline["aggregate"]["mean_auc"] - MEAN_AUC_MARGIN, 4)
    out["mean_f1_floor"] = round(baseline["aggregate"]["mean_f1"] - MEAN_F1_MARGIN, 4)
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
        if m["auc"] < fl["auc_floor"]:
            violations.append(f"{name}: AUC {m['auc']:.4f} < floor {fl['auc_floor']:.4f}")
        if m["f1"] < fl["f1_floor"]:
            violations.append(f"{name}: F1 {m['f1']:.4f} < floor {fl['f1_floor']:.4f}")
    if measured["aggregate"]["mean_auc"] < floors["mean_auc_floor"]:
        violations.append(
            f"mean AUC {measured['aggregate']['mean_auc']:.4f} < floor {floors['mean_auc_floor']:.4f}"
        )
    if measured["aggregate"]["mean_f1"] < floors["mean_f1_floor"]:
        violations.append(
            f"mean F1 {measured['aggregate']['mean_f1']:.4f} < floor {floors['mean_f1_floor']:.4f}"
        )
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-measure and re-pin the baseline")
    ap.add_argument("--check", action="store_true", help="fail if any metric is below its floor")
    args = ap.parse_args()

    if args.update:
        baseline = evaluate()
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"baseline written: {BASELINE_PATH}")
        print(
            f"  mean AUC={baseline['aggregate']['mean_auc']:.4f}  "
            f"mean F1={baseline['aggregate']['mean_f1']:.4f}"
        )
        return 0

    if args.check:
        violations = check()
        if violations:
            print("ANOMALY REGRESSION GUARD: FAIL")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("ANOMALY REGRESSION GUARD: PASS (all metrics >= floors)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic fusion + conformal regression guard (WS5/WS6 CI gate).

Why this exists
---------------
PR #269 shipped the fusion train→eval pipeline and committed AUC/ECE sweep
artifacts, but **no CI gate pinned the fusion model's AUC/F1 or the conformal
coverage floor** — so a regression in the fusion stack (a broken detector
feature, a calibration bug, a coverage collapse) could land silently. This guard
closes that: it deterministically trains-and-evaluates the real fusion path on a
fixed seeded synthetic corpus and fails non-zero if AUC/F1 fall below a pinned
floor or empirical conformal coverage drops below ``target − margin``.

Why train rather than load the shipped checkpoint
-------------------------------------------------
Loading ``default_fusion.pt`` and scoring it is cheaper, but the checkpoint
round-trip drifts per-sample probabilities (AUC survives at Δ≈0.002, but absolute
probabilities — and therefore F1@0.5 / conformal sets — drift by up to ≈0.76),
and the shipped checkpoint underperforms in-distribution because base-detector
state is not persisted (see ROADMAP v1.7.x deferred item #16). Training in-process
with a fixed seed and ``symbolic_weight=0.0`` gives a bit-stable measurement of
the fusion stack's *achievable* performance, which is exactly what a regression
gate must pin, and sidesteps the round-trip drift entirely. The synthetic path is
fully offline (no network / ADBench), so the lane is deterministic and CI-safe;
the strong real-data fusion numbers remain in
``benchmarks/fusion_capacity/sweep_real_v5.json`` (network-gated).

Usage::

    python benchmarks/fusion_regression_guard.py --update   # re-pin baseline + emit artifact
    python benchmarks/fusion_regression_guard.py --check     # CI gate (exit 1 on regression)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from scripts.train_default_fusion import SEED, _stratified_split, build_dataset

BASELINE_PATH = _HERE / "fusion_capacity" / "fusion_gate_baseline.json"
ARTIFACT_DIR = _ROOT / "artifacts" / "fusion"

# Deterministic gate config. Few epochs keep the lane fast; early stopping +
# symbolic_weight=0.0 make the metrics bit-stable (the co-trained "adaptive"
# path adds ~0.002 AUC noise, which the margins below absorb).
GATE_EPOCHS = 8
GATE_COVERAGE_TARGET = 0.9

# Floors are measured-minus-margin. Margins absorb cross-environment numerical
# drift (BLAS/seed/epoch jitter) while still catching real regressions, which
# are typically >> these margins.
AUC_MARGIN = 0.05
F1_MARGIN = 0.07
COVERAGE_MARGIN = 0.05


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _f1_at(probs: np.ndarray[Any, Any], y: np.ndarray[Any, Any], threshold: float = 0.5) -> float:
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate(epochs: int = GATE_EPOCHS, seed: int = SEED) -> dict[str, Any]:
    """Deterministically train + evaluate the fusion path; return metrics.

    Trains on a 60% split, evaluates AUC/F1 on a disjoint 20% test split, and
    measures empirical conformal coverage from a 20% calibration split — all from
    the same seeded synthetic distribution the shipped checkpoint trains on.
    """
    import torch

    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y = build_dataset(seed)
    train_idx, rest_idx = _stratified_split(y, train_frac=0.6, seed=seed)
    cal_local, test_local = _stratified_split(y[rest_idx], train_frac=0.5, seed=seed)
    cal_idx, test_idx = rest_idx[cal_local], rest_idx[test_local]

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        x[train_idx],
        y[train_idx],
        epochs=epochs,
        batch_size=64,
        early_stopping_patience=15,
        symbolic_weight=0.0,  # bit-stable path (no adaptive co-training noise)
    )

    probs = np.asarray(engine.score_fusion(x[test_idx]))
    y_test = y[test_idx]
    auc = float(roc_auc_score(y_test, probs))
    f1 = _f1_at(probs, y_test)

    engine.calibrate_fusion_conformal(x[cal_idx], y[cal_idx], coverage=GATE_COVERAGE_TARGET)
    # ``calibrate_fusion_conformal`` populates ``_fusion_conformal``; assert so
    # mypy can narrow the ``Optional`` (and the call fails closed if it did not).
    assert engine._fusion_conformal is not None
    report = engine._fusion_conformal.coverage_report(
        np.asarray(engine.score_fusion(x[test_idx])), y_test
    )

    return {
        "metadata": {
            "purpose": (
                "Deterministic regression guard for the fusion AUC/F1 + conformal "
                "coverage floor on a fixed seeded synthetic corpus. Offline / "
                "network-free; the real-data fusion numbers live in "
                "benchmarks/fusion_capacity/sweep_real_v5.json."
            ),
            "seed": seed,
            "epochs": epochs,
            "symbolic_weight": 0.0,
            "n_train": len(train_idx),
            "n_cal": len(cal_idx),
            "n_test": len(test_idx),
            "commit": _git_commit(),
            "python": sys.version.split()[0],
            "margins": {
                "auc": AUC_MARGIN,
                "f1": F1_MARGIN,
                "coverage": COVERAGE_MARGIN,
            },
        },
        "auc": auc,
        "f1": f1,
        "conformal": {
            "target_coverage": float(report["target_coverage"]),
            "empirical_coverage": float(report["empirical_coverage"]),
            "average_set_size": float(report["average_set_size"]),
        },
    }


def _floors_from(baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "auc_floor": round(baseline["auc"] - AUC_MARGIN, 4),
        "f1_floor": round(baseline["f1"] - F1_MARGIN, 4),
        # Coverage floor is target − margin (a *coverage* guarantee floor), not
        # baseline − margin: conformal must keep covering at ~target.
        "coverage_floor": round(baseline["conformal"]["target_coverage"] - COVERAGE_MARGIN, 4),
    }


def check(measured: dict[str, Any] | None = None) -> list[str]:
    """Return human-readable violations (empty == pass)."""
    if not BASELINE_PATH.exists():
        return [f"baseline missing: {BASELINE_PATH} (run with --update)"]
    baseline = json.loads(BASELINE_PATH.read_text())
    floors = _floors_from(baseline)
    if measured is None:
        measured = evaluate()
    violations: list[str] = []
    if measured["auc"] < floors["auc_floor"]:
        violations.append(f"AUC {measured['auc']:.4f} < floor {floors['auc_floor']:.4f}")
    if measured["f1"] < floors["f1_floor"]:
        violations.append(f"F1 {measured['f1']:.4f} < floor {floors['f1_floor']:.4f}")
    emp = measured["conformal"]["empirical_coverage"]
    if emp < floors["coverage_floor"]:
        violations.append(f"conformal coverage {emp:.4f} < floor {floors['coverage_floor']:.4f}")
    return violations


def _emit_artifact(metrics: dict[str, Any]) -> Path:
    """Write a timestamped, committable metrics artifact under artifacts/fusion/."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return out_path


def main() -> int:
    """CLI entry point: ``--update`` re-pins the baseline + emits an artifact; ``--check`` gates."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--update", action="store_true", help="re-measure, re-pin baseline + emit artifact"
    )
    ap.add_argument("--check", action="store_true", help="fail if any metric is below its floor")
    args = ap.parse_args()

    if args.update:
        metrics = evaluate()
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        artifact = _emit_artifact(metrics)
        print(f"baseline written: {BASELINE_PATH}")
        print(f"artifact written: {artifact}")
        print(
            f"  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
            f"coverage={metrics['conformal']['empirical_coverage']:.4f} "
            f"(target {metrics['conformal']['target_coverage']:.2f})"
        )
        return 0

    if args.check:
        violations = check()
        if violations:
            print("FUSION REGRESSION GUARD: FAIL")
            for v in violations:
                print(f"  - {v}")
            return 1
        print("FUSION REGRESSION GUARD: PASS (AUC/F1 >= floors, coverage >= floor)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

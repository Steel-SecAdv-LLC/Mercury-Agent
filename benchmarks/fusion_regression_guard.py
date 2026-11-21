# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seeded fusion + conformal regression guard (WS5/WS6 CI gate).

Why this exists
---------------
PR #269 shipped the fusion train→eval pipeline and committed AUC/ECE sweep
artifacts, but **no CI gate pinned the fusion model's AUC/F1 or the conformal
coverage floor** — so a regression in the fusion stack (a broken detector
feature, a calibration bug, a coverage collapse) could land silently. This guard
closes that: it trains-and-evaluates the real fusion path on a fixed seeded
synthetic corpus and fails non-zero if AUC/F1 fall below a pinned floor or
empirical conformal coverage drops below ``target − margin``.

What is gated is the **median over several seeded fits**, at one torch thread.
One fit is one draw: at the gate's eight epochs, the same data and code give
F1 from 0.835 to 1.000 across fit seeds (two of the first seven below 0.87),
and one seed read 0.985, 0.943 and 0.868 at four, two and one threads. A
single-realization pin therefore passed or failed on the runner's core count.
The median of ``GATE_FIT_SEEDS`` fits moves only when most of them do, which
is what a regression in the stack looks like; every realization is recorded
beside it.

Why train rather than load the shipped checkpoint
-------------------------------------------------
Training in-process with fixed seeds and ``symbolic_weight=0.0`` measures the
fusion stack's *achievable* performance — what a regression gate must pin —
independent of any shipped artifact. (Historically
this also sidestepped a checkpoint round-trip defect: per-sample probabilities
drifted by up to ≈0.76 on load because base-detector state was not persisted.
That defect is **closed** — checkpoints persist base-detector/model fitted
state and the conformal serving surface (measured save→load max|ΔProb| ≈ 6e-8),
locked by ``tests/test_fusion_checkpoint_roundtrip.py`` — but the in-process
design remains correct: the gate measures the stack, not one artifact.) The synthetic
path is fully offline (no network / ADBench), so the lane is seeded and
CI-safe; the strong real-data fusion numbers remain in
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

# Gate config. Few epochs keep the lane fast, which is also why one fit is a
# noisy draw; ``symbolic_weight=0.0`` keeps the co-trained path out of it.
GATE_EPOCHS = 8
GATE_COVERAGE_TARGET = 0.9
# Fit seeds whose median is gated (an odd count, so the median is a
# realization). The data and its splits are the same for all of them.
GATE_FIT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
# One torch thread: summation order, and so the fitted model, does not then
# depend on how many cores the runner has.
GATE_TORCH_THREADS = 1

# Floors are measured-minus-margin. Margins absorb cross-environment numerical
# drift (BLAS, CPU) while still catching real regressions, which are typically
# >> these margins. The AUC margin was 0.05 on a single fit; on the steadier
# median 0.04 keeps the floor above the one it replaces.
AUC_MARGIN = 0.04
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


def _evaluate_one(fit_seed: int, epochs: int, seed: int) -> dict[str, Any]:
    """One seeded fit of the fusion path on the gate corpus, and its metrics.

    Trains on a 60% split, evaluates AUC/F1 on a disjoint 20% test split, and
    measures empirical conformal coverage from a 20% calibration split — all from
    the same seeded synthetic distribution the shipped checkpoint trains on.
    ``seed`` fixes the data and its splits; ``fit_seed`` is the fit's own.
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
        symbolic_weight=0.0,  # no adaptive co-training in the gate
        seed=fit_seed,
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
        "fit_seed": fit_seed,
        "n_train": len(train_idx),
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
        "auc": auc,
        "f1": f1,
        "conformal": {
            "target_coverage": float(report["target_coverage"]),
            "empirical_coverage": float(report["empirical_coverage"]),
            "average_set_size": float(report["average_set_size"]),
        },
    }


def summarize(realizations: list[dict[str, Any]], epochs: int, seed: int) -> dict[str, Any]:
    """The gated metrics: each one's median over the seeded fits.

    Each metric's median is taken on its own, so the gated AUC, F1 and
    coverage need not come from the same fit; the fits are all listed under
    ``realizations``.
    """
    if not realizations:
        raise ValueError("the guard needs at least one fit to summarize")

    def median(pick: Any) -> float:
        return float(np.median([pick(one) for one in realizations]))

    first = realizations[0]
    return {
        "metadata": {
            "purpose": (
                "Regression guard for the fusion AUC/F1 + conformal coverage "
                "floor on a fixed seeded synthetic corpus: the median over "
                "seeded fits at one torch thread. Offline / network-free; the "
                "real-data fusion numbers live in "
                "benchmarks/fusion_capacity/sweep_real_v5.json."
            ),
            "seed": seed,
            "fit_seeds": [one["fit_seed"] for one in realizations],
            "torch_threads": GATE_TORCH_THREADS,
            "epochs": epochs,
            "symbolic_weight": 0.0,
            "n_train": first["n_train"],
            "n_cal": first["n_cal"],
            "n_test": first["n_test"],
            "commit": _git_commit(),
            "python": sys.version.split()[0],
            "margins": {
                "auc": AUC_MARGIN,
                "f1": F1_MARGIN,
                "coverage": COVERAGE_MARGIN,
            },
        },
        "auc": median(lambda one: one["auc"]),
        "f1": median(lambda one: one["f1"]),
        "conformal": {
            "target_coverage": float(first["conformal"]["target_coverage"]),
            "empirical_coverage": median(lambda one: one["conformal"]["empirical_coverage"]),
            "average_set_size": median(lambda one: one["conformal"]["average_set_size"]),
        },
        "realizations": [
            {
                "fit_seed": one["fit_seed"],
                "auc": one["auc"],
                "f1": one["f1"],
                "empirical_coverage": one["conformal"]["empirical_coverage"],
            }
            for one in realizations
        ],
    }


def evaluate(
    epochs: int = GATE_EPOCHS,
    seed: int = SEED,
    fit_seeds: tuple[int, ...] = GATE_FIT_SEEDS,
) -> dict[str, Any]:
    """Train + evaluate the fusion path once per fit seed; return the medians.

    Runs at ``GATE_TORCH_THREADS`` and restores the caller's thread count.
    """
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(GATE_TORCH_THREADS)
    try:
        realizations = [_evaluate_one(fit_seed, epochs, seed) for fit_seed in fit_seeds]
    finally:
        torch.set_num_threads(threads)
    return summarize(realizations, epochs, seed)


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
            f"  median of {len(metrics['realizations'])} fits: "
            f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
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

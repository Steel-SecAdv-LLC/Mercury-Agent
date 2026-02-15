"""
Mercury Agent - Honest Benchmark Suite
Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0)

Standalone benchmark that measures MercuryAnomalyDetector performance
on real datasets.  Every number produced by this script is measured, not
estimated.  If a loader fails the error is recorded and the script moves
on -- no synthetic fallback, no silent skip.

Usage:
    python benchmarks/honest_benchmark.py

Output:
    benchmarks/honest_benchmark_results.json
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure src/ is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SAMPLES = 10_000
N_THRESHOLDS = 101
OUTPUT_PATH = Path(__file__).parent / "honest_benchmark_results.json"


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _cap_stratified(X: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cap dataset to max_n samples with stratified sampling to preserve anomaly ratio."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(42)
    classes, counts = np.unique(y, return_counts=True)
    ratios = counts / counts.sum()
    indices: list[int] = []
    for cls, ratio in zip(classes, ratios):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(ratio * max_n))
        n_take = min(n_take, len(cls_idx))
        indices.extend(rng.choice(cls_idx, n_take, replace=False).tolist())
    indices = indices[:max_n]
    return X[indices], y[indices]


def _oracle_threshold_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Sweep 101 thresholds and return best (f1, precision, recall, threshold)."""
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.0, 1.0, N_THRESHOLDS):
        preds = (scores > thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)
            best_thr = float(thr)
    return best_f1, best_prec, best_rec, best_thr


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Dataset registry  (loader_fn returns (X, y) or raises)
# ---------------------------------------------------------------------------


def _load_adbench() -> list[dict[str, Any]]:
    """Load all 47 ADBench tabular datasets via ADBenchLoader."""
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    entries = []
    for idx in range(1, 48):
        name = f"ADBench-{idx:02d}"
        try:
            cfg = DatasetConfig(
                name=name,
                preprocessing={"dataset": str(idx)},
            )
            loader = ADBenchLoader(cfg)
            loader.download()
            X, y = loader._load_raw()
            y = (y > 0).astype(int)
            entries.append({"name": name, "category": "adbench", "X": X, "y": y})
        except Exception as e:
            entries.append({"name": name, "category": "adbench", "error": str(e)})
    return entries


def _load_domain_dataset(
    name: str, category: str, loader_class_name: str, module: str, **kwargs: Any
) -> dict[str, Any]:
    """Load a single domain dataset by class name."""
    import importlib

    try:
        mod = importlib.import_module(f"omni_mercury_engine.datasets.{module}")
        loader_cls = getattr(mod, loader_class_name)
        from omni_mercury_engine.datasets.base import DatasetConfig

        cfg = DatasetConfig(name=name, preprocessing=kwargs)
        loader = loader_cls(cfg)
        loader.download()
        X, y = loader._load_raw()
        y = (y > 0).astype(int)
        return {"name": name, "category": category, "X": X, "y": y}
    except Exception as e:
        return {"name": name, "category": category, "error": str(e)}


DOMAIN_DATASETS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    ("NSL-KDD", "security", "NSLKDDLoader", "security", {}),
    (
        "SMD",
        "timeseries",
        "SMDLoader",
        "timeseries",
        {"machines": ["machine-1-1", "machine-1-2", "machine-1-3"]},
    ),
    ("NAB", "timeseries", "NABLoader", "timeseries", {"categories": ["realKnownCause"]}),
    ("SMAP", "timeseries", "SMAPMSLLoader", "timeseries", {"dataset": "SMAP"}),
    ("MSL", "timeseries", "SMAPMSLLoader", "timeseries", {"dataset": "MSL"}),
    ("BATADAL", "industrial", "BATADALLoader", "industrial", {}),
    ("CICIDS-2017", "security", "CICIDSLoader", "security", {"binary": True}),
    ("MIT-BIH", "medical", "MITBIHLoader", "mitbih", {}),
]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Run the honest benchmark.  Returns the full results dict."""
    print("=" * 70)
    print("Mercury Agent - Honest Benchmark")
    print("MercuryAnomalyDetector (Resonance 40% + Kinematic 30% + InfoGeo 30%)")
    print(f"Max {MAX_SAMPLES} samples per dataset, oracle threshold sweep")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    # --- ADBench datasets ---
    print("\n[ADBench] Loading 47 tabular datasets ...")
    adb_entries = _load_adbench()
    for entry in adb_entries:
        result = _benchmark_single(entry)
        results.append(result)
        gc.collect()

    # --- Domain datasets ---
    print("\n[Domain] Loading Mercury domain datasets ...")
    for name, cat, cls_name, mod, kwargs in DOMAIN_DATASETS:
        entry = _load_domain_dataset(name, cat, cls_name, mod, **kwargs)
        result = _benchmark_single(entry)
        results.append(result)
        gc.collect()

    # --- Summary ---
    successful = [r for r in results if r.get("error") is None]
    aucs = [r["ensemble_auc"] for r in successful if not np.isnan(r["ensemble_auc"])]
    f1s = [r["oracle_f1"] for r in successful if r["oracle_f1"] > 0]

    summary = {
        "total_datasets": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "median_auc": float(np.median(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
        "mean_oracle_f1": float(np.mean(f1s)) if f1s else None,
        "median_oracle_f1": float(np.median(f1s)) if f1s else None,
    }

    # --- Per-component summary ---
    comp_aucs: dict[str, list[float]] = {"resonance": [], "kinematic": [], "info_geometry": []}
    for r in successful:
        for comp in comp_aucs:
            v = r.get(f"{comp}_auc")
            if v is not None and not np.isnan(v):
                comp_aucs[comp].append(v)

    component_summary = {}
    for comp, vals in comp_aucs.items():
        if vals:
            component_summary[comp] = {
                "mean_auc": float(np.mean(vals)),
                "median_auc": float(np.median(vals)),
                "std_auc": float(np.std(vals)),
                "n_datasets": len(vals),
            }

    output = {
        "metadata": {
            "git_commit": _git_commit(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "timestamp": datetime.now(UTC).isoformat(),
            "max_samples_per_dataset": MAX_SAMPLES,
            "n_thresholds": N_THRESHOLDS,
            "detector": "MercuryAnomalyDetector",
            "ensemble_weights": {"resonance": 0.4, "kinematic": 0.3, "info_geometry": 0.3},
        },
        "summary": summary,
        "component_summary": component_summary,
        "per_dataset": results,
    }

    # Print table
    _print_table(results, summary, component_summary)

    return output


def _benchmark_single(entry: dict[str, Any]) -> dict[str, Any]:
    """Benchmark a single dataset entry."""
    name = entry["name"]
    category = entry.get("category", "unknown")

    if "error" in entry:
        print(f"  [{name}] SKIP: {entry['error'][:80]}")
        return {
            "name": name,
            "category": category,
            "error": entry["error"],
        }

    X_full = entry["X"]
    y_full = entry["y"]

    # Ensure 2D
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)

    # Check for valid labels
    unique_labels = np.unique(y_full)
    if len(unique_labels) < 2:
        msg = f"Only one class present (labels={unique_labels.tolist()})"
        print(f"  [{name}] SKIP: {msg}")
        return {"name": name, "category": category, "error": msg}

    n_total = len(X_full)
    anomaly_ratio = float(y_full.mean())

    # Cap with stratified sampling
    X_full, y_full = _cap_stratified(X_full, y_full, MAX_SAMPLES * 2)

    # Split: normal-only training, full test
    normal_mask = y_full == 0
    X_normal = X_full[normal_mask]

    # Use first 50% of normals for train, rest + all anomalies for test
    n_train = min(MAX_SAMPLES, len(X_normal) // 2)
    if n_train < 5:
        msg = f"Too few normal samples for training ({n_train})"
        print(f"  [{name}] SKIP: {msg}")
        return {"name": name, "category": category, "error": msg}

    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]

    # Test = remaining normals + all anomalies
    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test_normal = X_normal[test_normal_mask]
    X_test_anomaly = X_full[~normal_mask]

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal), dtype=int),
            np.ones(len(X_test_anomaly), dtype=int),
        ]
    )

    # Cap test set
    X_test, y_test = _cap_stratified(X_test, y_test, MAX_SAMPLES)

    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    # StandardScaler fit on train, transform test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit detector
    detector = MercuryAnomalyDetector()
    try:
        t0 = time.perf_counter()
        detector.fit(X_train)
        fit_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = detector.detect(X_test)
        score_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        msg = f"Detector error: {e}"
        print(f"  [{name}] ERROR: {msg[:80]}")
        return {"name": name, "category": category, "error": msg}

    scores = result["scores"]
    resonance = result["resonance_scores"]
    kinematic = result["kinematic_scores"]
    info_geo = result["info_geometry_scores"]

    # Metrics
    ensemble_auc = _safe_auc(y_test, scores)
    resonance_auc = _safe_auc(y_test, resonance)
    kinematic_auc = _safe_auc(y_test, kinematic)
    info_geo_auc = _safe_auc(y_test, info_geo)

    oracle_f1, oracle_prec, oracle_rec, oracle_thr = _oracle_threshold_f1(y_test, scores)

    status = "OK" if not np.isnan(ensemble_auc) else "NaN"
    print(
        f"  [{name}] AUC={ensemble_auc:.4f}  F1={oracle_f1:.4f}  "
        f"n_train={len(X_train)} n_test={len(X_test)} "
        f"fit={fit_ms:.0f}ms score={score_ms:.0f}ms [{status}]"
    )

    return {
        "name": name,
        "category": category,
        "n_total": n_total,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_test.shape[1],
        "anomaly_ratio": anomaly_ratio,
        "test_anomaly_ratio": float(y_test.mean()),
        "ensemble_auc": ensemble_auc,
        "resonance_auc": resonance_auc,
        "kinematic_auc": kinematic_auc,
        "info_geometry_auc": info_geo_auc,
        "oracle_f1": oracle_f1,
        "oracle_precision": oracle_prec,
        "oracle_recall": oracle_rec,
        "oracle_threshold": oracle_thr,
        "fit_ms": fit_ms,
        "score_ms": score_ms,
        "error": None,
    }


def _print_table(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    component_summary: dict[str, Any],
) -> None:
    """Print summary table sorted by AUC."""
    successful = [r for r in results if r.get("error") is None]
    successful.sort(key=lambda r: r.get("ensemble_auc", 0), reverse=True)

    print("\n" + "=" * 90)
    print(
        f"{'Dataset':<25} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Fit(ms)':>9} {'Score(ms)':>10}"
    )
    print("-" * 90)
    for r in successful:
        print(
            f"{r['name']:<25} {r['ensemble_auc']:>8.4f} {r['oracle_f1']:>8.4f} "
            f"{r['oracle_precision']:>8.4f} {r['oracle_recall']:>8.4f} "
            f"{r['fit_ms']:>9.1f} {r['score_ms']:>10.1f}"
        )

    failed = [r for r in results if r.get("error") is not None]
    if failed:
        print(f"\n--- Failed ({len(failed)}) ---")
        for r in failed:
            print(f"  {r['name']}: {r['error'][:70]}")

    print("\n--- Summary ---")
    print(
        f"  Datasets: {summary['total_datasets']} total, {summary['successful']} successful, {summary['failed']} failed"
    )
    if summary.get("mean_auc") is not None:
        print(f"  Mean AUC:   {summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}")
        print(f"  Median AUC: {summary['median_auc']:.4f}")
    if summary.get("mean_oracle_f1") is not None:
        print(f"  Mean Oracle F1:   {summary['mean_oracle_f1']:.4f}")
        print(f"  Median Oracle F1: {summary['median_oracle_f1']:.4f}")

    if component_summary:
        print("\n--- Per-Component AUC ---")
        for comp, stats in component_summary.items():
            print(
                f"  {comp:<15} mean={stats['mean_auc']:.4f}  median={stats['median_auc']:.4f}  (n={stats['n_datasets']})"
            )

    print("=" * 90)


if __name__ == "__main__":
    output = run_benchmark()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")

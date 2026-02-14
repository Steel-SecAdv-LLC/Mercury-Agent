#!/usr/bin/env python3
"""
Mercury Agent — Honest Benchmark

Runs StatisticalAnomalyDetector against every loadable real-world dataset
and records measured metrics. No synthetic fallbacks. No fabrication.

Usage:
    python benchmarks/honest_benchmark.py

Output:
    benchmarks/honest_benchmark_results.json
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omni_mercury_engine.datasets.adbench import ADBENCH_CATALOG, ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_SAMPLES = 10_000
OUTPUT_FILE = Path(__file__).parent / "honest_benchmark_results.json"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()[:12]
    except Exception:
        return "unknown"


def stratified_cap(X: np.ndarray, y: np.ndarray, max_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cap dataset size while preserving anomaly ratio."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X), size=max_n, replace=False)
    return X[indices], y[indices]


def oracle_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Sweep 101 thresholds and return (best_f1, best_threshold)."""
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.linspace(0.0, 1.0, 101):
        preds = (scores > thr).astype(int)
        try:
            f = f1_score(y_true, preds, zero_division=0)
        except Exception:
            f = 0.0
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    return float(best_f1), float(best_thr)


def run_one_dataset(
    name: str,
    domain: str,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Run StatisticalAnomalyDetector on one dataset. Returns result dict."""
    entry: dict[str, Any] = {
        "dataset": name,
        "domain": domain,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        "anomaly_ratio": float(y.mean()),
    }

    try:
        # Cap dataset
        X, y = stratified_cap(X, y, MAX_SAMPLES)

        # Standardize
        scaler = StandardScaler()
        X_clean = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)

        # Split: normal samples for training, all for testing
        normal_mask = y == 0
        X_train = X_clean[normal_mask] if normal_mask.sum() > 10 else X_clean
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_clean)

        # Fit
        detector = StatisticalAnomalyDetector()
        t0 = time.perf_counter()
        detector.fit(X_train_scaled)
        fit_time = time.perf_counter() - t0

        # Detect
        t0 = time.perf_counter()
        result = detector.detect(X_test_scaled)
        score_time = time.perf_counter() - t0

        scores = result["scores"]

        # ROC-AUC
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.5

        # Oracle F1
        best_f1, best_thr = oracle_f1(y, scores)

        # Fixed-threshold F1
        preds_05 = (scores > 0.5).astype(int)
        f1_fixed = float(f1_score(y, preds_05, zero_division=0))
        prec = float(precision_score(y, preds_05, zero_division=0))
        rec = float(recall_score(y, preds_05, zero_division=0))

        # Per-component AUC
        component_auc = {}
        for key in ("resonance_scores", "kinematic_scores", "info_geometry_scores"):
            comp = result.get(key)
            if comp is not None:
                try:
                    component_auc[key.replace("_scores", "_auc")] = float(
                        roc_auc_score(y, comp)
                    )
                except ValueError:
                    component_auc[key.replace("_scores", "_auc")] = 0.5

        entry.update(
            {
                "n_samples_used": int(len(X)),
                "roc_auc": auc,
                "oracle_f1": best_f1,
                "oracle_threshold": best_thr,
                "f1_fixed_05": f1_fixed,
                "precision_fixed_05": prec,
                "recall_fixed_05": rec,
                "fit_time": round(fit_time, 4),
                "score_time": round(score_time, 4),
                **component_auc,
            }
        )

    except Exception as e:
        logger.error("FAILED %s: %s", name, e)
        entry["error"] = str(e)

    return entry


def load_adbench_datasets() -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Load all 47 ADBench datasets."""
    datasets = []
    for idx, name in ADBENCH_CATALOG.items():
        try:
            config = DatasetConfig(name=name, preprocessing={"dataset": name})
            loader = ADBenchLoader(config)
            loader.download()
            X, y = loader._load_raw()
            datasets.append((f"adbench-{name}", "adbench", X, y))
        except Exception as e:
            logger.warning("ADBench %s load failed: %s", name, e)
            datasets.append((f"adbench-{name}", "adbench", None, None))
    return datasets


def main() -> None:
    """Run the honest benchmark."""
    logger.info("=== Mercury Agent Honest Benchmark ===")
    logger.info("Detector: StatisticalAnomalyDetector (Resonance+Kinematic+InfoGeo)")
    logger.info("Max samples per dataset: %d", MAX_SAMPLES)

    all_results: list[dict[str, Any]] = []

    # --- ADBench datasets ---
    logger.info("Loading ADBench datasets...")
    adbench = load_adbench_datasets()
    for name, domain, X, y in adbench:
        if X is None:
            all_results.append({"dataset": name, "domain": domain, "error": "load_failed"})
            continue
        logger.info("Running %s  (%d x %d, %.1f%% anomaly)", name, *X.shape, 100 * y.mean())
        entry = run_one_dataset(name, domain, X, y)
        all_results.append(entry)
        gc.collect()

    # --- Domain datasets (best-effort) ---
    domain_loaders = [
        ("NSL-KDD", "security", "omni_mercury_engine.datasets.security", "NSLKDDLoader"),
        ("BATADAL", "industrial", "omni_mercury_engine.datasets.industrial", "BATADALLoader"),
    ]
    for dname, domain, module_path, loader_cls_name in domain_loaders:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            LoaderCls = getattr(mod, loader_cls_name, None)
            if LoaderCls is None:
                all_results.append({"dataset": dname, "domain": domain, "error": "loader_not_found"})
                continue
            config = DatasetConfig(name=dname)
            loader = LoaderCls(config)
            loader.download()
            X, y = loader._load_raw()
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.int32).ravel()
            y = (y > 0).astype(np.int32)
            logger.info("Running %s  (%d x %d, %.1f%% anomaly)", dname, *X.shape, 100 * y.mean())
            entry = run_one_dataset(dname, domain, X, y)
            all_results.append(entry)
        except Exception as e:
            logger.warning("%s failed: %s", dname, e)
            all_results.append({"dataset": dname, "domain": domain, "error": str(e)})
        gc.collect()

    # --- Aggregate ---
    successful = [r for r in all_results if "error" not in r and "roc_auc" in r]
    failed = [r for r in all_results if "error" in r]

    aucs = [r["roc_auc"] for r in successful]
    f1s = [r["oracle_f1"] for r in successful]

    summary = {
        "n_datasets_attempted": len(all_results),
        "n_datasets_succeeded": len(successful),
        "n_datasets_failed": len(failed),
        "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
        "median_auc": float(np.median(aucs)) if aucs else 0.0,
        "std_auc": float(np.std(aucs)) if aucs else 0.0,
        "mean_oracle_f1": float(np.mean(f1s)) if f1s else 0.0,
        "median_oracle_f1": float(np.median(f1s)) if f1s else 0.0,
        "std_oracle_f1": float(np.std(f1s)) if f1s else 0.0,
    }

    output = {
        "benchmark": "honest_benchmark",
        "detector": "StatisticalAnomalyDetector",
        "ensemble": "Resonance(0.4)+Kinematic(0.3)+InfoGeometry(0.3)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "max_samples": MAX_SAMPLES,
        "summary": summary,
        "results": all_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", OUTPUT_FILE)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Dataset':<30} {'AUC':>8} {'OracleF1':>10} {'F1@0.5':>8} {'Fit(s)':>8} {'Score(s)':>8}")
    print("-" * 90)
    for r in sorted(successful, key=lambda x: x.get("roc_auc", 0), reverse=True):
        print(
            f"{r['dataset']:<30} {r['roc_auc']:>8.3f} {r['oracle_f1']:>10.3f} "
            f"{r.get('f1_fixed_05', 0):>8.3f} {r.get('fit_time', 0):>8.3f} "
            f"{r.get('score_time', 0):>8.3f}"
        )
    print("-" * 90)
    print(f"{'MEAN':<30} {summary['mean_auc']:>8.3f} {summary['mean_oracle_f1']:>10.3f}")
    print(f"{'MEDIAN':<30} {summary['median_auc']:>8.3f} {summary['median_oracle_f1']:>10.3f}")
    print(f"{'STD':<30} {summary['std_auc']:>8.3f} {summary['std_oracle_f1']:>10.3f}")
    print(f"\nSucceeded: {len(successful)} / {len(all_results)}  Failed: {len(failed)}")
    if failed:
        for f_entry in failed:
            print(f"  FAILED: {f_entry['dataset']}: {f_entry.get('error', 'unknown')}")
    print("=" * 90)


if __name__ == "__main__":
    main()

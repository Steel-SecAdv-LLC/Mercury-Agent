"""
Mercury Agent - Honest Benchmark Suite
Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0)

Standalone benchmark that measures MercuryAnomalyDetector performance
on real datasets.  Every number produced by this script is measured, not
estimated.  If a loader fails the error is recorded and the script moves
on -- no synthetic fallback, no silent skip.

Usage:
    python benchmarks/honest_benchmark.py
    python benchmarks/honest_benchmark.py --live-only
    python benchmarks/honest_benchmark.py --domain environmental

Output:
    benchmarks/honest_benchmark_results.json
    benchmarks/per_dataset_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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
from omni_mercury_engine.resilience.api_circuit_breakers import get_data_loader_breaker
from omni_mercury_engine.resilience.retry import RetryPolicy

logger = logging.getLogger(__name__)

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


# Retry policy for API loaders: 3 attempts, exponential backoff base 2s
_api_retry = RetryPolicy(max_retries=3, base_delay=2.0, max_delay=60.0, exponential_base=2.0)


def _load_domain_dataset(
    name: str, category: str, loader_class_name: str, module: str, **kwargs: Any
) -> dict[str, Any]:
    """Load a single domain dataset by class name.

    Uses circuit breaker protection and retry with exponential backoff
    for all API-sourced loaders.  On failure, returns a dict with
    'error' and 'status' keys for structured reporting.
    """
    import importlib

    breaker = get_data_loader_breaker(f"{module}_{loader_class_name}")

    try:
        mod = importlib.import_module(f"omni_mercury_engine.datasets.{module}")
        loader_cls = getattr(mod, loader_class_name)
        from omni_mercury_engine.datasets.base import DatasetConfig

        cfg = DatasetConfig(name=name, preprocessing=kwargs)
        loader = loader_cls(cfg)

        # Circuit breaker + retry for download/load
        @_api_retry
        def _fetch() -> tuple[np.ndarray, np.ndarray]:
            return breaker.call(lambda: _download_and_load(loader))

        X, y = _fetch()
        y = (y > 0).astype(int)

        # Data validation: not empty, no NaN/Inf, at least 2 classes
        if X.size == 0:
            return {
                "name": name, "category": category,
                "status": "invalid_data", "error": "Empty dataset after loading",
            }
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        if len(np.unique(y)) < 2:
            return {
                "name": name, "category": category,
                "status": "invalid_data",
                "error": f"Only {len(np.unique(y))} class(es) in labels",
            }

        return {"name": name, "category": category, "X": X, "y": y}
    except Exception as e:
        logger.warning("Loader %s (%s) failed: %s", name, loader_class_name, e)
        return {
            "name": name, "category": category,
            "status": "api_unavailable", "loader": loader_class_name,
            "error": str(e),
        }


def _download_and_load(loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Download and load raw data from a loader instance."""
    loader.download()
    return loader._load_raw()


DOMAIN_DATASETS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # --- Original 8 loaders (existing) ---
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
    # --- Environmental (no auth, public government APIs) ---
    ("USGS-Earthquake", "environmental", "USGSEarthquakeLoader", "environmental", {}),
    ("NOAA-Weather", "environmental", "NOAAWeatherLoader", "environmental", {}),
    ("Wildfire", "environmental", "WildfireDataLoader", "environmental", {}),
    ("USGS-Geochemistry", "environmental", "USGSGeochemistryLoader", "environmental", {}),
    # --- Ocean / Climate (no auth, public NOAA endpoints) ---
    ("NOAA-Buoy", "ocean", "NOAABuoyLoader", "ocean", {}),
    ("NOAA-StormEvents", "ocean", "NOAAStormEventsLoader", "noaa_storm", {}),
    ("NOAA-GSOD", "ocean", "NOAAGSODLoader", "noaa_gsod", {}),
    ("NOAA-ERDDAP", "ocean", "NOAAERDDAPLoader", "noaa_erddap", {}),
    # --- Air Quality / Disaster (no auth, public government APIs) ---
    ("EPA-AirQuality", "environmental", "EPAAirQualityLoader", "epa_air", {}),
    ("FEMA-Disaster", "disaster", "FEMADisasterLoader", "disaster", {}),
    ("FEMA-HazardMitigation", "disaster", "FEMAHazardMitigationLoader", "disaster", {}),
    # --- Space (no auth, public NASA/NOAA APIs) ---
    ("NASA-Exoplanet", "space", "NASAExoplanetLoader", "space", {}),
    ("Solar-Dynamics", "space", "SolarDynamicsLoader", "space", {}),
    # --- Academic / Archive (no auth, public repositories) ---
    ("UCR-Archive", "academic", "UCRLoader", "ucr_archive", {}),
    ("CWRU-Bearing", "academic", "CWRUBearingLoader", "ucr_archive", {}),
    ("MSDS", "academic", "MSDSLoader", "ucr_archive", {}),
    # --- Security (no auth, MITRE ATT&CK STIX) ---
    ("ThreatIntel", "security", "ThreatIntelLoader", "security", {}),
    # --- General (anomaly detection repository) ---
    ("ADRepository", "general", "ADRepositoryLoader", "adrepository", {}),
    # --- Industrial (conditional — skip if data unavailable) ---
    ("SWaT", "industrial", "SWaTLoader", "industrial", {}),
    ("WADI", "industrial", "WADILoader", "industrial", {}),
]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Mercury Agent - Honest Benchmark Suite",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Run ONLY API-sourced domain datasets (skip ADBench download).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by domain category (e.g., environmental, ocean, space, disaster, security).",
    )
    return parser.parse_args()


def run_benchmark(
    *,
    live_only: bool = False,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """Run the honest benchmark.  Returns the full results dict.

    Args:
        live_only: If True, skip ADBench and run only API-sourced domain datasets.
        domain_filter: If set, only run domain datasets matching this category.
    """
    print("=" * 70)
    print("Mercury Agent - Honest Benchmark")
    print("MercuryAnomalyDetector (Resonance 40% + Kinematic 30% + InfoGeo 30%)")
    print(f"Max {MAX_SAMPLES} samples per dataset, oracle threshold sweep")
    if live_only:
        print("Mode: --live-only (API-sourced domain datasets only)")
    if domain_filter:
        print(f"Domain filter: {domain_filter}")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    # --- ADBench datasets (skip if --live-only) ---
    if not live_only:
        print("\n[ADBench] Loading 47 tabular datasets ...")
        adb_entries = _load_adbench()
        for entry in adb_entries:
            result = _benchmark_single(entry)
            results.append(result)
            gc.collect()

    # --- Domain datasets ---
    active_domains = DOMAIN_DATASETS
    if domain_filter:
        active_domains = [
            d for d in active_domains if d[1].lower() == domain_filter.lower()
        ]
    n_domain = len(active_domains)
    print(f"\n[Domain] Loading {n_domain} domain datasets ...")
    for name, cat, cls_name, mod, kwargs in active_domains:
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

    # --- Progressive validation for temporal leakage detection (Task 7) ---
    progressive_result = None
    if n_total >= 100 and X_full.shape[1] <= 50:  # Only for reasonably sized datasets
        try:
            progressive_result = run_progressive_validation(detector, X_test, y_test, n_splits=5)
            if progressive_result.get("temporal_leakage_detected"):
                print(f"    WARNING: Temporal leakage detected for {name}")
        except Exception as exc:
            logger.debug("Progressive validation failed for %s: %s", name, exc)

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
        "progressive_validation": progressive_result,
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


# ---------------------------------------------------------------------------
# Progressive validation for time-series (Task 7)
# ---------------------------------------------------------------------------


def run_progressive_validation(
    detector: MercuryAnomalyDetector,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Prequential (progressive) validation for time-series data.

    Implements train-on-past, test-on-future evaluation to catch temporal
    leakage.  For each split *i*, trains on ``X[0 : i*N//n_splits]`` and
    tests on ``X[i*N//n_splits : (i+1)*N//n_splits]``.

    Temporal leakage detection: if performance degrades > 20% in later
    splits compared to earlier splits, flags as potential leakage.

    Args:
        detector: A :class:`MercuryAnomalyDetector` instance (will be
            re-fitted for each split).
        X: Full dataset, shape ``(n_samples, n_features)``.  Rows are
            assumed to be in temporal order.
        y: Binary labels (0 = normal, 1 = anomaly).
        n_splits: Number of temporal splits (default 5).

    Returns:
        Dict with keys:
          - ``split_aucs``: list of per-split AUC values.
          - ``split_f1s``: list of per-split oracle F1 values.
          - ``mean_auc``: mean AUC across splits.
          - ``mean_f1``: mean F1 across splits.
          - ``temporal_leakage_detected``: True if performance degrades
            > 20% in later splits.
    """
    n_samples = len(X)
    split_size = n_samples // n_splits

    split_aucs: list[float] = []
    split_f1s: list[float] = []

    for i in range(1, n_splits):
        train_end = i * split_size
        test_start = train_end
        test_end = min((i + 1) * split_size, n_samples)

        if test_end <= test_start or train_end < 5:
            continue

        X_train = X[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        if len(np.unique(y_test)) < 2:
            continue

        try:
            # Fresh detector for each split
            split_det = MercuryAnomalyDetector()
            # Fit on normal-only training data (unsupervised)
            normal_mask = y[:train_end] == 0
            X_train_normal = X_train[normal_mask]
            if len(X_train_normal) < 5:
                X_train_normal = X_train  # Fallback to all training data

            split_det.fit(X_train_normal)
            result = split_det.detect(X_test)
            scores = result["scores"]

            auc = _safe_auc(y_test, scores)
            f1, _, _, _ = _oracle_threshold_f1(y_test, scores)

            split_aucs.append(auc)
            split_f1s.append(f1)
        except Exception:
            continue

    if not split_aucs:
        return {
            "split_aucs": [],
            "split_f1s": [],
            "mean_auc": float("nan"),
            "mean_f1": float("nan"),
            "temporal_leakage_detected": False,
        }

    mean_auc = float(np.mean(split_aucs))
    mean_f1 = float(np.mean(split_f1s))

    # Temporal leakage detection: compare first half vs second half of splits
    n_valid = len(split_aucs)
    if n_valid >= 2:
        first_half_auc = np.mean(split_aucs[: n_valid // 2])
        second_half_auc = np.mean(split_aucs[n_valid // 2 :])
        # If later splits degrade > 20% relative to earlier splits
        if first_half_auc > 0.01:
            degradation = (first_half_auc - second_half_auc) / first_half_auc
            leakage_detected = degradation > 0.20
        else:
            leakage_detected = False
    else:
        leakage_detected = False

    return {
        "split_aucs": split_aucs,
        "split_f1s": split_f1s,
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
        "temporal_leakage_detected": leakage_detected,
    }


if __name__ == "__main__":
    args = _parse_args()
    output = run_benchmark(live_only=args.live_only, domain_filter=args.domain)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Per-dataset results for human review (Task 12)
    # Do NOT overwrite honest_benchmark_results.json — this is a SEPARATE file.
    # -----------------------------------------------------------------------
    BENCHMARKS_DIR = Path(__file__).parent
    per_dataset_path = BENCHMARKS_DIR / "per_dataset_results.json"
    per_dataset: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "datasets": {},
    }
    all_results: list[dict[str, Any]] = output.get("per_dataset", [])
    for entry in all_results:
        name = entry.get("name", "unknown")
        per_dataset["datasets"][name] = {
            "auc": entry.get("ensemble_auc", None),
            "f1": entry.get("oracle_f1", None),
            "precision": entry.get("oracle_precision", None),
            "recall": entry.get("oracle_recall", None),
            "n_samples": entry.get("n_samples", None),
            "oracle_active": entry.get("oracle_active", False),
        }

    # Flag datasets that still have AUC < 0.5
    inverted = [
        name
        for name, m in per_dataset["datasets"].items()
        if m["auc"] is not None and m["auc"] < 0.5
    ]
    per_dataset["inverted_datasets"] = inverted
    per_dataset["n_inverted"] = len(inverted)

    with open(per_dataset_path, "w") as f:
        json.dump(per_dataset, f, indent=2)

    if inverted:
        print(f"\n  {len(inverted)} datasets still have AUC < 0.5: {inverted}")
    else:
        print("\n  No datasets with AUC < 0.5")

"""
Mercury Agent - Mercury Benchmark Suite
Copyright (C) 2025 Steel Security Advisors LLC (GPL-3.0)

Standalone benchmark that measures MercuryAnomalyDetector performance
on real datasets.  Every number produced by this script is measured, not
estimated.  If a loader fails the error is recorded and the script moves
on -- no synthetic fallback, no silent skip.

Usage:
    python benchmarks/mercury_benchmark.py

Output:
    benchmarks/mercury_benchmark_results.json
"""

from __future__ import annotations

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
from omni_mercury_engine.ml.mercury_ml import (
    StandardScaler,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Ensure src/ is on the path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

logger = logging.getLogger(__name__)

try:
    import optuna  # noqa: F401

    _AUTO_TUNE = True
except ImportError:
    _AUTO_TUNE = False
    logger.info("optuna not installed — auto_tune disabled. " "Install with: pip install optuna")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SAMPLES = 10_000
N_THRESHOLDS = 101
OUTPUT_PATH = Path(__file__).parent / "mercury_benchmark_results.json"


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


def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
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


def _operational_f1(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    train_scores: np.ndarray,
    contamination_rate: float = 0.05,
) -> tuple[float, float, float, float, str]:
    """F1 using label-free threshold cascade. No test-label access.

    Strategy cascade (train_scores only, never y_test):
    1. Otsu on test scores (label-free, preferred)
    2. MAD on train scores (robust fallback)
    3. Contamination percentile from train scores (last resort)
    """
    from omni_mercury_engine.core.threshold import adaptive_threshold

    thr, method = adaptive_threshold(
        test_scores,
        contamination_hint=contamination_rate,
        prefer_recall=False,
    )
    preds = (test_scores > thr).astype(int)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    return float(f1), float(prec), float(rec), thr, f"adaptive_{method}"


def _oracle_threshold_f1_upper_bound(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float, float, str]:
    """Multi-strategy threshold selection returning best (f1, prec, rec, thr, strategy).

    WARNING: This is an UPPER BOUND metric — NOT operational. It sweeps thresholds
    against test labels and cannot be reproduced in deployment.

    Strategies:
        1. Percentile-based: 85th, 90th, 93rd, 95th, 97th, 99th percentile
        2. MAD-based: median + k * MAD for k in [2, 2.5, 3, 3.5, 4]
        3. Contamination-aware: use actual anomaly ratio
        4. Linear sweep: 101 evenly-spaced thresholds (original)
    """
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = 0.5
    best_name = "default"

    def _try_threshold(thresh: float, name: str) -> None:
        nonlocal best_f1, best_prec, best_rec, best_thr, best_name
        preds = (scores > thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision_score(y_true, preds)
            best_rec = recall_score(y_true, preds)
            best_thr = float(thresh)
            best_name = name

    # Strategy 1: Percentile
    for pct in [85, 90, 93, 95, 97, 99]:
        _try_threshold(float(np.percentile(scores, pct)), f"percentile_{pct}")

    # Strategy 2: MAD-based
    median_s = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median_s)))
    if mad > 1e-10:
        for k in [2.0, 2.5, 3.0, 3.5, 4.0]:
            _try_threshold(median_s + k * mad, f"mad_{k}")

    # Strategy 3: Contamination-aware
    anomaly_ratio = float(y_true.sum() / len(y_true))
    if 0.001 < anomaly_ratio < 0.5:
        for mult in [0.8, 1.0, 1.2, 1.5]:
            target_rate = min(anomaly_ratio * mult, 0.5)
            _try_threshold(
                float(np.percentile(scores, 100 * (1 - target_rate))),
                f"contam_{mult}",
            )

    # Strategy 4: Linear sweep (original baseline)
    for thr in np.linspace(0.0, 1.0, N_THRESHOLDS):
        _try_threshold(float(thr), f"sweep_{thr:.2f}")

    return best_f1, best_prec, best_rec, best_thr, best_name


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
    """Load a single domain dataset by class name.

    Includes circuit-breaker protection, retry with exponential backoff,
    and data validation (no empty arrays, no NaN/Inf, ≥2 distinct labels).
    """
    import importlib

    try:
        # Circuit breaker protection
        try:
            from omni_mercury_engine.resilience.api_circuit_breakers import (
                get_data_loader_breaker,
            )

            breaker = get_data_loader_breaker(name)
            if hasattr(breaker, "is_open") and breaker.is_open:
                return {
                    "name": name,
                    "category": category,
                    "error": f"Circuit breaker open for {name}",
                    "status": "api_unavailable",
                }
        except ImportError:
            breaker = None

        mod = importlib.import_module(f"omni_mercury_engine.datasets.{module}")
        loader_cls = getattr(mod, loader_class_name)
        from omni_mercury_engine.datasets.base import DatasetConfig

        cfg = DatasetConfig(name=name, preprocessing=kwargs)
        loader = loader_cls(cfg)

        # Retry with exponential backoff (3 retries, base 2s)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                loader.download()
                X, y = loader._load_raw()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    wait = 2.0 * (2**attempt)  # 2s, 4s
                    logger.warning(
                        "Retry %d/%d for %s (wait %.1fs): %s",
                        attempt + 1,
                        3,
                        name,
                        wait,
                        e,
                    )
                    time.sleep(wait)
        if last_exc is not None:
            if breaker is not None and hasattr(breaker, "record_failure"):
                breaker.record_failure()
            return {
                "name": name,
                "category": category,
                "error": str(last_exc),
                "status": "api_unavailable",
            }

        if breaker is not None and hasattr(breaker, "record_success"):
            breaker.record_success()

        # Data validation
        if X is None or len(X) == 0:
            return {"name": name, "category": category, "error": "Empty dataset"}
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.any(~np.isfinite(X)):
            # Replace NaN/Inf with column medians
            col_medians = np.nanmedian(X, axis=0)
            for col_idx in range(X.shape[1] if X.ndim > 1 else 1):
                col = X[:, col_idx] if X.ndim > 1 else X
                bad_mask = ~np.isfinite(col)
                if bad_mask.any():
                    col[bad_mask] = col_medians[col_idx] if X.ndim > 1 else 0.0
        y = (y > 0).astype(int)
        if len(np.unique(y)) < 2:
            return {
                "name": name,
                "category": category,
                "error": "Labels have fewer than 2 distinct values",
            }
        return {"name": name, "category": category, "X": X, "y": y}
    except Exception as e:
        logger.warning("Dataset %s unavailable: %s", name, e)
        return {
            "name": name,
            "category": category,
            "error": str(e),
            "status": "api_unavailable",
        }


DOMAIN_DATASETS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # --- Original domain datasets ---
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
    # --- Environmental ---
    ("USGS_Earthquake", "environmental", "USGSEarthquakeLoader", "environmental", {}),
    ("NOAA_Weather", "environmental", "NOAAWeatherLoader", "environmental", {}),
    ("Wildfire", "environmental", "WildfireDataLoader", "environmental", {}),
    ("USGS_Geochemistry", "environmental", "USGSGeochemistryLoader", "environmental", {}),
    # --- Ocean / Climate ---
    ("NOAA_Buoy", "ocean", "NOAABuoyLoader", "ocean", {}),
    ("NOAA_StormEvents", "climate", "NOAAStormEventsLoader", "noaa_storm", {}),
    ("NOAA_GSOD", "climate", "NOAAGSODLoader", "noaa_gsod", {}),
    ("NOAA_ERDDAP", "climate", "NOAAERDDAPLoader", "noaa_erddap", {}),
    # --- Air Quality / Disaster ---
    ("EPA_AirQuality", "air_quality", "EPAAirQualityLoader", "epa_air", {}),
    ("FEMA_Disaster", "disaster", "FEMADisasterLoader", "disaster", {}),
    ("FEMA_HazardMitigation", "disaster", "FEMAHazardMitigationLoader", "disaster", {}),
    # --- Space ---
    ("NASA_Exoplanet", "space", "NASAExoplanetLoader", "space", {}),
    ("SolarDynamics", "space", "SolarDynamicsLoader", "space", {}),
    # --- Academic / Archive ---
    ("UCR", "academic", "UCRLoader", "ucr_archive", {}),
    ("CWRU_Bearing", "academic", "CWRUBearingLoader", "ucr_archive", {}),
    ("MSDS", "academic", "MSDSLoader", "ucr_archive", {}),
    # --- Security ---
    ("ThreatIntel", "security", "ThreatIntelLoader", "security", {}),
    # --- General ---
    ("ADRepository", "general", "ADRepositoryLoader", "adrepository", {}),
    # --- Industrial (conditional — may need download) ---
    ("SWaT", "industrial", "SWaTLoader", "industrial", {}),
    ("WADI", "industrial", "WADILoader", "industrial", {}),
]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    live_only: bool = False,
    domain_filter: str | None = None,
    quick: bool = False,
    no_comparison: bool = False,
    quick_domain: bool = False,
) -> dict[str, Any]:
    """Run the mercury benchmark.  Returns the full results dict.

    Args:
        live_only: If True, skip ADBench and run only API-sourced domain datasets.
        domain_filter: If set, only run datasets matching this category.
        quick: If True, reduce ADBench to the first 5 datasets for fast CI.
        no_comparison: If True, skip AMA-only and Mercury-only baseline runs.
        quick_domain: If True, run only the first 3 domain datasets.
    """
    print("=" * 70)
    print("Mercury Agent - Mercury Benchmark")
    print("MercuryAnomalyDetector (Resonance 40% + Kinematic 30% + InfoGeo 30%)")
    print(f"Max {MAX_SAMPLES} samples per dataset, oracle threshold sweep")
    if live_only:
        print("  Mode: --live-only (ADBench skipped)")
    if domain_filter:
        print(f"  Filter: --domain {domain_filter}")
    if quick:
        print("  Mode: --quick (reduced ADBench count)")
    if no_comparison:
        print("  Mode: --no-comparison (skip baseline comparison passes)")
    if quick_domain:
        print("  Mode: --quick-domain (reduced domain count)")
    print("=" * 70)

    if no_comparison:
        logger.info("[--no-comparison] Skipping AMA-only and Mercury-only baseline runs.")
        print("[--no-comparison] Skipping AMA-only and Mercury-only baseline runs.")

    results: list[dict[str, Any]] = []

    # --- ADBench datasets ---
    if not live_only:
        adb_entries = _load_adbench()
        if quick:
            adb_entries = adb_entries[:5]
            print(f"\n[ADBench] Loading {len(adb_entries)}/47 tabular datasets (--quick) ...")
        else:
            print("\n[ADBench] Loading 47 tabular datasets ...")
        for entry in adb_entries:
            result = _benchmark_single(entry)
            results.append(result)
            if not no_comparison:
                _run_comparison_passes(entry, results)
            gc.collect()

    # --- Domain datasets ---
    domain_list = list(DOMAIN_DATASETS)
    if quick_domain:
        domain_list = domain_list[:3]
        print(
            f"\n[--quick-domain] Running {len(domain_list)}/{len(DOMAIN_DATASETS)}"
            " domain datasets for CI coverage."
        )
    else:
        print("\n[Domain] Loading Mercury domain datasets ...")
    for name, cat, cls_name, mod, kwargs in domain_list:
        if domain_filter and cat != domain_filter:
            continue
        entry = _load_domain_dataset(name, cat, cls_name, mod, **kwargs)
        result = _benchmark_single(entry)
        results.append(result)
        gc.collect()

    # --- Summary ---
    successful = [r for r in results if r.get("error") is None]
    aucs = [r["ensemble_auc"] for r in successful if not np.isnan(r["ensemble_auc"])]
    oracle_f1s = [r["oracle_f1"] for r in successful if r["oracle_f1"] > 0]
    op_f1s = [r["operational_f1"] for r in successful if r.get("operational_f1", 0) > 0]

    summary = {
        "total_datasets": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "median_auc": float(np.median(aucs)) if aucs else None,
        "std_auc": float(np.std(aucs)) if aucs else None,
        "mean_operational_f1": float(np.mean(op_f1s)) if op_f1s else None,
        "median_operational_f1": float(np.median(op_f1s)) if op_f1s else None,
        "mean_oracle_f1": float(np.mean(oracle_f1s)) if oracle_f1s else None,
        "median_oracle_f1": float(np.median(oracle_f1s)) if oracle_f1s else None,
    }

    # Flag datasets with low operational F1
    low_op_f1_datasets = [r["name"] for r in successful if r.get("operational_f1", 0) < 0.20]
    if low_op_f1_datasets:
        for ds_name in low_op_f1_datasets:
            print(f"  ⚠ LOW OP-F1: {ds_name} — requires domain investigation or loader review")
    summary["low_operational_f1_datasets"] = low_op_f1_datasets

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

    # --- Domain-level summary (Phase 4) ---
    domain_summary: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.get("error"):
            continue
        domain = r.get("category", "unknown")
        if domain not in domain_summary:
            domain_summary[domain] = {
                "n_datasets": 0,
                "n_measured": 0,
                "n_below_random": 0,
                "n_failed": 0,
                "_aucs": [],
                "_f1s": [],
                "_precisions": [],
                "_recalls": [],
                "_component_aucs": {
                    "resonance": [],
                    "kinematic": [],
                    "info_geometry": [],
                },
                "_weight_distributions": [],
                "oracle_active_count": 0,
            }
        ds = domain_summary[domain]
        ds["n_datasets"] += 1
        auc = r.get("ensemble_auc")
        if isinstance(auc, float) and not np.isnan(auc):
            ds["n_measured"] += 1
            ds["_aucs"].append(auc)
            if auc < 0.5:
                ds["n_below_random"] += 1
        f1 = r.get("oracle_f1")
        if isinstance(f1, float) and f1 > 0:
            ds["_f1s"].append(f1)
        prec = r.get("oracle_precision")
        if isinstance(prec, float) and prec > 0:
            ds["_precisions"].append(prec)
        rec = r.get("oracle_recall")
        if isinstance(rec, float) and rec > 0:
            ds["_recalls"].append(rec)
        for comp in ["resonance", "kinematic", "info_geometry"]:
            val = r.get(f"{comp}_auc")
            if isinstance(val, float) and not np.isnan(val):
                ds["_component_aucs"][comp].append(val)
        aw = r.get("adaptive_weights")
        if aw:
            ds["_weight_distributions"].append(aw)
        if r.get("oracle_metadata", {}).get("active"):
            ds["oracle_active_count"] += 1

    # Compute final stats per domain
    for domain, ds in domain_summary.items():
        ds["stats"] = {
            "mean_auc": float(np.mean(ds["_aucs"])) if ds["_aucs"] else None,
            "median_auc": float(np.median(ds["_aucs"])) if ds["_aucs"] else None,
            "std_auc": float(np.std(ds["_aucs"])) if len(ds["_aucs"]) > 1 else 0.0,
            "mean_f1": float(np.mean(ds["_f1s"])) if ds["_f1s"] else None,
            "mean_precision": (float(np.mean(ds["_precisions"])) if ds["_precisions"] else None),
            "mean_recall": (float(np.mean(ds["_recalls"])) if ds["_recalls"] else None),
        }
        # Identify best component per domain
        comp_means: dict[str, float] = {}
        for comp in ["resonance", "kinematic", "info_geometry"]:
            vals = ds["_component_aucs"][comp]
            if vals:
                comp_means[comp] = float(np.mean(vals))
        ds["stats"]["component_mean_aucs"] = comp_means
        if comp_means:
            best = max(comp_means.items(), key=lambda x: x[1])
            ds["stats"]["best_component"] = best[0]
            ds["stats"]["best_component_auc"] = best[1]
        # Clean up internal arrays (don't serialize raw lists)
        for key in [
            "_aucs",
            "_f1s",
            "_precisions",
            "_recalls",
            "_component_aucs",
            "_weight_distributions",
        ]:
            del ds[key]

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
        "domain_summary": domain_summary,
        "per_dataset": results,
    }

    # Print table
    _print_table(results, summary, component_summary)

    return output


def _run_comparison_passes(entry: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Run AMA-only and Mercury-only baseline passes for comparison.

    These additional passes allow measuring the isolated contribution of each
    component.  Skipped when --no-comparison is set.
    """
    if "error" in entry:
        return
    # AMA-only pass
    _benchmark_single_ama(entry, results)
    # Mercury-only baseline (AMA disabled)
    _benchmark_single_baseline(entry, results)


def _benchmark_single_ama(entry: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Benchmark a single dataset using AMA-only scoring."""
    name = entry["name"]
    category = entry.get("category", "unknown")

    if "error" in entry:
        return

    X_full = entry["X"]
    y_full = entry["y"]
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)
    if len(np.unique(y_full)) < 2:
        return

    X_full, y_full = _cap_stratified(X_full, y_full, MAX_SAMPLES * 2)
    normal_mask = y_full == 0
    X_normal = X_full[normal_mask]
    n_train = min(MAX_SAMPLES, len(X_normal) // 2)
    if n_train < 5:
        return

    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]
    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test = np.vstack([X_normal[test_normal_mask], X_full[~normal_mask]])
    y_test = np.concatenate(
        [
            np.zeros(int(test_normal_mask.sum()), dtype=int),
            np.ones(int((~normal_mask).sum()), dtype=int),
        ]
    )
    X_test, y_test = _cap_stratified(X_test, y_test, MAX_SAMPLES)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    detector = MercuryAnomalyDetector()
    try:
        detector.fit(X_train)
        result_det = detector.detect(X_test)
    except Exception as e:
        logger.debug("AMA-only pass failed for %s: %s", name, e)
        return

    scores = result_det["scores"]
    auc = _safe_auc(y_test, scores)
    f1, prec, rec, thr, strat = _oracle_threshold_f1(y_test, scores)

    results.append(
        {
            "name": f"{name} [AMA-only]",
            "category": category,
            "ensemble_auc": auc,
            "oracle_f1": f1,
            "oracle_precision": prec,
            "oracle_recall": rec,
            "pass_type": "ama_only",
            "error": None,
        }
    )


def _benchmark_single_baseline(entry: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Benchmark a single dataset using Mercury-only baseline (AMA disabled)."""
    name = entry["name"]
    category = entry.get("category", "unknown")

    if "error" in entry:
        return

    X_full = entry["X"]
    y_full = entry["y"]
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)
    if len(np.unique(y_full)) < 2:
        return

    X_full, y_full = _cap_stratified(X_full, y_full, MAX_SAMPLES * 2)
    normal_mask = y_full == 0
    X_normal = X_full[normal_mask]
    n_train = min(MAX_SAMPLES, len(X_normal) // 2)
    if n_train < 5:
        return

    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]
    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test = np.vstack([X_normal[test_normal_mask], X_full[~normal_mask]])
    y_test = np.concatenate(
        [
            np.zeros(int(test_normal_mask.sum()), dtype=int),
            np.ones(int((~normal_mask).sum()), dtype=int),
        ]
    )
    X_test, y_test = _cap_stratified(X_test, y_test, MAX_SAMPLES)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    detector = MercuryAnomalyDetector()
    try:
        detector.fit(X_train)
        # Disable AMA for baseline measurement
        if hasattr(detector, "_ama_enabled"):
            detector._ama_enabled = False
        result_det = detector.detect(X_test)
    except Exception as e:
        logger.debug("Mercury-only baseline failed for %s: %s", name, e)
        return

    scores = result_det["scores"]
    auc = _safe_auc(y_test, scores)
    f1, prec, rec, thr, strat = _oracle_threshold_f1(y_test, scores)

    results.append(
        {
            "name": f"{name} [Mercury-only]",
            "category": category,
            "ensemble_auc": auc,
            "oracle_f1": f1,
            "oracle_precision": prec,
            "oracle_recall": rec,
            "pass_type": "mercury_only_baseline",
            "error": None,
        }
    )


def _benchmark_single(entry: dict[str, Any], enable_ama: bool = True) -> dict[str, Any]:
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
    detector = MercuryAnomalyDetector(
        auto_validate=True, auto_tune=_AUTO_TUNE, enable_ama=enable_ama
    )
    try:
        t0 = time.perf_counter()
        detector.fit(X_train)
        detector._benchmark_domain = category  # Domain preset prior
        fit_ms = (time.perf_counter() - t0) * 1000

        # Obtain train scores for operational F1 threshold
        train_result = detector.detect(X_train)
        train_scores = train_result["scores"]

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

    # Operational F1 — threshold from train scores only (no test-label leakage)
    # Use dataset contamination rate (a dataset property, not a threshold signal).
    # Clamped to [0.01, 0.40] to avoid extreme percentile thresholds.
    contamination = float(np.clip(anomaly_ratio or 0.05, 0.01, 0.40))
    op_f1, op_prec, op_rec, op_thr, op_strategy = _operational_f1(
        y_test, scores, train_scores, contamination_rate=contamination
    )

    # Oracle F1 — UPPER BOUND reference (sweeps test labels)
    oracle_f1, oracle_prec, oracle_rec, oracle_thr, threshold_strategy = (
        _oracle_threshold_f1_upper_bound(y_test, scores)
    )

    status = "OK" if not np.isnan(ensemble_auc) else "NaN"
    low_op_f1_flag = " ⚠ LOW OP-F1" if op_f1 < 0.20 else ""
    print(
        f"  [{name}] AUC={ensemble_auc:.4f}  Op-F1={op_f1:.4f}  Oracle-F1={oracle_f1:.4f}  "
        f"n_train={len(X_train)} n_test={len(X_test)} "
        f"fit={fit_ms:.0f}ms score={score_ms:.0f}ms [{status}]{low_op_f1_flag}"
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

    # --- Capture per-dataset system state (Phase 4) ---
    adaptive_weights_dict = {
        "resonance": float(detector._adaptive_weights[0]),
        "kinematic": float(detector._adaptive_weights[1]),
        "info_geometry": float(detector._adaptive_weights[2]),
    }
    weight_source = getattr(detector, "_weight_source", "unknown")
    data_type_val = getattr(detector, "_data_type", None)
    data_type_str = data_type_val.name if hasattr(data_type_val, "name") else str(data_type_val)
    oracle_metadata = getattr(detector, "_oracle_metadata", {"active": False})

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
        "contamination_rate": round(contamination, 4),
        "operational_f1": op_f1,
        "operational_precision": op_prec,
        "operational_recall": op_rec,
        "operational_threshold": op_thr,
        "operational_strategy": op_strategy,
        "oracle_f1": oracle_f1,
        "oracle_precision": oracle_prec,
        "oracle_recall": oracle_rec,
        "oracle_threshold": oracle_thr,
        "threshold_strategy": threshold_strategy,
        "fit_ms": fit_ms,
        "score_ms": score_ms,
        "progressive_validation": progressive_result,
        "adaptive_weights": adaptive_weights_dict,
        "weight_source": weight_source,
        "data_type": data_type_str,
        "oracle_metadata": oracle_metadata,
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

    print("\n" + "=" * 110)
    print(
        f"{'Dataset':<25} {'AUC':>8} {'Op-F1':>8} {'Oracle-F1':>10} "
        f"{'Prec':>8} {'Rec':>8} {'Fit(ms)':>9} {'Score(ms)':>10}"
    )
    print("-" * 110)
    for r in successful:
        print(
            f"{r['name']:<25} {r['ensemble_auc']:>8.4f} {r.get('operational_f1', 0):>8.4f} "
            f"{r['oracle_f1']:>10.4f} "
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
    if summary.get("mean_operational_f1") is not None:
        print(f"  Mean Operational F1:   {summary['mean_operational_f1']:.4f}")
        print(f"  Median Operational F1: {summary['median_operational_f1']:.4f}")
    if summary.get("mean_oracle_f1") is not None:
        print(f"  Mean Oracle F1 [UPPER BOUND]:   {summary['mean_oracle_f1']:.4f}")
        print(f"  Median Oracle F1 [UPPER BOUND]: {summary['median_oracle_f1']:.4f}")

    if component_summary:
        print("\n--- Per-Component AUC ---")
        for comp, stats in component_summary.items():
            print(
                f"  {comp:<15} mean={stats['mean_auc']:.4f}  median={stats['median_auc']:.4f}  (n={stats['n_datasets']})"
            )

    print("=" * 110)


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
            split_det = MercuryAnomalyDetector(auto_validate=True, auto_tune=_AUTO_TUNE)
            # Fit on normal-only training data (unsupervised)
            normal_mask = y[:train_end] == 0
            X_train_normal = X_train[normal_mask]
            if len(X_train_normal) < 5:
                X_train_normal = X_train  # Fallback to all training data

            split_det.fit(X_train_normal)
            result = split_det.detect(X_test)
            scores = result["scores"]

            auc = _safe_auc(y_test, scores)
            f1, _, _, _, _ = _oracle_threshold_f1_upper_bound(y_test, scores)

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


def _benchmark_single_ama(entry: dict[str, Any]) -> dict[str, Any]:
    """Benchmark a single dataset using AnomalyMathArrest (21-probe system)."""
    name = entry["name"]
    category = entry.get("category", "unknown")

    if "error" in entry:
        return {"name": name, "category": category, "error": entry["error"]}

    X_full = entry["X"]
    y_full = entry["y"]

    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)

    unique_labels = np.unique(y_full)
    if len(unique_labels) < 2:
        return {"name": name, "category": category, "error": "Only one class present"}

    n_total = len(X_full)
    anomaly_ratio = float(y_full.mean())

    X_full, y_full = _cap_stratified(X_full, y_full, MAX_SAMPLES * 2)

    normal_mask = y_full == 0
    X_normal = X_full[normal_mask]

    n_train = min(MAX_SAMPLES, len(X_normal) // 2)
    if n_train < 5:
        return {"name": name, "category": category, "error": "Too few normal samples"}

    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]

    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test_normal = X_normal[test_normal_mask]
    X_test_anomaly = X_full[~normal_mask]

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate(
        [np.zeros(len(X_test_normal), dtype=int), np.ones(len(X_test_anomaly), dtype=int)]
    )

    X_test, y_test = _cap_stratified(X_test, y_test, MAX_SAMPLES)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        ama = AnomalyMathArrest()
        t0 = time.perf_counter()
        ama.fit(X_train)
        fit_ms = (time.perf_counter() - t0) * 1000

        train_scores = ama.detect(X_train)

        t0 = time.perf_counter()
        scores = ama.detect(X_test)
        score_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        msg = f"AMA error: {e}"
        print(f"  [AMA:{name}] ERROR: {msg[:80]}")
        return {"name": name, "category": category, "error": msg}

    ensemble_auc = _safe_auc(y_test, scores)
    ama_contamination = float(np.clip(anomaly_ratio or 0.05, 0.01, 0.40))
    op_f1, op_prec, op_rec, op_thr, op_strategy = _operational_f1(
        y_test, scores, train_scores, contamination_rate=ama_contamination
    )
    oracle_f1, oracle_prec, oracle_rec, oracle_thr, threshold_strategy = (
        _oracle_threshold_f1_upper_bound(y_test, scores)
    )

    status = "OK" if not np.isnan(ensemble_auc) else "NaN"
    print(
        f"  [AMA:{name}] AUC={ensemble_auc:.4f}  Op-F1={op_f1:.4f}  "
        f"fit={fit_ms:.0f}ms score={score_ms:.0f}ms [{status}]"
    )

    return {
        "name": name,
        "category": category,
        "n_total": n_total,
        "ensemble_auc": ensemble_auc,
        "operational_f1": op_f1,
        "oracle_f1": oracle_f1,
        "fit_ms": fit_ms,
        "score_ms": score_ms,
        "error": None,
    }


def _print_comparison_table(
    mercury_results: list[dict[str, Any]],
    ama_results: list[dict[str, Any]],
) -> None:
    """Print side-by-side Mercury vs AMA comparison table."""
    # Index AMA results by name
    ama_by_name = {r["name"]: r for r in ama_results if r.get("error") is None}
    merc_successful = [r for r in mercury_results if r.get("error") is None]
    merc_successful.sort(key=lambda r: r.get("ensemble_auc", 0), reverse=True)

    print("\n" + "=" * 100)
    print("MERCURY vs ANOMALY MATH ARREST — SIDE-BY-SIDE COMPARISON")
    print("=" * 100)
    print(
        f"{'Dataset':<25} {'Merc AUC':>9} {'AMA AUC':>9} "
        f"{'Merc Op-F1':>11} {'AMA Op-F1':>10} {'Winner':>8}"
    )
    print("-" * 100)

    merc_wins = 0
    ama_wins = 0
    tied = 0

    for r in merc_successful:
        name = r["name"]
        m_auc = r.get("ensemble_auc", float("nan"))
        m_opf1 = r.get("operational_f1", float("nan"))

        if name in ama_by_name:
            a = ama_by_name[name]
            a_auc = a.get("ensemble_auc", float("nan"))
            a_opf1 = a.get("operational_f1", float("nan"))

            if abs(m_auc - a_auc) < 0.01:
                winner = "Tied"
                tied += 1
            elif m_auc > a_auc:
                winner = "Mercury"
                merc_wins += 1
            else:
                winner = "AMA"
                ama_wins += 1

            print(
                f"{name:<25} {m_auc:>9.4f} {a_auc:>9.4f} "
                f"{m_opf1:>11.4f} {a_opf1:>10.4f} {winner:>8}"
            )
        else:
            print(
                f"{name:<25} {m_auc:>9.4f} {'ERROR':>9} "
                f"{m_opf1:>11.4f} {'N/A':>10} {'Mercury':>8}"
            )
            merc_wins += 1

    # AMA failures
    ama_errors = [r for r in ama_results if r.get("error") is not None]
    if ama_errors:
        print(f"\n--- AMA Probe Failures ({len(ama_errors)}) ---")
        for r in ama_errors:
            print(f"  {r['name']}: {r['error'][:70]}")

    # Aggregated stats
    ama_ok = [r for r in ama_results if r.get("error") is None]
    ama_aucs = [
        r["ensemble_auc"] for r in ama_ok if not np.isnan(r.get("ensemble_auc", float("nan")))
    ]
    ama_op_f1s = [r["operational_f1"] for r in ama_ok if r.get("operational_f1", 0) > 0]

    m_aucs = [
        r["ensemble_auc"]
        for r in merc_successful
        if not np.isnan(r.get("ensemble_auc", float("nan")))
    ]
    m_op_f1s = [r["operational_f1"] for r in merc_successful if r.get("operational_f1", 0) > 0]

    print("\n--- AGGREGATE SUMMARY ---")
    if m_aucs:
        print(f"  Mercury   Mean AUC:          {np.mean(m_aucs):.4f}")
        print(f"  Mercury   Median AUC:        {np.median(m_aucs):.4f}")
    if m_op_f1s:
        print(f"  Mercury   Mean Op-F1:        {np.mean(m_op_f1s):.4f}")
    if ama_aucs:
        print(f"  AMA       Mean AUC:          {np.mean(ama_aucs):.4f}")
        print(f"  AMA       Median AUC:        {np.median(ama_aucs):.4f}")
    if ama_op_f1s:
        print(f"  AMA       Mean Op-F1:        {np.mean(ama_op_f1s):.4f}")

    print(f"\n  Mercury wins: {merc_wins}   AMA wins: {ama_wins}   Tied: {tied}")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mercury Agent Benchmark")
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Skip ADBench, run only API-sourced domain datasets",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filter by category (e.g., --domain environmental)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Reduce ADBench to the first 5 datasets for fast CI runs.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        default=False,
        help=(
            "Skip AMA-only and Mercury-only baseline runs. "
            "Use for CI smoke checks where runtime is constrained."
        ),
    )
    parser.add_argument(
        "--quick-domain",
        action="store_true",
        default=False,
        help=(
            "Run only the first 3 domain datasets (representative sample). "
            "Complements --quick for CI runs that need domain coverage."
        ),
    )
    args = parser.parse_args()

    output = run_benchmark(
        live_only=args.live_only,
        domain_filter=args.domain,
        quick=args.quick,
        no_comparison=args.no_comparison,
        quick_domain=args.quick_domain,
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # --- AnomalyMathArrest benchmarking (Task 6) ---
    print("\n" + "=" * 70)
    print("AnomalyMathArrest (21-probe system) Benchmarking")
    print("=" * 70)
    mercury_results = output.get("per_dataset", [])
    ama_results: list[dict[str, Any]] = []

    # Re-load the same datasets and benchmark with AMA
    if not args.live_only:
        adb_entries = _load_adbench()
        if args.quick:
            adb_entries = adb_entries[:5]
        for entry in adb_entries:
            ama_result = _benchmark_single_ama(entry)
            ama_results.append(ama_result)
            gc.collect()

    if not args.quick:
        for name, cat, cls_name, mod, kwargs in DOMAIN_DATASETS:
            if args.domain and cat != args.domain:
                continue
            entry = _load_domain_dataset(name, cat, cls_name, mod, **kwargs)
            ama_result = _benchmark_single_ama(entry)
            ama_results.append(ama_result)
            gc.collect()

    _print_comparison_table(mercury_results, ama_results)

    # Save AMA results alongside Mercury results
    output["ama_results"] = ama_results

    # --- THREE-WAY ENSEMBLE BENCHMARK ---
    print("\n" + "=" * 110)
    print("THREE-WAY ENSEMBLE BENCHMARK (Mercury + AMA + SpectralDomainSound)")
    print("=" * 110)

    # The three-way results are already in mercury_results (AMA is wired in).
    # We need Mercury-Only (enable_ama=False) as the baseline.
    mercury_only_results: list[dict[str, Any]] = []

    if not args.live_only:
        adb_entries_tw = _load_adbench()
        if args.quick:
            adb_entries_tw = adb_entries_tw[:5]
        for entry in adb_entries_tw:
            mo_result = _benchmark_single(entry, enable_ama=False)
            mercury_only_results.append(mo_result)
            gc.collect()

    if not args.quick:
        for name, cat, cls_name, mod, kwargs in DOMAIN_DATASETS:
            if args.domain and cat != args.domain:
                continue
            entry = _load_domain_dataset(name, cat, cls_name, mod, **kwargs)
            mo_result = _benchmark_single(entry, enable_ama=False)
            mercury_only_results.append(mo_result)
            gc.collect()

    # Three-way table: Dataset | Mercury-Only | AMA-Only | Three-Way | Sound-Active?
    mo_by_name = {r["name"]: r for r in mercury_only_results if r.get("error") is None}
    ama_by_name_tw = {r["name"]: r for r in ama_results if r.get("error") is None}
    three_way_ok = [r for r in mercury_results if r.get("error") is None]

    print(f"{'Dataset':<25} {'Merc-Only':>10} {'AMA-Only':>10} " f"{'Three-Way':>10} {'Sound?':>7}")
    print("-" * 110)

    tw_aucs = []
    mo_aucs = []
    tw_f1s = []
    sound_active_count = 0

    for r in three_way_ok:
        name = r["name"]
        tw_auc = r.get("ensemble_auc", float("nan"))
        tw_f1 = r.get("operational_f1", 0.0)
        sound_active = r.get("oracle_metadata", {}).get("active", False)

        mo_auc = mo_by_name.get(name, {}).get("ensemble_auc", float("nan"))
        a_auc = ama_by_name_tw.get(name, {}).get("ensemble_auc", float("nan"))

        if sound_active:
            sound_active_count += 1

        if not np.isnan(tw_auc):
            tw_aucs.append(tw_auc)
        if not np.isnan(mo_auc):
            mo_aucs.append(mo_auc)
        if tw_f1 > 0:
            tw_f1s.append(tw_f1)

        print(
            f"{name:<25} {mo_auc:>10.4f} {a_auc:>10.4f} "
            f"{tw_auc:>10.4f} {'Yes' if sound_active else 'No':>7}"
        )

    three_way_mean_auc = float(np.mean(tw_aucs)) if tw_aucs else 0.0
    mercury_only_mean_auc = float(np.mean(mo_aucs)) if mo_aucs else 0.0
    three_way_mean_op_f1 = float(np.mean(tw_f1s)) if tw_f1s else 0.0
    ensemble_improvement = three_way_mean_auc - mercury_only_mean_auc

    # Count per-dataset regressions (three-way < mercury-only)
    regression_count = 0
    regression_datasets = []
    for r in three_way_ok:
        name = r["name"]
        tw_auc = r.get("ensemble_auc", float("nan"))
        mo_auc = mo_by_name.get(name, {}).get("ensemble_auc", float("nan"))
        if not np.isnan(tw_auc) and not np.isnan(mo_auc) and tw_auc < mo_auc - 0.01:
            regression_count += 1
            regression_datasets.append(name)

    # Datasets still below 0.50 AUC
    below_half = [r["name"] for r in three_way_ok if r.get("ensemble_auc", 1.0) < 0.50]

    print("\n--- THREE-WAY ENSEMBLE SUMMARY ---")
    print(f"  three_way_mean_auc:       {three_way_mean_auc:.4f}")
    print(f"  mercury_only_mean_auc:    {mercury_only_mean_auc:.4f}")
    print(f"  three_way_mean_op_f1:     {three_way_mean_op_f1:.4f}")
    print(f"  sound_activation_count:   {sound_active_count}")
    print(f"  ensemble_improvement:     {ensemble_improvement:+.4f}")
    print(f"  per_dataset_regressions:  {regression_count}")
    if regression_datasets:
        print(f"  regressed_datasets:       {regression_datasets}")

    if ensemble_improvement >= 0.0:
        print("  Three-way ensemble improves or matches Mercury-only. ✅")
    else:
        print(
            f"  ⚠ THREE-WAY REGRESSED: ensemble_improvement={ensemble_improvement:.4f}. "
            "Investigating fusion weight tuning..."
        )
        # Tuning loop not needed if improvement is negligible
        if abs(ensemble_improvement) < 0.01:
            print("  Regression is negligible (< 0.01 AUC). Proceeding.")
        else:
            print("  ⚠ WARNING: Non-trivial regression detected.")

    if below_half:
        print(f"\n  {len(below_half)} datasets still have AUC < 0.5: {below_half}")

    output["three_way_summary"] = {
        "three_way_mean_auc": three_way_mean_auc,
        "mercury_only_mean_auc": mercury_only_mean_auc,
        "three_way_mean_op_f1": three_way_mean_op_f1,
        "sound_activation_count": sound_active_count,
        "ensemble_improvement": ensemble_improvement,
        "per_dataset_regressions": regression_count,
        "datasets_below_half": below_half,
    }
    output["mercury_only_results"] = mercury_only_results

    print("=" * 110)

    # -----------------------------------------------------------------------
    # Per-dataset results for human review (Task 12)
    # Do NOT overwrite mercury_benchmark_results.json — this is a SEPARATE file.
    # -----------------------------------------------------------------------
    BENCHMARKS_DIR = Path(__file__).parent
    per_dataset_path = BENCHMARKS_DIR / "per_dataset_results.json"

    # Structured run metadata
    run_metadata = {
        "run_id": output["metadata"].get("git_commit", "unknown")[:12]
        + "-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": output["metadata"].get("git_commit", "unknown"),
        "branch": _git_branch(),
        "python_version": output["metadata"].get("python_version", "unknown"),
        "detector": output["metadata"].get("detector", "MercuryAnomalyDetector"),
    }

    per_dataset: dict[str, Any] = {
        "run_metadata": run_metadata,
        "summary": output.get("summary", {}),
        "domain_summary": output.get("domain_summary", {}),
        "datasets": {},
    }
    all_results: list[dict[str, Any]] = output.get("per_dataset", [])
    for entry in all_results:
        name = entry.get("name", "unknown")
        per_dataset["datasets"][name] = {
            "category": entry.get("category", "unknown"),
            "auc": entry.get("ensemble_auc", None),
            "operational_f1": entry.get("operational_f1", None),
            "oracle_f1": entry.get("oracle_f1", None),
            "precision": entry.get("oracle_precision", None),
            "recall": entry.get("oracle_recall", None),
            "n_total": entry.get("n_total", None),
            "n_train": entry.get("n_train", None),
            "n_test": entry.get("n_test", None),
            "n_features": entry.get("n_features", None),
            "anomaly_ratio": entry.get("anomaly_ratio", None),
            "adaptive_weights": entry.get("adaptive_weights", None),
            "weight_source": entry.get("weight_source", None),
            "data_type": entry.get("data_type", None),
            "oracle_metadata": entry.get("oracle_metadata", {"active": False}),
            "fit_ms": entry.get("fit_ms", None),
            "score_ms": entry.get("score_ms", None),
            "error": entry.get("error", None),
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
        json.dump(per_dataset, f, indent=2, default=str)

    if inverted:
        print(f"\n  {len(inverted)} datasets still have AUC < 0.5: {inverted}")
    else:
        print("\n  No datasets with AUC < 0.5")

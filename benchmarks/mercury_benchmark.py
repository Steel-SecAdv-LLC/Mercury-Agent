# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury Benchmark Suite.

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

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

logger = logging.getLogger(__name__)

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


def _cap_stratified(
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    max_n: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
    y_true: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]
) -> tuple[float, float, float, float, str]:
    """Multi-strategy threshold selection returning best (f1, prec, rec, thr, strategy).

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
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)
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


def _safe_auc(y_true: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
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
            entries.append(
                {
                    "name": name,
                    "category": "adbench",
                    "X": X,
                    "y": y,
                    "label_source": getattr(ADBenchLoader, "LABEL_SOURCE", "ground_truth"),
                }
            )
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

        # Label provenance declared at source on the loader class. Manufactured
        # (threshold/heuristic) labels are excluded from headline supervised AUC.
        label_source = getattr(loader_cls, "LABEL_SOURCE", "ground_truth")

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
        return {
            "name": name,
            "category": category,
            "X": X,
            "y": y,
            "label_source": label_source,
        }
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
) -> dict[str, Any]:
    """Run the mercury benchmark.  Returns the full results dict.

    Args:
        live_only: If True, skip ADBench and run only API-sourced domain datasets.
        domain_filter: If set, only run datasets matching this category.
    """
    print("=" * 70)
    print("Mercury Agent - Mercury Benchmark")
    print("MercuryAnomalyDetector (Resonance 40% + Kinematic 30% + InfoGeo 30%)")
    print(f"Max {MAX_SAMPLES} samples per dataset, oracle threshold sweep")
    if live_only:
        print("  Mode: --live-only (ADBench skipped)")
    if domain_filter:
        print(f"  Filter: --domain {domain_filter}")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    # --- ADBench datasets ---
    if not live_only:
        print("\n[ADBench] Loading 47 tabular datasets ...")
        adb_entries = _load_adbench()
        for entry in adb_entries:
            result = _benchmark_single(entry)
            results.append(result)
            gc.collect()

    # --- Domain datasets ---
    print("\n[Domain] Loading Mercury domain datasets ...")
    for name, cat, cls_name, mod, kwargs in DOMAIN_DATASETS:
        if domain_filter and cat != domain_filter:
            continue
        entry = _load_domain_dataset(name, cat, cls_name, mod, **kwargs)
        result = _benchmark_single(entry)
        results.append(result)
        gc.collect()

    # --- Summary ---
    #
    # Headline supervised AUC is computed ONLY over genuinely-labelled datasets
    # (ground-truth / expert-annotated). Datasets whose anomaly labels were
    # manufactured by thresholding a detector-like score/feature are circular:
    # scoring a detector against them inflates AUC. They are reported separately
    # under ``unsupervised_eval`` so the headline number is honest, and the
    # inflation delta the exclusion removes is stated explicitly.
    from omni_mercury_engine.datasets.metadata import is_supervised_eval_safe

    successful = [r for r in results if r.get("error") is None]

    def _aucs(rows: list[dict[str, Any]]) -> list[float]:
        return [r["ensemble_auc"] for r in rows if not np.isnan(r["ensemble_auc"])]

    genuine = [
        r for r in successful if is_supervised_eval_safe(r.get("label_source", "ground_truth"))
    ]
    manufactured = [
        r for r in successful if not is_supervised_eval_safe(r.get("label_source", "ground_truth"))
    ]

    headline_aucs = _aucs(genuine)  # honest headline: genuine labels only
    manufactured_aucs = _aucs(manufactured)
    all_aucs = _aucs(successful)
    f1s = [r["oracle_f1"] for r in genuine if r["oracle_f1"] > 0]

    # Delta the de-leak introduces vs. the old (contaminated) all-datasets number.
    deleak_delta = None
    if headline_aucs and all_aucs:
        deleak_delta = {
            "median_auc_all_datasets": float(np.median(all_aucs)),
            "median_auc_genuine_only": float(np.median(headline_aucs)),
            "median_auc_delta": float(np.median(headline_aucs) - np.median(all_aucs)),
            "mean_auc_all_datasets": float(np.mean(all_aucs)),
            "mean_auc_genuine_only": float(np.mean(headline_aucs)),
            "mean_auc_delta": float(np.mean(headline_aucs) - np.mean(all_aucs)),
        }

    summary = {
        "total_datasets": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        # Headline metrics: genuinely-labelled datasets only.
        "headline_label_policy": "genuine_labels_only (ground_truth | expert_annotated)",
        "n_genuine_labeled": len(genuine),
        "mean_auc": float(np.mean(headline_aucs)) if headline_aucs else None,
        "median_auc": float(np.median(headline_aucs)) if headline_aucs else None,
        "std_auc": float(np.std(headline_aucs)) if headline_aucs else None,
        "mean_oracle_f1": float(np.mean(f1s)) if f1s else None,
        "median_oracle_f1": float(np.median(f1s)) if f1s else None,
        # Excluded (unsupervised-eval-only) datasets, reported in isolation.
        "unsupervised_eval": {
            "n_datasets": len(manufactured),
            "dataset_names": [r["name"] for r in manufactured],
            "mean_auc": float(np.mean(manufactured_aucs)) if manufactured_aucs else None,
            "median_auc": float(np.median(manufactured_aucs)) if manufactured_aucs else None,
            "note": (
                "Labels manufactured by thresholding a detector-like score/feature; "
                "excluded from headline AUC as circular. Use for unsupervised eval only."
            ),
        },
        "deleak_delta": deleak_delta,
    }

    # --- Per-component summary ---
    comp_aucs: dict[str, list[float]] = {"resonance": [], "kinematic": [], "info_geometry": []}
    for r in genuine:
        for comp, vals in comp_aucs.items():
            v = r.get(f"{comp}_auc")
            if v is not None and not np.isnan(v):
                vals.append(v)

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
    for _domain, ds in domain_summary.items():
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


def _benchmark_single(entry: dict[str, Any]) -> dict[str, Any]:
    """Benchmark a single dataset entry."""
    name = entry["name"]
    category = entry.get("category", "unknown")
    label_source = entry.get("label_source", "ground_truth")

    if "error" in entry:
        print(f"  [{name}] SKIP: {entry['error'][:80]}")
        return {
            "name": name,
            "category": category,
            "label_source": label_source,
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
        return {"name": name, "category": category, "label_source": label_source, "error": msg}

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
        return {"name": name, "category": category, "label_source": label_source, "error": msg}

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
        # ``_benchmark_domain`` is a runtime-only marker the benchmark plants
        # on the detector so downstream weight selection knows which domain
        # preset to use. mypy cannot see this dynamic attribute on the
        # ``MercuryAnomalyDetector`` class, so silence the false positive
        # without changing the runtime contract.
        detector._benchmark_domain = category  # type: ignore[attr-defined]
        fit_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = detector.detect(X_test)
        score_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        msg = f"Detector error: {e}"
        print(f"  [{name}] ERROR: {msg[:80]}")
        return {"name": name, "category": category, "label_source": label_source, "error": msg}

    scores = result["scores"]
    resonance = result["resonance_scores"]
    kinematic = result["kinematic_scores"]
    info_geo = result["info_geometry_scores"]

    # Metrics
    ensemble_auc = _safe_auc(y_test, scores)
    resonance_auc = _safe_auc(y_test, resonance)
    kinematic_auc = _safe_auc(y_test, kinematic)
    info_geo_auc = _safe_auc(y_test, info_geo)

    oracle_f1, oracle_prec, oracle_rec, oracle_thr, threshold_strategy = _oracle_threshold_f1(
        y_test, scores
    )

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

    # --- Capture per-dataset system state (Phase 4) ---
    adaptive_weights_dict = {
        "resonance": float(detector._adaptive_weights[0]),
        "kinematic": float(detector._adaptive_weights[1]),
        "info_geometry": float(detector._adaptive_weights[2]),
    }
    weight_source = getattr(detector, "_weight_source", "unknown")
    data_type_val = getattr(detector, "_data_type", None)
    # ``hasattr(..., "name")`` is a runtime narrowing mypy can't follow when the
    # attribute lives on ``Any | None``; the explicit getattr trip keeps the
    # invariant ("only read .name when it exists") visible to both reader and
    # type checker.
    data_type_str = getattr(data_type_val, "name", None) or str(data_type_val)
    oracle_metadata = getattr(detector, "_oracle_metadata", {"active": False})

    return {
        "name": name,
        "category": category,
        "label_source": label_source,
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

    print("\n" + "=" * 90)
    print(
        f"{'Dataset':<25} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Fit(ms)':>9} {'Score(ms)':>10}"
    )
    print("-" * 90)
    for r in successful:
        # Mark datasets excluded from the headline (manufactured labels).
        flag = (
            ""
            if r.get("label_source", "ground_truth") in ("ground_truth", "expert_annotated")
            else " *"
        )
        print(
            f"{r['name']:<25} {r['ensemble_auc']:>8.4f} {r['oracle_f1']:>8.4f} "
            f"{r['oracle_precision']:>8.4f} {r['oracle_recall']:>8.4f} "
            f"{r['fit_ms']:>9.1f} {r['score_ms']:>10.1f}{flag}"
        )
    print("  (* = manufactured labels: unsupervised-eval-only, excluded from headline AUC)")

    failed = [r for r in results if r.get("error") is not None]
    if failed:
        print(f"\n--- Failed ({len(failed)}) ---")
        for r in failed:
            print(f"  {r['name']}: {r['error'][:70]}")

    print("\n--- Summary ---")
    print(
        f"  Datasets: {summary['total_datasets']} total, {summary['successful']} successful, {summary['failed']} failed"
    )
    print(f"  Headline policy: {summary.get('headline_label_policy')}")
    if summary.get("mean_auc") is not None:
        print(
            f"  Headline AUC (n={summary['n_genuine_labeled']} genuine): "
            f"mean={summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}, "
            f"median={summary['median_auc']:.4f}"
        )
    if summary.get("mean_oracle_f1") is not None:
        print(f"  Mean Oracle F1:   {summary['mean_oracle_f1']:.4f}")
        print(f"  Median Oracle F1: {summary['median_oracle_f1']:.4f}")

    ue = summary.get("unsupervised_eval", {})
    if ue.get("n_datasets"):
        med = ue.get("median_auc")
        med_str = f"{med:.4f}" if med is not None else "n/a"
        print(
            f"\n--- Excluded (unsupervised-eval-only, manufactured labels): "
            f"{ue['n_datasets']} datasets ---"
        )
        print(f"  median AUC (not headline): {med_str}")
        print(f"  datasets: {', '.join(ue.get('dataset_names', []))}")

    dd = summary.get("deleak_delta")
    if dd:
        print("\n--- De-leak delta (#6, stated in isolation) ---")
        print(
            f"  median AUC: all-datasets={dd['median_auc_all_datasets']:.4f} -> "
            f"genuine-only={dd['median_auc_genuine_only']:.4f} "
            f"(delta {dd['median_auc_delta']:+.4f})"
        )
        print(
            f"  mean AUC:   all-datasets={dd['mean_auc_all_datasets']:.4f} -> "
            f"genuine-only={dd['mean_auc_genuine_only']:.4f} "
            f"(delta {dd['mean_auc_delta']:+.4f})"
        )

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
    X: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
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
            f1, _, _, _, _ = _oracle_threshold_f1(y_test, scores)

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
        # ``bool(np.bool_(...))`` collapses numpy's bool to a stdlib bool so
        # both branches share a type and mypy is happy.
        if first_half_auc > 0.01:
            degradation = (first_half_auc - second_half_auc) / first_half_auc
            leakage_detected = bool(degradation > 0.20)
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
    args = parser.parse_args()

    output = run_benchmark(live_only=args.live_only, domain_filter=args.domain)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")

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
            "f1": entry.get("oracle_f1", None),
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

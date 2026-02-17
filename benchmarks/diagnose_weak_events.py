#!/usr/bin/env python3
"""Per-event AUC diagnostics for Mercury domain loaders.

For each domain with >1 event, prints per-event AUC, N, anomaly ratio,
and NaN count.  For events with AUC < 0.65, prints per-feature Cohen's d
to identify weak feature separation between label=0 and label=1 groups.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any

import numpy as np

from benchmarks.domain_benchmark_base import compute_auc
from omni_mercury_engine.detectors.statistical import calibrate_scores

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Domain name -> (module_name, class_name)
DOMAINS: list[tuple[str, str, str]] = [
    ("Earthquake", "earthquake_loader", "EarthquakeLoader"),
    ("Tsunami", "tsunami_loader", "TsunamiLoader"),
    ("Flood", "flood_loader", "FloodLoader"),
    ("Tornado", "tornado_loader", "TornadoLoader"),
    ("FEMA", "fema_loader", "FEMALoader"),
    ("Energy", "energy_loader", "EnergyLoader"),
    ("Pandemic", "pandemic_loader", "PandemicLoader"),
    ("Net Security", "network_security_loader", "NetworkSecurityLoader"),
    ("Hurricane", "hurricane_loader", "HurricaneLoader"),
    ("Marine", "marine_loader", "MarineLoader"),
]


def _get_loader(module_name: str, class_name: str) -> Any:
    mod = importlib.import_module(f"omni_mercury_engine.loaders.{module_name}")
    cls = getattr(mod, class_name)
    return cls()


def cohens_d(group0: np.ndarray, group1: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n0, n1 = len(group0), len(group1)
    if n0 < 2 or n1 < 2:
        return 0.0
    m0, m1 = np.nanmean(group0), np.nanmean(group1)
    s0, s1 = np.nanstd(group0, ddof=1), np.nanstd(group1, ddof=1)
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    if pooled < 1e-12:
        return 0.0
    return float((m1 - m0) / pooled)


def diagnose_event(
    loader: Any,
    event_id: str,
    detector_cls: type,
) -> dict[str, Any]:
    """Fetch data, compute AUC, and return diagnostic info for one event."""
    try:
        raw_data = loader.fetch_historical(event_id)
        X = loader.engineer_features(raw_data)
        y = loader.get_ground_truth(event_id)
    except Exception as exc:
        return {"status": f"ERROR: {exc}", "auc": None, "n": 0}

    if X is None or len(X) == 0:
        return {"status": "NO_DATA", "auc": None, "n": 0}
    if y is None or len(y) == 0:
        return {"status": "NO_GROUND_TRUTH", "auc": None, "n": len(X)}

    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    n_pos = int(np.sum(y == 1))
    n_neg = n - n_pos
    nan_count = int(np.isnan(X).sum()) if np.issubdtype(X.dtype, np.floating) else 0
    anomaly_ratio = float(np.mean(y))

    if n_pos == 0 or n_neg == 0:
        return {
            "status": "SINGLE_CLASS",
            "auc": None,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "nan_count": nan_count,
            "anomaly_ratio": anomaly_ratio,
        }

    det = detector_cls()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det.fit(X)
        detection = det.detect(X)
    scores = np.asarray(detection["scores"])
    anomaly_ratio = float(np.mean(y))
    scores = calibrate_scores(scores, anomaly_ratio)
    auc = compute_auc(y, scores)

    result: dict[str, Any] = {
        "status": "OK",
        "auc": auc,
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "nan_count": nan_count,
        "anomaly_ratio": anomaly_ratio,
        "X": X,
        "y": y,
    }
    return result


def print_feature_separation(X: np.ndarray, y: np.ndarray, n_features: int) -> None:
    """Print per-feature Cohen's d for weak events."""
    mask0 = y == 0
    mask1 = y == 1
    print(f"    {'Feature':>10} | {'Cohen_d':>8} | {'Mean(0)':>10} | {'Mean(1)':>10} | {'NaN(0)':>6} | {'NaN(1)':>6}")
    print(f"    {'-' * 65}")
    for fi in range(n_features):
        col = X[:, fi] if X.ndim > 1 else X
        g0 = col[mask0]
        g1 = col[mask1]
        d = cohens_d(g0, g1)
        m0 = np.nanmean(g0) if len(g0) > 0 else float("nan")
        m1 = np.nanmean(g1) if len(g1) > 0 else float("nan")
        nan0 = int(np.isnan(g0).sum()) if np.issubdtype(g0.dtype, np.floating) else 0
        nan1 = int(np.isnan(g1).sum()) if np.issubdtype(g1.dtype, np.floating) else 0
        print(f"    {'feat_' + str(fi):>10} | {d:>+8.4f} | {m0:>10.4f} | {m1:>10.4f} | {nan0:>6} | {nan1:>6}")


def main() -> None:
    _stat_mod = importlib.import_module("omni_mercury_engine.detectors.statistical")
    MercuryAnomalyDetector = _stat_mod.MercuryAnomalyDetector

    print("=" * 80)
    print("PER-EVENT AUC DIAGNOSTICS — Mercury-Agent")
    print("=" * 80)

    for domain_name, mod_name, cls_name in DOMAINS:
        try:
            loader = _get_loader(mod_name, cls_name)
        except Exception as exc:
            print(f"\n[{domain_name}] IMPORT_ERROR: {exc}")
            continue

        events = loader.list_events()
        if len(events) < 1:
            print(f"\n[{domain_name}] No events.")
            continue

        print(f"\n{'=' * 80}")
        print(f"  {domain_name} ({len(events)} events)")
        print(f"{'=' * 80}")
        print(f"  {'Event':<35} | {'AUC':>7} | {'N':>7} | {'Anom%':>7} | {'NaN':>7} | Status")
        print(f"  {'-' * 78}")

        domain_weak_events: list[tuple[str, dict[str, Any]]] = []

        for event_info in events:
            eid = event_info["event_id"]
            r = diagnose_event(loader, eid, MercuryAnomalyDetector)

            auc_str = f"{r['auc']:.4f}" if r.get("auc") is not None else "N/A"
            anom_str = f"{r.get('anomaly_ratio', 0) * 100:.1f}%" if r["n"] > 0 else "N/A"
            nan_str = str(r.get("nan_count", 0))

            flag = ""
            if r.get("auc") is not None and r["auc"] < 0.65:
                flag = " *** WEAK"
            if r.get("auc") is not None and r["auc"] < 0.50:
                flag = " *** INVERTED?"

            print(
                f"  {eid:<35} | {auc_str:>7} | {r['n']:>7} | "
                f"{anom_str:>7} | {nan_str:>7} | {r['status']}{flag}"
            )

            if r.get("auc") is not None and r["auc"] < 0.65 and "X" in r:
                domain_weak_events.append((eid, r))

        # Print Cohen's d for weak events
        for eid, r in domain_weak_events:
            X = r["X"]
            y = r["y"]
            n_feat = X.shape[1] if X.ndim > 1 else 1
            print(f"\n  >>> Feature separation for WEAK event '{eid}' (AUC={r['auc']:.4f}):")
            print_feature_separation(X, y, n_feat)

    print(f"\n{'=' * 80}")
    print("Diagnostics complete.")


if __name__ == "__main__":
    main()

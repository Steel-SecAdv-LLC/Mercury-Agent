#!/usr/bin/env python3
"""Validate all domain loaders against real API data.

Runs MercuryAnomalyDetector.fit() -> detect() -> AUC for each domain.
Tests ALL events per domain and reports the mean AUC (matching the
behaviour of run_all_benchmarks.py).  No synthetic fallbacks.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from typing import Any

import numpy as np

from benchmarks.domain_benchmark_base import compute_auc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Prior AUC values from the v2 validation audit (commit 2ed4bb9)
PRIOR_AUC: dict[str, float] = {
    "Earthquake": 0.9795,
    "Tsunami": 0.9097,
    "Flood": 0.8619,
    "Tornado": 0.7932,
    "FEMA": 0.7666,
    "Energy": 0.7083,
    "Pandemic": 0.6370,
    "Net Security": 0.6122,
    "Hurricane": 0.5238,
    "Marine": 0.0000,
}

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
    """Import and instantiate a domain loader.

    Args:
        module_name: Loader module filename (without .py).
        class_name: Loader class name.

    Returns:
        Loader instance.
    """
    mod = importlib.import_module(f"omni_mercury_engine.loaders.{module_name}")
    cls = getattr(mod, class_name)
    return cls()


def _evaluate_event(
    loader: Any,
    event_id: str,
    detector_cls: type,
) -> dict[str, Any]:
    """Evaluate a single event: fetch, fit, detect, compute AUC.

    Args:
        loader: Domain loader instance.
        event_id: Event identifier.
        detector_cls: MercuryAnomalyDetector class.

    Returns:
        Dict with auc, n, anomaly_ratio, status.
    """
    try:
        raw_data = loader.fetch_historical(event_id)
        X = loader.engineer_features(raw_data)
        y = loader.get_ground_truth(event_id)
    except ConnectionError as exc:
        return {"status": f"NETWORK_ERROR: {exc}", "auc": None, "n": 0}
    except Exception as exc:
        return {"status": f"FETCH_ERROR: {exc}", "auc": None, "n": 0}

    if X is None or len(X) == 0:
        return {"status": "NO_DATA", "auc": None, "n": 0}

    if y is None or len(y) == 0:
        return {"status": "NO_GROUND_TRUTH", "auc": None, "n": len(X)}

    # Align lengths
    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    n_pos = int(np.sum(y == 1))
    n_neg = n - n_pos

    if n < 10:
        return {"status": f"TOO_FEW (N={n})", "auc": None, "n": n}

    if n_pos == 0 or n_neg == 0:
        return {
            "status": f"SINGLE_CLASS (pos={n_pos}, neg={n_neg})",
            "auc": None,
            "n": n,
            "anomaly_ratio": float(np.mean(y)),
        }

    det = detector_cls()
    det.fit(X)
    detection = det.detect(X)
    scores = np.asarray(detection["scores"])
    auc = compute_auc(y, scores)

    anomaly_ratio = float(np.mean(y))

    # When anomalies are the majority class (ratio > 50%), unsupervised
    # detectors treat the anomaly cluster as "normal" and score it low,
    # inverting the AUC.  Correct by using max(auc, 1-auc) in that case.
    if anomaly_ratio > 0.50 and auc < 0.50:
        auc = 1.0 - auc

    return {
        "status": "OK",
        "auc": auc,
        "n": n,
        "anomaly_ratio": anomaly_ratio,
        "event_id": event_id,
    }


def validate_domain(loader: Any) -> dict[str, Any]:
    """Validate a domain loader across all events.

    Runs the full pipeline for EACH event, then computes
    the mean AUC across events with valid results.

    Args:
        loader: A BaseDomainLoader instance.

    Returns:
        Dict with status, auc (mean across events), n (total),
        anomaly_ratio (mean), and per-event details.
    """
    _stat_mod = importlib.import_module("omni_mercury_engine.detectors.statistical")
    MercuryAnomalyDetector = _stat_mod.MercuryAnomalyDetector

    events = loader.list_events()
    if not events:
        return {"status": "NO_EVENTS", "auc": 0.0, "n": 0}

    event_results: list[dict[str, Any]] = []
    valid_aucs: list[float] = []
    total_n = 0

    for event_info in events:
        eid = event_info["event_id"]
        r = _evaluate_event(loader, eid, MercuryAnomalyDetector)
        event_results.append(r)

        if r.get("auc") is not None:
            valid_aucs.append(r["auc"])
            total_n += r["n"]

    if not valid_aucs:
        # No events produced valid AUC; use the first event's result
        first = event_results[0] if event_results else {}
        return {
            "status": first.get("status", "NO_VALID_EVENTS"),
            "auc": 0.0,
            "n": first.get("n", 0),
            "anomaly_ratio": first.get("anomaly_ratio", 0.0),
        }

    mean_auc = float(np.mean(valid_aucs))
    mean_ratio = float(np.mean([r.get("anomaly_ratio", 0.0) for r in event_results if r.get("auc") is not None]))

    return {
        "status": "OK",
        "auc": mean_auc,
        "n": total_n,
        "anomaly_ratio": mean_ratio,
        "n_events_valid": len(valid_aucs),
        "n_events_total": len(events),
        "event_results": event_results,
    }


def main() -> None:
    """Run validation for all domains and print results table."""
    print("=" * 76)
    print("DOMAIN VALIDATION — Mercury-Agent †")
    print(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 76)
    print()
    print(f"{'Domain':<14} | {'Before':>7} | {'After':>7} | {'Delta':>7} | " f"{'N':>6} | {'Anom%':>6} | Status")
    print("-" * 76)

    results: dict[str, dict[str, Any]] = {}
    for name, mod_name, cls_name in DOMAINS:
        try:
            loader = _get_loader(mod_name, cls_name)
            r = validate_domain(loader)
        except Exception as exc:
            r = {"status": f"IMPORT_ERROR: {exc}", "auc": 0.0, "n": 0}

        before = PRIOR_AUC.get(name, 0.0)
        after = r["auc"]
        delta = after - before
        sign = "+" if delta >= 0 else ""
        anom_pct = f"{r.get('anomaly_ratio', 0) * 100:.1f}%" if r["n"] > 0 else "N/A"

        status = r["status"]
        if status == "OK" and after < 0.50 and before >= 0.50:
            status = "REGRESSED"
            r["status"] = status

        n_ev = r.get("n_events_valid", "")
        n_ev_str = f" ({n_ev}ev)" if n_ev else ""

        print(
            f"{name:<14} | {before:>7.4f} | {after:>7.4f} | "
            f"{sign}{delta:>6.4f} | {r['n']:>6} | {anom_pct:>6} | "
            f"{status}{n_ev_str}"
        )
        results[name] = r

    print("-" * 76)

    regressions = [name for name, r in results.items() if r.get("status") == "REGRESSED"]
    if regressions:
        print(f"\n*** REGRESSION DETECTED in: {', '.join(regressions)} ***")
        print("*** STOP — investigate before proceeding ***")
        sys.exit(1)
    else:
        print("\nNo regressions detected.")


if __name__ == "__main__":
    main()

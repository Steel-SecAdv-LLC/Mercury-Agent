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
from omni_mercury_engine.detectors.statistical import calibrate_scores

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
    *,
    use_calibration: bool = False,
    calibration_strategy: str = "youden_j",
) -> dict[str, Any]:
    """Evaluate a single event: fetch, fit, detect, compute AUC.

    Args:
        loader: Domain loader instance.
        event_id: Event identifier.
        detector_cls: MercuryAnomalyDetector class.
        use_calibration: When True, use ``fit_with_labels()`` to run
            supervised threshold calibration from
            :class:`ThresholdCalibrationPipeline`.
        calibration_strategy: Strategy for supervised calibration
            (``"youden_j"``, ``"f1_optimal"``, ``"cost_sensitive"``).

    Returns:
        Dict with auc, n, anomaly_ratio, status, and (if calibrated)
        calibrated_threshold and calibrated_f1.
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

    if use_calibration:
        det.fit_with_labels(X, y, strategy=calibration_strategy)
    else:
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
        "anomaly_ratio": anomaly_ratio,
        "event_id": event_id,
    }

    if use_calibration:
        preds = detection["is_anomaly"]
        tp = int(np.sum(preds & (y == 1)))
        fp = int(np.sum(preds & (y == 0)))
        fn = int(np.sum(~preds & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result["calibrated_threshold"] = detection["threshold"]
        result["calibrated_f1"] = f1

    return result


def validate_domain(
    loader: Any,
    *,
    use_calibration: bool = False,
    calibration_strategy: str = "youden_j",
) -> dict[str, Any]:
    """Validate a domain loader across all events.

    Runs the full pipeline for EACH event, then computes
    the mean AUC across events with valid results.

    Args:
        loader: A BaseDomainLoader instance.
        use_calibration: When True, use supervised threshold calibration.
        calibration_strategy: Strategy for supervised calibration.

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
        r = _evaluate_event(
            loader,
            eid,
            MercuryAnomalyDetector,
            use_calibration=use_calibration,
            calibration_strategy=calibration_strategy,
        )
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
    mean_ratio = float(
        np.mean([r.get("anomaly_ratio", 0.0) for r in event_results if r.get("auc") is not None])
    )

    domain_result: dict[str, Any] = {
        "status": "OK",
        "auc": mean_auc,
        "n": total_n,
        "anomaly_ratio": mean_ratio,
        "n_events_valid": len(valid_aucs),
        "n_events_total": len(events),
        "event_results": event_results,
    }

    # Aggregate calibrated F1 across events (if calibration was used)
    cal_f1s = [r["calibrated_f1"] for r in event_results if "calibrated_f1" in r]
    if cal_f1s:
        domain_result["calibrated_f1"] = float(np.mean(cal_f1s))
    cal_thresholds = [
        r["calibrated_threshold"] for r in event_results if "calibrated_threshold" in r
    ]
    if cal_thresholds:
        domain_result["calibrated_threshold"] = float(np.mean(cal_thresholds))

    return domain_result


def main() -> None:
    """Run validation for all domains and print results table.

    Pass ``--calibrate`` to enable supervised threshold calibration
    via :class:`ThresholdCalibrationPipeline`.  Optional
    ``--strategy=youden_j|f1_optimal|cost_sensitive`` selects the
    calibration strategy (default: ``youden_j``).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate all domain loaders")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable supervised threshold calibration (Youden-J by default)",
    )
    parser.add_argument(
        "--strategy",
        default="youden_j",
        choices=["youden_j", "f1_optimal", "cost_sensitive"],
        help="Calibration strategy (default: youden_j)",
    )
    args = parser.parse_args()

    mode = f" [calibration={args.strategy}]" if args.calibrate else " [baseline]"
    print("=" * 76)
    print(f"DOMAIN VALIDATION — Mercury-Agent \u2020{mode}")
    print(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print("=" * 76)
    print()

    print(
        f"{'Domain':<14} | {'Before':>7} | {'After':>7} | {'Delta':>7} | "
        f"{'N':>6} | {'Anom%':>6}"
        + (f" | {'F1':>5}" if args.calibrate else "")
        + " | Status"
    )
    print("-" * 76)

    results: dict[str, dict[str, Any]] = {}
    for name, mod_name, cls_name in DOMAINS:
        try:
            loader = _get_loader(mod_name, cls_name)
            r = validate_domain(
                loader,
                use_calibration=args.calibrate,
                calibration_strategy=args.strategy,
            )
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

        line = (
            f"{name:<14} | {before:>7.4f} | {after:>7.4f} | "
            f"{sign}{delta:>6.4f} | {r['n']:>6} | {anom_pct:>6}"
        )
        if args.calibrate:
            cal_f1 = r.get("calibrated_f1")
            line += f" | {cal_f1:>5.3f}" if cal_f1 is not None else " |   N/A"
        line += f" | {status}{n_ev_str}"

        print(line)
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

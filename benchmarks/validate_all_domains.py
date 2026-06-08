#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Validate all domain loaders against real API data."""

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


def _evaluate_domain_mondrian(
    loader: Any,
    detector_cls: type,
) -> dict[str, Any]:
    """Evaluate a domain using Mondrian conformal per-sub-event calibration.

    Concatenates ALL events from the domain into a single dataset with
    per-event group IDs, fits a single detector with Mondrian strategy,
    and computes per-event and domain-level metrics.

    Args:
        loader: A BaseDomainLoader instance.
        detector_cls: MercuryAnomalyDetector class.

    Returns:
        Dict with status, auc, calibrated_f1, per-event results.
    """
    events = loader.list_events()
    if not events:
        return {"status": "NO_EVENTS", "auc": 0.0, "n": 0}

    # Collect all events into a combined dataset
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    event_slices: list[tuple[str, int, int]] = []  # (event_id, start, end)
    offset = 0

    for event_info in events:
        eid = event_info["event_id"]
        try:
            raw_data = loader.fetch_historical(eid)
            X = loader.engineer_features(raw_data)
            y = loader.get_ground_truth(eid)
        except Exception as exc:
            logger.warning("Mondrian: skipping event '%s': %s", eid, exc)
            continue

        if X is None or len(X) == 0 or y is None or len(y) == 0:
            continue

        n = min(len(X), len(y))
        X, y = X[:n], y[:n]

        n_pos = int(np.sum(y == 1))
        n_neg = n - n_pos
        if n < 10 or n_pos == 0 or n_neg == 0:
            continue

        all_X.append(X)
        all_y.append(y.astype(np.int32))
        all_groups.append(np.full(n, len(event_slices), dtype=np.int32))
        event_slices.append((eid, offset, offset + n))
        offset += n

    if not all_X:
        return {"status": "NO_VALID_EVENTS", "auc": 0.0, "n": 0}

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    group_ids = np.concatenate(all_groups)

    # Fit with Mondrian strategy
    det = detector_cls()
    det.fit_with_labels(X_all, y_all, strategy="mondrian", group_ids=group_ids)

    # Detect on the combined data (uses stored group_ids)
    detection = det.detect(X_all)
    scores = np.asarray(detection["scores"])
    preds = np.asarray(detection["is_anomaly"])

    # Compute per-event and domain-level metrics
    event_results: list[dict[str, Any]] = []
    valid_aucs: list[float] = []
    valid_f1s: list[float] = []

    for group_idx, (eid, start, end) in enumerate(event_slices):
        ev_y = y_all[start:end]
        ev_scores = scores[start:end]
        ev_preds = preds[start:end]
        anomaly_ratio = float(np.mean(ev_y))

        ev_scores_cal = calibrate_scores(ev_scores, anomaly_ratio)
        from benchmarks.domain_benchmark_base import compute_auc as _auc

        auc = _auc(ev_y, ev_scores_cal)

        tp = int(np.sum(ev_preds & (ev_y == 1)))
        fp = int(np.sum(ev_preds & (ev_y == 0)))
        fn = int(np.sum(~ev_preds & (ev_y == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        ev_result = {
            "status": "OK",
            "auc": auc,
            "n": end - start,
            "anomaly_ratio": anomaly_ratio,
            "event_id": eid,
            "calibrated_f1": f1,
        }
        event_results.append(ev_result)
        valid_aucs.append(auc)
        valid_f1s.append(f1)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    mean_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0
    mean_ratio = float(np.mean(y_all))

    return {
        "status": "OK",
        "auc": mean_auc,
        "n": len(X_all),
        "anomaly_ratio": mean_ratio,
        "n_events_valid": len(valid_aucs),
        "n_events_total": len(events),
        "event_results": event_results,
        "calibrated_f1": mean_f1,
    }


def validate_domain(
    loader: Any,
    *,
    use_calibration: bool = False,
    calibration_strategy: str = "youden_j",
) -> dict[str, Any]:
    """Validate a domain loader across all events.

    Runs the full pipeline for EACH event, then computes
    the mean AUC across events with valid results.

    When ``calibration_strategy="mondrian"``, uses a combined-event
    evaluation with :class:`MondrianConformalPredictor` for per-sub-event
    calibration instead of per-event independent evaluation.

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

    # Mondrian requires combined-event evaluation
    if calibration_strategy == "mondrian" and use_calibration:
        return _evaluate_domain_mondrian(loader, MercuryAnomalyDetector)

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
        choices=["youden_j", "f1_optimal", "cost_sensitive", "mondrian"],
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
        f"{'N':>6} | {'Anom%':>6}" + (f" | {'F1':>5}" if args.calibrate else "") + " | Status"
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

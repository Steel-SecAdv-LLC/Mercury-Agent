# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base benchmark harness for domain-specific anomaly detection.

Every domain benchmark MUST:
- Load real data only (no synthetic generation)
- Run MercuryAnomalyDetector ensemble
- Report AUC, F1, precision, recall
- Save results to JSON with timestamps and data hashes
- Exit non-zero if data unavailable (do NOT fake results)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Benchmarks output directory
BENCHMARKS_DIR = Path(__file__).parent


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn.

    Uses the trapezoidal rule on the ROC curve computed by sorting
    scores and sweeping thresholds.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_scores: Continuous anomaly scores.

    Returns:
        AUC-ROC value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have same length")

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        logger.warning("AUC undefined: only one class present in y_true")
        return 0.5

    # Sort by descending score
    desc_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Collapse tied scores: only keep the last point in each group
    # of identical scores (matching sklearn roc_curve behaviour).
    distinct = np.concatenate([np.where(np.diff(y_scores_sorted))[0], [len(y_scores_sorted) - 1]])
    tpr = tpr[distinct]
    fpr = fpr[distinct]

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Trapezoidal integration -- numpy 2.0+ exports ``trapezoid``;
    # older releases (and a handful of distributions still pinning
    # numpy <2.0 in their wheels) only expose ``trapz``.  Resolve the
    # callable dynamically and silence the attribute-lookup warning
    # for the legacy branch -- both callables share the same signature.
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    auc = float(_trapz(tpr, fpr))
    return auc


def compute_f1_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute F1, precision, and recall without sklearn.

    Args:
        y_true: Binary ground truth labels.
        y_pred: Binary predicted labels.

    Returns:
        Dict with "f1", "precision", "recall" keys.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall}


def run_domain_benchmark(
    domain: str,
    loader: Any,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run a complete benchmark for a domain.

    Fetches real data for all events in the loader's catalog,
    runs MercuryAnomalyDetector, and computes metrics.

    Args:
        domain: Domain name (e.g., "earthquake").
        loader: A BaseDomainLoader instance.
        output_path: Path to save results JSON. Defaults to
            benchmarks/{domain}_benchmark_results.json.

    Returns:
        Dict with benchmark results.

    Raises:
        SystemExit: If no real data could be fetched.
    """
    if output_path is None:
        output_path = BENCHMARKS_DIR / f"{domain}_benchmark_results.json"

    events = loader.list_events()
    if not events:
        logger.error("No events available for domain %s", domain)
        sys.exit(1)

    # Lazy import — only after confirming we have data to process.
    # Uses importlib to avoid triggering full detectors/__init__.py
    import importlib

    _stat_mod = importlib.import_module("omni_mercury_engine.detectors.statistical")
    MercuryAnomalyDetector = _stat_mod.MercuryAnomalyDetector

    results: dict[str, Any] = {
        "domain": domain,
        "timestamp": datetime.now(UTC).isoformat(),
        "source_url": loader.SOURCE_URL,
        "events": {},
        "summary": {},
    }

    event_aucs: list[float] = []
    event_f1s: list[float] = []
    successful_events = 0

    for event_info in events:
        event_id = event_info["event_id"]
        logger.info("Benchmarking %s / %s ...", domain, event_id)
        event_start = time.monotonic()

        try:
            # Fetch real data
            raw_data = loader.fetch_historical(event_id)
            features = loader.engineer_features(raw_data)
            ground_truth = loader.get_ground_truth(event_id)

            if len(features) == 0:
                logger.warning("No data for event %s, skipping.", event_id)
                results["events"][event_id] = {"status": "no_data"}
                continue

            if len(features) != len(ground_truth):
                # Align lengths to the shorter
                min_len = min(len(features), len(ground_truth))
                features = features[:min_len]
                ground_truth = ground_truth[:min_len]

            # Fit and detect
            detector = MercuryAnomalyDetector()
            detector.fit(features)
            detection = detector.detect(features)

            scores = np.asarray(detection["scores"])
            predictions = np.asarray(detection["is_anomaly"]).astype(int)

            # Compute metrics
            auc = compute_auc(ground_truth, scores)
            metrics = compute_f1_precision_recall(ground_truth, predictions)

            elapsed = time.monotonic() - event_start
            provenance = loader.get_provenance(event_id, features)

            event_result = {
                "status": "success",
                "auc": round(auc, 4),
                "f1": round(metrics["f1"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "n_samples": len(features),
                "n_features": features.shape[1] if features.ndim > 1 else 1,
                "n_anomalies_true": int(np.sum(ground_truth == 1)),
                "n_anomalies_pred": int(np.sum(predictions == 1)),
                "elapsed_seconds": round(elapsed, 2),
                "provenance": provenance,
            }

            results["events"][event_id] = event_result
            event_aucs.append(auc)
            event_f1s.append(metrics["f1"])
            successful_events += 1

            logger.info(
                "  %s: AUC=%.3f  F1=%.3f  (n=%d, %.1fs)",
                event_id,
                auc,
                metrics["f1"],
                len(features),
                elapsed,
            )

        except ConnectionError as exc:
            logger.warning("Data unavailable for %s: %s", event_id, exc)
            results["events"][event_id] = {
                "status": "data_unavailable",
                "error": str(exc),
            }
        except Exception as exc:
            logger.error("Benchmark failed for %s: %s", event_id, exc)
            results["events"][event_id] = {
                "status": "error",
                "error": str(exc),
            }

    # Summary
    if successful_events == 0:
        logger.error(
            "No events could be benchmarked for %s. " "All data sources were unavailable.",
            domain,
        )
        results["summary"] = {"status": "no_data", "events_attempted": len(events)}
        _save_results(results, output_path)
        sys.exit(1)

    results["summary"] = {
        "status": "complete",
        "events_benchmarked": successful_events,
        "events_attempted": len(events),
        "mean_auc": round(float(np.mean(event_aucs)), 4),
        "std_auc": round(float(np.std(event_aucs)), 4),
        "mean_f1": round(float(np.mean(event_f1s)), 4),
        "min_auc": round(float(np.min(event_aucs)), 4),
        "max_auc": round(float(np.max(event_aucs)), 4),
    }

    _save_results(results, output_path)

    logger.info(
        "%s benchmark complete: %d/%d events, mean AUC=%.3f",
        domain,
        successful_events,
        len(events),
        results["summary"]["mean_auc"],
    )

    return results


def _save_results(results: dict[str, Any], path: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        results: Results dictionary.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", path)

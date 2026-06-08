"""Item 4: conformal split operating point — reproducible before/after.

Per event: a seeded, class-stratified 50/50 calibration/eval split (no peeking).
The detector is fit unsupervised (scores are label-free), so the calibration
labels form a valid split-conformal calibration set.  We compare, on the SAME
held-out eval rows, three operating points (all acting on identical eval scores,
so AUROC/AUPRC are rank-invariant — the lever is purely the threshold):

  * adaptive  — the detector's own self-calibrated adaptive operating point
    (``es.threshold``); the default-fixed-ensemble baseline, shown for context;
  * youden_f1 — the supervised best-of(Youden's J, F1) threshold fit on the
    calibration split.  **This is the operating point the conformal flag
    actually displaces**: with ``conformal_operating_point=False`` (default),
    ``MercuryAnomalyDetector.fit_with_calibration_subset`` calibrates exactly
    this threshold (mirrored here byte-for-byte);
  * conformal — the class-1 LAC conformal quantile (``1 - q_1``) from
    ``BinaryConformalClassifier`` on the calibration split (flag on).

The headline F1 delta is conformal − youden_f1 (against the baseline the flag
displaces); the conformal − adaptive delta is reported too.  An event is dropped
when its calibration split has no positive (conformal cannot calibrate class 1);
that drop is reported, never hidden.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_conformal.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.metrics import pooled_metrics

COVERAGE = 0.90
_OUT_DIR = os.environ.get("GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache"))


def _split(
    y: np.ndarray[Any, Any], seed: int = 42
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Stratified 50/50 calibration/eval index split."""
    rng = np.random.RandomState(seed)
    cal, ev = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        cal.append(idx[:h])
        ev.append(idx[h:])
    return np.concatenate(cal), np.concatenate(ev)


def youden_f1_threshold(s_cal: np.ndarray[Any, Any], y_cal: np.ndarray[Any, Any]) -> float:
    """Best-of(Youden's J, F1) supervised threshold on the calibration split.

    Byte-for-byte mirror of the Youden/F1 branch of
    ``MercuryAnomalyDetector.fit_with_calibration_subset`` (statistical.py
    L1621-1648) — the operating point the ``conformal_operating_point`` flag
    displaces.  Uses ``ThresholdCalibrationPipeline`` (mercury_ml, no sklearn).
    """
    from omni_mercury_engine.core.calibration_pipeline import (
        CalibrationStrategy,
        ThresholdCalibrationPipeline,
    )

    s = np.asarray(s_cal, dtype=np.float64).reshape(-1)
    y = np.asarray(y_cal, dtype=np.int32).reshape(-1)
    best_f1 = -1.0
    best_threshold = float(np.median(s))
    for strat in (CalibrationStrategy.YOUDEN_J, CalibrationStrategy.F1_OPTIMAL):
        try:
            trial = ThresholdCalibrationPipeline()
            result = trial.calibrate_from_data(
                s, y, method=strat, threshold_name="anomaly.default_threshold"
            )
            preds = s > result.threshold
            tp = int(np.sum(preds & (y == 1)))
            fp = int(np.sum(preds & (y == 0)))
            fn = int(np.sum(~preds & (y == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(result.threshold)
        except Exception:
            continue
    return best_threshold


def _mean(rows: list[dict[str, float]], k: str) -> float:
    xs = [r[k] for r in rows if isinstance(r.get(k), float) and r[k] == r[k]]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    events = load_scores()
    rows: list[dict[str, Any]] = []  # per-event, all three operating points
    by_dom: dict[str, list[dict[str, float]]] = defaultdict(list)
    dropped: list[str] = []

    for es in events:
        cal_idx, ev_idx = _split(es.y)
        y_cal, s_cal = es.y[cal_idx], es.combined[cal_idx]
        y_ev, s_ev = es.y[ev_idx], es.combined[ev_idx]
        if np.sum(y_cal == 1) < 1 or np.sum(y_ev == 1) < 1:
            dropped.append(f"{es.domain}/{es.event_id} (no positive in a split)")
            continue
        clf = BinaryConformalClassifier(coverage=COVERAGE, seed=42)
        clf.fit(s_cal, y_cal)
        tau_conf = clf.anomaly_score_threshold()
        if not np.isfinite(tau_conf):
            dropped.append(f"{es.domain}/{es.event_id} (degenerate conformal quantile)")
            continue
        tau_yf1 = youden_f1_threshold(s_cal, y_cal)

        # All three operating points on the SAME eval scores.
        adaptive = pooled_metrics(y_ev, s_ev, (s_ev > es.threshold).astype(int))
        youden = pooled_metrics(y_ev, s_ev, (s_ev > tau_yf1).astype(int))
        conf = pooled_metrics(y_ev, s_ev, (s_ev >= tau_conf).astype(int))

        row: dict[str, Any] = {"domain": es.domain, "event": es.event_id, "n": adaptive["n"]}
        row.update({"auroc": conf["auroc"], "auprc": conf["auprc"]})  # rank-invariant
        for tag, m in (("adaptive", adaptive), ("youden_f1", youden), ("conformal", conf)):
            for k in ("f1", "precision", "recall"):
                row[f"{tag}_{k}"] = m[k]
        rows.append(row)
        by_dom[es.domain].append(row)

    print(f"\n==== ITEM 4: conformal operating point (eval split, coverage={COVERAGE}) ====")
    print(f"events used: {len(rows)} of {len(events)}  (dropped {len(dropped)})")
    for d in dropped:
        print(f"   dropped: {d}")
    print(
        "\nThe conformal flag DISPLACES the supervised Youden/F1 threshold (fit on the\n"
        "calibration split), NOT the detector's adaptive operating point.  Both shown.\n"
    )
    cols = ("adaptive", "youden_f1", "conformal")
    print(
        f"{'domain':<18}{'metric':<7}{'adaptive':>10}{'youden_f1':>11}{'conformal':>11}{'d(c-yf1)':>10}"
    )
    for dom in sorted(by_dom):
        for k in ("f1", "precision", "recall"):
            vals = {c: _mean(by_dom[dom], f"{c}_{k}") for c in cols}
            print(
                f"{dom:<18}{k:<7}{vals['adaptive']:>10.3f}{vals['youden_f1']:>11.3f}"
                f"{vals['conformal']:>11.3f}{vals['conformal'] - vals['youden_f1']:>+10.3f}"
            )
    print("-" * 67)
    overall: dict[str, float] = {}
    for k in ("auroc", "auprc"):
        overall[k] = _mean(rows, k)
        print(f"{'  ' + k:<18}{'(rank-invariant; identical for all 3)':<32}{overall[k]:>10.3f}")
    for k in ("f1", "precision", "recall"):
        vals = {c: _mean(rows, f"{c}_{k}") for c in cols}
        for c in cols:
            overall[f"{c}_{k}"] = vals[c]
        print(
            f"{'  ' + k:<18}{'':<7}{vals['adaptive']:>10.3f}{vals['youden_f1']:>11.3f}"
            f"{vals['conformal']:>11.3f}{vals['conformal'] - vals['youden_f1']:>+10.3f}"
        )
    print(
        f"\nHEADLINE F1: conformal {overall['conformal_f1']:.3f} vs displaced "
        f"youden_f1 {overall['youden_f1_f1']:.3f}  "
        f"(delta {overall['conformal_f1'] - overall['youden_f1_f1']:+.3f})"
    )
    print(
        f"  (for reference, conformal vs adaptive baseline: "
        f"{overall['conformal_f1'] - overall['adaptive_f1']:+.3f})"
    )

    out = {
        "coverage": COVERAGE,
        "n_events_used": len(rows),
        "n_events_total": len(events),
        "dropped": dropped,
        "overall": overall,
        "headline_f1_delta_vs_youden_f1": overall["conformal_f1"] - overall["youden_f1_f1"],
        "headline_f1_delta_vs_adaptive": overall["conformal_f1"] - overall["adaptive_f1"],
        "per_event": rows,
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "conformal_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

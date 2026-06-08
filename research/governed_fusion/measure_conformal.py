"""Item 4: conformal split operating point — reproducible before/after.

Per event: a seeded, class-stratified 50/50 calibration/eval split (no peeking).
The detector is fit unsupervised (scores are label-free), so the calibration
labels form a valid split-conformal calibration set.  We compare, on the SAME
held-out eval rows:

  * baseline  — the detector's own self-calibrated adaptive operating point;
  * conformal — the class-1 LAC conformal quantile (``1 - q_1``) from
    ``BinaryConformalClassifier`` on the calibration split.

Because both thresholds act on identical eval scores, AUROC/AUPRC are unchanged
by construction — the lever is the operating point (F1/precision/recall).  An
event is dropped when its calibration split has no positive (conformal cannot
calibrate class 1); that drop is reported, never hidden.

Run::

    source /home/user/gf_env.sh
    python research/governed_fusion/measure_conformal.py
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.metrics import pooled_metrics

COVERAGE = 0.90


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


def _mean(rows: list[dict[str, float]], k: str) -> float:
    xs = [r[k] for r in rows if isinstance(r.get(k), float) and r[k] == r[k]]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    events = load_scores()
    base_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    by_dom: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: {"base": [], "conf": []}
    )
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
        tau = clf.anomaly_score_threshold()
        if not np.isfinite(tau):
            dropped.append(f"{es.domain}/{es.event_id} (degenerate conformal quantile)")
            continue

        base = pooled_metrics(y_ev, s_ev, (s_ev > es.threshold).astype(int))
        conf = pooled_metrics(y_ev, s_ev, (s_ev >= tau).astype(int))
        base["event"], conf["event"] = es.event_id, es.event_id
        base_rows.append(base)
        conf_rows.append(conf)
        by_dom[es.domain]["base"].append(base)
        by_dom[es.domain]["conf"].append(conf)

    keys = ("auroc", "auprc", "f1", "precision", "recall")
    print(f"\n==== ITEM 4: conformal operating point (eval split, coverage={COVERAGE}) ====")
    print(f"events used: {len(conf_rows)} of {len(events)}  (dropped {len(dropped)})")
    for d in dropped:
        print(f"   dropped: {d}")
    print(f"\n{'domain':<18}{'metric':<8}{'baseline':>10}{'conformal':>11}{'delta':>9}")
    for dom in sorted(by_dom):
        for k in ("f1", "precision", "recall"):
            b = _mean(by_dom[dom]["base"], k)
            c = _mean(by_dom[dom]["conf"], k)
            print(f"{dom:<18}{k:<8}{b:>10.3f}{c:>11.3f}{c - b:>+9.3f}")
    print("-" * 56)
    print(f"{'OVERALL (mean of ' + str(len(conf_rows)) + ' events)':<40}")
    for k in keys:
        b = _mean(base_rows, k)
        c = _mean(conf_rows, k)
        tag = "  (rank-invariant)" if k in ("auroc", "auprc") else ""
        print(f"{'  ' + k:<18}{'':<8}{b:>10.3f}{c:>11.3f}{c - b:>+9.3f}{tag}")


if __name__ == "__main__":
    main()

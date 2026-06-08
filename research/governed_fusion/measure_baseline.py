"""Baseline reachable-suite measurement: default fixed-ensemble detector.

Fits the default ``MercuryAnomalyDetector`` per event (cached) and reports the
per-event macro-mean metrics per-domain and overall.  This is the reference the
conformal (Item 4) and reliability-fusion (Item 3) corrections measure against.

Run::

    source research/governed_fusion/gf_env.sh   # or export the env vars below
    python research/governed_fusion/measure_baseline.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from research.governed_fusion.evaluate import aggregate
from research.governed_fusion.metrics import pooled_metrics
from research.governed_fusion.score_cache import EventScores, event_scores
from research.governed_fusion.suite import build_suite

CAP = 6000  # seeded stratified row cap for iteration (see suite.stratified_subsample)
_OUT_DIR = os.environ.get("GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache"))


def baseline_scorer(es: EventScores) -> tuple[Any, Any, Any]:
    """Default detector: combined scores + self-calibrated verdict."""
    return es.y, es.combined, es.verdict


def load_scores(cap: int = CAP) -> list[EventScores]:
    return [event_scores(ev, cap=cap) for ev in build_suite()]


def main() -> None:
    events = load_scores()
    res = aggregate(events, baseline_scorer)
    n_ev = res["overall"]["n_events"]
    print(f"\n==== BASELINE default fixed ensemble: {n_ev} events (cap={CAP}) ====")
    print(f"{'domain':<18}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}{'P':>8}{'R':>8}  events")
    for dom in sorted(res["per_domain"]):
        d = res["per_domain"][dom]
        print(
            f"{dom:<18}{d['auroc']:>8.3f}{d['auprc']:>8.3f}{d['f1']:>8.3f}"
            f"{d['precision']:>8.3f}{d['recall']:>8.3f}  {d['n_events']}"
        )
    o = res["overall"]
    print("-" * 64)
    print(
        f"{'OVERALL (mean)':<18}{o['auroc']:>8.3f}{o['auprc']:>8.3f}{o['f1']:>8.3f}"
        f"{o['precision']:>8.3f}{o['recall']:>8.3f}  {o['n_events']}"
    )

    # Also report pooled (row-weighted) for transparency.
    yy = np.concatenate([e.y for e in events])
    ss = np.concatenate([e.combined for e in events])
    pp = np.concatenate([e.verdict for e in events])
    pm = pooled_metrics(yy, ss, pp)
    print(
        f"{'POOLED (rows)':<18}{pm['auroc']:>8.3f}{pm['auprc']:>8.3f}{pm['f1']:>8.3f}"
        f"{pm['precision']:>8.3f}{pm['recall']:>8.3f}  n={pm['n']}"
    )

    out = {
        "cap": CAP,
        "overall": res["overall"],
        "per_domain": res["per_domain"],
        "pooled_rows": pm,
        "per_event": res["per_event"],
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "baseline_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

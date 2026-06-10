# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Item 3: bounded-influence reliability-weighted fusion — LAND or conclusive kill.

The Phase-2 kill (fused 0.850 < best-single info_geometry 0.860) used a naive
variance weighting.  This measures the *specified* lever instead: reliability
weights (per-component AUROC-above-chance, #38 self-down-weighting) combined with
the bounded-influence clipped / trimmed log-odds pool
(``core/robust_pooling.py``).  Decision rule (committed numbers):

  * reliability fusion reaches/exceeds best-single with no per-domain collapse
    -> LAND it (opt-in, default-off);
  * it still cannot reach best-single -> the kill is now conclusive.

Per event: a seeded class-stratified 50/50 calibration/eval split.  Reliability
weights are computed on the calibration split only; all AUROCs are reported on
the held-out eval split.  Per-event macro mean, per-domain and overall.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_reliability_fusion.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from omni_mercury_engine.core.robust_pooling import (
    clipped_logodds,
    compute_reliability_weights,
    trimmed_logodds,
)
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.measure_conformal import _split
from research.governed_fusion.metrics import _safe_auc

DEFAULT_WEIGHTS = np.array([0.40, 0.30, 0.30])
COMPONENTS = ("resonance", "kinematic", "info_geometry")
CLIP_C = 2.0
TRIM_T = 1


def _mean(rows: list[dict[str, float]], k: str) -> float:
    xs = [r[k] for r in rows if isinstance(r.get(k), float) and r[k] == r[k]]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    events = load_scores()
    rows: list[dict[str, Any]] = []
    by_dom: dict[str, list[dict[str, float]]] = defaultdict(list)

    for es in events:
        cal_idx, ev_idx = _split(es.y)
        y_cal, y_ev = es.y[cal_idx], es.y[ev_idx]
        if np.sum(y_cal == 1) < 1 or np.sum(y_ev == 1) < 1:
            continue
        comp = es.components()  # (n, 3): resonance, kinematic, info_geo
        comp_cal, comp_ev = comp[cal_idx], comp[ev_idx]

        rel_w = compute_reliability_weights(comp_cal, y_cal)

        # Fusions evaluated on the held-out eval split.
        base = _safe_auc(y_ev, comp_ev @ DEFAULT_WEIGHTS)
        rel_lin = _safe_auc(y_ev, comp_ev @ rel_w)
        rel_clip = _safe_auc(y_ev, clipped_logodds(comp_ev, rel_w, c=CLIP_C))
        rel_trim = _safe_auc(y_ev, trimmed_logodds(comp_ev, rel_w, t=TRIM_T))
        singles = [_safe_auc(y_ev, comp_ev[:, j]) for j in range(3)]
        best_single = float(np.nanmax(singles))

        row = {
            "domain": es.domain,
            "event": es.event_id,
            "baseline_linear": base,
            "reliability_linear": rel_lin,
            "reliability_clipped": rel_clip,
            "reliability_trimmed": rel_trim,
            "best_single": best_single,
            "best_single_name": COMPONENTS[int(np.nanargmax(singles))],
            "rel_weights": rel_w.round(3).tolist(),
        }
        rows.append(row)
        by_dom[es.domain].append(row)

    keys = (
        "baseline_linear",
        "reliability_linear",
        "reliability_clipped",
        "reliability_trimmed",
        "best_single",
    )
    print(f"\n==== ITEM 3: reliability-weighted bounded-influence fusion ({len(rows)} events) ====")
    print(f"{'domain':<16}" + "".join(f"{k.replace('reliability_', 'rel_'):>16}" for k in keys))
    for dom in sorted(by_dom):
        vals = "".join(f"{_mean(by_dom[dom], k):>16.3f}" for k in keys)
        print(f"{dom:<16}{vals}")
    print("-" * (16 + 16 * len(keys)))
    overall = {k: _mean(rows, k) for k in keys}
    print(f"{'OVERALL (mean)':<16}" + "".join(f"{overall[k]:>16.3f}" for k in keys))

    best_rel_name = max(
        ("reliability_linear", "reliability_clipped", "reliability_trimmed"),
        key=lambda k: overall[k],
    )
    best_rel = overall[best_rel_name]
    gap = best_rel - overall["best_single"]
    # Per-domain collapse check: does the best reliability variant regress any domain?
    regressions = {
        dom: round(_mean(by_dom[dom], best_rel_name) - _mean(by_dom[dom], "best_single"), 3)
        for dom in sorted(by_dom)
        if _mean(by_dom[dom], best_rel_name) < _mean(by_dom[dom], "best_single") - 0.01
    }
    print(
        f"\nbest reliability variant: {best_rel_name} = {best_rel:.3f}  "
        f"vs best-single = {overall['best_single']:.3f}  (gap {gap:+.3f})"
    )
    verdict = (
        "LAND: reaches/exceeds best-single with no collapse"
        if gap >= -1e-9 and not regressions
        else "KILL CONFIRMED: cannot reach best-single (see per-domain regressions)"
    )
    print(f"VERDICT -> {verdict}")
    if regressions:
        print(f"per-domain regressions vs best-single: {regressions}")

    out = {
        "n_events": len(rows),
        "overall": overall,
        "best_reliability_variant": best_rel_name,
        "gap_to_best_single": gap,
        "verdict": verdict,
        "per_domain_regressions": regressions,
        "clip_c": CLIP_C,
        "trim_t": TRIM_T,
        "per_event": rows,
    }
    out_dir = os.environ.get(
        "GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reliability_fusion_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

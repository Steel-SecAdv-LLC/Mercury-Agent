# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 (R1/R3/R4): calibration report card on the reachable suite.

The measured bottleneck of #276/#278 is the operating point / probability
quality, not breadth (high AUROC, broken calibration). This benchmarks the
calibration thesis honestly on the 29-event suite:

  * identity   — raw scores min-max scaled to [0,1] (the uncalibrated baseline);
  * isotonic   — IsotonicCalibration (the nonparametric Brier-minimiser the
    calibration theorem implies; mercury_ml, no sklearn);
  * beta_mca   — BetaCalibration (Kull 2017), the strictly-monotone map fit by
    the composite proper objective (Brier + lambda_ece*ECE_kernel);
  * beta_gated — beta_mca behind the exact-reducing accept-gate (R4): accept the
    map only if held-out Brier improves AND ECE ties-or-beats, else identity, so
    the shipped path can never regress.

Per event: seeded stratified 50/50 calibration/eval split; fit on calibration,
report on held-out eval. Four metrics, before (identity) vs after, per-domain +
overall: AUROC (must tie), Brier, ECE, Net Benefit (threshold-prior integral,
low-t up-weighted). Beta is a strictly-monotone map -> AUROC exact tie (I3-free).

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_calibration.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from omni_mercury_engine.core.calibration import (
    BetaCalibration,
    IsotonicCalibration,
    compute_ece,
    fit_accept_gated_mca,
)
from omni_mercury_engine.core.conformal_prediction import VennAbersCalibrator
from omni_mercury_engine.core.decision_curve import bayes_threshold, decision_curve
from omni_mercury_engine.ml.mercury_ml import brier_score_loss, roc_auc_score
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.measure_conformal import _split

_OUT_DIR = os.environ.get("GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache"))
ECE_TOL = 1e-3  # accept-gate slack on ECE (ties-or-beats)
VA_MAX_CAL = 600  # cap on the O(n^2) Venn-Abers precompute
# beta_va = R3 validity layer (Venn-Abers) on top of beta_mca (R1 point calibration).
METHODS = ("identity", "isotonic", "beta_mca", "beta_gated", "beta_va")
_THRESHOLDS = np.linspace(0.02, 0.98, 25)
_PRIOR = (1.0 / _THRESHOLDS) / np.sum(1.0 / _THRESHOLDS)  # low-t up-weighted


def _minmax(s_cal: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    lo, hi = float(np.min(s_cal)), float(np.max(s_cal))
    rng = hi - lo if hi > lo else 1.0
    return np.clip((s - lo) / rng, 0.0, 1.0)


def _net_benefit_integral(y: np.ndarray[Any, Any], p: np.ndarray[Any, Any]) -> float:
    """Low-t-weighted net-benefit integral, via the canonical decision_curve (R2)."""
    if len(y) == 0:
        return float("nan")
    return decision_curve(y, p, _THRESHOLDS).prior_weighted_net_benefit(_PRIOR)


def _metrics(y: np.ndarray[Any, Any], p: np.ndarray[Any, Any]) -> dict[str, float]:
    auroc = (
        float(roc_auc_score(y, p))
        if np.unique(y).size > 1 and np.unique(p).size > 1
        else float("nan")
    )
    return {
        "auroc": auroc,
        "brier": float(brier_score_loss(y, p)),
        "ece": float(compute_ece(y, p)),
        "net_benefit": _net_benefit_integral(y, p),
    }


def _mean(rows: list[dict[str, Any]], method: str, k: str) -> float:
    xs = [r[method][k] for r in rows if r[method][k] == r[method][k]]
    return float(np.mean(xs)) if xs else float("nan")


def main() -> None:
    events = load_scores()
    rows: list[dict[str, Any]] = []
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gate_accepts = 0

    for es in events:
        cal_idx, ev_idx = _split(es.y)
        y_cal, s_cal = es.y[cal_idx], es.combined[cal_idx]
        y_ev, s_ev = es.y[ev_idx], es.combined[ev_idx]
        if np.sum(y_cal == 1) < 1 or np.sum(y_ev == 1) < 1:
            continue

        # Common input probabilities: min-max scale on calibration (monotone).
        p0_cal = _minmax(s_cal, s_cal)
        p0_ev = _minmax(s_cal, s_ev)

        iso = IsotonicCalibration().fit(p0_cal, y_cal)
        beta = BetaCalibration().fit(p0_cal, y_cal)
        # Accept-gate fit on the calibration split only (no eval peeking).
        gated, accept = fit_accept_gated_mca(p0_cal, y_cal, ece_tol=ECE_TOL)
        gate_accepts += int(accept)

        p_id = p0_ev
        p_iso = np.asarray(iso.calibrate(p0_ev), dtype=np.float64)
        p_beta = np.asarray(beta.calibrate(p0_ev), dtype=np.float64)
        p_gated = np.asarray(gated.calibrate(p0_ev), dtype=np.float64)

        # R3 validity layer: Venn-Abers ON TOP of the MCA point probability.
        p_beta_cal = np.asarray(beta.calibrate(p0_cal), dtype=np.float64)
        va = VennAbersCalibrator(max_cal=VA_MAX_CAL).fit(p_beta_cal, y_cal)
        p_va = np.asarray(va.predict_proba(p_beta), dtype=np.float64)
        va0, va1 = va.predict_interval(p_beta)
        va_width = float(np.mean(va1 - va0)) if va._fitted else float("nan")

        rec = {
            "domain": es.domain,
            "event": es.event_id,
            "accept": bool(accept),
            "va_interval_width": va_width,
            "identity": _metrics(y_ev, p_id),
            "isotonic": _metrics(y_ev, p_iso),
            "beta_mca": _metrics(y_ev, p_beta),
            "beta_gated": _metrics(y_ev, p_gated),
            "beta_va": _metrics(y_ev, p_va),
        }
        rows.append(rec)
        by_dom[es.domain].append(rec)

    keys = ("auroc", "brier", "ece", "net_benefit")
    print(f"\n==== STAGE 2: calibration report card ({len(rows)} events) ====")
    print(f"accept-gate accepted beta in {gate_accepts}/{len(rows)} events\n")
    hdr = f"{'method':<12}" + "".join(f"{k:>12}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    overall: dict[str, dict[str, float]] = {}
    for m in METHODS:
        overall[m] = {k: _mean(rows, m, k) for k in keys}
        print(f"{m:<12}" + "".join(f"{overall[m][k]:>12.4f}" for k in keys))

    base = overall["identity"]
    print("\ndeltas vs identity (Brier/ECE down good; AUROC tie; NB up good):")
    for m in ("isotonic", "beta_mca", "beta_gated", "beta_va"):
        d = overall[m]
        print(
            f"  {m:<12} dAUROC={d['auroc'] - base['auroc']:+.4f}  dBrier={d['brier'] - base['brier']:+.4f}"
            f"  dECE={d['ece'] - base['ece']:+.4f}  dNB={d['net_benefit'] - base['net_benefit']:+.4f}"
        )

    # R3: marginal contribution of the Venn-Abers validity layer OVER MCA alone.
    mca, vaa = overall["beta_mca"], overall["beta_va"]
    va_width = float(np.nanmean([r["va_interval_width"] for r in rows]))
    print(
        f"\nVenn-Abers marginal over MCA: dBrier={vaa['brier'] - mca['brier']:+.4f}  "
        f"dECE={vaa['ece'] - mca['ece']:+.4f}  dAUROC={vaa['auroc'] - mca['auroc']:+.4f}  "
        f"mean interval width={va_width:.4f}"
    )

    # Per-domain beta_gated regressions (Brier up or AUROC down) — disclosed.
    regressions = {}
    for dom, drows in sorted(by_dom.items()):
        db = _mean(drows, "beta_gated", "brier") - _mean(drows, "identity", "brier")
        da = _mean(drows, "beta_gated", "auroc") - _mean(drows, "identity", "auroc")
        if db > 1e-4 or da < -1e-4:
            regressions[dom] = {"dBrier": round(db, 4), "dAUROC": round(da, 4)}
    print(f"\nbeta_gated per-domain regressions vs identity: {regressions or 'none'}")

    # R2: the SINGLE operating-point pathway (reconciled with Item 4).
    t_star = bayes_threshold(cost_fp=1.0, benefit_tp=10.0)  # b=10c: a miss ~10x costlier
    print(
        f"\nR2 single operating point: MCA-calibrated prob >= cost-driven Bayes "
        f"t*={t_star:.3f} (b=10c, missed-detection-catastrophic). The conformal / "
        f"Venn-Abers layer is a coverage floor, NOT a second competing threshold."
    )

    out = {
        "n_events": len(rows),
        "gate_accepts": gate_accepts,
        "ece_tol": ECE_TOL,
        "operating_point": {
            "pathway": "MCA-calibrated probability >= cost-driven Bayes t*",
            "bayes_t_star_b10c": t_star,
            "coverage_layer": "conformal/Venn-Abers recall floor (not a second threshold)",
        },
        "overall": overall,
        "deltas_vs_identity": {
            m: {k: overall[m][k] - overall["identity"][k] for k in keys}
            for m in ("isotonic", "beta_mca", "beta_gated", "beta_va")
        },
        "venn_abers_marginal_over_mca": {
            k: overall["beta_va"][k] - overall["beta_mca"][k] for k in keys
        },
        "venn_abers_mean_interval_width": float(np.nanmean([r["va_interval_width"] for r in rows])),
        "beta_gated_domain_regressions": regressions,
        "per_event": rows,
        "per_domain": {
            dom: {m: {k: _mean(drows, m, k) for k in keys} for m in METHODS}
            for dom, drows in by_dom.items()
        },
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "calibration_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

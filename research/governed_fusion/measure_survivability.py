"""Item 2: Mercury adversarial-survivability on the REAL fused anomaly score.

For one representative event per reachable domain: fit the real
``MercuryAnomalyDetector`` on the event's normal rows, expose its **real fused
ensemble anomaly score** as the attack target ``score_fn`` (no toy ``||x-loc||``),
and run the fixed-budget battery.  Reports, per domain and overall:

  * clean vs worst-case fused AUROC (survivability);
  * the controlled-channel floor curve (worst-case AUROC vs budget ``m``);
  * the cubic-moment escape (D_phi AUC vs the Gaussian floor) with a Gaussian
    control so the escape is calibrated (~0 on Gaussian, >0 on real skew).

These are Mercury's own numbers — FINDOYOU's are not cited.  Research/test-only;
zero runtime change.

Run::

    source /home/user/gf_env.sh
    python research/governed_fusion/measure_survivability.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from research.adversarial.governed_attacks import cubic_moment_escape, floor_curve
from research.governed_fusion.suite import build_suite, stratified_subsample

_OUT_DIR = os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache")
ROW_CAP = 160  # keep the finite-difference battery tractable
EPS = 0.6  # L2 per-row perturbation budget (standardized feature units)


def _representative_per_domain(events: list[Any]) -> dict[str, Any]:
    """One event per domain — the one with the most positives (richest signal)."""
    best: dict[str, Any] = {}
    for ev in events:
        if ev.domain not in best or ev.n_pos > best[ev.domain].n_pos:
            best[ev.domain] = ev
    return best


def _make_score_fn(normal_ref: np.ndarray[Any, Any]):
    """Mercury's real fused ensemble anomaly score, fit on the normal reference."""
    det = MercuryAnomalyDetector().fit(normal_ref)

    def score_fn(batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(
            det.detect(np.asarray(batch, dtype=np.float64))["scores"], dtype=np.float64
        )

    return score_fn


def _gaussian_control_escape(d: int = 6, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(600, d))
    anom = rng.normal(0.0, 1.0, size=(150, d))
    anom[:, : d // 2] += 1.5  # pure mean shift -> Gaussian, no escape expected
    x = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(600, int), np.ones(150, int)])
    return cubic_moment_escape(x, y)


def main() -> None:
    reps = _representative_per_domain(build_suite())
    out: dict[str, Any] = {"per_domain": {}, "eps": EPS, "row_cap": ROW_CAP}
    clean_aucs: list[float] = []
    worst_half_aucs: list[float] = []

    print(f"\n==== ITEM 2: Mercury survivability on the REAL fused score (eps={EPS}) ====")
    ctrl = _gaussian_control_escape()
    print(
        f"cubic-moment Gaussian control: floor={ctrl['floor_auc']:.3f} "
        f"cubic={ctrl['cubic_auc']:.3f} escape={ctrl['escape']:+.3f} (must be ~0)\n"
    )
    print(
        f"{'domain':<16}{'event':<18}{'k':>3}{'clean':>8}{'floor_curve worst-case AUROC vs m':>40}"
    )

    for domain in sorted(reps):
        ev = reps[domain]
        X, y = stratified_subsample(ev.X, ev.y, ROW_CAP, seed=42)
        k = X.shape[1]
        normal_ref = X[y == 0]
        if len(normal_ref) < 8 or np.sum(y == 1) < 1:
            continue
        score_fn = _make_score_fn(normal_ref)
        curve = floor_curve(score_fn, X, y, normal_reference=normal_ref, eps=EPS, seed=0)
        escape = cubic_moment_escape(X, y)

        clean = curve[0]["worst_case_auroc"]
        half = next((c for c in curve if c["m"] == k // 2), curve[-1])
        clean_aucs.append(clean)
        worst_half_aucs.append(half["worst_case_auroc"])

        curve_str = "  ".join(f"m={c['m']}:{c['worst_case_auroc']:.3f}" for c in curve)
        print(f"{domain:<16}{ev.event_id[:17]:<18}{k:>3}{clean:>8.3f}   {curve_str}")
        out["per_domain"][domain] = {
            "event": ev.event_id,
            "k": int(k),
            "clean_auroc": clean,
            "floor_curve": curve,
            "worst_case_half_channel_auroc": half["worst_case_auroc"],
            "cubic_moment": escape,
        }

    print("-" * 96)
    if clean_aucs:
        print(
            f"OVERALL  mean clean AUROC = {np.mean(clean_aucs):.3f}   "
            f"mean worst-case (m=k/2) AUROC = {np.mean(worst_half_aucs):.3f}   "
            f"mean drop = {np.mean(clean_aucs) - np.mean(worst_half_aucs):+.3f}"
        )
    print("\ncubic-moment escape (D_phi AUC - floor AUC) per domain:")
    for domain, rec in out["per_domain"].items():
        cm = rec["cubic_moment"]
        print(
            f"  {domain:<16} floor={cm['floor_auc']:.3f} cubic={cm['cubic_auc']:.3f} "
            f"escape={cm['escape']:+.3f}"
        )
    out["overall"] = {
        "mean_clean_auroc": float(np.mean(clean_aucs)) if clean_aucs else float("nan"),
        "mean_worst_case_half_channel_auroc": (
            float(np.mean(worst_half_aucs)) if worst_half_aucs else float("nan")
        ),
        "gaussian_control_escape": ctrl["escape"],
    }
    out_path = os.path.join(_OUT_DIR, "survivability_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

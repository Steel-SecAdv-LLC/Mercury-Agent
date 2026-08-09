# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 2 calibration lever probe (held-out Brier/ECE vs the shipped default).

Can the composite objective, the accept-gate tolerance, or the beta warm-start
lower **held-out** Brier+ECE on the 23-event live suite WITHOUT breaking the
AUROC exact tie or the no-regression accept-gate?  Three levers named in the
Stage-2 brief are swept on the same seeded 50/50 splits ``measure_calibration.py``
uses:

  * ``lambda_ece`` -- the ECE-surrogate weight in ``Brier + lambda_ece*ECE``;
  * ``lambda_nb``  -- the net-benefit term ``- lambda_nb*NB`` (default 0);
  * ``warm_start`` -- the L-BFGS-B start: ``identity`` (the shipped ``[1,1,0]``,
    i.e. ``p == u``) vs ``mle`` (a closed-form beta MLE: Newton-fit logistic of
    ``y`` on ``[ln u, -ln(1-u)]`` clipped to ``a,b>=0``) -- the start the shipped
    docstring used to *claim*;
  * the accept-gate ``ece_tol`` is swept separately (loosen/tighten acceptance).

Every config is a monotone map (``a,b>=0``), so it preserves each event's score
ranking exactly; ``d_auroc`` (paired mean of config minus identity per-event
AUROC) is reported as evidence the tie is not broken (~0).  A config is a LAND
candidate only if it strictly beats the shipped default
(``lambda_ece=1, lambda_nb=0, identity``) on held-out Brier AND ECE; otherwise the
negative is recorded with numbers.  Read-only research; changes no shipped default.

Run::

    source research/governed_fusion/gf_env.sh
    python research/governed_fusion/measure_calibration_levers.py
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
from scipy import optimize

from omni_mercury_engine.core.calibration import (
    _ece_kernel_surrogate,
    _net_benefit_integral,
    compute_ece,
    fit_accept_gated_mca,
)
from omni_mercury_engine.ml.mercury_ml import brier_score_loss, roc_auc_score
from research.governed_fusion.measure_baseline import load_scores
from research.governed_fusion.measure_calibration import _minmax
from research.governed_fusion.measure_conformal import _split

_OUT_DIR = os.environ.get("GF_RESULTS_DIR", os.environ.get("GF_CACHE_DIR", "/home/user/gf_cache"))
_THRESHOLDS = np.linspace(0.02, 0.98, 25)
_PRIOR = 1.0 / _THRESHOLDS
_EPS = 1e-6

# (label, lambda_ece, lambda_nb, warm_start). The first row is the shipped default.
_CONFIGS: tuple[tuple[str, float, float, str], ...] = (
    ("default(l_ece=1,l_nb=0,identity)", 1.0, 0.0, "identity"),
    ("l_ece=0.0", 0.0, 0.0, "identity"),
    ("l_ece=0.5", 0.5, 0.0, "identity"),
    ("l_ece=2.0", 2.0, 0.0, "identity"),
    ("l_ece=4.0", 4.0, 0.0, "identity"),
    ("l_nb=0.5", 1.0, 0.5, "identity"),
    ("l_nb=1.0", 1.0, 1.0, "identity"),
    ("warm=mle", 1.0, 0.0, "mle"),
    ("warm=mle,l_ece=2.0", 2.0, 0.0, "mle"),
)
_ECE_TOLS = (1e-3, 1e-2, 5e-2)  # accept-gate slack sweep (shipped default 1e-3)


def _mle_start(u: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Closed-form beta MLE warm start: Newton-fit logistic of y on the lifted features.

    Solves the unconstrained logistic fit on ``[ln u, -ln(1-u), 1]`` by Newton
    steps and clips ``a, b`` to ``>= 0`` so the start is a valid monotone map.
    """
    feat = np.column_stack([np.log(u), -np.log1p(-u), np.ones_like(u)])
    theta = np.zeros(3)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-np.clip(feat @ theta, -50, 50)))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = feat.T @ (p - y)
        hess = (feat * w[:, None]).T @ feat + 1e-6 * np.eye(3)
        theta = theta - np.linalg.solve(hess, grad)
    return np.array([max(theta[0], 0.0), max(theta[1], 0.0), theta[2]])


def _fit(
    p_cal: np.ndarray[Any, Any],
    y_cal: np.ndarray[Any, Any],
    lambda_ece: float,
    lambda_nb: float,
    warm_start: str,
) -> tuple[float, float, float]:
    """Fit (a,b,c) for one config; mirrors BetaCalibration.fit with a choosable x0."""
    s_min, s_max = float(np.min(p_cal)), float(np.max(p_cal))
    rng = s_max - s_min if s_max > s_min else 1.0
    u = np.clip((p_cal - s_min) / rng, _EPS, 1.0 - _EPS)
    ln_u, ln_1mu = np.log(u), np.log1p(-u)
    y = y_cal.astype(np.float64)

    def objective(theta: np.ndarray[Any, Any]) -> float:
        a, b, c = float(theta[0]), float(theta[1]), float(theta[2])
        p = 1.0 / (1.0 + np.exp(-np.clip(c + a * ln_u - b * ln_1mu, -50.0, 50.0)))
        loss = float(np.mean((p - y) ** 2))
        if lambda_ece > 0:
            loss += lambda_ece * _ece_kernel_surrogate(p, y, 0.1)
        if lambda_nb > 0:
            loss -= lambda_nb * _net_benefit_integral(p, y, _THRESHOLDS, _PRIOR)
        return loss

    x0 = _mle_start(u, y) if warm_start == "mle" else np.array([1.0, 1.0, 0.0])
    res = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None), (None, None)],
        options={"maxiter": 200},
    )
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def _apply(
    p: np.ndarray[Any, Any],
    abc: tuple[float, float, float],
    lohi: tuple[float, float],
) -> np.ndarray[Any, Any]:
    """Apply a fitted (a,b,c) beta map to probabilities scaled by ``lohi``."""
    a, b, c = abc
    lo, hi = lohi
    rng = hi - lo if hi > lo else 1.0
    u = np.clip((p - lo) / rng, _EPS, 1.0 - _EPS)
    return 1.0 / (1.0 + np.exp(-np.clip(c + a * np.log(u) - b * np.log1p(-u), -50.0, 50.0)))


def _auroc(y: np.ndarray[Any, Any], p: np.ndarray[Any, Any]) -> float:
    """AUROC when computable, else NaN."""
    if np.unique(y).size > 1 and np.unique(p).size > 1:
        return float(roc_auc_score(y, p))
    return float("nan")


def _holdout_metrics() -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Per-config held-out Brier/ECE + paired AUROC-tie deviation + accept-gate sweep."""
    events = load_scores(kind="real")
    per_cfg: dict[str, list[tuple[float, float, float]]] = {c[0]: [] for c in _CONFIGS}
    gate: dict[float, list[tuple[float, float]]] = {t: [] for t in _ECE_TOLS}

    for es in events:
        cal_idx, ev_idx = _split(es.y)
        y_cal, y_ev = es.y[cal_idx], es.y[ev_idx]
        s_cal, s_ev = es.combined[cal_idx], es.combined[ev_idx]
        if np.sum(y_cal == 1) < 1 or np.sum(y_ev == 1) < 1:
            continue
        p0_cal, p0_ev = _minmax(s_cal, s_cal), _minmax(s_cal, s_ev)
        lohi = (float(np.min(p0_cal)), float(np.max(p0_cal)))
        auroc_id = _auroc(y_ev, p0_ev)  # identity reference for the tie check
        for label, l_ece, l_nb, warm in _CONFIGS:
            p_ev = _apply(p0_ev, _fit(p0_cal, y_cal, l_ece, l_nb, warm), lohi)
            au = _auroc(y_ev, p_ev)
            # paired tie deviation: only when BOTH identity and config are computable.
            # `math.isnan` rather than the `x == x` NaN idiom -- same semantics,
            # and it does not read as an accidental self-comparison.
            computable = not math.isnan(au) and not math.isnan(auroc_id)
            d_au = (au - auroc_id) if computable else float("nan")
            per_cfg[label].append(
                (float(brier_score_loss(y_ev, p_ev)), float(compute_ece(y_ev, p_ev)), d_au)
            )
        for tol in _ECE_TOLS:
            gated, accept = fit_accept_gated_mca(p0_cal, y_cal, ece_tol=tol)
            p_g = np.asarray(gated.calibrate(p0_ev), dtype=np.float64)
            gate[tol].append((float(accept), float(brier_score_loss(y_ev, p_g))))

    def _mean(rows: list[tuple[float, ...]], i: int) -> float:
        xs = [r[i] for r in rows if r[i] == r[i]]
        return float(np.mean(xs)) if xs else float("nan")

    cfg_summary = {
        label: {
            "brier": _mean(per_cfg[label], 0),
            "ece": _mean(per_cfg[label], 1),
            "d_auroc_vs_identity": _mean(per_cfg[label], 2),
            "n": len(per_cfg[label]),
        }
        for label, *_ in _CONFIGS
    }
    gate_summary = {
        f"ece_tol={tol:g}": {
            "accept_rate": float(np.mean([g[0] for g in gate[tol]])) if gate[tol] else float("nan"),
            "holdout_brier": _mean(gate[tol], 1),
            "n": len(gate[tol]),
        }
        for tol in _ECE_TOLS
    }
    return cfg_summary, gate_summary


def main() -> None:
    """Run the lever sweep, print the table, and write the results JSON."""
    cfg_summary, gate_summary = _holdout_metrics()
    base = cfg_summary["default(l_ece=1,l_nb=0,identity)"]

    print("\n==== STAGE 2 LEVER PROBE (held-out, 23-event live suite) ====")
    print(f"{'config':<34}{'brier':>10}{'ece':>10}{'d_auroc':>10}{'dBrier':>10}{'dECE':>10}")
    landed = []
    for label, *_ in _CONFIGS:
        m = cfg_summary[label]
        db, de = m["brier"] - base["brier"], m["ece"] - base["ece"]
        print(
            f"{label:<34}{m['brier']:>10.4f}{m['ece']:>10.4f}{m['d_auroc_vs_identity']:>+10.4f}"
            f"{db:>+10.4f}{de:>+10.4f}"
        )
        if db < -1e-4 and de < -1e-4:
            landed.append(label)

    print("\naccept-gate ece_tol sweep (accept_rate / held-out Brier of gated map):")
    for label, g in gate_summary.items():
        print(
            f"  {label:<14} accept={g['accept_rate']:.3f}  holdout_brier={g['holdout_brier']:.4f}"
        )

    verdict = (
        f"LAND candidates (beat default on BOTH Brier and ECE held-out): {landed}"
        if landed
        else "NEGATIVE: no lever beats the shipped default on held-out Brier AND ECE"
    )
    print(f"\nVERDICT -> {verdict}")

    out = {
        "suite": "live headline (real) 23 events",
        "default_config": "lambda_ece=1, lambda_nb=0, warm_start=identity",
        "note_d_auroc": "paired (config - identity) per-event AUROC; ~0 confirms the monotone tie",
        "configs": cfg_summary,
        "accept_gate_tol_sweep": gate_summary,
        "land_candidates": landed,
        "verdict": verdict,
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, "calibration_levers_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

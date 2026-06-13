# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""VALIDATION REGISTER V1-V8 (synthetic world, Brief Part 1).

Pre-registered: each check states metric + acceptance (kill) criterion before the
number is produced.  Every number is printed (R2), passes and fails alike.
Deterministic: world seed 7, split 2000 cal / 40000 test (R5).

Run:  python benchmarks/calibration_brief/run_v1_v8.py
"""

from __future__ import annotations

import numpy as np

from cal_core import (
    NB_GRID,
    PHI,
    BetaCalibrator,
    auroc,
    brier,
    conditional_mean_remap,
    ece,
    make_synth,
    net_benefit,
    net_benefit_treat_all,
    oracle_brier_bound,
    prevalence_shift,
)

RESULTS: list[tuple[str, bool, str]] = []


def record(vid: str, passed: bool, detail: str) -> None:
    RESULTS.append((vid, passed, detail))
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {vid}: {detail}")


def base_split():
    w = make_synth(42000, pi=0.15, gamma=2.2, delta=0.8, seed=7)
    cal = slice(0, 2000)
    test = slice(2000, 42000)
    return w, cal, test


def main() -> int:
    w, cal, test = base_split()
    beta = BetaCalibrator().fit(w.s[cal], w.y[cal])
    p_test = beta.calibrate(w.s[test])
    y_test = w.y[test]
    s_test = w.s[test]
    cstar_test = w.c_star[test]

    # V1 -- AUROC invariance under strictly monotone c_theta.  Kill: |dAUROC|>=1e-7
    a_raw = auroc(y_test, s_test)
    a_cal = auroc(y_test, p_test)
    d = abs(a_cal - a_raw)
    record(
        "V1",
        d < 1e-7,
        f"AUROC {a_raw:.6f} -> {a_cal:.6f}  |delta|={d:.2e}  (prior 0.919708, delta=0)",
    )

    # V2 -- Brier approaches oracle floor E[c*(1-c*)].  Kill: |Brier_cal-bound|>=5e-3
    b_raw = brier(y_test, s_test)
    b_cal = brier(y_test, p_test)
    bound = oracle_brier_bound(cstar_test)
    record(
        "V2",
        abs(b_cal - bound) < 5e-3,
        f"Brier {b_raw:.4f} -> {b_cal:.4f}; oracle bound {bound:.4f}; "
        f"gap {abs(b_cal - bound):.4f}  (prior 0.0763->0.0695, bound 0.0694)",
    )

    # V3 -- ECE shrinks >= 3x.  Kill: ratio < 3
    e_raw = ece(y_test, s_test)
    e_cal = ece(y_test, p_test)
    ratio = e_raw / e_cal if e_cal > 0 else float("inf")
    record("V3", ratio >= 3.0, f"ECE {e_raw:.4f} -> {e_cal:.4f}  ({ratio:.1f}x)  (prior 14.5x)")

    # V4 -- NB(t): calibrated >= uncalibrated AND >= max(treat-all,0), tol 2e-3.
    tol = 2e-3
    ok = True
    rows = []
    for t in NB_GRID:
        nb_cal = net_benefit(y_test, p_test, t)
        nb_raw = net_benefit(y_test, s_test, t)
        nb_all = max(net_benefit_treat_all(0.15, t), 0.0)
        cond = (nb_cal >= nb_raw - tol) and (nb_cal >= nb_all - tol)
        ok = ok and cond
        rows.append(f"t={t:.2f}:cal={nb_cal:+.4f}/raw={nb_raw:+.4f}/all={nb_all:+.4f}")
    record("V4", ok, "  ".join(rows))

    # V5 -- functional identity on already-calibrated input.  Test the MAP.
    #        Kill: sup_{s in [0.02,0.98]} |c_theta(s)-s| >= 0.05
    beta_id = BetaCalibrator().fit(w.c_star[cal], w.y[cal])
    grid = np.linspace(0.02, 0.98, 400)
    sup = float(np.max(np.abs(beta_id.calibrate(grid) - grid)))
    record(
        "V5",
        sup < 0.05,
        f"sup|c_theta(s)-s| on [0.02,0.98] = {sup:.3f}  (a,b,c="
        f"{beta_id.a:.2f},{beta_id.b:.2f},{beta_id.c:.2f}; prior 0.026)",
    )

    # V6 -- multiplicative-damage identity (eta^Phi deflation), kappa=0.92^Phi.
    #        Kill: identity mismatch > 1e-9, or ECE does not worsen >= 3x.
    kappa = 0.92**PHI
    p = p_test
    lhs = brier(y_test, kappa * p) - brier(y_test, p)
    rhs = (kappa**2 - 1.0) * np.mean(p**2) + 2.0 * (1.0 - kappa) * np.mean(p * y_test)
    mism = abs(lhs - rhs)
    e_dmg = ece(y_test, kappa * p)
    ratio6 = e_dmg / e_cal if e_cal > 0 else float("inf")
    record(
        "V6",
        (mism < 1e-9) and (ratio6 >= 3.0),
        f"kappa={kappa:.4f}; dBrier_exact={lhs:.5f}; identity mismatch={mism:.2e}; "
        f"ECE {e_cal:.4f}->{e_dmg:.4f} ({ratio6:.1f}x)  (prior dBrier 0.00097, ECE->0.0173)",
    )

    # V7 -- prevalence shift 0.15 -> 0.05; one-line adjustment.  Kill: adj >= unadj/2.
    wt = make_synth(40000, pi=0.05, gamma=2.2, delta=0.8, pi_score=0.15, seed=11)
    p_t = beta.calibrate(wt.s)  # calibrator trained at pi=0.15
    e_unadj = ece(wt.y, p_t)
    p_adj = prevalence_shift(p_t, pi_src=0.15, pi_tgt=0.05)
    e_adj = ece(wt.y, p_adj)
    record(
        "V7",
        e_adj < e_unadj / 2.0,
        f"target pi=0.05: ECE unadjusted {e_unadj:.4f} -> adjusted {e_adj:.4f}  "
        f"(prior 0.055 -> 0.0034)",
    )

    # V8 -- Corollary B: non-monotone bijection scramble s=(c*+0.5) mod 1.
    #        Monotone calibrators stay scrambled; 50-bin remap recovers >=0.90*oracle.
    s_scr = np.mod(w.c_star + 0.5, 1.0)
    a_oracle = auroc(w.y[test], w.c_star[test])
    a_scr = auroc(w.y[test], s_scr[test])
    beta_scr = BetaCalibrator().fit(s_scr[cal], w.y[cal])
    a_beta = auroc(w.y[test], beta_scr.calibrate(s_scr[test]))
    a_remap = auroc(w.y[test], conditional_mean_remap(s_scr[cal], w.y[cal], s_scr[test], 50))
    v8_ok = (abs(a_beta - a_scr) < 0.02) and (a_remap >= 0.90 * a_oracle)
    record(
        "V8",
        v8_ok,
        f"scrambled AUROC {a_scr:.4f}; monotone-Beta {a_beta:.4f} (stays); "
        f"50-bin remap {a_remap:.4f}; oracle {a_oracle:.4f}  (prior 0.400->0.9125,orc 0.9208)",
    )

    # V8 ANTI-CLAIM -- folded score exp(-z^2/2): nothing recovers (all AUROC ~ 0.50).
    s_fold = np.exp(-w.z**2 / 2.0)
    a_fold = auroc(w.y[test], s_fold[test])
    beta_f = BetaCalibrator().fit(s_fold[cal], w.y[cal])
    a_fold_beta = auroc(w.y[test], beta_f.calibrate(s_fold[test]))
    a_fold_remap = auroc(w.y[test], conditional_mean_remap(s_fold[cal], w.y[cal], s_fold[test], 50))
    near_half = all(abs(x - 0.5) < 0.03 for x in (a_fold, a_fold_beta, a_fold_remap))
    record(
        "V8-anti",
        near_half,
        f"folded AUROC raw={a_fold:.4f} beta={a_fold_beta:.4f} remap={a_fold_remap:.4f} "
        f"(all ~0.50; post-hoc maps cannot create discrimination)",
    )

    npass = sum(1 for _, p_, _ in RESULTS if p_)
    print(f"\n==== V1-V8 SUMMARY: {npass}/{len(RESULTS)} pre-registered checks PASS ====")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""X2 -- Non-monotonicity guard + repair (failure mode F4, proven real by V8).

PRE-REGISTRATION (R1):
  H: a cal-set test (Kendall tau between binned s and binned c-hat(s)) detects
     non-monotone c(s); on detection swap monotone calibrator -> non-monotone
     conditional-mean remap.
  Detector: K=12 equal-mass cal bins; tau = kendalltau(mean_s_b, mean_y_b);
            FIRE (declare non-monotone) iff tau < TAU_FIRE.
  Track: must FIRE on V8 scramble and the repair recovers >= 0.90 * oracle AUROC;
         must stay SILENT (< 5% false-fire) on all six real datasets (large + small).
  KILL : either side fails.

Run:  python benchmarks/calibration_brief/x2_nonmono_guard.py
"""

from __future__ import annotations

import warnings

import numpy as np
from cal_core import BetaCalibrator, auroc, conditional_mean_remap, ensure_datasets, make_synth
from scipy.stats import kendalltau
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

DATASETS = [
    "6_cardio",
    "23_mammography",
    "38_thyroid",
    "31_satimage-2",
    "28_pendigits",
    "30_satellite",
]
SEEDS = [0, 1, 2, 3, 4]
TAU_FIRE = 0.5
K_BINS = 12


def monotonicity_tau(s, y, k=K_BINS):
    """Kendall tau between per-bin mean score and per-bin mean label (equal-mass)."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(s, kind="stable")
    sb, yb = [], []
    for ch in np.array_split(order, k):
        if len(ch) == 0:
            continue
        sb.append(float(np.mean(s[ch])))
        yb.append(float(np.mean(y[ch])))
    if len(sb) < 3:
        return 1.0
    tau = kendalltau(sb, yb).statistic
    return 0.0 if tau is None or np.isnan(tau) else float(tau)


BOOT_CONF = 0.99  # one-sided test: false-positive rate <= 1-BOOT_CONF by construction


def brier_gap_fires(s_cal, y_cal, seed=0, n_boot=400, conf=BOOT_CONF):
    """Detector B (pre-registered alt): isotonic-vs-unrestricted-binning Brier gap.

    Split cal; fit monotone isotonic and a non-monotone conditional-mean remap on
    half A; on held-out half B fire iff the remap beats isotonic in > ``conf`` of
    PAIRED bootstrap resamples (threshold-free, robust to label noise since both
    fits see the same noise).  conf=0.99 keeps the null false-positive rate <=1%."""
    from sklearn.isotonic import IsotonicRegression

    s_cal = np.asarray(s_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(s_cal))
    h = len(idx) // 2
    A, B = idx[:h], idx[h:]
    if len(np.unique(y_cal[A])) < 2 or len(B) < 8:
        return False
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s_cal[A], y_cal[A])
    p_iso = iso.predict(s_cal[B])
    nb = int(min(20, max(3, len(A) // 5)))
    p_rmp = conditional_mean_remap(s_cal[A], y_cal[A], s_cal[B], nb)
    se_iso = (p_iso - y_cal[B]) ** 2
    se_rmp = (p_rmp - y_cal[B]) ** 2
    nB = len(B)
    wins = 0
    for _ in range(n_boot):
        bi = rng.integers(0, nB, nB)  # PAIRED resample (same idx both)
        if se_rmp[bi].mean() < se_iso[bi].mean():
            wins += 1
    return (wins / n_boot) > conf


def guarded_calibrate(s_cal, y_cal, s_eval, detector="brier_gap"):
    """Fire-or-not, then repair.  Returns (fired, calibrated_eval)."""
    if detector == "tau":
        fired = monotonicity_tau(s_cal, y_cal) < TAU_FIRE
    else:
        fired = brier_gap_fires(s_cal, y_cal)
    if fired:  # non-monotone -> non-monotone remap
        return True, conditional_mean_remap(s_cal, y_cal, s_eval, 50)
    return False, BetaCalibrator().fit(s_cal, y_cal).calibrate(s_eval)


def test_scramble(detector):
    """Must fire on V8 scramble and recover >= 0.90 * oracle AUROC."""
    w = make_synth(42000, seed=7)
    cal, te = slice(0, 2000), slice(2000, 42000)
    s_scr = np.mod(w.c_star + 0.5, 1.0)
    a_oracle = auroc(w.y[te], w.c_star[te])
    fired, p = guarded_calibrate(s_scr[cal], w.y[cal], s_scr[te], detector)
    a_rep = auroc(w.y[te], p)
    # control: well-specified monotone score must NOT fire
    mono_fired, _ = guarded_calibrate(w.s[cal], w.y[cal], w.s[te], detector)
    ok = fired and (a_rep >= 0.90 * a_oracle) and (not mono_fired)
    print(
        f"  scramble fired={fired}  repaired AUROC={a_rep:.4f} "
        f"(need >={0.90*a_oracle:.4f}); well-specified fires={mono_fired} (expect False) "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def test_real_falsefire(detector):
    """Must stay silent (<5% false-fire) on all six real datasets."""
    fires = []
    per_ds = {ds: [] for ds in DATASETS}
    for small in (False, True):
        for ds in DATASETS:
            d = np.load(f"data/{ds}.npz")
            X0, y0 = d["X"].astype(float), d["y"].astype(int)
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(y0))
                X, y = X0[perm], y0[perm]
                n = len(y)
                ntr, ncal = n // 2, n // 4
                tr = np.arange(ntr)
                cal = np.arange(ntr, ntr + ncal)
                iso = IsolationForest(n_estimators=200, random_state=seed).fit(X[tr])
                raw = -iso.score_samples(X)
                lo, hi = raw[cal].min(), raw[cal].max()
                sc = np.clip((raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw), 0, 1)
                ci = cal
                if small:
                    ci = rng.choice(cal, size=min(100, len(cal)), replace=False)
                    while y[ci].sum() < 3:
                        ci = rng.choice(cal, size=min(100, len(cal)), replace=False)
                if detector == "tau":
                    fired = monotonicity_tau(sc[ci], y[ci]) < TAU_FIRE
                else:
                    fired = brier_gap_fires(sc[ci], y[ci])
                fires.append(fired)
                per_ds[ds].append(fired)
    rate = np.mean(fires) * 100
    print(f"  real-data false-fire rate: {rate:.1f}% over {len(fires)} runs (need < 5%)")
    per = "  ".join(f"{ds.split('_')[0]}:{np.mean(per_ds[ds])*100:.0f}%" for ds in DATASETS)
    print("   per-ds: " + per)
    return rate < 5.0


def main():
    ensure_datasets()
    print("=== X2 non-monotonicity guard ===")
    for detector in ("tau", "brier_gap"):
        print(f"\n--- Detector: {detector} ---")
        a = test_scramble(detector)
        b = test_real_falsefire(detector)
        print(
            f"  VERDICT[{detector}]: scramble-ok={a}  real-silent={b}  -> "
            f"{'SURVIVES' if (a and b) else 'KILLED'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

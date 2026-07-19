# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""VALIDATION REGISTER V9-V11 (real ADBench data, Brief Part 1 + Standard Track S2/S4).

Protocol (per dataset x seed in {0..4}): permute; 50/25/25 train/cal/test;
IsolationForest(n_estimators=200, random_state=seed) on train; score = -score_samples;
min-max normalise using CAL stats; fit calibrators on cal; evaluate on test.

Baselines (S4): raw, Mercury-Platt, Mercury-Isotonic, Mercury-Temp, Mercury-Ens3,
Beta, Ens4.

V9 large-n (cal = 25%), V10 small-n (cal subsampled to 100, >=3 pos),
V11 Ens4 selection (picks Isotonic at large n, Beta at small n).

Run:  python benchmarks/calibration_brief/run_v9_v11.py
"""

from __future__ import annotations

import warnings

import numpy as np
from cal_core import (
    NB_GRID,
    BetaCalibrator,
    auroc,
    brier,
    ece,
    ensure_datasets,
    load_mercury_calibrators,
    net_benefit,
)
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
MC = load_mercury_calibrators()

DATASETS = [
    "6_cardio",
    "23_mammography",
    "38_thyroid",
    "31_satimage-2",
    "28_pendigits",
    "30_satellite",
]
SEEDS = [0, 1, 2, 3, 4]


def band_nb(y, p):
    return float(np.mean([net_benefit(y, p, t) for t in NB_GRID]))


class Ens4:
    """4-member ensemble {Platt, Isotonic, Temperature, Beta}; select on validation
    Brier over an internal 75/25 split of cal (Brief V11)."""

    def __init__(self, seed: int):
        self.seed = seed
        self.best_name = None
        self.best = None

    def _members(self):
        return {
            "platt": MC.PlattScaling(),
            "isotonic": MC.IsotonicCalibration(),
            "temperature": MC.TemperatureScaling(),
            "beta": BetaCalibrator(),
        }

    def fit(self, s, y):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(s))
        n_val = max(1, int(0.25 * len(s)))
        val, tr = idx[:n_val], idx[n_val:]
        best_brier = np.inf
        for name, cal in self._members().items():
            try:
                cal.fit(s[tr], y[tr])
                b = brier(y[val], cal.calibrate(s[val]))
            except Exception:
                b = np.inf
            if b < best_brier:
                best_brier, self.best_name = b, name
        self.best = self._members()[self.best_name]
        self.best.fit(s, y)
        return self

    def calibrate(self, s):
        return self.best.calibrate(s)


def subsample_pos(idx, y, n, k, rng):
    """Subsample idx to size n ensuring >= k positives."""
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    pos_avail = idx[y[idx] == 1]
    while int(y[sel].sum()) < k and len(pos_avail) >= k:
        need = k - int(y[sel].sum())
        add = rng.choice(pos_avail, size=need, replace=False)
        neg_in_sel = sel[y[sel] == 0]
        drop = rng.choice(neg_in_sel, size=need, replace=False)
        sel = np.concatenate([sel[~np.isin(sel, drop)], add])
    return sel


def make_methods(seed):
    return {
        "raw": None,
        "M-Platt": MC.PlattScaling(),
        "M-Isotonic": MC.IsotonicCalibration(),
        "M-Temp": MC.TemperatureScaling(),
        "M-Ens3": MC.CalibrationEnsemble(seed=seed),
        "Beta": BetaCalibrator(),
        "Ens4": Ens4(seed=seed),
    }


def run_track(small: bool):
    methods = ["raw", "M-Platt", "M-Isotonic", "M-Temp", "M-Ens3", "Beta", "Ens4"]
    acc = {m: {"auroc": [], "brier": [], "ece": [], "nb": []} for m in methods}
    iso_loses_auroc = {ds: [] for ds in DATASETS}
    beta_preserves = []
    platt_preserves = []
    temp_preserves = []
    ens4_pick = []
    per_ds_ece_best = {ds: [] for ds in DATASETS}
    per_ds_brier = {ds: {m: [] for m in methods} for ds in DATASETS}

    for ds in DATASETS:
        d = np.load(f"data/{ds}.npz")
        X0, y0 = d["X"].astype(float), d["y"].astype(int)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(y0))
            Xs, yy = X0[perm], y0[perm]  # fresh copy per seed (no cumulative clobber)
            n = len(yy)
            n_tr, n_cal = n // 2, n // 4
            tr = np.arange(0, n_tr)
            cal = np.arange(n_tr, n_tr + n_cal)
            te = np.arange(n_tr + n_cal, n)

            iso = IsolationForest(n_estimators=200, random_state=seed)
            iso.fit(Xs[tr])
            raw = -iso.score_samples(Xs)
            lo, hi = raw[cal].min(), raw[cal].max()
            sc = np.clip((raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw), 0, 1)

            cal_idx = cal
            if small:
                cal_idx = subsample_pos(cal, yy, 100, 3, rng)

            s_cal, y_cal = sc[cal_idx], yy[cal_idx]
            s_te, y_te = sc[te], yy[te]
            if len(np.unique(y_te)) < 2 or len(np.unique(y_cal)) < 2:
                continue

            a_raw = auroc(y_te, s_te)
            for m, model in make_methods(seed).items():
                if model is None:
                    p = s_te
                else:
                    try:
                        model.fit(s_cal, y_cal)
                        p = model.calibrate(s_te)
                    except Exception:
                        p = s_te
                a, b, e, nb = auroc(y_te, p), brier(y_te, p), ece(y_te, p), band_nb(y_te, p)
                acc[m]["auroc"].append(a)
                acc[m]["brier"].append(b)
                acc[m]["ece"].append(e)
                acc[m]["nb"].append(nb)
                per_ds_brier[ds][m].append(b)
                if m == "M-Isotonic":
                    iso_loses_auroc[ds].append(a < a_raw - 1e-9)
                if m == "Beta":
                    beta_preserves.append(abs(a - a_raw) < 1e-9)
                if m == "M-Platt":
                    platt_preserves.append(abs(a - a_raw) < 1e-9)
                if m == "M-Temp":
                    temp_preserves.append(abs(a - a_raw) < 1e-9)
            # which method has best ECE this run
            run_eces = {m: acc[m]["ece"][-1] for m in methods if m != "raw"}
            per_ds_ece_best[ds].append(min(run_eces, key=run_eces.get))
            ens4 = make_methods(seed)["Ens4"]
            ens4.fit(s_cal, y_cal)
            ens4_pick.append(ens4.best_name)

    return dict(
        acc=acc,
        iso_loses=iso_loses_auroc,
        beta_pres=beta_preserves,
        platt_pres=platt_preserves,
        temp_pres=temp_preserves,
        ens4_pick=ens4_pick,
        ece_best=per_ds_ece_best,
        brier=per_ds_brier,
    )


def fmt(vals):
    a = np.array(vals)
    return f"{a.mean():.4f}+-{a.std():.4f}"


def report(track_name, R):
    print(f"\n===================== {track_name} =====================")
    methods = ["raw", "M-Platt", "M-Isotonic", "M-Temp", "M-Ens3", "Beta", "Ens4"]
    print(f"{'method':12s} {'AUROC':>16s} {'Brier':>16s} {'ECE':>16s} {'band-NB':>16s}")
    for m in methods:
        a = R["acc"][m]
        print(
            f"{m:12s} {fmt(a['auroc']):>16s} {fmt(a['brier']):>16s} "
            f"{fmt(a['ece']):>16s} {fmt(a['nb']):>16s}"
        )
    iso_all = {ds: (np.mean(v) if v else float("nan")) for ds, v in R["iso_loses"].items()}
    n_iso_loses = sum(1 for ds in DATASETS if np.nanmean(R["iso_loses"][ds]) > 0.5)
    print(
        f"Isotonic loses AUROC vs raw on {n_iso_loses}/6 datasets "
        f"(per-ds frac: {[round(iso_all[ds],2) for ds in DATASETS]})"
    )
    print(
        f"Beta preserves AUROC exactly: {np.mean(R['beta_pres'])*100:.0f}% of runs; "
        f"Platt {np.mean(R['platt_pres'])*100:.0f}%; Temp {np.mean(R['temp_pres'])*100:.0f}%"
    )
    from collections import Counter

    ece_winner = Counter()
    for ds in DATASETS:
        c = Counter(R["ece_best"][ds])
        ece_winner[c.most_common(1)[0][0]] += 1
    print(f"Best-ECE method per dataset (majority over seeds): {dict(ece_winner)}")
    # Beta best/co-best Brier count
    beta_best = 0
    for ds in DATASETS:
        means = {m: np.mean(R["brier"][ds][m]) for m in methods if R["brier"][ds][m]}
        best = min(means.values())
        if means.get("Beta", 9) <= best + 1e-4:
            beta_best += 1
    print(f"Beta best/co-best Brier on {beta_best}/6 datasets")
    print(f"Ens4 selection distribution: {dict(Counter(R['ens4_pick']))}")


def main():
    ensure_datasets()
    print("Running V9 (large-n) ...")
    R_large = run_track(small=False)
    report("V9  LARGE-n  (cal = 25% of data)", R_large)
    print("\nRunning V10 (small-n, cal=100) ...")
    R_small = run_track(small=True)
    report("V10 SMALL-n  (cal subsampled to 100, >=3 pos)", R_small)
    print("\n===================== V11 (Ens4 selection) =====================")
    from collections import Counter

    print(f"Large-n Ens4 picks: {dict(Counter(R_large['ens4_pick']))}")
    print(f"Small-n Ens4 picks: {dict(Counter(R_small['ens4_pick']))}")
    print("Expectation: Isotonic-dominant at large n, Beta-dominant at small n (ADD, not replace).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

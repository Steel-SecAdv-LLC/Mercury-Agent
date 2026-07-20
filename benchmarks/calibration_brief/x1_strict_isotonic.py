# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""X1 -- Strict isotonic: kill the tie problem, keep the flexibility.

PRE-REGISTRATION (R1):
  H: centered isotonic regression (CIR), or isotonic + eps*s tie-break
     (eps=1e-6 strictly increasing perturbation), retains isotonic's ECE while
     restoring EXACT AUROC.
  Metric: AUROC preservation vs raw (|dAUROC| at 1e-9), ECE vs vanilla isotonic.
  Track : Standard Track real battery (6 ADBench, large + small, seeds 0..4)
          plus the V1 synthetic AUROC-invariance probe.
  KILL  : AUROC |delta| != 0 at 1e-9, OR ECE worse than vanilla isotonic by >10%.

Run:  python benchmarks/calibration_brief/x1_strict_isotonic.py
"""

from __future__ import annotations

import warnings

import numpy as np
from cal_core import auroc, ece, ensure_datasets, make_synth
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

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
EPS = 1e-6


class VanillaIsotonic:
    def fit(self, s, y):
        self.m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
        return self

    def calibrate(self, s):
        return self.m.predict(s)


class IsotonicEps:
    """Isotonic + eps*s strictly-increasing tie-break (restores rank order).

    NOTE: a naive ``clip(g + eps*s, 0, 1)`` re-saturates exactly the points
    isotonic flattened to {0,1}, destroying the tie-break.  Instead we squeeze g
    into the open interval [d, 1-d] (d=eps) so the perturbed output never leaves
    (0,1) and no clip is required -> strictly monotone in s by construction."""

    def fit(self, s, y):
        self.m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
        return self

    def calibrate(self, s):
        g = self.m.predict(s)
        d = EPS
        return d + (1.0 - 2.0 * d) * g + EPS * (np.asarray(s, dtype=float) - 0.5)


class CenteredIsotonic:
    """Centered isotonic regression (Oron & Flournoy): linear interpolation
    between the centroids of isotonic level sets -> strictly increasing, smoother."""

    def fit(self, s, y):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(s, kind="stable")
        ss, yy = s[order], y[order]
        fitted = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(ss, yy)
        g = fitted.predict(ss)
        # collapse level sets -> (centroid_s, level_value)
        xs, ys = [], []
        i = 0
        n = len(ss)
        while i < n:
            j = i
            while j + 1 < n and abs(g[j + 1] - g[i]) < 1e-12:
                j += 1
            xs.append(float(np.mean(ss[i : j + 1])))
            ys.append(float(g[i]))
            i = j + 1
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 2:  # degenerate: fall back to identity-ish
            xs = np.array([0.0, 1.0])
            ys = np.array([ys[0] if len(ys) else 0.5, ys[-1] if len(ys) else 0.5])
        # ensure strictly increasing x for interpolation
        keep = np.concatenate([[True], np.diff(xs) > 0])
        self.xs, self.ys = xs[keep], ys[keep]
        return self

    def calibrate(self, s):
        return np.clip(np.interp(np.asarray(s, dtype=float), self.xs, self.ys), 0.0, 1.0)


def synthetic_probe():
    """V1-style: strictly monotone calibrator must leave AUROC identical."""
    w = make_synth(42000, seed=7)
    cal, te = slice(0, 2000), slice(2000, 42000)
    a_raw = auroc(w.y[te], w.s[te])
    out = {}
    for name, M in [
        ("Vanilla", VanillaIsotonic),
        ("Iso+eps", IsotonicEps),
        ("CIR", CenteredIsotonic),
    ]:
        m = M().fit(w.s[cal], w.y[cal])
        a = auroc(w.y[te], m.calibrate(w.s[te]))
        out[name] = (a, abs(a - a_raw))
    return a_raw, out


def real_track():
    methods = ["Vanilla", "Iso+eps", "CIR"]
    auroc_pres = {m: [] for m in methods}  # |dAUROC| vs raw
    ece_vals = {m: [] for m in methods}
    for track_small in (False, True):
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
                te = np.arange(ntr + ncal, n)
                iso = IsolationForest(n_estimators=200, random_state=seed).fit(X[tr])
                raw = -iso.score_samples(X)
                lo, hi = raw[cal].min(), raw[cal].max()
                sc = np.clip((raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw), 0, 1)
                ci = cal
                if track_small:
                    ci = rng.choice(cal, size=min(100, len(cal)), replace=False)
                    while y[ci].sum() < 3:
                        ci = rng.choice(cal, size=min(100, len(cal)), replace=False)
                if len(np.unique(y[te])) < 2:
                    continue
                a_raw = auroc(y[te], sc[te])
                for name, M in [
                    ("Vanilla", VanillaIsotonic),
                    ("Iso+eps", IsotonicEps),
                    ("CIR", CenteredIsotonic),
                ]:
                    m = M().fit(sc[ci], y[ci])
                    p = m.calibrate(sc[te])
                    auroc_pres[name].append(abs(auroc(y[te], p) - a_raw))
                    ece_vals[name].append(ece(y[te], p))
    return auroc_pres, ece_vals


def main():
    ensure_datasets()
    print("=== X1 synthetic AUROC-invariance probe (V1 world) ===")
    a_raw, out = synthetic_probe()
    print(f"raw AUROC = {a_raw:.6f}")
    for name, (a, d) in out.items():
        print(f"  {name:9s} AUROC={a:.6f}  |delta|={d:.2e}")

    print("\n=== X1 real Standard Track (6 datasets x large+small x seeds 0..4) ===")
    ap, ev = real_track()
    print(
        f"{'method':9s} {'mean|dAUROC|':>13s} {'max|dAUROC|':>12s} {'meanECE':>9s} "
        f"{'exact-AUROC runs':>17s}"
    )
    base_ece = np.mean(ev["Vanilla"])
    verdict = {}
    for m in ["Vanilla", "Iso+eps", "CIR"]:
        mean_d = np.mean(ap[m])
        max_d = np.max(ap[m])
        mece = np.mean(ev[m])
        exact = np.mean(np.array(ap[m]) < 1e-9) * 100
        print(f"{m:9s} {mean_d:13.2e} {max_d:12.2e} {mece:9.4f} {exact:14.0f}%")
        if m != "Vanilla":
            restores = max_d < 1e-9
            ece_ok = mece <= base_ece * 1.10
            verdict[m] = (restores, ece_ok)

    print("\n=== VERDICT (Kill: not-exact-AUROC OR ECE > vanilla*1.10) ===")
    for m, (restores, ece_ok) in verdict.items():
        status = "SURVIVES" if (restores and ece_ok) else "KILLED"
        print(f"  {m:9s}: restores exact AUROC={restores}  ECE<=110% vanilla={ece_ok}  -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""X8 -- Settle Phi with data (resolves the in-repo UNJUSTIFIED flag and V12b).

PRE-REGISTRATION (R1, R3):
  H0: golden-ratio fusion weights are indistinguishable from learned weights.
  Metric: held-out fused-score AUROC (primary) + Brier, nested CV (weights chosen
          on cal, scored on test).  Paired over 6 datasets x seeds {0..4}.
  Compare: golden-A (Phi+2 -> 0.447/0.276/0.276), golden-B (2Phi -> 0.500/0.309/
           0.191), equal (1/3,1/3,1/3), learned-optimal (cal-AUROC argmax on a
           simplex grid).
  KILL: n/a -- ANY outcome resolves the flag (Phi survives => keep as documented
        default WITH this evidence; Phi loses => demote to initialisation).

CAVEAT (G3): the real 3R sub-scores (recursion/resonance/optimisation) need the
torch ThreeR stack; here R/H/O are PROXIED by three classical detectors
(IsolationForest / LOF / kNN-distance).  This tests whether the *weighting
constant* is special, NOT detection power -- no detection claim is made.

Run:  python benchmarks/calibration_brief/x8_phi_sweep.py
"""

from __future__ import annotations

import warnings

import numpy as np
from cal_core import PHI, auroc, brier, ensure_datasets
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

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

GOLDEN_A = np.array([PHI / (PHI + 2), 1 / (PHI + 2), 1 / (PHI + 2)])  # fusion.py:139
GOLDEN_B = np.array([PHI / (2 * PHI), 1 / (2 * PHI), (1 / PHI) / (2 * PHI)])  # fusion.py:367/430
EQUAL = np.array([1 / 3, 1 / 3, 1 / 3])


def simplex_grid(step=0.05):
    pts = []
    ks = round(1 / float(step))  # float() -> Python int from round(), robust to a numpy-scalar step
    for i in range(ks + 1):
        for j in range(ks + 1 - i):
            k = ks - i - j
            pts.append(np.array([i, j, k]) / ks)
    return np.array(pts)


def three_proxy_scores(Xtr, Xall, seed):
    """R/H/O proxies: IsolationForest, LOF, kNN-distance.  Higher = more anomalous."""
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    r = -iso.score_samples(Xall)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(Xtr) - 1), novelty=True).fit(Xtr)
    h = -lof.score_samples(Xall)
    k = min(10, len(Xtr) - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    o = nn.kneighbors(Xall)[0].mean(axis=1)
    return np.column_stack([r, h, o])


def norm_cols(M, ref):
    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((M - lo) / rng, 0, 1)


def main():
    ensure_datasets()
    grid = simplex_grid(0.05)
    rows = {"golden-A": [], "golden-B": [], "equal": [], "learned": []}
    brows = {k: [] for k in rows}
    learned_pts = []
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
            S = three_proxy_scores(X[tr], X, seed)
            Sn = norm_cols(S, S[cal])
            if len(np.unique(y[te])) < 2 or len(np.unique(y[cal])) < 2:
                continue
            # learned-optimal weights: argmax cal AUROC over simplex grid
            cal_aurocs = np.array([auroc(y[cal], Sn[cal] @ w) for w in grid])
            w_star = grid[int(np.argmax(cal_aurocs))]
            learned_pts.append(w_star)
            for name, w in [
                ("golden-A", GOLDEN_A),
                ("golden-B", GOLDEN_B),
                ("equal", EQUAL),
                ("learned", w_star),
            ]:
                fused = Sn[te] @ w
                rows[name].append(auroc(y[te], fused))
                brows[name].append(brier(y[te], fused / fused.max() if fused.max() > 0 else fused))

    print("=== X8 Phi sweep (3-detector proxy; held-out fused AUROC) ===")
    print(f"{'weights':10s} {'test AUROC (mean+-sd)':>24s}")
    base = np.array(rows["golden-A"])
    for name in ["golden-A", "golden-B", "equal", "learned"]:
        a = np.array(rows[name])
        print(f"{name:10s} {a.mean():.4f} +- {a.std():.4f}")
    learned = np.array(rows["learned"])
    diff = learned - base
    print(
        f"\nlearned - golden-A : mean {diff.mean():+.4f}  sd {diff.std():.4f}  "
        f"(paired over {len(diff)} runs)"
    )
    # simple paired t-like effect size
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    t = diff.mean() / se if se > 0 else float("nan")
    print(f"paired effect: t~{t:.2f}  (|t|<2 => golden indistinguishable from learned)")
    lp = np.array(learned_pts)
    print(
        f"\nlearned-optimum centroid (w_R,w_H,w_O) = "
        f"({lp[:, 0].mean():.3f},{lp[:, 1].mean():.3f},{lp[:, 2].mean():.3f})"
    )
    print(f"  golden-A = ({GOLDEN_A[0]:.3f},{GOLDEN_A[1]:.3f},{GOLDEN_A[2]:.3f})")
    print(f"  golden-B = ({GOLDEN_B[0]:.3f},{GOLDEN_B[1]:.3f},{GOLDEN_B[2]:.3f})")
    dist_a = np.linalg.norm(lp.mean(axis=0) - GOLDEN_A)
    print(f"  ||centroid - golden-A|| = {dist_a:.3f}")
    print(
        "\nRESOLUTION: "
        + (
            "Phi INDISTINGUISHABLE from learned (|t|<2) -> keep as documented default, "
            "BUT reconcile to ONE derivation; it is an initialisation, not a proven optimum."
            if abs(t) < 2
            else "learned SIGNIFICANTLY beats golden (|t|>=2) -> DEMOTE Phi to initialisation."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

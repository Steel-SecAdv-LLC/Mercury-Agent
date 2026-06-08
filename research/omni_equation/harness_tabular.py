"""Real-signal measurement harness for the Mercury omni-equation decision.

Runs the REAL Mercury detector (MercuryAnomalyDetector -> resonance / kinematic /
info-geometry streams) on REAL datasets (omni_mercury_engine.datasets.load_dataset),
then answers the questions that gate the omni-equation build:

  Q-ACC1  Are Mercury's existing streams independent, or redundant?  (corr matrix)
  Q-ACC2  Does fusing the streams beat the best single stream?       (lift vs best)
  Q-ACC3  Does the real outer equation S=w_N*N+w_E*OAE beat raw N?    (equation lift)
  Q-GATE  Does the eta^Phi ethics gate ever change a verdict?         (flip rate)

No reimplementation: real detector, real datasets, real equation_profiles.
Protocol: standard transductive AD (fit unlabeled on X, score X, AUROC vs y).
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
# Mercury builds its own ML (mercury_ml); scikit-learn is never imported in the
# Mercury codebase — it is only a conceptual benchmark baseline, not a dependency.
from omni_mercury_engine.core.equation_profiles import score_runtime_equation_profile
from omni_mercury_engine.datasets import load_dataset
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import (
    LogisticRegression,
    StandardScaler,
    roc_auc_score,
)

DATASETS = ["thyroid", "campaign", "fraud", "backdoor", "smd", "epilepsy", "dsads", "donors"]
MAX_SAMPLES = 4000
RNG = np.random.default_rng(7)


def auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def run_one(name):
    t0 = time.time()
    X, y, meta = load_dataset(name, max_samples=MAX_SAMPLES)
    X = np.asarray(X, float); y = np.asarray(y).astype(int).reshape(-1)
    synthetic = bool(meta.get("synthetic", meta.get("is_synthetic", False)))
    Xs = StandardScaler().fit_transform(X)

    det = MercuryAnomalyDetector()
    det.fit(Xs)
    res = det.detect(Xs)
    keys = list(res.keys()) if isinstance(res, dict) else type(res)
    # component extraction (defensive about key names)
    def grab(*cands):
        for c in cands:
            if isinstance(res, dict) and c in res and res[c] is not None:
                v = np.asarray(res[c], float).reshape(-1)
                if v.shape[0] == y.shape[0]:
                    return v
        return None
    N   = grab("scores", "anomaly_scores", "score")
    rR  = grab("resonance_scores", "resonance")
    rH  = grab("kinematic_scores", "kinematic")
    rO  = grab("info_geometry_scores", "info_geometry", "infogeometry_scores")
    comps = {"resonance": rR, "kinematic": rH, "info_geom": rO}
    comps = {k: v for k, v in comps.items() if v is not None}

    out = {"dataset": name, "n": int(X.shape[0]), "d": int(X.shape[1]),
           "anom_rate": round(float(y.mean()), 4), "synthetic": synthetic,
           "secs": round(time.time() - t0, 1), "keys": keys}
    if N is None:
        out["error"] = "no ensemble score key"; return out

    out["auroc_ensemble_N"] = round(auroc(y, N), 4)
    comp_auroc = {k: round(auroc(y, v), 4) for k, v in comps.items()}
    out["auroc_components"] = comp_auroc
    if comp_auroc:
        out["best_single"] = round(max(comp_auroc.values()), 4)

    # Q-ACC1: stream independence (|corr| among components)
    if len(comps) >= 2:
        M = np.vstack(list(comps.values()))
        C = np.corrcoef(M)
        offdiag = C[np.triu_indices(len(comps), 1)]
        out["mean_abs_corr"] = round(float(np.mean(np.abs(offdiag))), 3)

    # Q-ACC2: fusion vs best single  (mean fusion + ORACLE logistic upper bound)
    if len(comps) >= 2:
        F = np.vstack(list(comps.values())).T
        out["auroc_fuse_mean"] = round(auroc(y, F.mean(1)), 4)
        try:
            lr = LogisticRegression(max_iter=500).fit(StandardScaler().fit_transform(F), y)
            p = lr.predict_proba(StandardScaler().fit_transform(F))[:, 1]
            out["auroc_fuse_oracle"] = round(auroc(y, p), 4)   # uses labels = upper bound
            out["lift_fusion_oracle"] = round(out["auroc_fuse_oracle"] - out["best_single"], 4)
        except Exception as e:
            out["oracle_err"] = str(e)[:60]

    # Q-ACC3: real outer equation vs raw N
    # slot mapping mirrors the benchmark: recursion<-resonance, resonance<-kinematic, optimization<-info_geom
    def comp_or_N(v): return v if v is not None else N
    S_base, _ = score_runtime_equation_profile(
        N, comp_or_N(rR), comp_or_N(rH), comp_or_N(rO), eta=0.96, profile_id="baseline_original_v1")
    S_phi, _ = score_runtime_equation_profile(
        N, comp_or_N(rR), comp_or_N(rH), comp_or_N(rO), eta=0.96, profile_id="phi_fibring_v1")
    out["auroc_eq_baseline"] = round(auroc(y, S_base), 4)
    out["auroc_eq_phi_fibring"] = round(auroc(y, S_phi), 4)
    out["lift_equation_vs_N"] = round(out["auroc_eq_baseline"] - out["auroc_ensemble_N"], 4)

    # Q-GATE: does eta^Phi change a verdict? gate(0.96) vs no-gate(1.0)
    Sg, _ = score_runtime_equation_profile(N, comp_or_N(rR), comp_or_N(rH), comp_or_N(rO),
                                           eta=0.96, profile_id="baseline_original_v1")
    Sn, _ = score_runtime_equation_profile(N, comp_or_N(rR), comp_or_N(rH), comp_or_N(rO),
                                           eta=1.0, profile_id="baseline_original_v1")
    k = max(1, int(y.sum()))
    topk_g = set(np.argsort(-Sg)[:k]); topk_n = set(np.argsort(-Sn)[:k])
    out["gate_topk_flip"] = round(1 - len(topk_g & topk_n) / k, 4)           # rank-based verdict
    out["gate_abs0.5_flip"] = round(float(np.mean((Sg > 0.5) != (Sn > 0.5))), 4)  # absolute thresh
    out["auroc_gate_delta"] = round(auroc(y, Sg) - auroc(y, Sn), 6)
    return out


def main():
    rows = []
    for name in DATASETS:
        try:
            r = run_one(name)
        except Exception as e:
            r = {"dataset": name, "error": f"{type(e).__name__}: {str(e)[:140]}"}
        rows.append(r)
        print(json.dumps(r))
    with open("/home/user/eqlab/results.json", "w") as f:
        json.dump(rows, f, indent=2)

    ok = [r for r in rows if "auroc_ensemble_N" in r]
    if ok:
        print("\n==== SUMMARY (real Mercury detector + real datasets) ====")
        print(f"{'dataset':10} {'syn':3} {'N':>6} {'best1':>6} {'fuse_mean':>9} "
              f"{'fuse_orac':>9} {'eq_base':>7} {'eqΔvsN':>7} {'|corr|':>6} {'gate_rank':>9} {'gateΔAUC':>8}")
        for r in ok:
            print(f"{r['dataset'][:10]:10} {str(r['synthetic'])[0]:3} "
                  f"{r['auroc_ensemble_N']:>6} {r.get('best_single','-'):>6} "
                  f"{r.get('auroc_fuse_mean','-'):>9} {r.get('auroc_fuse_oracle','-'):>9} "
                  f"{r.get('auroc_eq_baseline','-'):>7} {r.get('lift_equation_vs_N','-'):>7} "
                  f"{r.get('mean_abs_corr','-'):>6} {r.get('gate_topk_flip','-'):>9} "
                  f"{r.get('auroc_gate_delta','-'):>8}")
        import statistics as st
        def avg(key):
            vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
            return round(st.mean(vals), 4) if vals else float("nan")
        print("\nmeans:",
              "N=", avg("auroc_ensemble_N"),
              "best_single=", avg("best_single"),
              "fuse_oracle=", avg("auroc_fuse_oracle"),
              "eq_baseline=", avg("auroc_eq_baseline"),
              "lift_eq_vs_N=", avg("lift_equation_vs_N"),
              "lift_fusion_oracle=", avg("lift_fusion_oracle"),
              "|corr|=", avg("mean_abs_corr"),
              "gate_rank_flip=", avg("gate_topk_flip"),
              "gate_abs_flip=", avg("gate_abs0.5_flip"))


if __name__ == "__main__":
    main()

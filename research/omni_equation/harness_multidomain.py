"""Per-stream measurement on REAL multi-domain Mercury data.

Real loaders -> real MercuryAnomalyDetector -> real equation_profiles.
Answers:
 Q-ACC1 stream redundancy: |corr| among the detector's internal streams (label-free)
 Q-ACC2 does Mercury's ensemble beat its best single stream?  (lift_ens_vs_best)
 Q-ACC3 does the real outer equation S=w_N*N+w_E*OAE beat raw ensemble N? (lift_eq_vs_N)
 Q-GATE does eta^Phi ever change a verdict? (rank top-k flip / abs-0.5 flip / AUC delta)
"""
from __future__ import annotations
import warnings, json, importlib
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from omni_mercury_engine.core.equation_profiles import score_runtime_equation_profile

stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
MercuryAnomalyDetector = stat.MercuryAnomalyDetector

LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "wildfire":   ("wildfire_loader", "WildfireLoader"),
    "hurricane":  ("hurricane_loader", "HurricaneLoader"),
    "flood":      ("flood_loader", "FloodLoader"),
    "marine":     ("marine_loader", "MarineLoader"),
    "tornado":    ("tornado_loader", "TornadoLoader"),
    "volcanic":   ("volcanic_loader", "VolcanicLoader"),
    "tsunami":    ("tsunami_loader", "TsunamiLoader"),
    "landslide":  ("landslide_loader", "LandslideLoader"),
}
STREAM_KEYS = ["resonance_scores", "kinematic_scores", "info_geometry_scores",
               "z_score_continuous", "iqr_scores", "isolation_forest_scores"]


def auroc(y, s):
    y = np.asarray(y).astype(int).reshape(-1); s = np.asarray(s, float).reshape(-1)
    if y.min() == y.max() or len(np.unique(s)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def run_event(feats, y):
    det = MercuryAnomalyDetector(); det.fit(feats); R = det.detect(feats)
    if not isinstance(R, dict) or "scores" not in R:
        return None
    n = len(y)
    N = np.asarray(R["scores"], float).reshape(-1)
    streams = {}
    for k in STREAM_KEYS:
        if k in R:
            v = np.asarray(R[k], float).reshape(-1)
            if v.shape[0] == n and np.ptp(v) > 0:
                streams[k] = v
    rec = {"n": int(n), "anoms": int(np.sum(np.asarray(y) == 1)), "n_streams": len(streams)}
    rec["auc_N"] = auroc(y, N)
    sa = {k: auroc(y, v) for k, v in streams.items()}
    rec["stream_aucs"] = {k: round(v, 3) for k, v in sa.items() if v == v}
    valid = [v for v in sa.values() if v == v]
    rec["best_single"] = round(max(valid), 3) if valid else float("nan")
    if len(streams) >= 2:
        M = np.vstack(list(streams.values())); C = np.corrcoef(M)
        od = C[np.triu_indices(len(streams), 1)]; od = od[np.isfinite(od)]
        rec["mean_abs_corr"] = round(float(np.mean(np.abs(od))), 3) if len(od) else float("nan")
    g = lambda k: streams.get(k, N)
    Sb, _ = score_runtime_equation_profile(N, g("resonance_scores"), g("kinematic_scores"),
                                           g("info_geometry_scores"), eta=0.96, profile_id="baseline_original_v1")
    Sp, _ = score_runtime_equation_profile(N, g("resonance_scores"), g("kinematic_scores"),
                                           g("info_geometry_scores"), eta=0.96, profile_id="phi_fibring_v1")
    Sn, _ = score_runtime_equation_profile(N, g("resonance_scores"), g("kinematic_scores"),
                                           g("info_geometry_scores"), eta=1.0, profile_id="baseline_original_v1")
    rec["auc_eq_base"] = auroc(y, Sb); rec["auc_eq_phi"] = auroc(y, Sp)
    if rec["auc_N"] == rec["auc_N"]:
        rec["lift_eq_vs_N"] = round(rec["auc_eq_base"] - rec["auc_N"], 3)
    if rec["best_single"] == rec["best_single"] and rec["auc_N"] == rec["auc_N"]:
        rec["lift_ens_vs_best"] = round(rec["auc_N"] - rec["best_single"], 3)
    k = max(1, int(np.sum(np.asarray(y) == 1)))
    tg = set(np.argsort(-Sb)[:k]); tn = set(np.argsort(-Sn)[:k])
    rec["gate_topk_flip"] = round(1 - len(tg & tn) / k, 3)
    rec["gate_abs_flip"] = round(float(np.mean((Sb > 0.5) != (Sn > 0.5))), 4)
    rec["gate_auc_delta"] = round(auroc(y, Sb) - auroc(y, Sn), 6)
    return rec


def main():
    out = {}
    for dom, (mod, cls) in LOADERS.items():
        try:
            L = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{mod}"), cls)()
            events = L.list_events()
        except Exception as e:
            print(dom, "LOADER_ERR", str(e)[:110], flush=True); continue
        ev_recs = []
        for ev in events[:6]:
            eid = ev.get("event_id") if isinstance(ev, dict) else str(ev)
            try:
                raw = L.fetch_historical(eid)
                feats = np.asarray(L.engineer_features(raw), float)
                y = np.asarray(L.get_ground_truth(eid)).astype(int).reshape(-1)
                m = min(len(feats), len(y)); feats = feats[:m]; y = y[:m]
                if len(feats) == 0 or y.min() == y.max():
                    print(dom, eid, "skip(no_var)", flush=True); continue
                r = run_event(feats, y)
                if r is None:
                    print(dom, eid, "skip(no_scores)", flush=True); continue
                r["event"] = eid; ev_recs.append(r)
                print(dom, eid, "auc_N=%.3f best=%.3f |corr|=%s eqd=%s gate_rank=%s gate_abs=%s gAUCd=%s"
                      % (r["auc_N"], r["best_single"], r.get("mean_abs_corr"),
                         r.get("lift_eq_vs_N"), r["gate_topk_flip"], r["gate_abs_flip"],
                         r["gate_auc_delta"]), flush=True)
            except Exception as e:
                print(dom, eid, "EVENT_ERR", type(e).__name__, str(e)[:90], flush=True)
        if ev_recs:
            out[dom] = ev_recs
    json.dump(out, open("/home/user/eqlab/real_results.json", "w"), indent=2, default=str)
    rows = [(d, r) for d, v in out.items() if isinstance(v, list) for r in v]

    def mean(key):
        xs = [r[key] for _, r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return round(float(np.mean(xs)), 4) if xs else float("nan")

    print("\n==== AGGREGATE  %d real events / %d domains ====" % (len(rows), len(set(d for d, _ in rows))), flush=True)
    print("mean auc_N (ensemble)                              =", mean("auc_N"))
    print("mean best_single stream                            =", mean("best_single"))
    print("mean lift_ens_vs_best (ensemble - best single)     =", mean("lift_ens_vs_best"))
    print("mean lift_eq_vs_N     (equation  - ensemble)       =", mean("lift_eq_vs_N"))
    print("mean |corr| among streams (0=indep 1=redundant)    =", mean("mean_abs_corr"))
    print("mean gate_topk_flip  (rank verdict change)         =", mean("gate_topk_flip"))
    print("mean gate_abs_flip   (abs-0.5 verdict change)      =", mean("gate_abs_flip"))
    print("mean gate_auc_delta  (AUC gate - nogate)           =", mean("gate_auc_delta"))


if __name__ == "__main__":
    main()

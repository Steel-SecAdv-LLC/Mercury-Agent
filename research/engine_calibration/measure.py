"""Canonical before/after measurement on cached real events (single pass).

Scores are deterministic, so before/after differ ONLY in the operating point:
  BEFORE = fixed 0.5 cut on the ensemble scores (prior default)
  AFTER  = detector's is_anomaly (distribution-adaptive cut, this change)
AUROC/AUPRC are rank metrics from the (identical) scores -> reported once.
Writes research/engine_calibration/results.json.
"""
from __future__ import annotations
import warnings, glob, os, json, importlib
warnings.filterwarnings("ignore")
import numpy as np
from omni_mercury_engine.ml.mercury_ml import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
)

CACHE = "/home/user/eqlab/cache"


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    if y.min() == y.max() or len(np.unique(s)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def ap(y, s):
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, np.asarray(s, float))) if y.sum() else float("nan")


def prf(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


def main():
    stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
    MAD = stat.MercuryAnomalyDetector
    rows = []
    for fn in sorted(glob.glob(f"{CACHE}/*.npz")):
        dom = os.path.basename(fn).split("__")[0]
        eid = os.path.basename(fn)[len(dom) + 2:-4]
        d = np.load(fn); X = d["X"]; y = d["y"].astype(int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.min() == y.max():
            continue
        det = MAD(); det.fit(X); R = det.detect(X)
        s = np.asarray(R["scores"], float).reshape(-1)
        after = np.asarray(R["is_anomaly"]).astype(int).reshape(-1)   # adaptive cut
        before = (s > 0.5).astype(int)                                # prior fixed cut
        bf = prf(y, before); af = prf(y, after)
        rows.append({"dom": dom, "event": eid, "n": int(len(y)), "pos": int(y.sum()),
                     "auroc": auc(y, s), "auprc": ap(y, s),
                     "f1_before": bf[0], "prec_before": bf[1], "rec_before": bf[2],
                     "f1_after": af[0], "prec_after": af[1], "rec_after": af[2],
                     "thr_after": float(R.get("threshold", float("nan")))})

    doms = sorted(set(r["dom"] for r in rows))

    def mean(key, dom=None):
        xs = [r[key] for r in rows if (dom is None or r["dom"] == dom)
              and isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return float(np.mean(xs)) if xs else float("nan")

    print(f"\n==== CANONICAL BEFORE/AFTER  {len(rows)} events / {len(doms)} domains ====")
    hdr = (f"{'domain':12s} {'nev':>3s} {'AUROC':>6s} {'AUPRC':>6s} | "
           f"{'F1.b':>6s} {'F1.a':>6s} | {'prec.b':>6s} {'prec.a':>6s} | {'rec.b':>6s} {'rec.a':>6s}")
    print(hdr)
    for dom in doms:
        nev = sum(1 for r in rows if r["dom"] == dom)
        print(f"{dom:12s} {nev:3d} {mean('auroc',dom):6.3f} {mean('auprc',dom):6.3f} | "
              f"{mean('f1_before',dom):6.3f} {mean('f1_after',dom):6.3f} | "
              f"{mean('prec_before',dom):6.3f} {mean('prec_after',dom):6.3f} | "
              f"{mean('rec_before',dom):6.3f} {mean('rec_after',dom):6.3f}")
    print("-" * len(hdr))
    print(f"{'OVERALL':12s} {len(rows):3d} {mean('auroc'):6.3f} {mean('auprc'):6.3f} | "
          f"{mean('f1_before'):6.3f} {mean('f1_after'):6.3f} | "
          f"{mean('prec_before'):6.3f} {mean('prec_after'):6.3f} | "
          f"{mean('rec_before'):6.3f} {mean('rec_after'):6.3f}")

    summary = {"n_events": len(rows), "domains": doms,
               "overall": {k: mean(k) for k in
                           ["auroc", "auprc", "f1_before", "f1_after",
                            "prec_before", "prec_after", "rec_before", "rec_after"]},
               "per_domain": {dom: {k: mean(k, dom) for k in
                              ["auroc", "auprc", "f1_before", "f1_after",
                               "prec_before", "prec_after", "rec_before", "rec_after"]}
                              for dom in doms},
               "events": rows}
    out = os.path.join(os.path.dirname(__file__), "results.json")
    json.dump(summary, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

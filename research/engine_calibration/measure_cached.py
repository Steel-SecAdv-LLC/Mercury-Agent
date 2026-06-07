"""Canonical before/after measurement using cached scores + the REAL detector.

Scores are rank-preserving under the calibration change, so the cached per-event
scores (scores.pkl, one detector pass) are reused and the detector's actual
_adaptive_operating_point() is applied to them. This reproduces the honest-
benchmark operating-point metrics exactly (verified against the live scripts)
without re-running detect() on the 148k-row NSL-KDD event.

  BEFORE = fixed 0.5 cut   AFTER = MercuryAnomalyDetector._adaptive_operating_point
AUROC/AUPRC are rank metrics (unchanged by the cut). Writes results.json.
"""
from __future__ import annotations
import warnings, pickle, os, json, importlib
warnings.filterwarnings("ignore")
import numpy as np
from omni_mercury_engine.ml.mercury_ml import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
)

stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
DET = stat.MercuryAnomalyDetector()
S = pickle.load(open("/home/user/eqlab/scores.pkl", "rb"))
doms = sorted({d for d, _ in S})


def _prf(y, pred):
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


rows = []
for (dom, eid), r in S.items():
    s = np.asarray(r["scores"], float); y = np.asarray(r["y"], int)
    thr, _ = DET._adaptive_operating_point(s)          # real detector logic
    bf = _prf(y, (s > 0.5).astype(int))
    af = _prf(y, (s > thr).astype(int))
    rows.append({
        "dom": dom, "event": eid, "n": int(y.size), "pos": int(y.sum()),
        "subsampled": bool(r.get("subsampled", False)),
        "auroc": float(roc_auc_score(y, s)) if y.min() != y.max() and len(np.unique(s)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, s)) if y.sum() else float("nan"),
        "f1_before": bf[0], "prec_before": bf[1], "rec_before": bf[2],
        "f1_after": af[0], "prec_after": af[1], "rec_after": af[2],
    })


def m(key, dom=None):
    xs = [r[key] for r in rows if (dom is None or r["dom"] == dom)
          and isinstance(r.get(key), (int, float)) and r[key] == r[key]]
    return float(np.mean(xs)) if xs else float("nan")


hdr = (f"{'domain':16s} {'nev':>3s} {'AUROC':>6s} {'AUPRC':>6s} | "
       f"{'F1.b':>5s} {'F1.a':>5s} | {'pr.b':>5s} {'pr.a':>5s} | {'rc.b':>5s} {'rc.a':>5s}")
print(f"events={len(rows)}  domains={len(doms)}  (unreachable excluded: wildfire/flood/volcanic/landslide/financial/sepsis)")
print(hdr)
for d in doms:
    nev = sum(1 for r in rows if r["dom"] == d)
    print(f"{d:16s} {nev:3d} {m('auroc',d):6.3f} {m('auprc',d):6.3f} | "
          f"{m('f1_before',d):5.3f} {m('f1_after',d):5.3f} | "
          f"{m('prec_before',d):5.3f} {m('prec_after',d):5.3f} | "
          f"{m('rec_before',d):5.3f} {m('rec_after',d):5.3f}")
print("-" * len(hdr))
print(f"{'OVERALL':16s} {len(rows):3d} {m('auroc'):6.3f} {m('auprc'):6.3f} | "
      f"{m('f1_before'):5.3f} {m('f1_after'):5.3f} | "
      f"{m('prec_before'):5.3f} {m('prec_after'):5.3f} | "
      f"{m('rec_before'):5.3f} {m('rec_after'):5.3f}")

summary = {
    "events": len(rows), "domains": doms,
    "overall": {k: round(m(k), 4) for k in
                ["auroc", "auprc", "f1_before", "f1_after",
                 "prec_before", "prec_after", "rec_before", "rec_after"]},
    "per_domain": {d: {k: round(m(k, d), 4) for k in
                   ["auroc", "auprc", "f1_before", "f1_after",
                    "prec_before", "prec_after", "rec_before", "rec_after"]} for d in doms},
    "per_event": rows,
}
out = os.path.join(os.path.dirname(__file__), "results.json")
json.dump(summary, open(out, "w"), indent=2, default=float)
print(f"\nwrote {out}")

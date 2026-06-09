# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline evaluation on cached real events.

Runs the REAL MercuryAnomalyDetector (fit+detect) on cached (X,y) events and
reports the suite-level metrics the task asks for:
  - AUROC, AUPRC  (rank metrics, from detection["scores"])
  - operating-point F1 / precision / recall (from detection["is_anomaly"])
Aggregated per-domain and overall.  Picks up any in-place code change to the
engine because it imports the detector fresh and uses the public detect() path.

Usage:
  eval_offline.py            -> current engine operating point
  eval_offline.py --sweep    -> also compare candidate unsupervised thresholds
"""

from __future__ import annotations
import warnings, glob, os, sys, importlib
warnings.filterwarnings("ignore")
import numpy as np
from omni_mercury_engine.ml.mercury_ml import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
)

CACHE = "/home/user/eqlab/cache"


def _auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    if y.min() == y.max() or len(np.unique(s)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _ap(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    try:
        return float(average_precision_score(y, np.asarray(s, float)))
    except Exception:
        return float("nan")


def _prf(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


# ---- candidate unsupervised threshold rules (label-free) -------------------
def _rule_thresholds(s):
    s = np.asarray(s, float)
    med = np.median(s); mad = np.median(np.abs(s - med))
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    mu, sd = s.mean(), s.std()
    out = {"fixed0.5": 0.5}
    for p in (90, 95, 97, 98, 99):
        out[f"pct{p}"] = float(np.percentile(s, p))
    for k in (2.0, 2.5, 3.0, 3.5):
        out[f"mad{k}"] = float(med + k * 1.4826 * mad) if mad > 1e-12 else float(np.percentile(s, 97))
    out["iqr1.5"] = float(q3 + 1.5 * iqr)
    out["mu+2sd"] = float(mu + 2 * sd)
    out["mu+3sd"] = float(mu + 3 * sd)
    # contamination estimate via IQR upper fence -> percentile
    est = float(np.clip(np.mean(s > (q3 + 1.5 * iqr)), 0.005, 0.5)) if iqr > 1e-12 else 0.05
    out["estfence"] = float(np.percentile(s, 100 * (1 - est)))
    return out


def main():
    sweep = "--sweep" in sys.argv
    stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
    importlib.reload(stat)
    MAD = stat.MercuryAnomalyDetector

    files = sorted(glob.glob(f"{CACHE}/*.npz"))
    rows = []
    rule_acc: dict[str, list] = {}
    for fn in files:
        dom = os.path.basename(fn).split("__")[0]
        eid = os.path.basename(fn)[len(dom) + 2:-4]
        d = np.load(fn); X = d["X"]; y = d["y"].astype(int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.min() == y.max():
            continue
        det = MAD(); det.fit(X); R = det.detect(X)
        s = np.asarray(R["scores"], float).reshape(-1)
        pred = np.asarray(R["is_anomaly"]).astype(int).reshape(-1)
        f1, pr, rc = _prf(y, pred)
        rec = {"dom": dom, "event": eid, "n": len(y), "pos": int(y.sum()),
               "auroc": _auc(y, s), "auprc": _ap(y, s),
               "f1": f1, "prec": pr, "rec": rc}
        rows.append(rec)
        if sweep:
            for name, thr in _rule_thresholds(s).items():
                ff, _, _ = _prf(y, (s > thr).astype(int))
                rule_acc.setdefault(name, []).append(ff)
            # oracle ceiling
            best = 0.0
            for thr in np.unique(s):
                ff, _, _ = _prf(y, (s >= thr).astype(int))
                best = max(best, ff)
            rule_acc.setdefault("ORACLE", []).append(best)

    def m(key, subset=None):
        xs = [r[key] for r in rows if (subset is None or r["dom"] == subset)
              and isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return float(np.mean(xs)) if xs else float("nan")

    doms = sorted(set(r["dom"] for r in rows))
    print(f"\n==== OFFLINE EVAL  {len(rows)} events / {len(doms)} domains ====")
    print(f"{'domain':12s} {'nev':>3s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'prec':>7s} {'rec':>7s}")
    for dom in doms:
        nev = sum(1 for r in rows if r["dom"] == dom)
        print(f"{dom:12s} {nev:3d} {m('auroc',dom):7.3f} {m('auprc',dom):7.3f} "
              f"{m('f1',dom):7.3f} {m('prec',dom):7.3f} {m('rec',dom):7.3f}")
    print("-" * 56)
    print(f"{'OVERALL':12s} {len(rows):3d} {m('auroc'):7.3f} {m('auprc'):7.3f} "
          f"{m('f1'):7.3f} {m('prec'):7.3f} {m('rec'):7.3f}")

    if sweep:
        print("\n---- candidate unsupervised threshold rules (mean F1 over all events) ----")
        for name in sorted(rule_acc, key=lambda k: -np.mean(rule_acc[k])):
            print(f"  {name:10s} meanF1={np.mean(rule_acc[name]):.4f}")
    return rows


if __name__ == "__main__":
    main()

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake-off of unsupervised threshold methods on cached real events.

For each cached event: run the REAL detector -> scores s, labels y.  Then apply
each LABEL-FREE threshold method and score operating-point F1/precision/recall.
Aggregates mean F1 per method (overall + per-domain) and the ORACLE ceiling.
Goal: pick the unsupervised rule that best converts good ranking -> good F1
across the full reachable suite (contamination 0.3%..72%).
"""

from __future__ import annotations
import warnings, glob, os, importlib
warnings.filterwarnings("ignore")
import numpy as np
from omni_mercury_engine.ml.mercury_ml import f1_score, precision_score, recall_score
from omni_mercury_engine.core.score_calibration import (
    AutoThresholdOptimizer, CalibrationMethod, ScoreDiagnostics,
)

CACHE = "/home/user/eqlab/cache"
_OPT = AutoThresholdOptimizer(default_contamination=0.05,
                              min_contamination=0.001, max_contamination=0.5)


def prf(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


# ---- candidate label-free threshold methods -> threshold value -------------
def scm(method, contam):
    def f(s):
        return float(_OPT.optimize(scores=np.asarray(s, float), method=method,
                                   contamination=contam).threshold)
    return f


def est_contam(s):
    """robust contamination estimate handling low & high contamination."""
    q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
    if iqr < 1e-12:
        return 0.05
    return float(np.clip(np.mean(s > (q3 + 1.5 * iqr)), 0.005, 0.5))


def mad_k(k):
    def f(s):
        med = np.median(s); mad = np.median(np.abs(s - med))
        if mad < 1e-12:
            return float(np.percentile(s, 97))
        return float(med + k * 1.4826 * mad)
    return f


def gap_split(s):
    """largest relative gap in the upper tail of sorted scores (1-D natural break)."""
    ss = np.sort(np.asarray(s, float))
    n = len(ss)
    if n < 8:
        return float(np.percentile(s, 95))
    lo = int(n * 0.5)  # search the upper half only
    gaps = np.diff(ss[lo:])
    if len(gaps) == 0 or gaps.max() <= 0:
        return float(np.percentile(s, 95))
    j = lo + int(np.argmax(gaps))
    return float((ss[j] + ss[j + 1]) / 2.0)


def _otsu(s):
    s = np.asarray(s, float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return float(hi)
    norm = ((s - lo) / (hi - lo) * 255).astype(int)
    hist = np.bincount(norm, minlength=256).astype(float)
    tot = hist.sum(); sumT = np.dot(np.arange(256), hist)
    wB = 0.0; sumB = 0.0; best = 0.0; bt = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = tot - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB; mF = (sumT - sumB) / wF
        v = wB * wF * (mB - mF) ** 2
        if v > best:
            best = v; bt = t
    return float(lo + (bt / 255) * (hi - lo))


def _valley_depth(s, t):
    """1 - density(at t)/peak_density.  ~1 => deep valley (bimodal); ~0 => unimodal."""
    s = np.asarray(s, float)
    nb = int(np.clip(np.sqrt(len(s)), 10, 40))
    hist, edges = np.histogram(s, bins=nb)
    if hist.max() == 0:
        return 0.0
    bi = int(np.clip(np.searchsorted(edges, t) - 1, 0, len(hist) - 1))
    return float(1.0 - hist[bi] / hist.max())


def adaptive(s, vd=0.45, low="gap"):
    """regime-adaptive: deep valley => Otsu (high-contam mode); else selective tail."""
    s = np.asarray(s, float)
    t_otsu = _otsu(s)
    if _valley_depth(s, t_otsu) >= vd:
        return t_otsu
    if low == "gap":
        return gap_split(s)
    if low == "mad25":
        return mad_k(2.5)(s)
    return float(np.percentile(s, 97))


def _gap_info(s):
    """largest gap in upper half: (midpoint, gap_size, median_gap)."""
    ss = np.sort(np.asarray(s, float)); n = len(ss)
    lo = int(n * 0.5)
    gaps = np.diff(ss[lo:])
    if gaps.size == 0 or gaps.max() <= 0:
        return float(np.percentile(s, 95)), 0.0, 1.0
    j = lo + int(np.argmax(gaps))
    nz = gaps[gaps > 0]
    med_gap = float(np.median(nz)) if nz.size else 1e-9
    return float((ss[j] + ss[j + 1]) / 2.0), float(gaps.max()), med_gap


def adapt3(s, vd=0.45, ratio=4.0, pct=92):
    """3-regime: bimodal=>Otsu; isolated(big gap)=>gap; smooth tail=>percentile."""
    s = np.asarray(s, float)
    t_otsu = _otsu(s)
    if _valley_depth(s, t_otsu) >= vd:
        return t_otsu
    t_gap, g_max, g_med = _gap_info(s)
    if g_max >= ratio * max(g_med, 1e-9):
        return t_gap
    return float(np.percentile(s, pct))


def mad_lower(k):
    """median + k*1.4826*MAD computed from the lower (normal-bulk) half only."""
    def f(s):
        s = np.asarray(s, float); med = np.median(s)
        low = s[s <= med]
        ml = np.median(np.abs(low - med)) if low.size else 0.0
        if ml < 1e-12:
            return float(np.percentile(s, 95))
        return float(med + k * 1.4826 * ml)
    return f


METHODS = {
    "fixed0.5": lambda s: 0.5,
    "SCM_auto_d05": scm(CalibrationMethod.AUTO, 0.05),
    "SCM_pct_d05": scm(CalibrationMethod.PERCENTILE, 0.05),
    "SCM_otsu": scm(CalibrationMethod.OTSU, 0.05),
    "SCM_mad_d05": scm(CalibrationMethod.MAD, 0.05),
    "SCM_knee": scm(CalibrationMethod.KNEE, 0.05),
    "SCM_aiqr": scm(CalibrationMethod.ADAPTIVE_IQR, 0.05),
    "SCM_pct_est": lambda s: float(np.percentile(s, 100 * (1 - est_contam(s)))),
    "SCM_auto_est": lambda s: float(_OPT.optimize(
        scores=np.asarray(s, float), method=CalibrationMethod.AUTO,
        contamination=est_contam(s)).threshold),
    "mad2.0": mad_k(2.0),
    "mad2.5": mad_k(2.5),
    "gap_split": gap_split,
    "adapt_gap": lambda s: adaptive(s, vd=0.45, low="gap"),
    "adapt_mad20": lambda s: (_otsu(s) if _valley_depth(s, _otsu(s)) >= 0.45 else mad_k(2.0)(s)),
    "adapt_mad25": lambda s: (_otsu(s) if _valley_depth(s, _otsu(s)) >= 0.45 else mad_k(2.5)(s)),
    "adapt_madlow25": lambda s: (_otsu(s) if _valley_depth(s, _otsu(s)) >= 0.45 else mad_lower(2.5)(s)),
    "adapt_madlow30": lambda s: (_otsu(s) if _valley_depth(s, _otsu(s)) >= 0.45 else mad_lower(3.0)(s)),
    "adapt_mad20cap": lambda s: _adapt_cap(s, 2.0),
    "adapt_maxom": lambda s: (max(_otsu(s), mad_k(2.0)(s)) if _valley_depth(s, _otsu(s)) >= 0.45
                              else mad_k(2.0)(s)),
}


def _adapt_cap(s, k):
    s = np.asarray(s, float)
    t = _otsu(s)
    if _valley_depth(s, t) >= 0.45 and np.mean(s > t) <= 0.45:
        return t
    return mad_k(k)(s)


def main():
    stat = importlib.import_module("omni_mercury_engine.detectors.statistical")
    MAD = stat.MercuryAnomalyDetector
    files = sorted(glob.glob(f"{CACHE}/*.npz"))
    # method -> list of (dom, f1, prec, rec)
    acc: dict[str, list] = {m: [] for m in METHODS}
    acc["ORACLE"] = []
    acc["TOPK"] = []
    diag_rows = []
    for fn in files:
        dom = os.path.basename(fn).split("__")[0]
        d = np.load(fn); X = d["X"]; y = d["y"].astype(int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.min() == y.max():
            continue
        det = MAD(); det.fit(X); R = det.detect(X)
        s = np.asarray(R["scores"], float).reshape(-1)
        diag_rows.append((dom, len(y), int(y.sum()), float(y.mean()),
                          ScoreDiagnostics._detect_bimodality(s),
                          round(ScoreDiagnostics._compute_kurtosis(s), 2)))
        for name, fn_thr in METHODS.items():
            try:
                thr = fn_thr(s)
            except Exception:
                thr = 0.5
            f1, pr, rc = prf(y, (s > thr).astype(int))
            acc[name].append((dom, f1, pr, rc))
        # oracle ceiling (vectorized best-F1 over all top-k cuts) + top-k ref
        order = np.argsort(-s)
        ys = y[order].astype(float)
        tp = np.cumsum(ys)
        kk = np.arange(1, len(s) + 1)
        prec = tp / kk
        rec = tp / max(int(y.sum()), 1)
        f1c = np.divide(2 * prec * rec, prec + rec,
                        out=np.zeros_like(prec), where=(prec + rec) > 0)
        acc["ORACLE"].append((dom, float(f1c.max()), 0, 0))
        k = int(y.sum())
        topk = np.zeros(len(s), int); topk[np.argsort(-s)[:k]] = 1
        ff, pr, rc = prf(y, topk)
        acc["TOPK"].append((dom, ff, pr, rc))

    doms = sorted(set(r[0] for r in acc["ORACLE"]))
    print(f"\n==== METHOD BAKE-OFF  {len(acc['ORACLE'])} events / {len(doms)} domains ====")
    print("event distribution diagnostics (dom n pos contam bimodal kurtosis):")
    for r in diag_rows:
        print("   ", r)
    # rank by overall mean F1
    def mF1(name, dom=None):
        xs = [x[1] for x in acc[name] if dom is None or x[0] == dom]
        return float(np.mean(xs)) if xs else float("nan")
    def mPR(name):
        return (float(np.mean([x[2] for x in acc[name]])),
                float(np.mean([x[3] for x in acc[name]])))
    print(f"\n{'method':14s} {'F1':>7s} {'prec':>7s} {'rec':>7s} | " + " ".join(f"{d[:5]:>6s}" for d in doms))
    for name in sorted(acc, key=lambda k: -mF1(k)):
        pr, rc = mPR(name)
        per = " ".join(f"{mF1(name,d):6.3f}" for d in doms)
        print(f"{name:14s} {mF1(name):7.3f} {pr:7.3f} {rc:7.3f} | {per}")


if __name__ == "__main__":
    main()

"""Fast unsupervised-threshold method comparison on cached detector scores.

Reads /home/user/eqlab/scores.pkl (scores+labels per event, produced once) so
candidate operating-point rules can be compared instantly. Reports per-domain
mean F1, the overall mean, and — crucially — how many domains REGRESS below the
fixed-0.5 baseline (the keep/revert gate).
"""
from __future__ import annotations
import pickle, numpy as np
from omni_mercury_engine.ml.mercury_ml import f1_score, precision_score, recall_score

S = pickle.load(open("/home/user/eqlab/scores.pkl", "rb"))


def prf(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


def otsu(s):
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return hi
    norm = ((s - lo) / (hi - lo) * 255).astype(int)
    h = np.bincount(norm, minlength=256).astype(float)
    tot = h.sum(); sumT = float(np.dot(np.arange(256), h))
    wB = sumB = best = 0.0; bt = 0
    for t in range(256):
        wB += h[t]
        if wB == 0:
            continue
        wF = tot - wB
        if wF == 0:
            break
        sumB += t * h[t]
        v = wB * wF * (sumB / wB - (sumT - sumB) / wF) ** 2
        if v > best:
            best, bt = v, t
    return lo + (bt / 255.0) * (hi - lo)


def valley(s, t):
    nb = int(np.clip(np.sqrt(s.size), 10, 40))
    h, e = np.histogram(s, bins=nb)
    if h.max() == 0:
        return 0.0
    bi = int(np.clip(np.searchsorted(e, t) - 1, 0, len(h) - 1))
    return 1.0 - float(h[bi]) / float(h.max())


def gap(s):
    ss = np.sort(s); n = ss.size
    if n < 8:
        return float(np.percentile(s, 95))
    lo = int(n * 0.5); g = np.diff(ss[lo:])
    if g.size == 0 or g.max() <= 0:
        return float(np.percentile(s, 95))
    j = lo + int(np.argmax(g))
    return float((ss[j] + ss[j + 1]) / 2.0)


def mad(s, k):
    m = np.median(s); d = np.median(np.abs(s - m))
    return float(np.percentile(s, 97)) if d < 1e-12 else float(m + k * 1.4826 * d)


def madlow(s, k):
    m = np.median(s); low = s[s <= m]
    ml = np.median(np.abs(low - m)) if low.size else 0.0
    return float(np.percentile(s, 95)) if ml < 1e-12 else float(m + k * 1.4826 * ml)


def adapt(s, low_fn, vd=0.45, cap=1.0):
    t = otsu(s)
    if valley(s, t) >= vd and np.mean(s > t) <= cap:
        return t
    return low_fn(s)


def est_c(s):
    """robust unsupervised contamination estimate: tail mass beyond median + 3*MAD_lower."""
    m = np.median(s); low = s[s <= m]
    ml = np.median(np.abs(low - m)) if low.size else 0.0
    if ml < 1e-12:
        return float(np.mean(s > np.percentile(s, 95)))
    return float(np.mean(s > m + 3.0 * 1.4826 * ml))


def adapt_c(s, c_high=0.15):
    """route by robust contamination estimate: high mode => Otsu; else MAD tail."""
    return otsu(s) if est_c(s) >= c_high else mad(s, 2.0)


def adapt_c3(s, c_low=0.015, c_high=0.15):
    """3-way: very low contam => gap (isolated); moderate => MAD; high => Otsu."""
    c = est_c(s)
    if c >= c_high:
        return otsu(s)
    if c <= c_low:
        return gap(s)
    return mad(s, 2.0)


def gap_dom(s):
    """largest upper-half gap as a fraction of (max - median): big => isolated outliers."""
    ss = np.sort(s); n = ss.size
    if n < 8:
        return 0.0, float(np.percentile(s, 95))
    lo = int(n * 0.5); g = np.diff(ss[lo:])
    if g.size == 0 or g.max() <= 0:
        return 0.0, float(np.percentile(s, 95))
    j = lo + int(np.argmax(g))
    spread = max(float(ss[-1] - np.median(s)), 1e-9)
    return float(g.max() / spread), float((ss[j] + ss[j + 1]) / 2.0)


def adapt_mad_gap(s, vd=0.45, gd=0.33):
    """Otsu if bimodal valley; else gap when an upper gap dominates the spread
    (isolated outliers), otherwise a robust MAD tail."""
    t = otsu(s)
    if valley(s, t) >= vd:
        return t
    dom, tg = gap_dom(s)
    return tg if dom >= gd else mad(s, 2.0)


def adapt_v(s, vd=0.45, c_gate=0.08, c_high=0.20, low="mad"):
    """Otsu if (a real valley AND non-trivial contamination) OR clearly high
    contamination; otherwise the low-regime rule."""
    t = otsu(s); c = est_c(s)
    if (valley(s, t) >= vd and c >= c_gate) or c >= c_high:
        return t
    return gap(s) if (low == "gap" and c <= 0.02) else mad(s, 2.0)


METHODS = {
    "fixed0.5": lambda s: 0.5,
    "otsu": otsu,
    "gap": gap,
    "mad2.0": lambda s: mad(s, 2.0),
    "adapt_mad20": lambda s: adapt(s, lambda x: mad(x, 2.0)),
    "adapt_c_h15": lambda s: adapt_c(s, c_high=0.15),
    "adapt_c_h12": lambda s: adapt_c(s, c_high=0.12),
    "adapt_c_h20": lambda s: adapt_c(s, c_high=0.20),
    "adapt_c3_l01": lambda s: adapt_c3(s, c_low=0.01, c_high=0.15),
    "adapt_mg_33": lambda s: adapt_mad_gap(s, vd=0.45, gd=0.33),
    "adapt_mg_40": lambda s: adapt_mad_gap(s, vd=0.45, gd=0.40),
    "adapt_mg_50": lambda s: adapt_mad_gap(s, vd=0.45, gd=0.50),
}


def oracle_f1(s, y):
    order = np.argsort(-s); ys = y[order].astype(float)
    tp = np.cumsum(ys); kk = np.arange(1, s.size + 1)
    p = tp / kk; r = tp / max(int(y.sum()), 1)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(f.max())


doms = sorted(set(d for d, _ in S))
base = {}  # (method, dom) -> list F1
prec = {}; rec = {}
for (dom, eid), rec_d in S.items():
    s = rec_d["scores"]; y = rec_d["y"]
    for name, fn in METHODS.items():
        try:
            thr = fn(s)
        except Exception:
            thr = 0.5
        f, p, r = prf(y, (s > thr).astype(int))
        base.setdefault(name, {}).setdefault(dom, []).append(f)
        prec.setdefault(name, []).append(p); rec.setdefault(name, []).append(r)
    base.setdefault("ORACLE", {}).setdefault(dom, []).append(oracle_f1(s, y))


def overall(name):
    return float(np.mean([f for dom in doms for f in base[name].get(dom, [])]))


def dom_mean(name, dom):
    xs = base[name].get(dom, [])
    return float(np.mean(xs)) if xs else float("nan")


print(f"events={len(S)}  domains={doms}")
print(f"\n{'method':22s} {'F1':>6s} {'prec':>6s} {'rec':>6s} {'reg':>3s} | " + " ".join(f"{d[:4]:>5s}" for d in doms))
basef = {d: dom_mean("fixed0.5", d) for d in doms}
for name in sorted(METHODS, key=lambda k: -overall(k)):
    regs = sum(1 for d in doms if dom_mean(name, d) < basef[d] - 1e-9)
    pp = float(np.mean(prec[name])); rr = float(np.mean(rec[name]))
    per = " ".join(f"{dom_mean(name,d):5.2f}" for d in doms)
    print(f"{name:22s} {overall(name):6.3f} {pp:6.3f} {rr:6.3f} {regs:3d} | {per}")
print(f"{'ORACLE':22s} {overall('ORACLE'):6.3f} {'':6s} {'':6s}     | " + " ".join(f"{dom_mean('ORACLE',d):5.2f}" for d in doms))

# vd-sensitivity probe for adapt_mad20
if __name__ == "__main__" and "--vd" in __import__("sys").argv:
    for vdt in (0.45, 0.55, 0.60, 0.65):
        per = {d: [] for d in doms}; allp = []; allr = []
        for (dom, eid), r in S.items():
            s = r["scores"]; y = r["y"]; t = otsu(s)
            thr = t if valley(s, t) >= vdt else mad(s, 2.0)
            f, p, rr = prf(y, (s > thr).astype(int)); per[dom].append(f); allp.append(p); allr.append(rr)
        ov = np.mean([f for d in doms for f in per[d]])
        regs = sum(1 for d in doms if np.mean(per[d]) < np.mean(base["fixed0.5"][d]) - 1e-9)
        print(f"vd={vdt}: F1={ov:.3f} prec={np.mean(allp):.3f} rec={np.mean(allr):.3f} reg={regs} | "
              + " ".join(f"{d[:4]}={np.mean(per[d]):.2f}" for d in doms))

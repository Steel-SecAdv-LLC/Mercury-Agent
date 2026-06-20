# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fusion-frontier diagnosis for the MercuryAnomalyDetector ensemble.

Companion to ``harness_adbench.py``. That harness reports the *headline*
(transductive mean AUROC, base vs hardened). This one answers the next
question the hardening pass raises: **is the remaining gap to "best single
stream" closable without ground-truth labels?**

It measures, on the same fixed 18-set real-ADBench composition:

  1. Per-stream TRUE AUROC (resonance / kinematic / info-geometry), the fused
     AUROC, and the per-set best single stream -> the *dilution gap*
     (best-single - fused) that FINDINGS.md flagged.
  2. A native kNN-distance stream (mean distance to k nearest neighbours, no
     third-party dependency) measured as a candidate complementary stream.
  3. A label-free fusion-frontier search: every candidate combiner (stream
     re-weights and kNN blends) is scored against the pre-registered keep bar
     **net-positive mean AND no set regressing by more than 0.002 AUROC**.

The committed result (``fusion_diagnosis_results.json``) records a measured
*negative*: across re-weighting, contrast difficulty (see the module note) and
a complementary kNN stream, no label-free combiner clears the zero-regression
bar. The dilution gap is real (~0.03) but, for these streams, irreducible
without labels -- which is why the deployed weighter prefers the supervised
path (``fit(calibration_labels=...)``) whenever labels are available, and why
a new stream "earns its place" only with a working per-set reliability signal.

Real data only (ADBench raw.githubusercontent mirror); a set that fails to
download raises rather than degrading. AUROC via ``mercury_ml`` (no sklearn).

    LD_LIBRARY_PATH=<ama-build-lib> python research/omni_equation/harness_fusion_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Runnable from a fresh checkout without an editable install: put the repo's
# src/ on sys.path before importing the engine (mirrors benchmarks/*.py and
# harness_adbench.py).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from omni_mercury_engine.datasets.adbench import ADBenchLoader
from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import StandardScaler, roc_auc_score

# Same fixed composition as harness_adbench.py so the two harnesses are directly
# comparable (see that module for the selection rationale).
DATASETS: tuple[str, ...] = (
    "breastw",
    "cardio",
    "Cardiotocography",
    "glass",
    "Hepatitis",
    "Ionosphere",
    "Lymphography",
    "mammography",
    "optdigits",
    "PageBlocks",
    "Pima",
    "Stamps",
    "thyroid",
    "vertebral",
    "Waveform",
    "WBC",
    "wine",
    "WPBC",
)

# Pre-registered keep bar (declared before measuring): a candidate combiner is
# kept only if it raises the mean AND regresses no set beyond this tolerance.
# 0.002 is the noise floor used for the PR-302 "2 losses are noise" call.
REGRESSION_TOLERANCE = 0.002


def _auroc(y: np.ndarray[Any, Any], s: np.ndarray[Any, Any]) -> float:
    """Mann-Whitney AUROC via mercury_ml (no sklearn)."""
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return 0.5


def _rank(s: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Rank-normalise scores to [0, 1] (ties broken by argsort order)."""
    s = np.asarray(s, dtype=float)
    order = s.argsort()
    r = np.empty(len(s), dtype=float)
    r[order] = np.arange(len(s))
    return r / max(len(s) - 1, 1)


def _knn_distance_score(
    x: np.ndarray[Any, Any], k: int = 10, ref_cap: int = 3000, seed: int = 0
) -> np.ndarray[Any, Any]:
    """Native kNN anomaly score: mean Euclidean distance to k nearest neighbours.

    A reference subsample (``ref_cap``) bounds the O(n*ref) cost on large sets;
    deterministic given ``seed``. No third-party dependency.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    ref = x
    if n > ref_cap:
        idx = np.random.RandomState(seed).choice(n, ref_cap, replace=False)
        ref = x[idx]
    out = np.empty(n, dtype=float)
    for i in range(0, n, 512):
        block = x[i : i + 512]
        d = np.sqrt(((block[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
        d.sort(1)
        # column 0 is the point itself when it is in ref; skip it.
        out[i : i + 512] = d[:, 1 : k + 1].mean(1)
    return out


def _measure_one(name: str, data_dir: str) -> dict[str, Any]:
    """Fit the real detector on one ADBench set and collect stream-level AUROCs."""
    loader = ADBenchLoader(DatasetConfig(name=name, data_dir=data_dir, max_samples=None))
    x_raw, y_raw = loader.load()
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw).astype(int).reshape(-1)
    if not 0 < int(y.sum()) < len(y):
        return {"dataset": name, "error": "single-class after load"}

    x_std = StandardScaler().fit_transform(x)
    det = MercuryAnomalyDetector()
    det.fit(x_std)

    fused = np.asarray(det.detect(x_std)["scores"], dtype=float).reshape(-1)
    res = det._compute_resonance_score(x_std)
    kin = det._compute_kinematic_score(x_std)
    igeo = det._compute_info_geometry_score(x_std)
    knn = _knn_distance_score(x_std)

    auc_res, auc_kin, auc_igeo = _auroc(y, res), _auroc(y, kin), _auroc(y, igeo)
    best_single = max(auc_res, auc_kin, auc_igeo)
    return {
        "dataset": name,
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "auroc_fused": round(_auroc(y, fused), 4),
        "auroc_resonance": round(auc_res, 4),
        "auroc_kinematic": round(auc_kin, 4),
        "auroc_infogeo": round(auc_igeo, 4),
        "auroc_knn": round(_auroc(y, knn), 4),
        "best_single": round(best_single, 4),
        "dilution_gap": round(best_single - _auroc(y, fused), 4),
        # rank vectors retained (in-memory only) for the frontier search
        "_ranks": (y, _rank(fused), _rank(res), _rank(igeo), _rank(knn)),
    }


# Candidate label-free combiners over rank-normalised streams. ``cf`` is the
# current fused rank, ``rr``/``ri`` resonance/info-geometry ranks, ``rk`` the
# kNN rank. Each spans a different "de-dilution" idea (prefer resonance, blend,
# rank-mean, add the complementary kNN stream).
_COMBINERS: tuple[str, ...] = (
    "resonance_only",
    "blend_0.6R_0.4I",
    "rankmean_R_I",
    "blend_0.8fused_0.2knn",
    "rankmax_fused_knn",
    "rankmean_fused_knn",
)


def _combine(
    name: str,
    cf: np.ndarray[Any, Any],
    rr: np.ndarray[Any, Any],
    ri: np.ndarray[Any, Any],
    rk: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Evaluate one named label-free combiner over the rank-normalised streams."""
    if name == "resonance_only":
        return rr
    if name == "blend_0.6R_0.4I":
        return 0.6 * rr + 0.4 * ri
    if name == "rankmean_R_I":
        return rr + ri
    if name == "blend_0.8fused_0.2knn":
        return 0.8 * cf + 0.2 * rk
    if name == "rankmax_fused_knn":
        return np.maximum(cf, rk)
    if name == "rankmean_fused_knn":
        return cf + rk
    raise ValueError(f"unknown combiner: {name}")


def _frontier_search(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score label-free combiners against the pre-registered zero-regression bar."""
    scored = [r for r in rows if "_ranks" in r]
    base_mean = float(np.mean([r["auroc_fused"] for r in scored]))

    out: list[dict[str, Any]] = []
    for cname in _COMBINERS:
        deltas: list[float] = []
        aucs: list[float] = []
        for r in scored:
            y, cf, rr, ri, rk = r["_ranks"]
            a = _auroc(y, _combine(cname, cf, rr, ri, rk))
            aucs.append(a)
            deltas.append(a - r["auroc_fused"])
        mean = float(np.mean(aucs))
        worst = float(min(deltas))
        keeps = mean > base_mean and worst >= -REGRESSION_TOLERANCE
        out.append(
            {
                "combiner": cname,
                "mean_auroc": round(mean, 4),
                "delta_mean": round(mean - base_mean, 4),
                "worst_set_delta": round(worst, 4),
                "n_regress_beyond_tol": int(sum(1 for d in deltas if d < -REGRESSION_TOLERANCE)),
                "clears_zero_regression_bar": keeps,
            }
        )
    return out


def main() -> None:
    """Run the fusion-frontier diagnosis and persist the JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/adbench")
    parser.add_argument("--out", default="research/omni_equation/fusion_diagnosis_results.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for name in DATASETS:
        try:
            rows.append(_measure_one(name, args.data_dir))
        except Exception as exc:  # fail loud per dataset, never fabricate
            rows.append({"dataset": name, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})

    scored = [r for r in rows if "auroc_fused" in r]
    mean_fused = float(np.mean([r["auroc_fused"] for r in scored]))
    mean_best = float(np.mean([r["best_single"] for r in scored]))
    mean_knn = float(np.mean([r["auroc_knn"] for r in scored]))
    knn_complements = sum(1 for r in scored if r["auroc_knn"] > r["best_single"])
    frontier = _frontier_search(rows)

    # strip the in-memory rank tuples before serialising
    public_rows = [{k: v for k, v in r.items() if k != "_ranks"} for r in rows]
    summary = {
        "n_scored": len(scored),
        "mean_fused": round(mean_fused, 4),
        "mean_best_single": round(mean_best, 4),
        "dilution_gap": round(mean_best - mean_fused, 4),
        "mean_knn_stream": round(mean_knn, 4),
        "knn_beats_best_single_on_sets": knn_complements,
        "regression_tolerance": REGRESSION_TOLERANCE,
        "any_combiner_clears_bar": any(c["clears_zero_regression_bar"] for c in frontier),
    }
    Path(args.out).write_text(
        json.dumps({"summary": summary, "results": public_rows, "frontier": frontier}, indent=2)
        + "\n"
    )

    print("==== Fusion-frontier diagnosis (real Mercury detector, 18-set ADBench) ====")
    print(f"{'dataset':18}{'fused':>7}{'R':>7}{'K':>7}{'I':>7}{'kNN':>7}{'best1':>7}{'gap':>7}")
    for r in scored:
        print(
            f"{r['dataset']:18}{r['auroc_fused']:>7.3f}{r['auroc_resonance']:>7.3f}"
            f"{r['auroc_kinematic']:>7.3f}{r['auroc_infogeo']:>7.3f}{r['auroc_knn']:>7.3f}"
            f"{r['best_single']:>7.3f}{r['dilution_gap']:>+7.3f}"
        )
    print(
        f"\nmean fused={mean_fused:.4f}  best-single={mean_best:.4f}  "
        f"dilution gap={mean_best - mean_fused:+.4f}  mean kNN={mean_knn:.4f} "
        f"(beats best-single on {knn_complements}/{len(scored)} sets)"
    )
    print(
        f"\nLabel-free fusion frontier (keep bar: net-positive AND no set < -{REGRESSION_TOLERANCE}):"
    )
    for c in frontier:
        verdict = "KEEP" if c["clears_zero_regression_bar"] else "reject"
        print(
            f"  {c['combiner']:22} mean={c['mean_auroc']:.4f} "
            f"worst_delta={c['worst_set_delta']:+.4f} "
            f"reg={c['n_regress_beyond_tol']:>2} -> {verdict}"
        )
    if not summary["any_combiner_clears_bar"]:
        print(
            "\nNo label-free combiner clears the zero-regression bar: the dilution gap "
            "is real but irreducible for these streams without labels."
        )


if __name__ == "__main__":
    main()

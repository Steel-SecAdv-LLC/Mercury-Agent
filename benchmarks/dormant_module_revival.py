"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Dormant-module revival: do the orphaned "cognitive" modules carry *real*
anomaly-detection signal on genuinely-labelled data?

This is the anti-theater gate for salvaging the dormant cognitive subsystem.
The audit (docs/NEUROSYMBOLIC.md / the dormancy report) found ~13 K LOC of
cognitive modules that are exported in the public API but never run in any live
path. A module's *interface* is not evidence it carries signal -- only a paired
measurement on real held-out labels is. This harness asks, per candidate
detector, the only honest question: **does it produce an anomaly score that
beats chance on real ADBench labels, and does it add anything beyond the
detector the live ensemble already has?**

Candidates (the only orphaned modules that expose a per-sample score over
tabular features X):

* ``predictive_coding``  -- ``PredictiveCodingDetector`` (prediction-error /
  free-energy surprise). Unsupervised: fit on train, score test.
* ``kmeans_distance``    -- ``neural_memory_layer.KMeansClusterer`` distance to
  the nearest learned centroid. Unsupervised: fit on train, score test.
* ``case_based_knn``     -- ``CaseBasedReasoner`` retrieval (supervised: the case
  base carries train labels), scored as 1 - P(normal | k nearest cases).

Reference: ``lof_reference`` -- the live ensemble's own distance/density detector
(``detectors.spatial``), so a candidate's salvage is judged on whether it *adds*
to what already ships, not merely on beating chance.

Ablation integrity (non-negotiable): metrics are real held-out ROC-AUC only. If
ADBench cannot be downloaded the run reports that and exits non-zero -- it never
fabricates a pass. A candidate is recommended for revival only if it clears a
pre-registered bar on real labels.

Usage::

    python -m benchmarks.dormant_module_revival
    python -m benchmarks.dormant_module_revival \\
        --datasets cardio thyroid breastw WBC Pima --seeds 0 1 2 \\
        --out artifacts/dormant_module_revival.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from typing import Any, Callable

import numpy as np

DEFAULT_DATASETS = ["cardio", "thyroid", "breastw", "WBC", "Pima"]
DEFAULT_SEEDS = [0, 1, 2]

# Pre-registered salvage bar. "Carries signal" is the weak gate (beats chance by
# a clear margin); "adds value" is the gate that actually justifies wiring the
# candidate into the live ensemble (it must beat the ensemble's existing
# distance/density detector, not merely chance).
_SIGNAL_AUC = 0.70  # mean held-out AUC to count as carrying real signal


def _load_dataset(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": name}))
    loader.download()
    data = loader.load()
    return np.asarray(data[0], dtype=np.float32), np.asarray(data[1]).astype(int).ravel()


def _stratified(
    y: np.ndarray[Any, Any], frac: float, rng: np.random.RandomState
) -> np.ndarray[Any, Any]:
    keep: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        keep.extend(idx[: max(1, int(len(idx) * frac))].tolist())
    return np.array(sorted(keep))


# -- candidate scorers: each fits on (X_tr, y_tr) and returns test scores -------


def _score_predictive_coding(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.cognitive.predictive_coding import PredictiveCodingDetector

    det = PredictiveCodingDetector(input_dim=X_tr.shape[1], anomaly_threshold=2.0)
    det.fit(X_tr, epochs=10)
    return np.array([det.detect(o, update_model=False)["anomaly_score"] for o in X_te])


def _score_kmeans_distance(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.cognitive.neural_memory_layer import KMeansClusterer

    km = KMeansClusterer(n_clusters=8).fit(X_tr)
    return np.asarray(km.get_cluster_distances(X_te)).min(axis=1)


def _score_case_based_knn(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    from omni_mercury_engine.cognitive.case_based_reasoning import (
        Case,
        CaseBasedReasoner,
        CaseOutcome,
        SimilarityMetric,
    )

    reasoner = CaseBasedReasoner(similarity_metric=SimilarityMetric.EUCLIDEAN)
    for i, (xi, yi) in enumerate(zip(X_tr, y_tr)):
        reasoner.add_case(
            Case(
                case_id=f"c{i}",
                problem_description="adbench",
                problem_features={f"f{j}": float(v) for j, v in enumerate(xi)},
                feature_vector=np.asarray(xi, dtype=float),
                solution={"label": int(yi)},
                outcome=CaseOutcome.SUCCESS if yi == 0 else CaseOutcome.FAILURE,
                outcome_score=0.0 if yi == 0 else 1.0,
                domain="general",
            )
        )
    scores = []
    for xi in X_te:
        res = reasoner.retrieve(
            Case(
                case_id="q",
                problem_description="adbench",
                problem_features={f"f{j}": float(v) for j, v in enumerate(xi)},
                feature_vector=np.asarray(xi, dtype=float),
                solution={},
                outcome=CaseOutcome.UNKNOWN,
                outcome_score=0.0,
                domain="general",
            ),
            k=5,
        )
        # Anomaly score = similarity-weighted mean anomaly label of the k nearest.
        nbrs = res.retrieved_cases
        num = sum(sim * float(c.outcome_score) for c, sim in nbrs)
        den = sum(sim for _, sim in nbrs) + 1e-9
        scores.append(num / den)
    return np.asarray(scores)


def _score_lof_reference(
    X_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], X_te: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The live ensemble's own distance/density detector, as the salvage baseline.

    ``decision_function`` follows the sklearn LOF convention (higher == more
    normal), so the anomaly score is its negation.
    """
    from omni_mercury_engine.detectors.spatial import _NativeLOF

    lof = _NativeLOF(n_neighbors=min(20, max(2, len(X_tr) - 1))).fit(X_tr)
    return -np.asarray(lof.decision_function(X_te))


CANDIDATES: dict[str, tuple[Callable[..., np.ndarray[Any, Any]], bool]] = {
    # name: (scorer, supervised?)
    "predictive_coding": (_score_predictive_coding, False),
    "kmeans_distance": (_score_kmeans_distance, False),
    "case_based_knn": (_score_case_based_knn, True),
    "lof_reference": (_score_lof_reference, False),
}


def run_candidate(name: str, datasets: list[str], seeds: list[int]) -> dict[str, Any]:
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    scorer, supervised = CANDIDATES[name]
    per_dataset: dict[str, float] = {}
    all_aucs: list[float] = []
    errors = 0
    for ds in datasets:
        try:
            X, y = _load_dataset(ds)
        except Exception:
            continue
        ds_aucs: list[float] = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            te = _stratified(y, 0.3, rng)
            mask = np.zeros(len(y), dtype=bool)
            mask[te] = True
            tr = np.where(~mask)[0]
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            mu = X[tr].mean(0)
            sd = X[tr].std(0)
            sd[sd < 1e-8] = 1.0
            X_tr, X_te = (X[tr] - mu) / sd, (X[te] - mu) / sd
            try:
                scores = scorer(X_tr, y[tr], X_te)
                auc = float(roc_auc_score(y[te], scores))
                # Raw held-out AUC. Each candidate's natural orientation is
                # higher-score == more-anomalous, so a value below 0.5 means the
                # signal is genuinely absent (not merely inverted); the >= 0.70
                # gate then correctly reads it as "no signal".
                ds_aucs.append(auc)
            except Exception:
                errors += 1
        if ds_aucs:
            per_dataset[ds] = float(np.mean(ds_aucs))
            all_aucs.extend(ds_aucs)
    return {
        "candidate": name,
        "supervised": supervised,
        "per_dataset_auc": per_dataset,
        "mean_auc": float(np.mean(all_aucs)) if all_aucs else float("nan"),
        "min_auc": float(np.min(all_aucs)) if all_aucs else float("nan"),
        "n_runs": len(all_aucs),
        "errors": errors,
    }


def derive_verdicts(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Verdict per candidate from measured standalone signal.

    REVIVE_CANDIDATE means the module produces a real anomaly score on held-out
    labels (mean AUC >= the signal bar) and is worth *promoting to the
    ensemble-marginal ablation* -- the separate, rigorous test of whether it adds
    over the full live fusion ensemble (the single ``lof_reference`` AUC is only
    informational context, not a fused-ensemble proxy). A candidate that does not
    clear the standalone bar is ARCHIVE: no honest detection salvage.
    """
    ref = next((r for r in results if r["candidate"] == "lof_reference"), None)
    ref_mean = ref["mean_auc"] if ref else float("nan")
    verdicts = {}
    for r in results:
        if r["candidate"] == "lof_reference":
            continue
        mean_auc = r["mean_auc"]
        carries = bool(not np.isnan(mean_auc) and mean_auc >= _SIGNAL_AUC)
        verdicts[r["candidate"]] = {
            "mean_auc": mean_auc,
            "carries_signal": carries,
            "verdict": (
                "REVIVE_CANDIDATE -- carries real signal; promote to ensemble-marginal ablation"
                if carries
                else "ARCHIVE -- no usable anomaly signal on real labels"
            ),
        }
    return {"lof_reference_mean_auc": ref_mean, "candidates": verdicts}


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", default="artifacts/dormant_module_revival.json", type=str)
    args = parser.parse_args()

    print("Dormant-module revival (standalone anomaly AUC on real ADBench labels)")
    print(f"datasets={args.datasets}  seeds={args.seeds}")
    print("-" * 80)

    results = [run_candidate(name, args.datasets, args.seeds) for name in CANDIDATES]
    measured = [r for r in results if r["n_runs"] > 0]
    if not measured:
        print("INTEGRITY FAILURE: no candidate could be measured (network unavailable?).")
        return 1

    for r in results:
        cells = "  ".join(f"{d}={a:.3f}" for d, a in r["per_dataset_auc"].items())
        print(
            f"  {r['candidate']:<18} mean_AUC={r['mean_auc']:.3f} "
            f"(n={r['n_runs']}, err={r['errors']})  {cells}"
        )

    verdicts = derive_verdicts(results)
    print("-" * 80)
    print(f"LOF reference mean AUC: {verdicts['lof_reference_mean_auc']:.3f}")
    for name, v in verdicts["candidates"].items():
        print(f"  {name:<18} {v['verdict']}")

    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "verdicts": verdicts}, indent=2, sort_keys=True))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

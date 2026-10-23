#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ten-minute maritime quickstart: one real dataset, one honest number.

Downloads the only public, expert-annotated real-AIS anomaly benchmark —
521 vessel trajectories from Danish waters (2021-12-13), 25 manually
labelled abnormal: colliding vessels, Search-and-Rescue responders, law
enforcement, and traffic forced off course by the accident — and scores
it with the stock :class:`MercuryAnomalyDetector`, unsupervised, in
seconds, under two seed-pinned protocols:

* **Harness protocol** -- the committed benchmark split, step for step, so
  its ROC-AUC reproduces the ``DTU_AIS_Abnormal`` row of
  ``benchmarks/mercury_benchmark_results.json`` exactly.  It fits on half
  the *normal* tracks (picked with the labels) and scores the other 273,
  so it does not score the full evaluation set.
* **Label-free cross-fit** -- every one of the 521 tracks is scored by a
  model fitted, without labels, on the other four of five folds.  No label
  reaches a fit; this is the number to read beside the published ones.

Published baselines on all 521 labelled tracks (Olesen et al., J. Mar.
Sci. Eng. 11(11):2085, 2023, Figure 12a): VRNN (the GeoTrackNet
architecture) 0.87, the paper's two-step clustering 0.73 (0.79 only after
removing stationary tracks, Figure 12b), RVAE 0.72.  The authors trained
every model on the Bornholm December 2021 unlabelled data except 13
December and tested on that day (section 4.7), so neither protocol here
is theirs: the cross-fit trains on the test day's own unlabelled traffic.
Olesen's DTU thesis (Fig. 6.1) is cited elsewhere for RAE 0.86, Seq2Seq
0.84 and Seq2Seq+attention 0.83; those are not in the JMSE paper and are
not shown until checked against the thesis.

Data: Olesen, K. V., Clemmensen, L. K. H., & Christensen, A. N. (2023),
Technical University of Denmark, CC BY 4.0,
https://doi.org/10.11583/DTU.21511815. Download is ~38 MB, digest-pinned,
and deserialised through the loader's fail-closed restricted unpickler
(tests/datasets/test_maritime_loader.py).

Usage::

    pip install -e .
    python examples/mercury_quickstart.py [--data-dir ./data]

No PQC build is needed for this demo: it runs entirely on the AMA-free
analytic leaves (dataset loader, statistical detector, metrics) under the
scoped production gate. The full Mercury *agent* — engine, CLI, API,
crypto — still requires the native AMA Cryptography backend and refuses
to import without it (``omni_mercury_engine/_pqc_gate.py``); install it
with ``pip install -e ".[pqc]"`` or ``bash scripts/build_ama_native.sh``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.maritime import FEATURE_NAMES, DTUAISLoader
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.ml.mercury_ml import (
    StandardScaler,
    average_precision_score,
    roc_auc_score,
)

#: ROC-AUC on all 521 labelled tracks, JMSE 11(11):2085 Figure 12a; each
#: model trained on Bornholm December 2021 except the 13th (section 4.7).
PUBLISHED_BASELINES: tuple[tuple[str, float, str], ...] = (
    ("VRNN (GeoTrackNet arch.)", 0.87, "JMSE 2023 Fig. 12a"),
    ("Two-step clustering", 0.73, "JMSE 2023 Fig. 12a (0.79 w/o stationary)"),
    ("RVAE", 0.72, "JMSE 2023 Fig. 12a"),
)


def benchmark_protocol_split(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the committed benchmark split (mercury_benchmark.py, seed 42).

    Fit on a random half of the normal trajectories; score the remaining
    normals plus every abnormal one, order de-leaked with the same fixed
    permutation the suite uses.

    Returns:
        ``(X_train, X_test, y_test)``.
    """
    normal_mask = labels == 0
    X_normal = features[normal_mask]
    n_train = len(X_normal) // 2
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_normal), n_train, replace=False)
    X_train = X_normal[train_idx]

    test_normal_mask = np.ones(len(X_normal), dtype=bool)
    test_normal_mask[train_idx] = False
    X_test = np.vstack([X_normal[test_normal_mask], features[~normal_mask]])
    y_test = np.concatenate(
        [
            np.zeros(int(test_normal_mask.sum()), dtype=int),
            np.ones(int((~normal_mask).sum()), dtype=int),
        ]
    )
    perm = np.random.RandomState(42).permutation(len(X_test))
    return X_train, X_test[perm], y_test[perm]


#: Folds of the label-free cross-fit.
CROSS_FIT_FOLDS = 5


def fit_and_score(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """The benchmark's own pipeline: scale on train, fit, score the test rows."""
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
    scaler = StandardScaler()
    detector = MercuryAnomalyDetector()
    detector.fit(scaler.fit_transform(X_train))
    # The benchmark suite plants the dataset category on the detector after
    # fitting so ensemble weight selection uses the matching domain preset;
    # mirror it exactly so the harness run reproduces the committed row.
    detector._benchmark_domain = "maritime"
    return np.asarray(detector.detect(scaler.transform(X_test))["scores"], dtype=float)


def label_free_cross_fit(
    features: np.ndarray, n_folds: int = CROSS_FIT_FOLDS, seed: int = 42
) -> np.ndarray:
    """One score per track, each from a model that never saw it or any label.

    The tracks are split into ``n_folds`` seeded folds; each fold is scored by
    a detector fitted on the other folds' features alone (abnormal tracks
    included, unlabelled, as in deployment).

    Returns:
        Scores aligned with ``features``.
    """
    folds = np.array_split(np.random.RandomState(seed).permutation(len(features)), n_folds)
    scores = np.empty(len(features), dtype=float)
    for k, held_out in enumerate(folds):
        fit_rows = np.concatenate([fold for j, fold in enumerate(folds) if j != k])
        scores[held_out] = fit_and_score(features[fit_rows], features[held_out])
    return scores


def bootstrap_auc_ci(
    y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95 % percentile bootstrap interval for ROC-AUC.

    Stratified: each resample keeps the positive and negative counts, so
    every resample has both classes (with 25 positives, the uncertainty that
    matters is which abnormal tracks were drawn).
    """
    rng = np.random.default_rng(seed)
    positives, negatives = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))]
        )
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def main(argv: list[str] | None = None) -> int:
    """Run the quickstart end to end; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Download/cache directory (default: ./data; ~38 MB on first run)",
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("MERCURY QUICKSTART")
    print("Real AIS trajectories, expert-annotated labels, published baselines")
    print("=" * 74)
    print(
        "\nData: Olesen, Clemmensen & Christensen (2023), Technical University"
        "\nof Denmark, CC BY 4.0, https://doi.org/10.11583/DTU.21511815"
    )

    t0 = time.perf_counter()
    loader = DTUAISLoader(DatasetConfig(name="DTU_AIS_Abnormal", data_dir=args.data_dir))
    cached = loader._members_verified()
    print(f"\n[1/4] {'Verifying cached' if cached else 'Downloading (~38 MB)'} dataset ...")
    loader.download()
    print(f"      digest-pinned members verified in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print("[2/4] Extracting trajectory features ...")
    features, labels = loader._load_raw()
    print(
        f"      {features.shape[0]} trajectories x {len(FEATURE_NAMES)} features, "
        f"{int(labels.sum())} expert-labelled abnormal ({time.perf_counter() - t1:.1f}s)"
    )

    t2 = time.perf_counter()
    print("[3/4] Harness protocol: fit on half the normal tracks (label-selected) ...")
    X_train, X_test, y_test = benchmark_protocol_split(features, labels)
    harness = fit_and_score(X_train, X_test)
    harness_auc = roc_auc_score(y_test, harness)
    harness_ci = bootstrap_auc_ci(y_test, harness)
    print(
        f"      fit on {len(X_train)} normals, scored {len(X_test)} tracks incl. all "
        f"{int(y_test.sum())} abnormal: ROC-AUC {harness_auc:.4f} "
        f"[95% CI {harness_ci[0]:.3f}-{harness_ci[1]:.3f}], "
        f"AP {average_precision_score(y_test, harness):.4f} ({time.perf_counter() - t2:.1f}s)"
    )

    t3 = time.perf_counter()
    print(f"[4/4] Label-free {CROSS_FIT_FOLDS}-fold cross-fit over all {len(labels)} tracks ...")
    cross = label_free_cross_fit(features)
    cross_auc = roc_auc_score(labels, cross)
    cross_ci = bootstrap_auc_ci(labels, cross)
    cross_ap = average_precision_score(labels, cross)
    print(
        f"      no label reaches a fit: ROC-AUC {cross_auc:.4f} "
        f"[95% CI {cross_ci[0]:.3f}-{cross_ci[1]:.3f}], AP {cross_ap:.4f} "
        f"(base rate {labels.mean():.3f}) ({time.perf_counter() - t3:.1f}s)\n"
    )

    print("      All 521 labelled tracks, ROC-AUC (protocols differ; see caveats):\n")
    rows = [("Mercury label-free cross-fit", cross_auc, "this run, no labels")]
    rows += [(name, value, note) for name, value, note in PUBLISHED_BASELINES]
    for name, value, note in sorted(rows, key=lambda r: -r[1]):
        marker = " <-- you" if name.startswith("Mercury") else ""
        print(f"      {name:<28s} ROC-AUC {value:.4f}   {note}{marker}")
    print(f"\n      Total wall time: {time.perf_counter() - t0:.1f}s")

    outside = [
        name for name, value, _ in PUBLISHED_BASELINES if not cross_ci[0] <= value <= cross_ci[1]
    ]
    print(
        f"\nCaveats: with {int(labels.sum())} positives the cross-fit's 95% interval spans "
        f"{cross_ci[1] - cross_ci[0]:.2f} of AUC"
    )
    if outside:
        print(f"({', '.join(outside)} fall outside it).", end=" ")
    print(
        "No ranking follows: the protocols differ.\n"
        "The published models were trained on Bornholm December 2021 except the\n"
        "13th; the cross-fit trains on the 13th's own (unlabelled) traffic and\n"
        "scores 22 trajectory-level features. The harness row scores 273 of the\n"
        "521 tracks after fitting on label-selected normals, so it is not\n"
        "comparable to the published figures."
    )
    print("\nThis run is deterministic (seed 42); the harness ROC-AUC reproduces")
    print("the DTU_AIS_Abnormal row of benchmarks/mercury_benchmark_results.json.")
    print("\nMERCURY QUICKSTART COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

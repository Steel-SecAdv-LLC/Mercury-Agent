# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Do the unreached ensemble heads beat a plain mean of real detector scores?

``docs/DORMANCY_LEDGER.md`` §2c lists four ensembles as "still unreached and
still unmeasured -- named, not assumed": ``ml/ensemble.py``,
``ml/ensemble_coordinator.py``, ``core/three_r/learnable_fusion.py`` and
``models/foundation/ensemble.py``. The ledger records a *hypothesis* for the
coordinator -- that the engine's own ``OmniFusionModel`` already fuses detector
scores better than any re-weighting of them, making it dead by construction --
and is explicit that it is a hypothesis until measured.

This harness measures the first necessary bar, which does not need the engine:
**does the head beat the plain mean of the same real detector scores?**

That bar is the right one to try first, and it is decisive in one direction. For
unsupervised outlier ensembles the plain mean is the dominant aggregation
(Aggarwal & Sathe, *Outlier Ensembles*, 2017) -- it is what the detector tier's
``consensus`` combiner and ``math_arrest``'s fusion default already use on that
authority. A head that cannot beat the mean of its own inputs cannot be worth
wiring, whatever a larger comparison would say. A head that *does* beat it is
not thereby promoted: it would still have to clear the engine's trained
``OmniFusionModel``, which §8 established as the baseline a fusion head must
out-separate, and that comparison needs a working engine.

Protocol, matching ``tests/detectors/test_recorded_adbench_tabular.py`` so the
inputs are the same real scores that module pins: five recorded ADBench tables,
stratified 70/30, three seeds, detectors fitted on the training rows and scoring
the held-out rows. Supervised heads are fitted on the *training* rows' scores
and labels only.

Run from the repository root::

    python research/ensembles/harness_unreached_heads.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

#: The five manifest detectors the recorded ADBench module measures. These are
#: the real inputs any fusion head would receive.
DETECTORS = ("adaptive_anomaly", "info_geometry", "spatial", "directive", "realtime_threat")
SEEDS = (0, 1, 2)


def _read_scores(name: str, detector: Any, rows: Any) -> Any:
    """The per-row score each seam returns; the contract differs by entry."""
    if name == "adaptive_anomaly":
        return np.asarray(detector.detect(rows).scores, dtype=float)
    if name == "info_geometry":
        return np.asarray(detector.predict(rows)["anomaly_scores"], dtype=float)
    if name == "realtime_threat":
        return np.asarray(detector.predict(rows)["scores"], dtype=float)
    return np.asarray(detector.detect(rows)["scores"], dtype=float)


def _fitted_detector(name: str, fit_rows: Any, seed: int) -> Any:
    """One manifest detector, constructed and fitted as the recorded module does."""
    import importlib

    import torch

    from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
    from tests.core.test_detector_manifest_seams import FAST_CONFIG

    entry = {e.name: e for e in DETECTOR_MANIFEST}[name]
    cls = getattr(importlib.import_module(entry.module_path), entry.class_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    detector = cls(FAST_CONFIG[name]) if name in FAST_CONFIG else cls()
    detector.fit(fit_rows)
    return detector


def _rank_normalize(scores: Any) -> Any:
    """Map scores to [0, 1] by rank.

    Detector scores live on incommensurable scales, so any combiner that
    averages them has to normalize first or it silently weights by scale. The
    plain-mean baseline and every head below receive the *same* normalized
    matrix, so the comparison isolates the combination rule.
    """
    order = np.argsort(np.argsort(scores))
    return order / max(1, len(scores) - 1)


def _score_matrix(X: Any, y: Any, seed: int) -> tuple[Any, Any, Any, Any]:
    """``(train_matrix, train_labels, test_matrix, test_labels)`` of real scores.

    Features are standardised on the training rows, exactly as
    ``test_recorded_adbench_tabular.measure_tabular`` does, so the columns here
    are the same real scores that module pins. Detectors are fitted once on the
    training rows and then score both parts, so a supervised head can be fitted
    on training-row scores without any held-out row reaching it.
    """
    train_idx, test_idx = stratified_split(y, seed)
    mean = X[train_idx].mean(axis=0)
    spread = X[train_idx].std(axis=0)
    spread[spread < 1e-8] = 1.0
    fit_rows = (X[train_idx] - mean) / spread
    test_rows = (X[test_idx] - mean) / spread

    train_columns, test_columns = [], []
    for name in DETECTORS:
        detector = _fitted_detector(name, fit_rows, seed)
        train_columns.append(_rank_normalize(_read_scores(name, detector, fit_rows)))
        test_columns.append(_rank_normalize(_read_scores(name, detector, test_rows)))
    return (
        np.column_stack(train_columns),
        y[train_idx],
        np.column_stack(test_columns),
        y[test_idx],
    )


# ---------------------------------------------------------------------------
# The heads under measurement
# ---------------------------------------------------------------------------
def head_plain_mean(train: Any, y_train: Any, test: Any) -> Any:
    """The baseline: unweighted mean of the detector columns."""
    return test.mean(axis=1)


def head_best_single(train: Any, y_train: Any, test: Any) -> Any:
    """Strongest single detector chosen on the TRAINING rows only.

    A label-free-at-test reference: if a head cannot beat picking one detector
    by training AUROC, re-weighting is buying nothing.
    """
    aurocs = [auroc(train[y_train == 1, j], train[y_train == 0, j]) for j in range(train.shape[1])]
    return test[:, int(np.argmax(aurocs))]


def head_voting_ensemble(train: Any, y_train: Any, test: Any) -> Any:
    """``ml/ensemble.VotingEnsemble`` soft voting, scored by its ``predictions``.

    The binary ``predictions`` field is used, not ``confidence``, because
    ``confidence`` is ``|weighted_avg - 0.5| * 2`` -- symmetric about the
    threshold, so a confidently-normal row and a confidently-anomalous row
    receive the same value. That is a defensible meaning for "confidence in a
    binary call" but it is **not** an anomaly score, and ranking by it measures
    0.7566 against the plain mean's 0.9026 on cardio/seed 0 because half the
    ordering is inverted. ``test_unreached_ensemble_heads.py`` pins the
    non-monotonicity so the field cannot be mistaken for a score.

    This is the structural finding about this head: it exposes no continuous
    fused score at all, so it cannot serve as a ranker whatever its weights
    are. A binary AUROC is the most its contract supports.
    """
    from omni_mercury_engine.ml.ensemble import VotingEnsemble

    outputs = {name: test[:, j] for j, name in enumerate(DETECTORS)}
    result = VotingEnsemble(voting_type="soft").predict(outputs)
    return np.asarray(result["predictions"], dtype=float)


def head_stacking_meta_learner(train: Any, y_train: Any, test: Any) -> Any:
    """``ml/ensemble.MetaLearner`` trained on training-row detector scores."""
    import torch
    from torch import nn

    from omni_mercury_engine.ml.ensemble import MetaLearner

    torch.manual_seed(0)
    model = MetaLearner(input_dim=train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    features = torch.tensor(train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    for _ in range(300):
        optimizer.zero_grad()
        predicted = model(features)
        output = predicted.reshape(-1, 1)
        loss = loss_fn(output.clamp(1e-6, 1 - 1e-6), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return model(torch.tensor(test, dtype=torch.float32)).reshape(-1).numpy().astype(float)


def _coordinator_weighted(optimizer: Any, train: Any, y_train: Any, test: Any) -> Any:
    """Apply a ``WeightOptimizer`` fitted on the training-row scores.

    The optimizer's contract is
    ``optimize(detector_scores, labels, current_weights) -> weights``, so it is
    given the training-row score columns and their labels and starts from the
    uniform weights the plain-mean baseline uses. Any lift is therefore
    attributable to the re-weighting and nothing else.
    """
    uniform = {name: 1.0 / len(DETECTORS) for name in DETECTORS}
    scores = {name: train[:, j] for j, name in enumerate(DETECTORS)}
    weights = optimizer.optimize(scores, y_train.astype(np.int64), uniform)
    vector = np.array([float(weights.get(name, 0.0)) for name in DETECTORS], dtype=float)
    if not np.isfinite(vector).all() or vector.sum() <= 0:
        vector = np.ones(len(DETECTORS), dtype=float)
    return test @ (vector / vector.sum())


def head_bayesian_weights(train: Any, y_train: Any, test: Any) -> Any:
    """``ml/ensemble_coordinator.BayesianWeightOptimizer`` (Thompson sampling)."""
    from omni_mercury_engine.ml.ensemble_coordinator import BayesianWeightOptimizer

    return _coordinator_weighted(BayesianWeightOptimizer(seed=0), train, y_train, test)


def head_gradient_weights(train: Any, y_train: Any, test: Any) -> Any:
    """``ml/ensemble_coordinator.GradientWeightOptimizer``."""
    from omni_mercury_engine.ml.ensemble_coordinator import GradientWeightOptimizer

    return _coordinator_weighted(GradientWeightOptimizer(), train, y_train, test)


HEADS = {
    "plain_mean (baseline)": head_plain_mean,
    "best_single_on_train": head_best_single,
    "ensemble.VotingEnsemble": head_voting_ensemble,
    "ensemble.MetaLearner": head_stacking_meta_learner,
    "coordinator.Bayesian": head_bayesian_weights,
    "coordinator.Gradient": head_gradient_weights,
}


def main() -> int:
    """Measure every head against the plain mean on the recorded tables."""
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", default=list(ADBENCH_NAMES))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--out", default="artifacts/unreached_heads.json")
    args = parser.parse_args()

    per_head: dict[str, list[float]] = {name: [] for name in HEADS}
    per_cell: dict[str, dict[str, float]] = {}
    failures: dict[str, str] = {}

    for table in args.tables:
        X, y = adbench(table)
        for seed in args.seeds:
            train, y_train, test, y_test = _score_matrix(X, y, seed)
            cell = f"{table}/s{seed}"
            per_cell[cell] = {}
            for name, head in HEADS.items():
                try:
                    scores = np.asarray(head(train, y_train, test), dtype=float)
                    value = auroc(scores[y_test == 1], scores[y_test == 0])
                except Exception as exc:  # a head that cannot run is evidence too
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    continue
                per_head[name].append(value)
                per_cell[cell][name] = value
            print(
                f"  {cell:<18s} "
                + "  ".join(f"{k.split()[0][:22]}={v:.4f}" for k, v in per_cell[cell].items())
            )

    print("\n" + "=" * 86)
    baseline = float(np.mean(per_head["plain_mean (baseline)"]))
    print(f"{'head':<28s} {'mean AUROC':>11s} {'vs plain mean':>14s} {'cells won':>11s}")
    print("-" * 86)
    summary: dict[str, Any] = {}
    base_cells = per_cell
    for name in HEADS:
        values = per_head[name]
        if not values:
            print(f"{name:<28s} {'DID NOT RUN':>11s}   {failures.get(name, '')[:34]}")
            summary[name] = {"ran": False, "error": failures.get(name)}
            continue
        mean = float(np.mean(values))
        won = sum(
            1
            for cell, row in base_cells.items()
            if name in row and row[name] > row.get("plain_mean (baseline)", -1)
        )
        print(f"{name:<28s} {mean:>11.4f} {mean - baseline:>+14.4f} {won:>8d}/{len(values)}")
        summary[name] = {
            "ran": True,
            "mean_auroc": mean,
            "delta_vs_plain_mean": mean - baseline,
            "cells_beating_plain_mean": won,
            "n_cells": len(values),
        }

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"per_cell": per_cell, "summary": summary, "detectors": list(DETECTORS)},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nreport -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

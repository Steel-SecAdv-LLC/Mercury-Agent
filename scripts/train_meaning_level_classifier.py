#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train Mercury's offline meaning-level harm classifier.

Fits the linear model that :mod:`omni_mercury_engine.cognitive.meaning_level`
serves, on the in-house corpus in ``benchmarks/meaning_level_corpus.py``, and
writes the auditable JSON weight artifact the package ships.

Deterministic end to end: weights start at zero, the optimizer is full-batch
gradient descent, and the corpus generator uses no RNG -- so the same source
always produces byte-identical weights. There is nothing to seed.

    # retrain and write the shipped artifact
    PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py

    # generalization protocol: hold out whole request frames / act verbs
    PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py --cross-validate

    # train without writing (inspect metrics only)
    PYTHONPATH=src:benchmarks python scripts/train_meaning_level_classifier.py --dry-run

Why leave-group-out and not a random split
------------------------------------------

A random split of a compositional corpus measures almost nothing: the held-out
rows share their frame and their act verb with hundreds of training rows, so
near-perfect scores are guaranteed and meaningless. ``--cross-validate`` instead
holds out **entire request frames** and **entire act verbs**, so the test rows
use a request shape or a verb the model has never seen in any row. That is the
property the classifier actually needs -- the production-verb class is open and
cannot be enumerated -- and it is the falsifiable claim recorded in the model's
provenance block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks"))

from meaning_level_corpus import (
    build_training_corpus,
    corpus_summary,
)

from omni_mercury_engine.cognitive.meaning_level import (
    FEATURE_VERSION,
    WEIGHTS_PATH,
    MeaningLevelModel,
    ordered_features,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from meaning_level_corpus import TrainingRow

# Hyperparameters. Fixed, not searched against the held-out slice -- searching
# them against it would silently turn the test set into a validation set.
LEARNING_RATE = 1.0
EPOCHS = 400
L2 = 2e-4
#: Features whose |weight| falls below this are dropped from the artifact. Keeps
#: the shipped JSON small and readable; the pruned model is re-measured after
#: pruning so the reported numbers describe what actually ships.
PRUNE_BELOW = 0.02


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    e = math.exp(max(z, -60.0))
    return e / (1.0 + e)


def train(
    rows: Sequence[TrainingRow],
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    l2: float = L2,
) -> tuple[dict[str, float], float]:
    """Fit L2-regularized logistic regression; return ``(weights, bias)``.

    Full-batch gradient descent from a zero initialization, so the result is a
    deterministic function of ``rows`` alone.
    """
    # ordered_features(), not extract_features(): set iteration order varies with
    # Python's per-process string hash seed, and float addition is not
    # associative, so an unordered iteration makes the fitted weights differ in
    # the last bit between runs. Sorting makes the artifact byte-reproducible.
    featurized = [(ordered_features(r.text), float(r.label)) for r in rows]
    weights: dict[str, float] = {}
    bias = 0.0
    n = float(len(featurized)) or 1.0

    for _ in range(epochs):
        grad: dict[str, float] = {}
        grad_bias = 0.0
        for feats, label in featurized:
            z = bias + sum(weights.get(f, 0.0) for f in feats)
            err = _sigmoid(z) - label
            grad_bias += err
            for f in feats:
                grad[f] = grad.get(f, 0.0) + err
        bias -= lr * (grad_bias / n)
        for f, g in grad.items():
            w = weights.get(f, 0.0)
            weights[f] = w - lr * (g / n + l2 * w)
    return weights, bias


def evaluate(model: MeaningLevelModel, rows: Sequence[TrainingRow], threshold: float = 0.5) -> dict:
    """Return accuracy / precision / recall / AUROC of ``model`` over ``rows``."""
    scored = [(model.score(r.text), r.label) for r in rows]
    tp = sum(1 for s, y in scored if s >= threshold and y == 1)
    fp = sum(1 for s, y in scored if s >= threshold and y == 0)
    fn = sum(1 for s, y in scored if s < threshold and y == 1)
    tn = sum(1 for s, y in scored if s < threshold and y == 0)
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    # Rank-based AUROC (ties count a half), exact rather than trapezoidal.
    auroc = 0.0
    if pos and neg:
        wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
        auroc = wins / (len(pos) * len(neg))
    total = tp + fp + fn + tn
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 1.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
        "auroc": round(auroc, 4),
    }


def _leave_group_out(rows: Sequence[TrainingRow], key: str, folds: int) -> list[dict]:
    """Hold out whole ``key`` values (``"frame"`` or ``"act"``) and re-measure."""
    values = sorted({getattr(r, key) for r in rows if getattr(r, key)})
    if len(values) < folds:
        folds = max(1, len(values))
    results = []
    for f in range(folds):
        held = {v for i, v in enumerate(values) if i % folds == f}
        train_rows = [r for r in rows if getattr(r, key) not in held]
        test_rows = [r for r in rows if getattr(r, key) in held]
        if not test_rows or not train_rows:
            continue
        w, b = train(train_rows)
        metrics = evaluate(MeaningLevelModel(w, b), test_rows)
        metrics["held_out"] = sorted(held)[:4]
        metrics["n_held_values"] = len(held)
        results.append(metrics)
    return results


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--dry-run", action="store_true", help="train and report, write nothing")
    ap.add_argument(
        "--cross-validate",
        action="store_true",
        help="run the leave-frame-out / leave-act-out generalization protocol",
    )
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    rows = build_training_corpus()
    summary = corpus_summary()
    print(
        f"corpus: {summary['total']} rows ({summary['offensive']} offensive / "
        f"{summary['benign']} benign), groups={summary['by_group']}"
    )

    generalization: dict[str, object] = {}
    if args.cross_validate:
        # Two fold counts for the act axis on purpose. A 4-fold split holds out
        # 25% of ALL act verbs at once -- including the defensive ones -- which
        # is a deliberately harsh stress condition: with a quarter of the
        # professional vocabulary unseen, the model falls back on the frame and
        # over-predicts offensive, so precision is the number that moves. The
        # 10-fold split is closer to the deployment reality of meeting one novel
        # verb at a time. Both are reported; neither is hidden.
        configurations = (
            ("frame", "leave-frame-out", args.folds),
            ("act", "leave-act-out", args.folds),
            ("act", "leave-act-out-fine", 10),
        )
        for key, label, folds in configurations:
            fold_results = _leave_group_out(rows, key, folds)
            agg = {
                "folds": len(fold_results),
                "mean_auroc": _mean([r["auroc"] for r in fold_results]),
                "mean_accuracy": _mean([r["accuracy"] for r in fold_results]),
                "mean_recall": _mean([r["recall"] for r in fold_results]),
                "mean_precision": _mean([r["precision"] for r in fold_results]),
            }
            generalization[label] = agg
            print(f"  {label}: {agg}")

    weights, bias = train(rows)
    full = MeaningLevelModel(weights, bias)
    before = evaluate(full, rows)

    pruned_weights = {f: round(w, 6) for f, w in weights.items() if abs(w) >= PRUNE_BELOW}
    model = MeaningLevelModel(
        pruned_weights,
        bias,
        FEATURE_VERSION,
        {
            "trainer": "scripts/train_meaning_level_classifier.py",
            "corpus": "benchmarks/meaning_level_corpus.py",
            "corpus_summary": summary,
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "l2": L2,
                "prune_below": PRUNE_BELOW,
            },
            "features_before_prune": len(weights),
            "features_shipped": len(pruned_weights),
            "fit_metrics": evaluate(MeaningLevelModel(pruned_weights, bias), rows),
            "generalization": generalization,
            "determinism": "zero-init full-batch gradient descent; no RNG anywhere",
        },
    )
    after = model.metadata["fit_metrics"]
    print(
        f"features: {len(weights)} fit -> {len(pruned_weights)} shipped " f"(|w| >= {PRUNE_BELOW})"
    )
    print(f"in-sample before prune: {before}")
    print(f"in-sample after  prune: {after}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=1, sort_keys=True)
        fh.write("\n")
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

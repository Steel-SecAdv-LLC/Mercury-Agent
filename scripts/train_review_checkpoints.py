# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train, calibrate and measure the shipped artifact-review checkpoints.

One checkpoint per kind (``review_document.pt``, ``review_image.pt``,
``review_audio.pt``), each made the same way:

1. **fit** the fusion network through the real :meth:`OmniMercuryEngine.fit_fusion`
   path on the self-standardized segments of reference artifacts, labelled by
   segment (spliced or not);
2. **calibrate** a batch-level conformal certificate
   (:meth:`OmniMercuryEngine.calibrate_fusion_conformal_batches`) on reference
   artifacts the fit never saw, one statistic per artifact, which also stores
   the family those artifacts form;
3. **measure** it on a third, disjoint set through the whole review path, and
   on artifacts unlike the family;
4. write what was measured into the checkpoint's provenance as its
   ``certificate_scope``: the statement every decision made with it repeats.

The reference artifacts are seeded and need no network
(``research/review/reference_artifacts.py``).  The independent re-measurement
is ``research/review/harness_artifact_review.py``.

Usage::

    python -m scripts.train_review_checkpoints                 # all three
    python -m scripts.train_review_checkpoints --kind audio    # one
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omni_mercury_engine import review
from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.ml.fusion_network import OmniFusionModel
from omni_mercury_engine.ml.inference import FusionInference
from omni_mercury_engine.review import features
from omni_mercury_engine.review.features import extract
from research.review import (
    harness_artifact_review as harness,
    reference_artifacts as R,
)

#: Reference artifacts per stage (half of each spliced).
N_TRAIN, N_CALIBRATION, N_HELD_OUT, N_OUTSIDE = 60, 300, 300, 24
EPOCHS = 60

#: Training draws tried per kind, in this order. The fit is deterministic
#: given a seed, and the document kind is fragile to the draw (measured
#: 2026-11-21: clean batch coverage from 0.28 to 1.00 over five seeds), so
#: the draw is selected. Selection is ordinary train / validation / report:
#: each candidate is fitted on the train split, its certificate is fitted on
#: the calibration split, and the quantities the published bars are about
#: (per-class batch coverage, wrong-decision rate) are computed on the
#: TRAINER'S held-out draw (seed 3303) -- the validation set. The published
#: numbers come from the independent harness draw (seed 4404), which takes
#: no part in selection; the two draws share the evaluation sources and
#: nothing else. Every candidate's validation numbers are written into the
#: provenance. Two cheaper criteria were tried first and recorded here so
#: they are not re-tried: batch-max AUROC on the calibration split picked an
#: image draw whose validation clean coverage was 79.3% (AUROC ranks
#: artifacts but ignores the clean score tail the conformal threshold sits
#: in), and two-fold cross-conformal coverage on the calibration split
#: picked a document draw at 79.3% as well (a 150-batch coverage estimate
#: carries about +/-5 points of noise, and the calibration split cannot say
#: how a fit behaves on sources it never saw, which is what the document
#: bars are about).
SEED_CANDIDATES = (0, 1, 2, 3, 4)
#: The fusion width the bundled default checkpoint ships with (see
#: scripts/train_default_fusion.py for the capacity analysis behind it).
HIDDEN_DIM = 64


def _segments(
    kind: str, count: int, seed: int, split: str
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int, np.ndarray[Any, Any]]]:
    """Reference artifacts as (standardized matrix, segment labels, artifact label, raw levels)."""
    built = []
    for reference in R.references(kind, count, seed, split):
        segmented = extract(reference.decoded)
        built.append(
            (
                segmented.matrix,
                R.segment_labels(segmented, reference),
                int(reference.spliced),
                segmented.levels,
            )
        )
    return built


def _fit_once(
    kind: str,
    rows: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    seed: int,
) -> tuple[OmniMercuryEngine, dict[str, Any]]:
    """One deterministic fit of the fusion network on the kind's segments."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    if engine.fusion_model.hidden_dim != HIDDEN_DIM:
        engine.fusion_model = OmniFusionModel(hidden_dim=HIDDEN_DIM).to(engine.device)
        engine.fusion_inference = FusionInference(
            model=engine.fusion_model, device=str(engine.device)
        )
    metrics = engine.fit_fusion(
        rows,
        labels,
        epochs=EPOCHS,
        batch_size=64,
        early_stopping_patience=10,
        conformal_calibrate=False,  # the certificate is batch-level, fitted after selection
        seed=seed,
    )
    return engine, metrics


def _validation_quality(
    engine: OmniMercuryEngine,
    calibration: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int, np.ndarray[Any, Any]]],
    validation: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int, np.ndarray[Any, Any]]],
) -> dict[str, Any]:
    """The bar quantities of one fitted candidate, on the validation draw.

    A certificate is fitted on the calibration artifacts exactly as the kept
    model's will be, and read on the validation artifacts: per-class coverage
    of the 90% set, the wrong-decision rate of grounded singletons, and the
    batch-max AUROC. The harness draw is not touched.
    """
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    def batch_scores(
        artifacts: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int, np.ndarray[Any, Any]]
        ],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        scores = [
            float(np.max(np.asarray(engine.score_fusion(matrix), dtype=np.float64)))
            for matrix, _, _, _ in artifacts
        ]
        return np.array(scores), np.array([label for _, _, label, _ in artifacts])

    conformal = BinaryConformalClassifier(coverage=harness.COVERAGE).fit(*batch_scores(calibration))
    scores, truth = batch_scores(validation)
    sets = conformal.predict(scores)
    kept = {0: sets.contains_normal, 1: sets.contains_anomaly}
    coverage = {
        "clean": float(np.mean(kept[0][truth == 0])),
        "spliced": float(np.mean(kept[1][truth == 1])),
    }
    singleton = sets.set_size == 1
    wrong = {
        "clean": float(np.mean(singleton[truth == 0] & kept[1][truth == 0])),
        "spliced": float(np.mean(singleton[truth == 1] & kept[0][truth == 1])),
    }
    return {
        "coverage": coverage,
        "grounded_wrong": wrong,
        "floor": float(min(coverage.values())),
        "worst_wrong": float(max(wrong.values())),
        "batch_auroc": float(roc_auc_score(truth, scores)),
    }


def train(kind: str, out: Path) -> dict[str, Any]:
    """Fit, calibrate, measure and save the review checkpoint for ``kind``.

    Returns:
        The provenance written into the checkpoint.
    """
    fit_set = _segments(kind, N_TRAIN, harness.SEEDS["train"], "train")
    rows = np.vstack([matrix for matrix, _, _, _ in fit_set])
    labels = np.concatenate([segment_labels for _, segment_labels, _, _ in fit_set])
    print(
        f"[{kind}] fitting on {rows.shape[0]} segments of {len(fit_set)} reference artifacts "
        f"({int(labels.sum())} spliced segments)..."
    )
    calibration = _segments(kind, N_CALIBRATION, harness.SEEDS["calibration"], "calibration")
    validation = _segments(kind, N_HELD_OUT, harness.SEEDS["trainer_held_out"], "evaluation")

    # Selection is train / validation / report (see SEED_CANDIDATES).
    candidates: dict[str, dict[str, Any]] = {}
    for offset in SEED_CANDIDATES:
        seed = harness.SEEDS["train"] + offset
        tried, _ = _fit_once(kind, rows, labels, seed)
        quality = _validation_quality(tried, calibration, validation)
        candidates[str(seed)] = quality
        print(
            f"[{kind}] seed {seed}: validation coverage clean "
            f"{quality['coverage']['clean']:.3f} / spliced {quality['coverage']['spliced']:.3f}, "
            f"worst wrong {quality['worst_wrong']:.3f}, batch AUROC {quality['batch_auroc']:.4f}"
        )
    selected = int(
        max(
            candidates,
            key=lambda key: (
                candidates[key]["floor"],
                -candidates[key]["worst_wrong"],
                candidates[key]["batch_auroc"],
            ),
        )
    )
    print(f"[{kind}] selected training seed {selected}")
    engine, metrics = _fit_once(kind, rows, labels, selected)
    # Each calibration artifact also says what it is like as a whole (its raw
    # feature levels, which self-standardization removed), so the family test
    # can tell an upload of another kind from one of these.
    report = engine.calibrate_fusion_conformal_batches(
        [matrix for matrix, _, _, _ in calibration],
        [label for _, _, label, _ in calibration],
        coverage=harness.COVERAGE,
        descriptors=[levels for _, _, _, levels in calibration],
        descriptor_names=[
            f"{name}:level" for name in getattr(features, f"{kind.upper()}_FEATURES")
        ],
    )
    print(
        f"[{kind}] batch-level certificate on {report['n_calibration']} artifacts: "
        f"achieved bound {report['coverage_achieved']:.3f}"
    )

    engine.enable_decision_layer()
    # The scope statement's own numbers come from a draw that took no part in
    # the selection above and none in the harness's published measurement.
    held_out = harness.measure(engine, kind, N_HELD_OUT, harness.SEEDS["trainer_report"])
    outside = harness.measure_outside(engine, kind, N_OUTSIDE, harness.SEEDS["trainer_report"] + 1)
    scope = harness.scope_statement(kind, report["n_calibration"], held_out, outside)
    provenance: dict[str, Any] = {
        "source": "review_reference_artifacts",
        "kind": kind,
        "features": list(getattr(features, f"{kind.upper()}_FEATURES")),
        "seeds": dict(harness.SEEDS),
        "training_seed_selection": {
            "candidates": candidates,
            "selected": selected,
            "criterion": (
                "highest per-class coverage minimum on the trainer's held-out draw "
                "(validation; seed 3303), then the lower wrong-decision maximum, then "
                "batch-max AUROC. The harness draw (seed 4404) that produces the "
                "published numbers takes no part; SEED_CANDIDATES records the two "
                "calibration-only criteria tried and dropped first"
            ),
        },
        "artifacts": {"fit": len(fit_set), "calibration": len(calibration), "held_out": N_HELD_OUT},
        # Documents are cut from the repository's documentation: which state of
        # it this checkpoint was fitted, calibrated and measured on.
        "corpus": R.corpus_digest() if kind == review.DOCUMENT else None,
        "fit": {
            "segments": int(rows.shape[0]),
            "spliced_segments": int(labels.sum()),
            "seed": selected,
            "best_loss": float(metrics["best_loss"]),
            "temperature": metrics.get("temperature"),
        },
        "conformal_calibration": {
            "regime": report["regime"],
            "coverage": harness.COVERAGE,
            "coverage_achieved": report["coverage_achieved"],
            "n_cal": report["n_calibration"],
            "held_out": {
                "coverage": held_out["coverage"],
                "rates": held_out["rates"],
                "row_auroc": held_out["row_auroc"],
                "localisation": held_out["localisation"],
            },
            "outside_the_family": outside,
        },
        "certificate_scope": scope,
    }
    engine._fusion_provenance = provenance
    out.parent.mkdir(parents=True, exist_ok=True)
    engine.save_model(str(out))

    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(str(out))
    assert fresh._fusion_conformal_regime == "batch_max", "the regime must survive the round trip"
    assert fresh._fusion_batch_family is not None, "the batch family must survive the round trip"
    assert fresh._fusion_provenance == provenance, "provenance must round-trip via the checkpoint"
    probe = calibration[0][0]
    drift = float(np.max(np.abs(engine.score_fusion(probe) - fresh.score_fusion(probe))))
    assert drift < 1e-3, f"save->load probability drift {drift} exceeds 1e-3"
    print(f"[{kind}] saved {out.name} ({out.stat().st_size / 1024:.0f} KB); scope: {scope}")
    return provenance


def main() -> int:
    """Train the requested review checkpoints."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--kind", choices=review.KINDS, action="append", help="default: all three")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    for kind in args.kind or review.KINDS:
        train(kind, review.checkpoint_path(kind))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

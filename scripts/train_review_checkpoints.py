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
#: The fusion width the bundled default checkpoint ships with (see
#: scripts/train_default_fusion.py for the capacity analysis behind it).
HIDDEN_DIM = 64


def _segments(
    kind: str, count: int, seed: int, split: str
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]]:
    """Reference artifacts as (standardized matrix, segment labels, artifact label)."""
    built = []
    for reference in R.references(kind, count, seed, split):
        segmented = extract(reference.decoded)
        built.append(
            (segmented.matrix, R.segment_labels(segmented, reference), int(reference.spliced))
        )
    return built


def train(kind: str, out: Path) -> dict[str, Any]:
    """Fit, calibrate, measure and save the review checkpoint for ``kind``.

    Returns:
        The provenance written into the checkpoint.
    """
    import torch

    torch.manual_seed(harness.SEEDS["train"])
    np.random.seed(harness.SEEDS["train"])
    fit_set = _segments(kind, N_TRAIN, harness.SEEDS["train"], "train")
    rows = np.vstack([matrix for matrix, _, _ in fit_set])
    labels = np.concatenate([segment_labels for _, segment_labels, _ in fit_set])
    print(
        f"[{kind}] fitting on {rows.shape[0]} segments of {len(fit_set)} reference artifacts "
        f"({int(labels.sum())} spliced segments)..."
    )
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
        conformal_calibrate=False,  # the certificate is batch-level, fitted below
        seed=harness.SEEDS["train"],
    )

    calibration = _segments(kind, N_CALIBRATION, harness.SEEDS["calibration"], "calibration")
    report = engine.calibrate_fusion_conformal_batches(
        [matrix for matrix, _, _ in calibration],
        [label for _, _, label in calibration],
        coverage=harness.COVERAGE,
    )
    print(
        f"[{kind}] batch-level certificate on {report['n_calibration']} artifacts: "
        f"achieved bound {report['coverage_achieved']:.3f}"
    )

    engine.enable_decision_layer()
    held_out = harness.measure(engine, kind, N_HELD_OUT, harness.SEEDS["trainer_held_out"])
    outside = harness.measure_outside(
        engine, kind, N_OUTSIDE, harness.SEEDS["trainer_held_out"] + 1
    )
    scope = harness.scope_statement(kind, report["n_calibration"], held_out, outside)
    provenance: dict[str, Any] = {
        "source": "review_reference_artifacts",
        "kind": kind,
        "features": list(getattr(features, f"{kind.upper()}_FEATURES")),
        "seeds": dict(harness.SEEDS),
        "artifacts": {"fit": len(fit_set), "calibration": len(calibration), "held_out": N_HELD_OUT},
        "fit": {
            "segments": int(rows.shape[0]),
            "spliced_segments": int(labels.sum()),
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

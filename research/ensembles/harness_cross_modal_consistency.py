# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-modal consistency as an abstention factor, measured before fusing.

The proposal: agreement among the engine's already-computed detector streams
as an additional demotion factor — a flagged row the streams disagree about
defers.  The streams score on different scales, so agreement is rank-based:
for each stream, the row's rank percentile within the batch; a row's
``agreement`` is the mean percentile across streams (1.0 = every stream
ranks it at its own top).

Measured on the five recorded ADBench tables x three seeds through the
engine's real ``_extract_detector_features`` path (detectors fitted on the
train split, scored on the held-out split; flagged = detector-mean ensemble
above the train 95th percentile): the demotion factor is adoptable only if
low agreement is enriched in FALSE positives — otherwise a floor demotes
true detections as often as false ones.

Measured verdict (recorded in ``artifacts/cross_modal_consistency.json``):
among flagged rows, mean agreement is ~0.943 on false positives vs ~0.953 on
true positives — the wrong-side-or-nothing regime (TP agreement higher on
only 5 of 9 evaluations with both classes above 0.93), so the factor is
REFUSED: it carries no usable false-positive discrimination on these tables.

Run from the repository root::

    python research/ensembles/harness_cross_modal_consistency.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.recorded_data import ADBENCH_NAMES, adbench, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "cross_modal_consistency.json"
SEEDS = (0, 1, 2)
FLAG_QUANTILE = 0.95


def _rank_percentile(column: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(column))
    return order / max(len(column) - 1, 1)


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from omni_mercury_engine.engine import OmniMercuryEngine

    per_dataset: dict[str, dict[str, float | int | None]] = {}
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            x_train = (features[train] - mean) / spread
            x_test = (features[test] - mean) / spread

            engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
            _, scores_train, _, _ = engine._extract_detector_features(x_train)
            _, scores_test, _, _ = engine._extract_detector_features(x_test)
            names = sorted(scores_test)
            train_streams = {k: np.asarray(scores_train[k], float).ravel() for k in names}
            test_streams = {k: np.asarray(scores_test[k], float).ravel() for k in names}
            ensemble_train = np.mean([train_streams[k] for k in names], axis=0)
            ensemble_test = np.mean([test_streams[k] for k in names], axis=0)
            flagged = ensemble_test >= float(np.quantile(ensemble_train, FLAG_QUANTILE))
            agreement = np.column_stack([_rank_percentile(test_streams[k]) for k in names]).mean(
                axis=1
            )

            y_test = labels[test]
            fp = flagged & (y_test == 0)
            tp = flagged & (y_test == 1)
            per_dataset[f"{table}/seed{seed}"] = {
                "n_flagged": int(flagged.sum()),
                "n_false_positive": int(fp.sum()),
                "n_true_positive": int(tp.sum()),
                "agreement_false_positive": (float(agreement[fp].mean()) if fp.any() else None),
                "agreement_true_positive": (float(agreement[tp].mean()) if tp.any() else None),
            }

    pairs = [
        (row["agreement_false_positive"], row["agreement_true_positive"])
        for row in per_dataset.values()
        if row["agreement_false_positive"] is not None
        and row["agreement_true_positive"] is not None
    ]
    fp_mean = float(np.mean([p[0] for p in pairs]))
    tp_mean = float(np.mean([p[1] for p in pairs]))
    tp_higher = int(sum(1 for fp_a, tp_a in pairs if tp_a > fp_a))
    # Adoptable only if low agreement is enriched in false positives: the FP
    # mean must sit meaningfully BELOW the TP mean, consistently.
    adopted = bool(fp_mean < tp_mean - 0.05 and tp_higher >= int(0.8 * len(pairs)))

    artifact = {
        "schema": "mercury.research.cross_modal_consistency/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "agreement": "mean rank percentile of the row across detector streams",
            "flagged": f"detector-mean ensemble >= train q{FLAG_QUANTILE}",
            "bar": "adopt only if FP agreement < TP agreement - 0.05 on >= 80% of evaluations",
            "n_evaluations": len(per_dataset),
        },
        "per_dataset": per_dataset,
        "mean_agreement_false_positive": round(fp_mean, 4),
        "mean_agreement_true_positive": round(tp_mean, 4),
        "tp_agreement_higher_count": tp_higher,
        "comparable_evaluations": len(pairs),
        "adopted": adopted,
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"agreement: FP {fp_mean:.4f} vs TP {tp_mean:.4f} "
        f"(TP higher on {tp_higher}/{len(pairs)}) -> "
        f"{'ADOPT' if adopted else 'refused'}"
    )
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

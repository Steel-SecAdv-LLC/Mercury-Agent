# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vector isotonic calibration + conformal sets, measured on the recorded tables.

The detector's fused score is a deviation magnitude, not a probability, and
every calibrator in the repository consumed one scalar. This harness measures
the opt-in vector calibration (``MathArrestDetector.fit_vector_calibration``):
the isotonic map over the weighted (probes + fused) index, applied in the
decision path, and the Mondrian conformal predictor's {normal, anomaly} sets.

Protocol, per recorded ADBench table x three seeds: stratified 49% fit / 21%
labelled calibration / 30% held-out test. Reported per condition:

* ECE (15-bin) and Brier of the decision score read as a probability,
* F1 at each condition's own operating point (the raw path's conformal cut;
  the calibrated path's 0.5),
* AUROC, plus mean rank between the two conditions across tables,
* conformal empirical coverage (overall and per class) and average set size
  at alpha = 0.1, against the finite-sample ``1 − alpha`` guarantee.

Results go to ``artifacts/math_arrest_vector_calibration.json``;
``tests/detectors/test_vector_calibration.py`` pins the guarantees and the
calibration gains.

Run from the repository root::

    python research/math_arrest/harness_vector_calibration.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.core.calibration import compute_ece
from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
from omni_mercury_engine.ml.mercury_ml import brier_score_loss
from research.math_arrest.harness_probe_redundancy import _git_commit
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "math_arrest_vector_calibration.json"
SEEDS = (0, 1, 2)
ALPHA = 0.1


def _f1(predictions: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    predictions = np.asarray(predictions, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    tp = int((predictions & labels).sum())
    fp = int((predictions & ~labels).sum())
    fn = int((~predictions & labels).sum())
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def run() -> dict[str, Any]:
    """The full measurement; returns the artifact payload."""
    rows: dict[str, dict[str, list[float]]] = {}

    def record(metric: str, table: str, value: float) -> None:
        rows.setdefault(metric, {}).setdefault(table, []).append(value)

    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train_all, test = stratified_split(labels, seed)
            fit_local, cal_local = stratified_split(labels[train_all], seed)
            fit_rows = train_all[fit_local]
            cal_rows = train_all[cal_local]

            mean = features[fit_rows].mean(axis=0)
            spread = features[fit_rows].std(axis=0)
            spread[spread < 1e-8] = 1.0

            def z(rows_idx: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
                return (features[rows_idx] - mean) / spread

            detector = MathArrestDetector()
            detector.fit(z(fit_rows))

            y_test = labels[test]
            raw = detector.detect(z(test))
            raw_scores = np.asarray(raw["scores"], dtype=np.float64)
            record("raw_auc", table, auroc(raw_scores[y_test == 1], raw_scores[y_test == 0]))
            record("raw_ece", table, compute_ece(y_test, raw_scores))
            record("raw_brier", table, float(brier_score_loss(y_test, raw_scores)))
            record("raw_f1", table, _f1(raw["is_anomaly"], y_test == 1))

            detector.fit_vector_calibration(z(cal_rows), labels[cal_rows], alpha=ALPHA)
            calibrated = detector.detect(z(test))
            cal_scores = np.asarray(calibrated["calibrated_scores"], dtype=np.float64)
            record("cal_auc", table, auroc(cal_scores[y_test == 1], cal_scores[y_test == 0]))
            record("cal_ece", table, compute_ece(y_test, cal_scores))
            record("cal_brier", table, float(brier_score_loss(y_test, cal_scores)))
            record("cal_f1", table, _f1(calibrated["is_anomaly"], y_test == 1))

            sets = np.asarray(calibrated["conformal_sets"], dtype=bool)
            covered = sets[np.arange(len(y_test)), y_test]
            record("coverage", table, float(covered.mean()))
            for cls in (0, 1):
                member = covered[y_test == cls]
                if member.size:
                    record(f"coverage_class_{cls}", table, float(member.mean()))
            record("avg_set_size", table, float(sets.sum(axis=1).mean()))

            # Cross-probe consistency gate impact: the served sets include the
            # don't-know routing; the ungated sets are the raw p-value cut.
            routed = np.asarray(calibrated["dont_know_routed"], dtype=bool)
            p_values = np.asarray(calibrated["conformal_p_values"], dtype=np.float64)
            ungated = p_values > ALPHA
            ungated_covered = ungated[np.arange(len(y_test)), y_test]
            record("routed_fraction", table, float(routed.mean()))
            record("coverage_ungated", table, float(ungated_covered.mean()))
            record("avg_set_size_ungated", table, float(ungated.sum(axis=1).mean()))

    per_table = {
        metric: {table: round(float(np.mean(v)), 4) for table, v in tables.items()}
        for metric, tables in rows.items()
    }
    means = {metric: round(float(np.mean(list(t.values()))), 4) for metric, t in per_table.items()}

    # Mean rank between raw and calibrated, by AUROC and by F1 (1 = better).
    ranks: dict[str, dict[str, float]] = {}
    for metric in ("auc", "f1"):
        raw_total = cal_total = 0.0
        for table in ADBENCH_NAMES:
            raw_value = per_table[f"raw_{metric}"][table]
            cal_value = per_table[f"cal_{metric}"][table]
            if raw_value == cal_value:
                raw_total += 1.5
                cal_total += 1.5
            elif cal_value > raw_value:
                cal_total += 1.0
                raw_total += 2.0
            else:
                cal_total += 2.0
                raw_total += 1.0
        ranks[metric] = {
            "raw": round(raw_total / len(ADBENCH_NAMES), 4),
            "calibrated": round(cal_total / len(ADBENCH_NAMES), 4),
        }

    return {
        "schema": "mercury.research.math_arrest_vector_calibration/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config": {"tables": list(ADBENCH_NAMES), "seeds": list(SEEDS), "alpha": ALPHA},
        "per_table": per_table,
        "means": means,
        "mean_rank": ranks,
    }


def main() -> int:
    artifact = run()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    means = artifact["means"]
    print(f"{'metric':12s} {'raw':>8s} {'calibrated':>11s}")
    for metric in ("ece", "brier", "f1", "auc"):
        print(f"{metric:12s} {means[f'raw_{metric}']:8.4f} {means[f'cal_{metric}']:11.4f}")
    print(
        f"\nconformal (alpha={ALPHA}): coverage {means['coverage']:.4f} "
        f"(class 0 {means['coverage_class_0']:.4f}, class 1 {means['coverage_class_1']:.4f}), "
        f"avg set size {means['avg_set_size']:.4f}"
    )
    print(
        f"consistency gate: routed {means['routed_fraction']:.4f} of samples; "
        f"ungated coverage {means['coverage_ungated']:.4f} -> gated {means['coverage']:.4f}, "
        f"ungated set size {means['avg_set_size_ungated']:.4f} -> {means['avg_set_size']:.4f}"
    )
    print(f"mean rank: {artifact['mean_rank']}")
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

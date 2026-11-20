# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The hub's sound calibration split and conformal certificate, measured.

The hub used to fit its neural encoder on every row — calibration rows
included — and then fit Platt on *neural* scores of those already-seen rows
while serving Platt over *fused* scores: a split-ordering leak and a
score-function mismatch, and the reason its docstring disclaimed any
conformal certificate.  The fit now draws the calibration split first,
fits every component on the non-calibration rows only, scores the
calibration rows through the served path, and sub-splits them (stratified)
into a Platt half and a conformal half, so the certificate's rows were
touched by no fitted component.

This harness measures what that soundness costs and what it buys, on the
five recorded ADBench tables x three seeds:

* held-out test AUC of the served fused score under the new scheme, beside
  an encoder fitted on ALL training rows (``use_calibration=False``) — the
  upper bound on what withholding the calibration rows costs;
* the Brier score of the served calibrated probability on the test rows;
* the certificate's stated achieved bound and its empirical held-out
  coverage, with singleton/ambiguous/empty rates.

Writes ``artifacts/hub_conformal.json``.
``tests/core/test_hub_conformal.py`` pins the soundness properties.

Run from the repository root::

    python research/neurosymbolic/harness_hub_conformal.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "hub_conformal.json"
SEEDS = (0, 1, 2)


def _evaluate(table: str, seed: int) -> dict[str, object]:
    features, labels = adbench(table)
    train, test = stratified_split(labels, seed)
    mean = features[train].mean(axis=0)
    spread = features[train].std(axis=0)
    spread[spread < 1e-8] = 1.0
    x_train = (features[train] - mean) / spread
    x_test = (features[test] - mean) / spread
    y_train, y_test = labels[train], labels[test]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hub = NeuroSymbolicHub(input_dim=features.shape[1], seed=seed)
        hub.fit(x_train, y_train)
        outputs = hub.predict(x_test, return_explanations=False)

        reference = NeuroSymbolicHub(input_dim=features.shape[1], seed=seed, use_calibration=False)
        reference.fit(x_train, y_train)
        ref_outputs = reference.predict(x_test, return_explanations=False)

    fused = np.array([float(o.anomaly_score) for o in outputs])
    ref_fused = np.array([float(o.anomaly_score) for o in ref_outputs])
    calibrated = np.array(
        [float(o.calibrated_score) if o.calibrated_score is not None else np.nan for o in outputs]
    )
    sets = [o.conformal for o in outputs]

    result: dict[str, object] = {
        "auc_sound_split": auroc(fused[y_test == 1], fused[y_test == 0]),
        "auc_encoder_on_all_train": auroc(ref_fused[y_test == 1], ref_fused[y_test == 0]),
    }
    if np.isfinite(calibrated).all():
        result["brier_calibrated"] = float(np.mean((calibrated - y_test) ** 2))
    if all(s is not None for s in sets) and sets:
        contains = np.array(
            [int(y) in set(s["prediction_set"]) for y, s in zip(y_test, sets, strict=True)]
        )
        sizes = np.array([s["set_size"] for s in sets], dtype=int)
        result["conformal"] = {
            "stated_coverage": float(sets[0]["coverage"]),
            "n_calibration": sets[0]["n_calibration"],
            "empirical_coverage": float(contains.mean()),
            "singleton_rate": float(np.mean(sizes == 1)),
            "ambiguous_rate": float(np.mean(sizes == 2)),
            "empty_rate": float(np.mean(sizes == 0)),
        }
    return result


def main() -> int:
    per_dataset: dict[str, dict[str, object]] = {}
    for table in ADBENCH_NAMES:
        for seed in SEEDS:
            name = f"{table}/seed{seed}"
            per_dataset[name] = _evaluate(table, seed)
            row = per_dataset[name]
            conformal = row.get("conformal") or {}
            print(
                f"{name:18s} AUC {row['auc_sound_split']:.4f} "
                f"(encoder-on-all {row['auc_encoder_on_all_train']:.4f}) "
                f"coverage {conformal.get('empirical_coverage', float('nan')):.4f} "
                f"stated {conformal.get('stated_coverage', float('nan')):.2f}"
            )

    sound = [float(r["auc_sound_split"]) for r in per_dataset.values()]
    full = [float(r["auc_encoder_on_all_train"]) for r in per_dataset.values()]
    covered = [
        r["conformal"]["empirical_coverage"]
        for r in per_dataset.values()
        if isinstance(r.get("conformal"), dict)
    ]
    stated = [
        r["conformal"]["stated_coverage"]
        for r in per_dataset.values()
        if isinstance(r.get("conformal"), dict)
    ]
    artifact = {
        "schema": "mercury.research.hub_conformal/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "split": "calibration drawn before the encoder fit; stratified "
            "Platt/conformal halves; certificate on rows no fitted component touched",
            "n_evaluations": len(per_dataset),
        },
        "per_dataset": per_dataset,
        "mean": {
            "auc_sound_split": round(float(np.mean(sound)), 4),
            "auc_encoder_on_all_train": round(float(np.mean(full)), 4),
            "split_cost_upper_bound": round(float(np.mean(full) - np.mean(sound)), 4),
            "empirical_coverage": round(float(np.mean(covered)), 4) if covered else None,
            "stated_coverage": round(float(np.mean(stated)), 4) if stated else None,
        },
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["mean"], indent=1))
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

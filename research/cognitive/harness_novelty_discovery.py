# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the CuriosityEngine's novelty score a real discovery instrument? Measured.

The engine scores an observation by its diagonal-Mahalanobis distance from
the running distribution of what it has seen.  It shipped opt-in and
advisory: no number said whether its score separates genuinely unseen
structure from the seen, so nothing could justify serving it.  This harness
closes that: on the five recorded ADBench tables x three seeds, the engine
observes the fit split's normal rows (exactly what the detector itself fits
on) and the held-out rows are scored two ways —

* ``read_only`` — :meth:`CuriosityEngine.novelty`, the measurement form:
  the estimate is fixed, every held-out row scored against it;
* ``absorbing`` — :meth:`CuriosityEngine.explore`, the serving form, which
  folds each scored row in (the estimate drifts toward what it scores).

The novelty score's AUROC against the table's labels is recorded per
evaluation, per protocol.  The signal is *measured* to discriminate (mean
read-only AUROC ~0.93); it is served as a signal in the record's cognitive
block, never as a verdict-mover — adopting it as a decision factor would
need its own harness against the standing keep bar.

The score discriminating says nothing about the VERDICT, ``is_novel``, which
is a cut.  So the verdict's operating point is measured too: the share of
held-out *normal* rows it calls novel (it should be small, and it should be
what the rule says it is) and the share of anomalies it calls novel.  Two
rules, on the same rows:

* ``score_cut`` — what shipped: novel when the score is at or above 0.7.
* ``rank`` — novel when the distance ranks in the top 5% of the reference
  distances (:func:`omni_mercury_engine.discovery.rank_p_value`).

Bar for the rank rule, set before it was run: normal rows called novel at
most 0.07 on average and at most 0.10 on any table.

Writes ``artifacts/novelty_discovery.json``.
``tests/cognitive/test_novelty_discovery.py`` pins the floor.

Run from the repository root::

    python research/cognitive/harness_novelty_discovery.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.cognitive.cognitive_evolution_engine import CuriosityEngine
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "novelty_discovery.json"
SEEDS = (0, 1, 2)


def main() -> int:
    per_dataset: dict[str, dict[str, float]] = {}
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            fit_rows = (features[train] - mean) / spread
            held_rows = (features[test] - mean) / spread
            held_labels = labels[test]

            engine = CuriosityEngine()
            for row in fit_rows[labels[train] == 0]:
                engine.explore("fit", row)

            scored = [engine.novelty("held", row) for row in held_rows]
            read_only = np.array([result.novelty_score for result in scored])
            # The verdict, both rules, on the same read-only scores.
            by_rank = np.array([result.is_novel for result in scored])
            by_cut = read_only >= engine.novelty_threshold
            normal, anomalous = held_labels == 0, held_labels == 1
            absorbing_engine = CuriosityEngine()
            for row in fit_rows[labels[train] == 0]:
                absorbing_engine.explore("fit", row)
            absorbing = np.array(
                [absorbing_engine.explore("held", row).novelty_score for row in held_rows]
            )

            name = f"{table}/seed{seed}"
            per_dataset[name] = {
                "read_only_auroc": auroc(read_only[held_labels == 1], read_only[held_labels == 0]),
                "absorbing_auroc": auroc(absorbing[held_labels == 1], absorbing[held_labels == 0]),
                "score_cut_normal_called_novel": float(by_cut[normal].mean()),
                "score_cut_anomalies_called_novel": float(by_cut[anomalous].mean()),
                "rank_normal_called_novel": float(by_rank[normal].mean()),
                "rank_anomalies_called_novel": float(by_rank[anomalous].mean()),
            }

    keys = (
        "read_only_auroc",
        "absorbing_auroc",
        "score_cut_normal_called_novel",
        "score_cut_anomalies_called_novel",
        "rank_normal_called_novel",
        "rank_anomalies_called_novel",
    )
    means = {key: float(np.mean([v[key] for v in per_dataset.values()])) for key in keys}
    by_table = {
        table: float(
            np.mean(
                [
                    v["rank_normal_called_novel"]
                    for name, v in per_dataset.items()
                    if name.startswith(f"{table}/")
                ]
            )
        )
        for table in ADBENCH_NAMES
    }
    worst_table = max(by_table.values())
    artifact = {
        "schema": "mercury.research.novelty_discovery/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "observed": "fit-split normal rows (the detector's own fit data)",
            "scored": "held-out rows, read-only (novelty()) and absorbing (explore())",
            "n_evaluations": len(per_dataset),
        },
        "per_dataset": per_dataset,
        "mean": {k: round(v, 4) for k, v in means.items()},
        "verdict": {
            "alpha": CuriosityEngine().alpha,
            "score_cut": CuriosityEngine().novelty_threshold,
            "bar": {"rank_normal_called_novel_mean_max": 0.07, "per_table_max": 0.10},
            "rank_normal_called_novel_by_table": {k: round(v, 4) for k, v in by_table.items()},
            "rank_normal_called_novel_worst_table": round(worst_table, 4),
            "bar_met": means["rank_normal_called_novel"] <= 0.07 and worst_table <= 0.10,
        },
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    for name, values in per_dataset.items():
        print(
            f"{name:20s} read-only {values['read_only_auroc']:.4f} "
            f"absorbing {values['absorbing_auroc']:.4f}"
        )
    print(f"MEAN read-only {means['read_only_auroc']:.4f} absorbing {means['absorbing_auroc']:.4f}")
    print(
        "VERDICT normal rows called novel: "
        f"score cut {means['score_cut_normal_called_novel']:.3f}, "
        f"rank {means['rank_normal_called_novel']:.3f} (worst table {worst_table:.3f}); "
        "anomalies called novel: "
        f"score cut {means['score_cut_anomalies_called_novel']:.3f}, "
        f"rank {means['rank_anomalies_called_novel']:.3f}"
    )
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

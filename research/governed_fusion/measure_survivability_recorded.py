# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the survivability drop a detector property, or one event's number?

``docs/DORMANCY_LEDGER.md`` and ``research/governed_fusion/FINDINGS.md`` record
the adversarial-survivability headline as a mean over seven events, split by
label provenance after it turned out six of them carry ``statistical`` labels
the repository's own ``GENUINE_LABEL_SOURCES`` excludes:

===========================  ======  ==========  ==================  ======
bucket                       events  mean clean  worst @ ``m = k/2``  drop
===========================  ======  ==========  ==================  ======
external_label (admissible)  1       0.9135      0.8172              0.0962
self_label (leakage-flagged) 6       0.9261      0.6889              0.2371
mixed (the headline)         7       0.9243      0.7073              0.2170
===========================  ======  ==========  ==================  ======

The 0.217 was reported as measured and flagged **not established** as a
detector property, because an admissible-label-only mean is *n = 1*. That is
the right disclosure and it leaves the question open: is the real figure the
optimistic 0.0962, or is that one event unrepresentative?

This harness answers it without touching the governed-fusion manifest. The
manifest's admissible set is small because it is gated on *hazard-domain
loaders* whose labels are audited; but the survivability claim is about the
**detector**, and this repository records five ADBench tables with published
ground-truth labels and live rows. Running the same battery on those gives
externally-labelled events that the n = 1 objection does not reach.

**What this is and is not.** It expands the evidence for the *detector
property*, not the manifest's domain coverage: ADBench tables are not hazard
domains and do not become governed-fusion events. The protocol is otherwise the
one ``measure_survivability.py`` uses -- the real ``MercuryAnomalyDetector``
fused score as the attack target, the same ``floor_curve`` battery, the same
``eps`` and row cap -- so the numbers are comparable to the table above.

Run from the repository root::

    python research/governed_fusion/measure_survivability_recorded.py
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

#: Matches ``measure_survivability.py`` so the figures are comparable.
ROW_CAP = 160
EPS = 0.6

#: The recorded buckets this harness is measured against.
RECORDED = {
    "external_label_n1": {"clean": 0.9135, "worst": 0.8172, "drop": 0.0962},
    "self_label_n6": {"clean": 0.9261, "worst": 0.6889, "drop": 0.2371},
    "mixed_headline_n7": {"clean": 0.9243, "worst": 0.7073, "drop": 0.2170},
}


def measure_table(name: str, seed: int) -> dict[str, Any] | None:
    """Clean and worst-case fused AUROC for one recorded table."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
    from research.adversarial.governed_attacks import floor_curve
    from tests.recorded_data import adbench, stratified_split

    features, labels = adbench(name)
    train_idx, test_idx = stratified_split(labels, seed)
    mean = features[train_idx].mean(axis=0)
    spread = features[train_idx].std(axis=0)
    spread[spread < 1e-8] = 1.0
    train_rows = (features[train_idx] - mean) / spread
    test_rows = (features[test_idx] - mean) / spread

    normal_reference = train_rows[labels[train_idx] == 0]
    if len(normal_reference) < 8:
        return None

    detector = MercuryAnomalyDetector().fit(normal_reference)

    def score_fn(batch: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(detector.detect(batch)["scores"], dtype=float)

    rng = np.random.RandomState(seed)
    keep = rng.permutation(len(test_rows))[:ROW_CAP]
    rows, row_labels = test_rows[keep], labels[test_idx][keep]
    if len(np.unique(row_labels)) < 2:
        return None

    curve = floor_curve(
        score_fn, rows, row_labels, normal_reference=normal_reference, eps=EPS, seed=seed
    )
    clean = float(curve[0]["worst_case_auroc"])
    # The reported operating point is m = k/2, which is the third entry of the
    # default sweep {0, k/4, k/2, 3k/4}.
    half = curve[min(2, len(curve) - 1)]
    worst = float(half["worst_case_auroc"])
    return {
        "table": name,
        "seed": seed,
        "n_features": int(rows.shape[1]),
        "n_rows": len(rows),
        "n_positive": int(row_labels.sum()),
        "clean_auroc": clean,
        "worst_case_auroc_at_half": worst,
        "drop": clean - worst,
        "m_at_half": int(half["m"]),
        "worst_attack": str(half.get("worst_attack", "")),
        "curve": [{"m": int(r["m"]), "auroc": float(r["worst_case_auroc"])} for r in curve],
    }


def main() -> int:
    """Measure survivability on every recorded externally-labelled table."""
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    from tests.recorded_data import ADBENCH_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", default=list(ADBENCH_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/survivability_recorded.json")
    args = parser.parse_args()

    print("Adversarial survivability on recorded, externally-labelled tables")
    print(f"(real MercuryAnomalyDetector fused score; eps={EPS}; row cap {ROW_CAP})")
    print("-" * 82)
    print(f"{'table':12s} {'k':>3s} {'rows/pos':>9s} {'clean':>8s} {'worst@k/2':>10s} {'drop':>8s}")

    results = []
    for table in args.tables:
        row = measure_table(table, args.seed)
        if row is None:
            print(f"  {table:12s} SKIP (degenerate split)")
            continue
        results.append(row)
        print(
            f"{row['table']:12s} {row['n_features']:3d} "
            f"{row['n_rows']:5d}/{row['n_positive']:<3d} "
            f"{row['clean_auroc']:8.4f} {row['worst_case_auroc_at_half']:10.4f} "
            f"{row['drop']:8.4f}"
        )

    if not results:
        print("INTEGRITY FAILURE: no table measured.")
        return 1

    clean = float(np.mean([r["clean_auroc"] for r in results]))
    worst = float(np.mean([r["worst_case_auroc_at_half"] for r in results]))
    drop = float(np.mean([r["drop"] for r in results]))
    print("-" * 82)
    print(f"{'MEAN':12s} {'':3s} {len(results):>9d} {clean:8.4f} {worst:10.4f} {drop:8.4f}")
    print()
    print("against the recorded buckets:")
    for bucket, values in RECORDED.items():
        print(
            f"  {bucket:22s} clean {values['clean']:.4f}  worst {values['worst']:.4f}  "
            f"drop {values['drop']:.4f}   (this run's drop differs by "
            f"{drop - values['drop']:+.4f})"
        )

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "per_table": results,
                "mean_clean_auroc": clean,
                "mean_worst_case_auroc_at_half": worst,
                "mean_drop": drop,
                "n_events": len(results),
                "recorded_buckets": RECORDED,
                "eps": EPS,
                "row_cap": ROW_CAP,
                "note": (
                    "Externally-labelled ADBench tables, live rows. Expands the "
                    "evidence for the detector property; does NOT add "
                    "governed-fusion manifest events."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nreport -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

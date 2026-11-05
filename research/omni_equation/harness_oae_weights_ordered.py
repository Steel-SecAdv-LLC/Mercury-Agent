# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the OAE's golden-ratio weighting beat uniform on corpora other than NAB?

``docs/MATH_SPEC.md`` §2.1.1 measured ``PHI : 1 : 1`` against alternatives for
the first time and found it loses to a plain mean by **0.0017** (0.8301 against
0.8318) on the four recorded NAB files. The default was deliberately left alone,
and the stated reason was the size of the evidence: a 0.0017 difference on four
files is not separable from noise, and replacing a ratified invariant to collect
a noise-level gain is churn.

That leaves the question open rather than answered. This harness widens the
evidence to the two **other** ordered corpora this repository records, both with
external labels:

* **CHB-MIT scalp EEG** (``tests/fixtures/eeg``) — channel FP1-F7 at 256 Hz,
  background against clinician-annotated seizure.
* **CWRU bearing vibration** (``tests/fixtures/cwru``) — 12 kHz drive-end
  accelerometer, healthy against machined 0.007-inch faults.

The three stream producers, the two weightings and the grid search are imported
from ``harness_oae_weights`` rather than copied, so the NAB numbers that module
publishes stay reproducible and both corpora are scored by identical code. The
``eta ** Phi`` gate is omitted for the same reason it is there: one scalar per
call is a monotone transform of the batch and cannot change within-batch
ranking.

**What differs from the NAB protocol, and why it is stated rather than hidden.**
NAB labels a bounded window inside one long stream, so a window's label comes
from whether an anomaly falls inside it. These two corpora label whole
recordings, so a window inherits its recording's class. Windows are therefore cut
**non-overlapping** here (stride = window) rather than at NAB's stride of 16:
with class-homogeneous signals, overlapping windows would put near-duplicate rows
on both sides of the train/test split and flatter the learned arm.

Run from the repository root::

    python research/omni_equation/harness_oae_weights_ordered.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.omni_equation.harness_oae_weights import (
    W_PHI,
    W_UNIFORM,
    learn_weights,
    streams,
)
from tests.recorded_data import auroc, cwru, eeg

#: Window length for the stream producers. 128 is the value
#: ``harness_oae_weights`` uses on NAB, kept so the three corpora are scored at
#: the same resolution.
WINDOW = 128

#: ``tests/detectors/test_recorded_cwru_bearings.py``'s faults.
CWRU_FAULTS = ("inner_race", "ball", "outer_race")
CWRU_SEGMENT = 2048
EEG_RECORDS = ("chb01_03", "chb01_04")

#: §2.1.1's NAB figures, for the comparison this harness exists to make.
RECORDED_NAB = {"phi": 0.8301, "uniform": 0.8318, "learned": 0.8916}


def _rows(signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Non-overlapping ``WINDOW``-length windows, each reduced to (R, H, O)."""
    count = signal.size // WINDOW
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    windows = signal[: count * WINDOW].reshape(count, WINDOW)
    return np.asarray([streams(window) for window in windows], dtype=float)


def _split_and_score(
    normal: np.ndarray[Any, Any], anomalous: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Fit the learned arm on the first half of each class, score the second.

    Both classes are split independently so each half carries both, which a
    single index split over a concatenation would not guarantee.
    """
    normal_half, anomalous_half = len(normal) // 2, len(anomalous) // 2
    if min(normal_half, anomalous_half) < 2:
        return {}

    train = np.vstack([normal[:normal_half], anomalous[:anomalous_half]])
    train_labels = np.concatenate(
        [np.zeros(normal_half, dtype=bool), np.ones(anomalous_half, dtype=bool)]
    )
    test = np.vstack([normal[normal_half:], anomalous[anomalous_half:]])
    test_labels = np.concatenate(
        [
            np.zeros(len(normal) - normal_half, dtype=bool),
            np.ones(len(anomalous) - anomalous_half, dtype=bool),
        ]
    )

    learned = learn_weights(train, train_labels)
    out: dict[str, Any] = {}
    for name, weights in (("phi", W_PHI), ("uniform", W_UNIFORM), ("learned", learned)):
        scored = test @ weights
        out[name] = auroc(scored[test_labels], scored[~test_labels])
    for index, name in enumerate(("R_alone", "H_alone", "O_alone")):
        out[name] = auroc(test[test_labels, index], test[~test_labels, index])
    out["_w_learned"] = [round(float(v), 2) for v in learned]
    out["_n_test"] = len(test_labels)
    out["_pos_test"] = int(test_labels.sum())
    return out


def measure_eeg() -> dict[str, dict[str, Any]]:
    """Seizure against background, per recording."""
    results: dict[str, dict[str, Any]] = {}
    for record in EEG_RECORDS:
        background, seizure = eeg(record)
        row = _split_and_score(_rows(background), _rows(seizure))
        if row:
            results[f"eeg/{record}"] = row
    return results


def measure_cwru() -> dict[str, dict[str, Any]]:
    """Each machined fault against healthy, per fault."""

    def segments(name: str, first: int, last: int) -> np.ndarray[Any, Any]:
        signal = cwru(name)
        return np.concatenate(
            [signal[i * CWRU_SEGMENT : (i + 1) * CWRU_SEGMENT] for i in range(first, last)]
        )

    healthy = _rows(np.concatenate([segments("normal_0hp", 0, 60), segments("normal_1hp", 0, 60)]))
    results: dict[str, dict[str, Any]] = {}
    for fault in CWRU_FAULTS:
        row = _split_and_score(healthy, _rows(segments(fault, 0, 20)))
        if row:
            results[f"cwru/{fault}"] = row
    return results


def main() -> int:
    """Measure the weightings on the two independent ordered corpora."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=["eeg", "cwru"])
    parser.add_argument("--out", default="artifacts/oae_weights_ordered.json")
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}
    if "eeg" in args.corpora:
        results.update(measure_eeg())
    if "cwru" in args.corpora:
        results.update(measure_cwru())
    if not results:
        print("INTEGRITY FAILURE: no unit measured.")
        return 1

    keys = ["phi", "uniform", "learned", "R_alone", "H_alone", "O_alone"]
    print(f"{'unit':24s} " + " ".join(f"{k:>9s}" for k in keys) + "   n/pos   learned w")
    print("-" * 110)
    for name, row in results.items():
        line = " ".join(f"{row[k]:9.4f}" for k in keys)
        print(
            f"{name:24s} {line}   {row['_n_test']}/{row['_pos_test']}   "
            f"{tuple(row['_w_learned'])}"
        )

    print("-" * 110)
    means = {k: float(np.mean([row[k] for row in results.values()])) for k in keys}
    print(f"{'MEAN':24s} " + " ".join(f"{means[k]:9.4f}" for k in keys))

    delta = means["phi"] - means["uniform"]
    print()
    print(f"phi - uniform on these corpora : {delta:+.4f}")
    print(
        "phi - uniform on NAB (MATH_SPEC 2.1.1): "
        f"{RECORDED_NAB['phi'] - RECORDED_NAB['uniform']:+.4f}"
    )
    wins = sum(1 for row in results.values() if row["phi"] > row["uniform"])
    print(f"units where phi beats uniform  : {wins}/{len(results)}")
    print(f"learned arm over uniform       : {means['learned'] - means['uniform']:+.4f}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "per_unit": results,
                "mean": means,
                "phi_minus_uniform": delta,
                "units_phi_beats_uniform": f"{wins}/{len(results)}",
                "window": WINDOW,
                "note": (
                    "Non-overlapping windows; whole-recording labels. See the "
                    "module docstring for how this differs from the NAB protocol."
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

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the sliding mean-shift lead generalize off the four NAB files?

``docs/DORMANCY_LEDGER.md`` §2e records a sliding mean shift at ``h = 16`` as
"the one lead worth naming": 0.6679 standalone on NAB against the shipped
``additive`` probe's 0.6848, nearly independent of it, and worth **+0.0188** to
a plain 21-column mean as a 22nd column. It was explicitly *not* promoted,
because ADBench cannot corroborate a change-point statistic (tabular rows carry
no ordering) and four files is below the bar this repository applies elsewhere.

That left the question open rather than answered, and the harness behind those
numbers was never committed -- so the figures could not be reproduced either.
This file closes both halves:

1. It reproduces the NAB measurement, so the §2e numbers become checkable.
2. It measures the same statistic on the two **other ordered corpora already
   recorded in this repository**, which ADBench's objection does not reach:

   * **CHB-MIT scalp EEG** (``tests/fixtures/eeg``) -- channel FP1-F7 at 256 Hz,
     background against clinician-annotated seizure. External labels.
   * **CWRU bearing vibration** (``tests/fixtures/cwru``) -- 12 kHz drive-end
     accelerometer, healthy against 0.007-inch inner-race / ball / outer-race
     faults. External labels (the fault is a machined ground truth).

Both are ordered series, so a change-point statistic is *admissible* on them in
a way it is not on ADBench. They differ from NAB in what a label marks, and the
difference is the point: NAB labels a bounded window inside one long stream,
while these label a whole window as faulty or not. A statistic that only wins on
NAB's labelling geometry is a NAB artifact; one that carries across both is a
property of the statistic.

Run from the repository root::

    python research/math_arrest/harness_mean_shift.py
    python research/math_arrest/harness_mean_shift.py --corpora nab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.base_probe import (
    BaseEquationProbe,
    ProbeResult,
)
from tests.recorded_data import auroc, cwru, eeg, nab_corpus

#: ``docs/DORMANCY_LEDGER.md`` §2e, reproduced by ``--corpora nab``.
RECORDED_NAB = {
    "additive_shipped": 0.6848,
    "mean_shift_h16": 0.6679,
    "mean_shift_h48": 0.5498,
    "column_mean_lift_h16": 0.0188,
}

#: CWRU protocol, from ``tests/detectors/test_recorded_cwru_bearings.py``.
CWRU_WINDOW = 2048
CWRU_FITTED = 40
CWRU_FAULTS = ("inner_race", "ball", "outer_race")

#: EEG protocol, from ``tests/detectors/test_recorded_eeg_seizures.py``.
EEG_WINDOW = 256
EEG_RECORDS = ("chb01_03", "chb01_04")


class SlidingMeanShiftProbe(BaseEquationProbe):
    """Magnitude of the local mean shift across each sample.

    At each index the mean of the ``h`` samples before it is compared against
    the mean of the ``h`` samples from it onward, scaled by the fit series'
    one-step dispersion. This is the statistic §2e names, kept here rather than
    in ``src/`` because it is a measurement subject, not a shipped probe: this
    harness exists to decide whether it earns promotion.
    """

    def __init__(self, half_window: int = 16) -> None:
        """Initialize with the half-window ``h``."""
        super().__init__(min_samples=4)
        self._h = int(half_window)
        self._scale = 1.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Store the fit series' one-step dispersion as the score scale."""
        x = self._to_1d(data)
        self._validate_data(x)
        step = np.diff(x) if x.size > 1 else np.zeros(1)
        self._scale = float(np.std(step)) + 1e-10
        self._is_fitted = True

    def _shift_magnitude(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``|mean(next h) - mean(previous h)|`` at every index, edges clamped."""
        h, n = self._h, x.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        padded = np.concatenate([np.full(h, x[0]), x, np.full(h, x[-1])])
        cumulative = np.concatenate([[0.0], np.cumsum(padded)])
        before = (cumulative[h : h + n] - cumulative[0:n]) / h
        after = (cumulative[2 * h : 2 * h + n] - cumulative[h : h + n]) / h
        magnitude: npt.NDArray[np.float64] = np.abs(after - before)
        return magnitude

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its local mean-shift magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        raw = self._shift_magnitude(x) / self._scale
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return ProbeResult(
            probe_name=f"mean_shift_h{self._h}",
            deviation_scores=self._normalize_scores(raw),
            confidence=1.0,
            trajectory_fit_quality=1.0,
            anomaly_geometry="level_shift",
        )


def _probe_matrix(arrest: AnomalyMathArrest, series: npt.NDArray[np.float64]) -> Any:
    """``(n_samples, n_probes)`` per-probe scores, trimmed to a common width."""
    columns = arrest.per_probe_scores(series)
    if not columns:
        return np.zeros((series.size, 0)), []
    width = min(len(c) for c in columns.values())
    matrix = np.column_stack([np.asarray(c[:width], dtype=np.float64) for c in columns.values()])
    return matrix, list(columns)


def _fit_mean_shift(reference: npt.NDArray[np.float64], h: int) -> SlidingMeanShiftProbe:
    probe = SlidingMeanShiftProbe(half_window=h)
    probe.fit_trajectory(reference)
    return probe


#: Samples the probes are fitted on, for the two window-labelled corpora.
#:
#: NAB fits each file on its probationary prefix, which is 169-282 points on
#: the four recorded files. The EEG and CWRU references are whole recordings --
#: 38,400 and 163,840 samples if handed over entire -- so fitting on all of
#: them would give those two corpora a 130-580x larger fit budget than NAB and
#: make the standalone AUROCs incomparable across corpora (it is also what made
#: the first run of this harness intractable: 21 trajectory probes, several
#: with Hankel/SVD steps, over a 38k series).
#:
#: Capped at the same order as NAB's probation. The *lift* this harness
#: measures is a within-corpus comparison -- the same fit serves the 21-column
#: and 22-column arms -- so the cap cannot bias it in either direction; it only
#: makes the standalone columns comparable between corpora.
FIT_CAP = 1024


# ---------------------------------------------------------------------------
# NAB: point-labelled windows inside one stream (the §2e protocol)
# ---------------------------------------------------------------------------
def measure_nab(halves: tuple[int, ...]) -> dict[str, Any]:
    """Reproduce §2e: standalone AUROC and the 22nd-column lift, per NAB file."""
    out: dict[str, Any] = {"per_file": {}, "halves": list(halves)}
    standalone: dict[int, list[float]] = {h: [] for h in halves}
    additive: list[float] = []
    base_mean: list[float] = []
    lifted: dict[int, list[float]] = {h: [] for h in halves}

    for series in nab_corpus():
        split = series.n_probation
        fit_part, score_part = series.values[:split], series.values[split:]
        positive = series.positive[split:]

        arrest = AnomalyMathArrest(weighting="uniform")
        arrest.fit(fit_part)
        matrix, names = _probe_matrix(arrest, score_part)
        width = matrix.shape[0]
        pos = positive[:width]

        file_row: dict[str, float] = {}
        if "additive" in names:
            column = matrix[:, names.index("additive")]
            value = auroc(column[pos], column[~pos])
            additive.append(value)
            file_row["additive"] = value

        plain = matrix.mean(axis=1)
        base = auroc(plain[pos], plain[~pos])
        base_mean.append(base)
        file_row["column_mean_21"] = base

        for h in halves:
            probe = _fit_mean_shift(fit_part, h)
            column = probe.deviation_score(score_part).deviation_scores[:width]
            value = auroc(column[pos], column[~pos])
            standalone[h].append(value)
            file_row[f"mean_shift_h{h}"] = value

            widened = np.column_stack([matrix, column]).mean(axis=1)
            with_probe = auroc(widened[pos], widened[~pos])
            lifted[h].append(with_probe)
            file_row[f"column_mean_22_h{h}"] = with_probe

        out["per_file"][series.name] = file_row

    out["mean_additive"] = float(np.mean(additive)) if additive else None
    out["mean_column_mean_21"] = float(np.mean(base_mean))
    out["mean_standalone"] = {h: float(np.mean(standalone[h])) for h in halves}
    out["mean_column_mean_22"] = {h: float(np.mean(lifted[h])) for h in halves}
    out["mean_lift"] = {h: float(np.mean(lifted[h]) - np.mean(base_mean)) for h in halves}
    return out


# ---------------------------------------------------------------------------
# Window-labelled ordered corpora (EEG, CWRU): one verdict per window
# ---------------------------------------------------------------------------
def _window_scores(
    arrest: AnomalyMathArrest,
    probes: dict[int, SlidingMeanShiftProbe],
    windows: npt.NDArray[np.float64],
) -> tuple[Any, dict[int, Any], list[str]]:
    """Per-window maxima: the 21 probe columns and each mean-shift column.

    ``max`` is the per-window summary ``MathArrestDetector.detect`` already
    publishes as ``anomaly_score``, so the reduction matches the served one.
    """
    rows: list[Any] = []
    shift_rows: dict[int, list[float]] = {h: [] for h in probes}
    names: list[str] = []
    for window in windows:
        matrix, names = _probe_matrix(arrest, window)
        rows.append(matrix.max(axis=0) if matrix.size else np.zeros(len(names)))
        for h, probe in probes.items():
            shift_rows[h].append(float(probe.deviation_score(window).deviation_scores.max()))
    return (
        np.asarray(rows, dtype=np.float64),
        {h: np.asarray(v, dtype=np.float64) for h, v in shift_rows.items()},
        names,
    )


def _measure_windowed(
    label: str,
    reference: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    anomalous: npt.NDArray[np.float64],
    halves: tuple[int, ...],
) -> dict[str, float]:
    """Fit on *reference* windows, then rank *anomalous* against *normal*."""
    fit_series = reference.reshape(-1)[:FIT_CAP]
    arrest = AnomalyMathArrest(weighting="uniform")
    arrest.fit(fit_series)
    probes = {h: _fit_mean_shift(fit_series, h) for h in halves}

    normal_matrix, normal_shift, names = _window_scores(arrest, probes, normal)
    anomalous_matrix, anomalous_shift, _ = _window_scores(arrest, probes, anomalous)

    row: dict[str, float] = {}
    if "additive" in names:
        index = names.index("additive")
        row["additive"] = auroc(anomalous_matrix[:, index], normal_matrix[:, index])

    base = auroc(anomalous_matrix.mean(axis=1), normal_matrix.mean(axis=1))
    row["column_mean_21"] = base
    for h in halves:
        row[f"mean_shift_h{h}"] = auroc(anomalous_shift[h], normal_shift[h])
        widened_anomalous = np.column_stack([anomalous_matrix, anomalous_shift[h]]).mean(axis=1)
        widened_normal = np.column_stack([normal_matrix, normal_shift[h]]).mean(axis=1)
        with_probe = auroc(widened_anomalous, widened_normal)
        row[f"column_mean_22_h{h}"] = with_probe
        row[f"lift_h{h}"] = with_probe - base
    print(f"  {label:<28s} " + "  ".join(f"{k}={v:+.4f}" for k, v in row.items()))
    return row


def _cwru_windows(name: str, first: int, last: int) -> npt.NDArray[np.float64]:
    signal = cwru(name)
    return np.stack([signal[i * CWRU_WINDOW : (i + 1) * CWRU_WINDOW] for i in range(first, last)])


def measure_cwru(halves: tuple[int, ...]) -> dict[str, Any]:
    """Healthy against each machined bearing fault, per the recorded protocol."""
    reference = np.concatenate(
        [_cwru_windows("normal_0hp", 0, CWRU_FITTED), _cwru_windows("normal_1hp", 0, CWRU_FITTED)]
    )
    healthy = np.concatenate(
        [
            _cwru_windows("normal_0hp", CWRU_FITTED, 60),
            _cwru_windows("normal_1hp", CWRU_FITTED, 60),
        ]
    )
    results = {
        fault: _measure_windowed(
            f"cwru/{fault}", reference, healthy, _cwru_windows(fault, 0, 20), halves
        )
        for fault in CWRU_FAULTS
    }
    return results


def measure_eeg(halves: tuple[int, ...]) -> dict[str, Any]:
    """Annotated seizure against background, per the recorded protocol."""

    def windows(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        usable = signal[: signal.size // EEG_WINDOW * EEG_WINDOW]
        return usable.reshape(-1, EEG_WINDOW)

    results: dict[str, Any] = {}
    for record in EEG_RECORDS:
        background, seizure = (windows(x) for x in eeg(record))
        # Fit on the first half of background; rank seizure against the rest,
        # so no scored window contributed to the fit.
        cut = len(background) // 2
        results[record] = _measure_windowed(
            f"eeg/{record}", background[:cut], background[cut:], seizure, halves
        )
    return results


def _aggregate(per_unit: dict[str, dict[str, float]], halves: tuple[int, ...]) -> dict[str, Any]:
    keys = ["additive", "column_mean_21", *[f"mean_shift_h{h}" for h in halves]]
    keys += [f"column_mean_22_h{h}" for h in halves] + [f"lift_h{h}" for h in halves]
    return {
        k: float(np.mean([row[k] for row in per_unit.values() if k in row]))
        for k in keys
        if any(k in row for row in per_unit.values())
    }


def main() -> int:
    """Measure the statistic on every selected ordered corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="*", default=["nab", "eeg", "cwru"])
    parser.add_argument("--halves", nargs="*", type=int, default=[16, 48])
    parser.add_argument("--out", default="artifacts/mean_shift_generalization.json")
    args = parser.parse_args()
    halves = tuple(args.halves)

    report: dict[str, Any] = {"halves": list(halves)}

    if "nab" in args.corpora:
        print("NAB (point-labelled windows in one stream) -- reproducing ledger 2e")
        print("-" * 78)
        nab = measure_nab(halves)
        report["nab"] = nab
        print(
            f"  additive (shipped)        {nab['mean_additive']:.4f}  "
            f"recorded {RECORDED_NAB['additive_shipped']:.4f}"
        )
        for h in halves:
            key = f"mean_shift_h{h}"
            recorded = RECORDED_NAB.get(key)
            suffix = f"  recorded {recorded:.4f}" if recorded else ""
            print(f"  {key:<25s} {nab['mean_standalone'][h]:.4f}{suffix}")
        print(f"  column mean, 21 probes    {nab['mean_column_mean_21']:.4f}")
        for h in halves:
            print(
                f"  + h={h} as 22nd column     {nab['mean_column_mean_22'][h]:.4f}"
                f"   lift {nab['mean_lift'][h]:+.4f}"
                + (f"   recorded {RECORDED_NAB['column_mean_lift_h16']:+.4f}" if h == 16 else "")
            )
        print()

    if "eeg" in args.corpora:
        print("CHB-MIT EEG (clinician-annotated seizure vs background) -- INDEPENDENT")
        print("-" * 78)
        per_record = measure_eeg(halves)
        report["eeg"] = {"per_record": per_record, "mean": _aggregate(per_record, halves)}
        print()

    if "cwru" in args.corpora:
        print("CWRU bearings (machined fault vs healthy) -- INDEPENDENT")
        print("-" * 78)
        per_fault = measure_cwru(halves)
        report["cwru"] = {"per_fault": per_fault, "mean": _aggregate(per_fault, halves)}
        print()

    print("=" * 78)
    print("Lift from adding the mean shift as a 22nd column, per corpus:")
    for corpus in ("nab", "eeg", "cwru"):
        if corpus not in report:
            continue
        for h in halves:
            if corpus == "nab":
                value = report["nab"]["mean_lift"][h]
            else:
                value = report[corpus]["mean"].get(f"lift_h{h}")
            if value is not None:
                print(f"  {corpus:<6s} h={h:<3d} {value:+.4f}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nreport -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

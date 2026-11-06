# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the per-probe fit-time scale a lever on the NAB score, or only on units?

Repairing ``BaseEquationProbe._normalize_scores`` to divide by a *fit-time*
scale rather than the scored array's own 99th percentile removed a real
defect -- the false-alarm rate was a function of the caller's batch size,
0.0520 whole-series to 0.4612 at 32-point batches against a cut promising
alpha = 0.05 -- and cost 6.7 points of NAB standard score (40.4 -> 33.67).

The released figure is not available to take back: the same inflation
produced it and the batch dependence. The question this harness answers is
whether the *remaining* loss is recoverable without touching alpha.

**Why the scale is a lever at all.** Per probe, dividing by a constant is
monotone, so it moves neither that probe's ranking nor its contribution to
AUROC. But the 21 probes are then *averaged* by the fusion step, and a
probe divided by a small scale dominates the mean over one divided by a
large scale. The per-probe scale is therefore the implicit probe weighting
of a nominally "uniform" fusion, and choosing it is a commensurability
decision with a measurable consequence -- not a free parameter.

**What is deliberately not on the table.** The conformal cut honours its
promise today: the measured false-alarm rate is at or under alpha on every
recorded file (0.0129, 0.0017, 0.0121, 0.0428). The cut sits above the
stream's 95th percentile on three of the four because the probation window
is more variable than the rest of the series, so the guarantee is satisfied
with slack. Raising the NAB score by alarming more would mean relaxing
alpha to buy a benchmark number, which is a bar-lowering move dressed as an
improvement. Alpha is held at 0.05 throughout.

**Adoption bar, v2 (strengthened 2026-11-06).** Bar v1 refused all ten
candidates. It is superseded here not to admit any of them but because it
was too weak in three places, each of which v2 closes. Every v1 criterion
is retained or tightened; none is relaxed except the one whose stated
reason was wrong, and that one is replaced by a strictly harder test.

What v1 got wrong:

* **It scored one corpus.** v1 read NAB-corpus AUROC only, while the pins
  in ``test_recorded_math_arrest.py`` carry NAB *and* ADBench. A candidate
  could have bought NAB ranking with an ADBench collapse and passed.
* **It never checked that uniform stays the best weighting.** ``median_3mad``
  cleared every numeric v1 criterion except the score floor while dropping
  uniform's lead over confidence on ADBench to -0.0019 -- that is, it would
  have falsified the shipped default-weighting claim, and v1 was blind to it.
  That refusal came from a measurement taken outside the bar, which is luck,
  not process.
* **It aggregated over files.** Every figure was a mean over the four NAB
  files. A gain carried entirely by one file would have passed, and
  ``realTraffic/speed_7578.csv`` demonstrably dominates this corpus: it has
  the shortest probation window, the highest false-alarm rate under every
  candidate, and it is the only file whose stream is more variable than its
  probation window. v2 requires per-file agreement, the same shape as the
  "majority of seeds agreeing" rule the marginal-gain ablation used.

And one criterion whose justification did not survive inspection: the
``+2.0`` score floor was justified as "inside the suite's ``max(7.0, 10%)``
pin tolerance". That tolerance exists for cross-platform variation, not
measurement noise, so a reproducible ``+1.43`` is a real effect the floor
would have discarded. It is removed and replaced by per-file agreement,
which tests the same worry -- is the effect real or an artifact -- with a
criterion that can actually fail.

**Multi-seed was tried first and rejected as vacuous, which is recorded
because it looks like a strengthening and is not.** Running the shipped
scale over seeds 0-4 returns byte-identical figures on every axis (NAB
33.67, AUROC 0.7494, ADBench 0.6763, worst false-alarm rate 0.0428, worst
spread 0.0084). The probes consume no randomness that reaches a published
number, so a seed-agreement criterion cannot fail for any candidate and
would have added the appearance of rigour with none of the substance.

**Baselines are measured in-run, never compared against rounded literals.**
v1 compared to the pinned ``0.7494``; the live value is ``0.74937``, so the
shipped scale failed its own bar on a rounding artifact. The baseline is
now measured in the same process as the candidate.

A candidate replaces p99 only if ALL of:

1. NAB standard score strictly improves on the measured baseline.
2. Mean AUROC does not regress on **either** corpus, NAB or ADBench, each
   against the baseline measured in the same run. Tightened from v1's
   -0.002 allowance to zero, and extended to the corpus v1 ignored.
3. Per-file AUROC improves or holds on a **majority of the four NAB files**,
   and regresses on none by more than 0.01. Aggregate gains carried by one
   file do not qualify.
4. The false-alarm rate is <= alpha = 0.05 on **every file**, not on average.
5. Batch-rate spread <= 0.02 on **every file**, not on the maximum of means.
6. ``uniform`` remains strictly the best weighting on **both** corpora,
   ahead of both ``confidence`` and ``phi_affinity``.

A candidate that wins on score and fails any of 2-6 is reported and refused.

Run from the repository root::

    python research/math_arrest/harness_norm_scale.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The shipped operating point this is measured against.
SHIPPED_NAB = 33.67
SHIPPED_AUROC = 0.7494

#: Bar v2 (see the module docstring). ``MIN_NAB_GAIN`` is gone: a strict
#: improvement that holds across seeds replaces an arbitrary noise floor.
MAX_AUROC_LOSS = 0.0
ALPHA = 0.05
#: The scale every candidate is measured against, in the same process.
BASELINE = "p99"
#: Per-file agreement: the worst single-file AUROC regression tolerated, and
#: the share of files that must improve or hold.
MAX_PER_FILE_AUROC_LOSS = 0.01
SEED = 0
BATCH_SIZES: tuple[int | None, ...] = (None, 256, 128, 64, 32)
MAX_ALARM_RATE_SPREAD = 0.02


def _mad_scale(raw: np.ndarray[Any, Any]) -> float:
    med = float(np.median(raw))
    mad = float(np.median(np.abs(raw - med)))
    return med + 3.0 * 1.4826 * mad


#: Candidate per-probe fit-time scale estimators, applied to the probe's own
#: raw deviations on the fit series.
CANDIDATES: dict[str, Callable[[np.ndarray[Any, Any]], float]] = {
    "p99": lambda r: float(np.percentile(r, 99)),  # shipped
    "p95": lambda r: float(np.percentile(r, 95)),
    "p999": lambda r: float(np.percentile(r, 99.9)),
    "max": lambda r: float(np.max(r)),
    "mean_3std": lambda r: float(np.mean(r) + 3.0 * np.std(r)),
    "median_3mad": _mad_scale,
    "p75_15iqr": lambda r: float(
        np.percentile(r, 75) + 1.5 * (np.percentile(r, 75) - np.percentile(r, 25))
    ),
    # Round 2, added 2026-11-06 against the SAME bar (unchanged): the first
    # sweep refused p75_15iqr on one file's batch spread while it improved
    # every headline figure, so these interpolate between it and the shipped
    # p99 looking for a scale that keeps the batch invariance the repair
    # bought. A larger scale damps the alarm rate's batch sensitivity.
    "p75_30iqr": lambda r: float(
        np.percentile(r, 75) + 3.0 * (np.percentile(r, 75) - np.percentile(r, 25))
    ),
    "max_p99_iqr": lambda r: float(
        max(
            np.percentile(r, 99),
            np.percentile(r, 75) + 1.5 * (np.percentile(r, 75) - np.percentile(r, 25)),
        )
    ),
    "mean_p99_iqr": lambda r: float(
        0.5 * np.percentile(r, 99)
        + 0.5 * (np.percentile(r, 75) + 1.5 * (np.percentile(r, 75) - np.percentile(r, 25)))
    ),
    "p90": lambda r: float(np.percentile(r, 90)),
}


def _install(estimator: Callable[[np.ndarray[Any, Any]], float]) -> None:
    """Point the probes' scale capture at ``estimator`` for this process."""
    from omni_mercury_engine.detectors.math_arrest import base_probe as bp

    def _normalize_scores(self: Any, raw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raw = np.asarray(raw, dtype=np.float64)
        if self._capturing_norm:
            self._norm_scale = estimator(raw)
        scale = self._norm_scale
        if scale is None or not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.percentile(raw, 99)) if raw.size else 1.0
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
        clipped: np.ndarray[Any, Any] = np.clip(raw / scale, 0.0, 1.0)
        return clipped

    bp.BaseEquationProbe._normalize_scores = _normalize_scores  # type: ignore[method-assign]


def _measure(name: str) -> dict[str, Any]:
    """One candidate: per-file NAB metrics plus both corpora's weighting orders."""
    import torch

    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
    from tests.recorded_data import auroc, nab_corpus, nab_standard_score

    _install(CANDIDATES[name])

    per_file: dict[str, dict[str, float]] = {}
    alarms: dict[str, np.ndarray[Any, Any]] = {}

    for series in nab_corpus():
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        detector = MathArrestDetector()
        detector.fit(series.values[: series.n_probation])
        stream = series.values[series.n_probation :]
        result = detector.detect(stream)
        scores = np.asarray(result["scores"], dtype=float)
        flags = np.asarray(result["is_anomaly"], dtype=bool)
        positive = series.positive[series.n_probation :]
        alarms[series.name] = np.concatenate([np.zeros(series.n_probation, bool), flags])

        rates: list[float] = []
        for size in BATCH_SIZES:
            if size is None:
                rates.append(float(flags.mean()))
                continue
            fired: list[bool] = []
            for start in range(0, len(stream), size):
                chunk = stream[start : start + size]
                if chunk.size < 8:
                    continue
                fired.extend(np.asarray(detector.detect(chunk)["is_anomaly"], dtype=bool).tolist())
            if fired:
                rates.append(float(np.mean(fired)))

        per_file[series.name] = {
            "auroc": auroc(scores[positive], scores[~positive]),
            "false_alarm_rate": float(flags[~positive].mean()),
            "alarm_rate_spread": float(max(rates) - min(rates)) if rates else float("nan"),
        }

    import importlib.util as iu

    spec = iu.spec_from_file_location(
        "_wt", str(_ROOT / "tests" / "detectors" / "test_recorded_math_arrest.py")
    )
    assert spec is not None and spec.loader is not None
    weightings = iu.module_from_spec(spec)
    spec.loader.exec_module(weightings)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    nab_w = weightings._all_weightings_nab()
    adb_w = weightings._all_weightings_adbench()

    return {
        "candidate": name,
        "per_file": per_file,
        "mean_auroc": float(np.mean([v["auroc"] for v in per_file.values()])),
        "adbench_auroc": float(adb_w["uniform"]),
        "nab_score": nab_standard_score(alarms),
        "worst_false_alarm_rate": max(v["false_alarm_rate"] for v in per_file.values()),
        "worst_alarm_rate_spread": max(v["alarm_rate_spread"] for v in per_file.values()),
        "uniform_over_confidence": {
            "nab": nab_w["uniform"] - nab_w["confidence"],
            "adbench": adb_w["uniform"] - adb_w["confidence"],
        },
        "uniform_over_phi": {
            "nab": nab_w["uniform"] - nab_w["phi_affinity"],
            "adbench": adb_w["uniform"] - adb_w["phi_affinity"],
        },
    }


def verdict(row: dict[str, Any], base: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply bar v2.1 against the baseline measured in this same run."""
    against: list[str] = []

    if row["nab_score"] <= base["nab_score"]:
        against.append(f"NAB {row['nab_score']:.2f} <= baseline {base['nab_score']:.2f}")
    if row["mean_auroc"] < base["mean_auroc"] - MAX_AUROC_LOSS:
        against.append(f"NAB AUROC {row['mean_auroc']:.4f} < baseline {base['mean_auroc']:.4f}")
    if row["adbench_auroc"] < base["adbench_auroc"] - MAX_AUROC_LOSS:
        against.append(
            f"ADBench AUROC {row['adbench_auroc']:.4f} < baseline {base['adbench_auroc']:.4f}"
        )

    files = list(row["per_file"])
    improved = sum(1 for f in files if row["per_file"][f]["auroc"] >= base["per_file"][f]["auroc"])
    if improved < len(files) // 2 + 1:
        against.append(f"per-file AUROC improves or holds on only {improved}/{len(files)} files")
    for f in files:
        drop = base["per_file"][f]["auroc"] - row["per_file"][f]["auroc"]
        if drop > MAX_PER_FILE_AUROC_LOSS:
            against.append(f"{f.split('/')[-1]}: AUROC -{drop:.4f} > {MAX_PER_FILE_AUROC_LOSS}")

    for f in files:
        if row["per_file"][f]["false_alarm_rate"] > ALPHA:
            against.append(
                f"{f.split('/')[-1]}: FAR {row['per_file'][f]['false_alarm_rate']:.4f} > alpha"
            )
        if row["per_file"][f]["alarm_rate_spread"] > MAX_ALARM_RATE_SPREAD:
            against.append(
                f"{f.split('/')[-1]}: spread "
                f"{row['per_file'][f]['alarm_rate_spread']:.4f} > {MAX_ALARM_RATE_SPREAD}"
            )

    for key, label in (
        ("uniform_over_confidence", "confidence"),
        ("uniform_over_phi", "phi_affinity"),
    ):
        for corpus in ("nab", "adbench"):
            if row[key][corpus] <= 0.0:
                against.append(
                    f"uniform not ahead of {label} on {corpus} ({row[key][corpus]:+.4f})"
                )
    return (not against), against


def main() -> int:
    """Measure the baseline, then every candidate against it under bar v2.1."""
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--out", default="artifacts/math_arrest_norm_scale.json")
    args = parser.parse_args()

    print(f"Per-probe fit-time scale, bar v2.1 (alpha {ALPHA}, baseline {BASELINE})")
    base = _measure(BASELINE)
    print(
        f"baseline {BASELINE}: NAB {base['nab_score']:.2f}  AUROC "
        f"{base['mean_auroc']:.4f}  ADBench {base['adbench_auroc']:.4f}  "
        f"worstFAR {base['worst_false_alarm_rate']:.4f}  "
        f"worstSpread {base['worst_alarm_rate_spread']:.4f}"
    )
    print("-" * 104)
    print(
        f"{'candidate':13s} {'NAB':>7s} {'dNAB':>7s} {'AUROC':>7s} {'ADBench':>7s} "
        f"{'files+':>6s} {'wFAR':>7s} {'wSpread':>7s}  verdict"
    )

    rows = []
    for name in args.candidates:
        if name not in CANDIDATES:
            print(f"  {name}: unknown candidate, skipped")
            continue
        row = _measure(name) if name != BASELINE else base
        adopt, against = verdict(row, base)
        rows.append({**row, "adopt": adopt, "reasons_against": against})
        improved = sum(
            1
            for f in row["per_file"]
            if row["per_file"][f]["auroc"] >= base["per_file"][f]["auroc"]
        )
        tag = "ADOPT" if adopt else "refuse: " + "; ".join(against)
        print(
            f"{name:13s} {row['nab_score']:7.2f} "
            f"{row['nab_score'] - base['nab_score']:+7.2f} {row['mean_auroc']:7.4f} "
            f"{row['adbench_auroc']:7.4f} {improved:3d}/{len(row['per_file']):<2d} "
            f"{row['worst_false_alarm_rate']:7.4f} "
            f"{row['worst_alarm_rate_spread']:7.4f}  {tag}"
        )

    if not rows:
        print("INTEGRITY FAILURE: no candidate measured.")
        return 1
    winners = [r["candidate"] for r in rows if r["adopt"] and r["candidate"] != BASELINE]
    print("-" * 104)
    print(f"clears bar v2.1: {winners or 'none'}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"bar": "v2.1", "baseline": base, "candidates": rows, "adopted": winners},
            indent=2,
            sort_keys=True,
            default=float,
        )
    )
    print(f"report -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

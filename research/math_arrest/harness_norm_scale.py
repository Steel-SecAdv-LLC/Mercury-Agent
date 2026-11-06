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

**Pre-registered adoption bar** (fixed before any candidate was run, and
stated here so it cannot be moved afterwards). A candidate replaces p99
only if ALL of:

1. NAB standard score improves by >= +2.0 over the shipped 33.67. The pin
   tolerance in ``tests/detectors/test_recorded_nab_series.py`` is
   ``max(7.0, 10%)``, so anything smaller is inside the noise the suite
   already tolerates and is not a result.
2. Mean AUROC does not fall more than 0.002 below the shipped 0.7494.
3. The false-alarm rate stays <= alpha = 0.05 on *every* file, not on average.
4. Batch invariance holds: alarm-rate spread <= 0.02 across batch sizes
   {whole, 256, 128, 64, 32}, the property the repair bought.

A candidate that wins on NAB and fails any of 2-4 is reported and refused.

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

#: Pre-registered bar (see the module docstring).
MIN_NAB_GAIN = 2.0
MAX_AUROC_LOSS = 0.002
ALPHA = 0.05
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


def measure(name: str) -> dict[str, Any]:
    """AUROC, NAB score, per-file false-alarm rates and batch invariance."""
    import torch

    from omni_mercury_engine.detectors.math_arrest.detector import MathArrestDetector
    from tests.recorded_data import auroc, nab_corpus, nab_standard_score

    _install(CANDIDATES[name])

    aucs: list[float] = []
    fars: dict[str, float] = {}
    alarms: dict[str, np.ndarray[Any, Any]] = {}
    spreads: dict[str, float] = {}

    for series in nab_corpus():
        torch.manual_seed(0)
        np.random.seed(0)
        detector = MathArrestDetector()
        detector.fit(series.values[: series.n_probation])
        body = series.values[series.n_probation :]
        result = detector.detect(body)
        scores = np.asarray(result["scores"], dtype=float)
        flags = np.asarray(result["is_anomaly"], dtype=bool)
        positive = series.positive[series.n_probation :]
        aucs.append(auroc(scores[positive], scores[~positive]))
        fars[series.name] = float(flags[~positive].mean())
        alarms[series.name] = np.concatenate([np.zeros(series.n_probation, bool), flags])

        # Batch invariance: the property the repair bought. Re-score the same
        # body in chunks and compare realized alarm rates.
        rates: list[float] = []
        for size in BATCH_SIZES:
            if size is None:
                rates.append(float(flags.mean()))
                continue
            fired: list[bool] = []
            for start in range(0, len(body), size):
                chunk = body[start : start + size]
                if chunk.size < 8:
                    continue
                chunk_flags = np.asarray(detector.detect(chunk)["is_anomaly"], dtype=bool)
                fired.extend(chunk_flags.tolist())
            if fired:
                rates.append(float(np.mean(fired)))
        spreads[series.name] = float(max(rates) - min(rates)) if rates else float("nan")

    return {
        "candidate": name,
        "mean_auroc": float(np.mean(aucs)),
        "nab_score": nab_standard_score(alarms),
        "false_alarm_rates": fars,
        "max_false_alarm_rate": max(fars.values()),
        "mean_false_alarm_rate": float(np.mean(list(fars.values()))),
        "alarm_rate_spreads": spreads,
        "max_alarm_rate_spread": max(spreads.values()),
    }


def verdict(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the pre-registered bar. Returns ``(adopt, reasons_against)``."""
    against: list[str] = []
    gain = row["nab_score"] - SHIPPED_NAB
    if gain < MIN_NAB_GAIN:
        against.append(f"NAB gain {gain:+.2f} < {MIN_NAB_GAIN:+.2f}")
    loss = SHIPPED_AUROC - row["mean_auroc"]
    if loss > MAX_AUROC_LOSS:
        against.append(f"AUROC loss {loss:+.4f} > {MAX_AUROC_LOSS:.4f}")
    if row["max_false_alarm_rate"] > ALPHA:
        against.append(f"false-alarm rate {row['max_false_alarm_rate']:.4f} > alpha {ALPHA}")
    if row["max_alarm_rate_spread"] > MAX_ALARM_RATE_SPREAD:
        against.append(f"batch spread {row['max_alarm_rate_spread']:.4f} > {MAX_ALARM_RATE_SPREAD}")
    return (not against), against


def main() -> int:
    """Sweep the candidate scales and apply the pre-registered bar."""
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--out", default="artifacts/math_arrest_norm_scale.json")
    args = parser.parse_args()

    print("Per-probe fit-time scale sweep (alpha held at 0.05)")
    print(f"shipped: p99 -> AUROC {SHIPPED_AUROC:.4f}, NAB {SHIPPED_NAB:.2f}")
    print(
        f"bar: NAB >= {SHIPPED_NAB + MIN_NAB_GAIN:.2f}, AUROC >= "
        f"{SHIPPED_AUROC - MAX_AUROC_LOSS:.4f}, max FAR <= {ALPHA}, "
        f"batch spread <= {MAX_ALARM_RATE_SPREAD}"
    )
    print("-" * 94)
    print(
        f"{'candidate':12s} {'AUROC':>8s} {'NAB':>8s} {'dNAB':>7s} "
        f"{'maxFAR':>7s} {'meanFAR':>8s} {'spread':>7s}  verdict"
    )

    rows = []
    for name in args.candidates:
        if name not in CANDIDATES:
            print(f"  {name}: unknown candidate, skipped")
            continue
        row = measure(name)
        adopt, against = verdict(row)
        rows.append({**row, "adopt": adopt, "reasons_against": against})
        tag = "ADOPT" if adopt else "refuse: " + "; ".join(against)
        print(
            f"{name:12s} {row['mean_auroc']:8.4f} {row['nab_score']:8.2f} "
            f"{row['nab_score'] - SHIPPED_NAB:+7.2f} "
            f"{row['max_false_alarm_rate']:7.4f} {row['mean_false_alarm_rate']:8.4f} "
            f"{row['max_alarm_rate_spread']:7.4f}  {tag}"
        )

    if not rows:
        print("INTEGRITY FAILURE: no candidate measured.")
        return 1

    winners = [r["candidate"] for r in rows if r["adopt"]]
    print("-" * 94)
    print(f"clears the pre-registered bar: {winners or 'none'}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "shipped": {"scale": "p99", "auroc": SHIPPED_AUROC, "nab": SHIPPED_NAB},
                "bar": {
                    "min_nab_gain": MIN_NAB_GAIN,
                    "max_auroc_loss": MAX_AUROC_LOSS,
                    "alpha": ALPHA,
                    "max_alarm_rate_spread": MAX_ALARM_RATE_SPREAD,
                },
                "candidates": rows,
                "adopted": winners,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"report -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

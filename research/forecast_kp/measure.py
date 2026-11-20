#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can tomorrow's geomagnetic storm be forecast from the Kp record alone? Measured, walk-forward.

The watch's forecast (``omni_mercury_engine/watch/forecast.py``) is the
probability that a UT day holds a storm at NOAA's G1 or above, made at 12:00
UT the day before from the 27 days of Kp that end then. This fits it and
measures it, on the record GFZ has published since 1932:

* **Walk-forward.** Each year from 1960 is forecast by a logistic regression
  fitted on every year before it, and on nothing later. The scores are of
  those forecasts: no day is forecast by a model that had seen it.
* **Against what.** Two reference forecasts, built the same way from the
  years before: the share of days that held a storm (*climatology*), and
  that share among days that had a storm in the 24 hours before the cut-off
  or had none (*persistence*). The forecast is worth having only where it
  beats both.
* **The model that ships** is fitted on every day through
  :data:`TRAINED_THROUGH`. The days after it in the record are forecast by
  it and scored apart: they are the only ones the shipped model has not seen.

The record is ``kp_three_hourly.txt`` beside this file: GFZ's distribution
(CC BY 4.0; Matzka et al. 2021, https://doi.org/10.1029/2020SW002641, data
https://doi.org/10.5880/Kp.0001) packed to one character per three-hour
interval by ``--pack``.

Run::

    python research/forecast_kp/measure.py                 # measure and print
    python research/forecast_kp/measure.py --write         # and record: evidence.json, the model
    python research/forecast_kp/measure.py --pack Kp_ap_since_1932.txt   # repack GFZ's file

Nothing here is random: the same record gives the same numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale
from omni_mercury_engine.watch.forecast import FEATURES, features, storm_day

HERE = Path(__file__).resolve().parent
RECORD = HERE / "kp_three_hourly.txt"
EVIDENCE = HERE / "evidence.json"
MODEL = HERE.parents[1] / "src" / "omni_mercury_engine" / "watch" / "kp_g1_model.json"
KP_URL = "https://kp.gfz.de/app/files/Kp_ap_since_1932.txt"

#: The first year that is forecast: the 28 before it are what its model is fitted on.
FIRST_FORECAST_YEAR = 1960
#: The last day the shipped model is fitted on.
TRAINED_THROUGH = date(2025, 12, 31)
#: The weight of the penalty on the size of the weights (not on the intercept).
L2 = 1e-3
_THIRDS = "0123456789abcdefghijklmnopqr"  # Kp in thirds, 0 to 27 (Kp 9)
_DAYS_PER_LINE = 32


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #


def pack(source: Path) -> None:
    """Repack GFZ's ``Kp_ap_since_1932.txt`` as :data:`RECORD`."""
    days: dict[date, list[str]] = {}
    definitive = date.min
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        part = line.split()
        day = date(int(part[0]), int(part[1]), int(part[2]))
        value = float(part[7])
        thirds = round(value * 3)
        days.setdefault(day, ["."] * 8)[int(float(part[3])) // 3] = (
            _THIRDS[thirds] if value >= 0 else "."
        )
        if part[9] == "1":
            definitive = max(definitive, day)
    first, last = min(days), max(days)
    if any(first + timedelta(days=n) not in days for n in range((last - first).days + 1)):
        raise ValueError("the record skips a day")
    packed = "".join("".join(days[first + timedelta(days=n)]) for n in range(len(days)))
    width = 8 * _DAYS_PER_LINE
    lines = [
        "# The planetary Kp index, one value per three-hour interval.",
        "# Geomagnetic Observatory Niemegk, GFZ Helmholtz Centre for Geosciences. CC BY 4.0.",
        "# Cite: Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O. and Morschhauser, A., 2021.",
        "# The geomagnetic Kp index and derived indices of geomagnetic activity. Space Weather,",
        "# https://doi.org/10.1029/2020SW002641. Data: https://doi.org/10.5880/Kp.0001",
        f"# Repacked from {KP_URL} by research/forecast_kp/measure.py --pack.",
        "# One character per interval, eight to a UT day, the days in order from `first`.",
        '# A character is Kp in thirds, 0-9 then a-r: "e" is 14 thirds, Kp 5-. "." is no value.',
        f"first {first.isoformat()}",
        f"last {last.isoformat()}",
        f"definitive_through {definitive.isoformat()}",
        *(packed[at : at + width] for at in range(0, len(packed), width)),
    ]
    RECORD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_record(path: Path = RECORD) -> tuple[dict[date, list[float | None]], dict[str, date]]:
    """Each UT day's eight Kp values from the packed record, and its three dates."""
    stated: dict[str, date] = {}
    packed: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, value = line.partition(" ")
        if value:
            stated[name] = date.fromisoformat(value)
        else:
            packed.append(line)
    values = "".join(packed)
    days: dict[date, list[float | None]] = {}
    for number in range(len(values) // 8):
        days[stated["first"] + timedelta(days=number)] = [
            None if char == "." else _THIRDS.index(char) / 3.0
            for char in values[8 * number : 8 * number + 8]
        ]
    if max(days) != stated["last"]:
        raise ValueError("the record does not end where it says it does")
    return days, stated


# --------------------------------------------------------------------------- #
# Fitting and scoring
# --------------------------------------------------------------------------- #


def dataset(
    record: dict[date, list[float | None]],
) -> tuple[list[date], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Every day the record lets a forecast be made for and resolved: its features and outcome."""
    days, rows, outcomes = [], [], []
    for day in sorted(record):
        if any(value is None for value in record[day]):
            continue
        made_from = features(record, day)
        if made_from is None:
            continue
        days.append(day)
        rows.append(made_from)
        outcomes.append(float(storm_day(record[day])))
    return days, np.array(rows), np.array(outcomes)


def fit(rows: np.ndarray[Any, Any], outcomes: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Logistic regression by Newton's method on standardised features."""
    mean, scale = rows.mean(axis=0), rows.std(axis=0)
    design = np.c_[np.ones(len(rows)), (rows - mean) / scale]
    penalty = L2 * np.diag(np.r_[0.0, np.ones(design.shape[1] - 1)])
    weights = np.zeros(design.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (p - outcomes) / len(outcomes) + penalty @ weights
        hessian = (design * (p * (1.0 - p))[:, None]).T @ design / len(outcomes) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.abs(step).max()) < 1e-12:
            break
    stormy = np.array([noaa_g_scale(float(v)) >= 1 for v in rows[:, FEATURES.index("max_24h")]])
    return {
        "mean": mean,
        "scale": scale,
        "bias": float(weights[0]),
        "weights": weights[1:],
        "climatology": float(outcomes.mean()),
        "after_storm": float(outcomes[stormy].mean()),
        "after_quiet": float(outcomes[~stormy].mean()),
    }


def forecast(model: dict[str, Any], rows: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
    """The model's probabilities for ``rows``, and the two references'."""
    z = model["bias"] + ((rows - model["mean"]) / model["scale"]) @ model["weights"]
    stormy = np.array([noaa_g_scale(float(v)) >= 1 for v in rows[:, FEATURES.index("max_24h")]])
    return {
        "forecast": 1.0 / (1.0 + np.exp(-z)),
        "climatology": np.full(len(rows), model["climatology"]),
        "persistence": np.where(stormy, model["after_storm"], model["after_quiet"]),
    }


def _brier(p: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    return float(np.mean((p - y) ** 2))


def _log_score(p: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _auroc(p: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float | None:
    positives = int(y.sum())
    if positives in (0, len(y)):
        return None
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for value in np.unique(p):  # tied forecasts share the mean of their ranks
        tied = p == value
        ranks[tied] = ranks[tied].mean()
    negatives = len(y) - positives
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def account(made: dict[str, np.ndarray[Any, Any]], y: np.ndarray[Any, Any]) -> dict[str, Any]:
    """How a set of forecasts did, against each reference."""
    said = made["forecast"]
    auroc = _auroc(said, y)
    told: dict[str, Any] = {
        "forecasts": len(y),
        "storm_days": int(y.sum()),
        "brier": round(_brier(said, y), 4),
        "log_score": round(_log_score(said, y), 4),
        "auroc": None if auroc is None else round(auroc, 4),
        "references": {},
        "reliability": [],
    }
    for name in ("climatology", "persistence"):
        reference = _brier(made[name], y)
        told["references"][name] = {
            "brier": round(reference, 4),
            "log_score": round(_log_score(made[name], y), 4),
            "skill": round(1.0 - _brier(said, y) / reference, 4),
        }
    for tenth in range(10):
        inside = (said >= tenth / 10.0) & ((said < (tenth + 1) / 10.0) | (tenth == 9))
        if inside.any():
            told["reliability"].append(
                {
                    "from": tenth / 10.0,
                    "to": (tenth + 1) / 10.0,
                    "forecasts": int(inside.sum()),
                    "mean_probability": round(float(said[inside].mean()), 3),
                    "observed_rate": round(float(y[inside].mean()), 3),
                }
            )
    return told


def measure() -> tuple[dict[str, Any], dict[str, Any]]:
    """The evidence, and the model that ships."""
    record, stated = read_record()
    days, rows, outcomes = dataset(record)
    years = np.array([day.year for day in days])
    made = {name: np.full(len(days), np.nan) for name in ("forecast", "climatology", "persistence")}
    for year in range(FIRST_FORECAST_YEAR, TRAINED_THROUGH.year + 1):
        before, this = years < year, years == year
        for name, values in forecast(fit(rows[before], outcomes[before]), rows[this]).items():
            made[name][this] = values
    walked = ~np.isnan(made["forecast"])
    periods = {}
    for first, last in ((1960, 1979), (1980, 1999), (2000, 2019), (2020, TRAINED_THROUGH.year)):
        inside = walked & (years >= first) & (years <= last)
        periods[f"{first}-{last}"] = account(
            {name: values[inside] for name, values in made.items()}, outcomes[inside]
        )
    fitted_on = np.array([day <= TRAINED_THROUGH for day in days])
    shipped = fit(rows[fitted_on], outcomes[fitted_on])
    after = ~fitted_on

    # Conformal certificate, uniform with the anomaly path's: fitted on the
    # walk-forward out-of-sample forecasts of the last two decades (each day
    # forecast by a model that never saw it; those late models are near the
    # shipped fit, which each year moves by one year in ~90), and measured
    # on the after-the-fit days the shipped model has never seen. The
    # held-out coverage below is the certificate's real content: a measured
    # guarantee, not an exchangeability theorem, because the walk-forward
    # score functions differ slightly from the shipped one.
    from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier

    recent = walked & (years >= 2000)
    certificate = BinaryConformalClassifier(coverage=0.9).fit(
        made["forecast"][recent], outcomes[recent].astype(int)
    )
    after_probs = forecast(shipped, rows[after])["forecast"]
    after_y = outcomes[after].astype(int)
    after_sets = certificate.predict(after_probs)
    sizes = np.asarray(after_sets.set_size, dtype=int)
    contains = np.array(
        [int(y) in set(labels) for y, labels in zip(after_y, after_sets.label_sets(), strict=True)]
    )
    conformal_evidence = {
        "calibrated_on": "walk-forward forecasts 2000-" + str(TRAINED_THROUGH.year),
        "n_calibration": int(certificate.n_calibration or 0),
        "coverage_target": 0.9,
        "coverage_achieved_bound": round(float(certificate.coverage_achieved), 4),
        "after_the_fit": {
            "days": int(after_y.size),
            "empirical_coverage": round(float(contains.mean()), 4),
            "empirical_coverage_storm_days": (
                round(float(contains[after_y == 1].mean()), 4) if (after_y == 1).any() else None
            ),
            "empirical_coverage_quiet_days": (
                round(float(contains[after_y == 0].mean()), 4) if (after_y == 0).any() else None
            ),
            "singleton_rate": round(float(np.mean(sizes == 1)), 4),
            "ambiguous_rate": round(float(np.mean(sizes == 2)), 4),
            "empty_rate": round(float(np.mean(sizes == 0)), 4),
        },
    }

    model = {
        "schema": "mercury.kp-g1-model/v1",
        "version": f"kp-g1/{TRAINED_THROUGH.year}",
        "features": list(FEATURES),
        "mean": [round(float(value), 6) for value in shipped["mean"]],
        "scale": [round(float(value), 6) for value in shipped["scale"]],
        "weights": [round(float(value), 6) for value in shipped["weights"]],
        "bias": round(shipped["bias"], 6),
        "climatology": round(shipped["climatology"], 6),
        "persistence": {
            "after_storm": round(shipped["after_storm"], 6),
            "after_quiet": round(shipped["after_quiet"], 6),
        },
        "trained_through": TRAINED_THROUGH.isoformat(),
        "fitted_on_days": int(fitted_on.sum()),
        "l2": L2,
        # The certificate rides the artifact in the exact layout the fusion
        # checkpoint's conformal_state uses, so the forecast path's conformal
        # output is uniform with the anomaly path's.
        "conformal": certificate.export_state(),
    }
    evidence = {
        "record": {
            "source": KP_URL,
            "first": stated["first"].isoformat(),
            "last": stated["last"].isoformat(),
            "definitive_through": stated["definitive_through"].isoformat(),
            "days": len(record),
            "forecastable_days": len(days),
            "storm_day_rate": round(float(outcomes.mean()), 4),
        },
        "walk_forward": {
            "first_year": FIRST_FORECAST_YEAR,
            "last_year": TRAINED_THROUGH.year,
            "all": account(
                {name: values[walked] for name, values in made.items()}, outcomes[walked]
            ),
            "by_period": periods,
        },
        "after_the_fit": {
            "first": min(day for day, later in zip(days, after, strict=True) if later).isoformat(),
            "last": max(days).isoformat(),
            **account(forecast(shipped, rows[after]), outcomes[after]),
        },
        "conformal": conformal_evidence,
        "model": model["version"],
    }
    return evidence, model


def main() -> None:
    """Measure, print, and with ``--write`` record."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--write", action="store_true", help="record the evidence and the model")
    parser.add_argument("--pack", type=Path, metavar="FILE", help="repack GFZ's file as the record")
    args = parser.parse_args()
    if args.pack:
        pack(args.pack)
        print(f"wrote {RECORD}")
        return
    evidence, model = measure()
    print(json.dumps(evidence, indent=2))
    if args.write:
        EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        MODEL.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {EVIDENCE} and {MODEL}")
    assert math.isfinite(evidence["walk_forward"]["all"]["brier"])


if __name__ == "__main__":
    main()

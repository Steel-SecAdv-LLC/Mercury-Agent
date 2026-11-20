# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A forecast the watch commits to before the day it is about, and is scored on after.

The watch reported what had happened. It said nothing of what would, and so
could never be wrong, or shown to be right. This is one forecast it makes
and is held to:

    the probability that a UT day will hold a geomagnetic storm: that the
    planetary Kp index reaches NOAA's G1 (Kp 5-, :func:`~omni_mercury_engine
    .evaluation.hazard_metrics.noaa_g_scale` at least 1) in any of its
    eight three-hour intervals.

* **From what.** The Kp record alone, as GFZ publishes it: the 27 days that
  end at 12:00 UT on the day before (:func:`features`). Persistence of a
  disturbance, the Sun's 27-day rotation bringing the same stream back, the
  level of the last month, and the season. Nothing of the Sun or the solar
  wind is read: it is a statistical forecast from the index's own past, and
  NOAA's Space Weather Prediction Center, which watches both, is who to ask
  about a particular storm.
* **Fitted on.** Every day from 1932 to the model's ``trained_through``, by
  logistic regression (``research/forecast_kp``). Scored there walk-forward:
  each year from 1960 forecast by a model fitted on the years before it.
* **Committed.** Issued between 12:30 and 24:00 UT on the day before, into
  the append-only ledger (:mod:`~omni_mercury_engine.watch.ledger`), which
  refuses a forecast of a day that has begun. With it go the two forecasts
  it has to beat, fixed before the day as it is: the long-run rate
  (*climatology*) and the long-run rate given a storm in the last 24 hours
  or none (*persistence*).
* **Resolved.** Three hours after the day ends, from the same record, and
  never revised: the values and whether GFZ had made them definitive are
  kept with the outcome.

A day for which nothing could be issued in its window (the watch was down,
the record could not be read) has no forecast. It is not made up afterwards,
and the score counts it as missed.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.evaluation.hazard_metrics import noaa_g_scale

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from omni_mercury_engine.watch.ledger import ForecastLedger

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURES",
    "ISSUE_FROM",
    "RESOLVE_AFTER",
    "RESOLVE_FOR",
    "TARGET",
    "Forecaster",
    "Model",
    "features",
    "fetch_kp",
    "load_model",
    "parse_gfz_json",
    "storm_day",
]

#: What is forecast, as the ledger names it.
TARGET = "kp-g1"
#: Where the Kp record is read from: GFZ's web service, which answers for a
#: range of days with a few kilobytes.
KP_URL = "https://kp.gfz.de/app/json/"
#: Set to ``false`` and the watch makes no forecast and reads no Kp record.
ENABLED_ENV = "MERCURY_WATCH_FORECASTS"
#: What a forecast is made from, in the order the model weighs them.
FEATURES = (
    "latest",
    "max_24h",
    "mean_24h",
    "max_72h",
    "recurrence_27d",
    "recurrence_26_28d",
    "mean_27d",
    "semiannual_cos",
    "semiannual_sin",
)
#: The hour (UT) on the day before at which the record a forecast is made
#: from ends.
CUTOFF_HOUR = 12
#: The earliest a forecast is issued: half an hour after that, by when GFZ
#: has published the interval that ends at the cut-off.
ISSUE_FROM = time(12, 30)
#: How long after a day ends its forecast is resolved: by then GFZ has
#: published the day's last interval.
RESOLVE_AFTER = timedelta(hours=3)
#: How long a forecast whose day's values GFZ has not published goes on being
#: asked about. After that it stays open, and is shown as open.
RESOLVE_FOR = timedelta(days=60)
#: How many days of the record a forecast needs before the day it is about.
HISTORY_DAYS = 28
_INTERVALS = 8
_MODEL_PATH = Path(__file__).with_name("kp_g1_model.json")
_MODEL_SCHEMA = "mercury.kp-g1-model/v1"


@dataclass(frozen=True)
class Model:
    """The fitted forecaster.

    Attributes:
        version: The artifact's version; a forecast names the model it came from.
        mean: Each feature's mean on the days it was fitted on.
        scale: Each feature's standard deviation there.
        weights: The weight of each standardised feature.
        bias: The intercept.
        climatology: The share of fitted days that held a storm.
        persistence: That share among days with a storm in the 24 hours
            before the cut-off (``"after_storm"``) and without (``"after_quiet"``).
        trained_through: The last day it was fitted on.
        conformal: Optional conformal certificate state, in exactly the
            layout the fusion checkpoint's ``conformal_state`` carries
            (``BinaryConformalClassifier.export_state``), so the forecast
            path's certificate is uniform with the anomaly path's.  Fitted
            and measured by ``research/forecast_kp/measure.py``; ``None`` on
            an artifact written before the certificate existed.
    """

    version: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    climatology: float
    persistence: Mapping[str, float]
    trained_through: str
    conformal: Mapping[str, Any] | None = None

    def probability(self, values: Sequence[float]) -> float:
        """The probability of a storm day, from one day's :func:`features`."""
        if len(values) != len(self.weights):
            raise ValueError(f"the model weighs {len(self.weights)} features, not {len(values)}")
        z = self.bias + sum(
            weight * (value - mean) / scale
            for weight, value, mean, scale in zip(
                self.weights, values, self.mean, self.scale, strict=True
            )
        )
        return 1.0 / (1.0 + math.exp(-z))

    def references(self, values: Sequence[float]) -> dict[str, float]:
        """The forecasts this one has to beat, for the same day."""
        stormy = noaa_g_scale(min(max(values[FEATURES.index("max_24h")], 0.0), 9.0)) >= 1
        return {
            "climatology": self.climatology,
            "persistence": self.persistence["after_storm" if stormy else "after_quiet"],
        }

    def prediction_set(self, probability: float) -> dict[str, Any] | None:
        """The conformal label set for one forecast probability.

        The same block shape the anomaly path serves ({``prediction_set``,
        ``set_size``, ``abstain``, ``coverage``}), plus the regime it was
        issued under: the certificate was calibrated on walk-forward
        out-of-sample forecasts and its held-out coverage is measured in
        ``research/forecast_kp/evidence.json`` — a measured guarantee, not an
        exchangeability theorem, because each walk-forward year's model
        differs slightly from the shipped one.

        Returns:
            The block, or ``None`` when the artifact carries no certificate.
        """
        if self.conformal is None:
            return None
        import numpy as np

        from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier

        classifier = BinaryConformalClassifier.from_state(dict(self.conformal))
        pred = classifier.predict(np.array([float(probability)]))
        size = int(pred.set_size[0])
        return {
            "prediction_set": pred.label_sets()[0],
            "set_size": size,
            "abstain": bool(size == 2),
            "coverage": float(pred.coverage_level),
            "regime": "per_day_walk_forward",
            "n_calibration": classifier.n_calibration,
        }


def load_model(path: Path | None = None) -> Model:
    """The model that ships with the package.

    Raises:
        ValueError: If the artifact is not one this code can read.
    """
    document = json.loads((path or _MODEL_PATH).read_text(encoding="utf-8"))
    if document.get("schema") != _MODEL_SCHEMA or tuple(document.get("features", ())) != FEATURES:
        raise ValueError("the Kp forecast model is not one this version reads")
    return Model(
        version=str(document["version"]),
        mean=tuple(float(value) for value in document["mean"]),
        scale=tuple(float(value) for value in document["scale"]),
        weights=tuple(float(value) for value in document["weights"]),
        bias=float(document["bias"]),
        climatology=float(document["climatology"]),
        persistence={
            name: float(document["persistence"][name]) for name in ("after_storm", "after_quiet")
        },
        trained_through=str(document["trained_through"]),
        conformal=document.get("conformal"),
    )


def storm_day(values: Sequence[float | None]) -> bool:
    """Whether a day's eight Kp values hold one at G1 or above.

    Raises:
        ValueError: If the day is not complete.
    """
    if len(values) != _INTERVALS or any(value is None for value in values):
        raise ValueError("a day is eight Kp values")
    return any(
        noaa_g_scale(min(max(float(value), 0.0), 9.0)) >= 1 for value in values if value is not None
    )


def features(kp: Mapping[date, Sequence[float | None]], target: date) -> tuple[float, ...] | None:
    """What the forecast for ``target`` is made from, or ``None`` if the record has a gap.

    The 216 three-hour values that end at :data:`CUTOFF_HOUR` on the day
    before ``target``, and nothing later: the forecast for a day never reads
    that day, or the afternoon before it.

    Args:
        kp: Each UT day's eight Kp values, oldest interval first.
        target: The day the forecast is about.
    """
    eve = target - timedelta(days=1)
    upto = CUTOFF_HOUR // 3
    history: list[float] = []
    for back in range(HISTORY_DAYS - 1, -1, -1):
        day = kp.get(eve - timedelta(days=back))
        if day is None or len(day) != _INTERVALS:
            return None
        part = day[upto:] if back == HISTORY_DAYS - 1 else day[:upto] if back == 0 else day
        if any(value is None for value in part):
            return None
        history.extend(float(value) for value in part if value is not None)
    returning: list[float] = []
    for back in (28, 27, 26):
        day = kp.get(target - timedelta(days=back))
        if day is None or len(day) != _INTERVALS or any(value is None for value in day):
            return None
        returning.append(max(float(value) for value in day if value is not None))
    angle = 4.0 * math.pi * target.timetuple().tm_yday / 365.25
    return (
        history[-1],
        max(history[-8:]),
        sum(history[-8:]) / 8.0,
        max(history[-24:]),
        returning[1],
        max(returning),
        sum(history) / len(history),
        math.cos(angle),
        math.sin(angle),
    )


def parse_gfz_json(document: Mapping[str, Any]) -> dict[date, dict[str, list[Any]]]:
    """Each day's eight values and their statuses, from the answer GFZ's service gives.

    Returns:
        ``{day: {"kp": [eight values or None], "status": [eight of "def", "pre" or ""]}}``.
        An interval GFZ did not give is ``None``.

    Raises:
        ValueError: If the answer is not the service's shape.
    """
    times, values = document.get("datetime"), document.get("Kp")
    statuses = document.get("status") or []
    if not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values):
        raise ValueError("GFZ's answer has no Kp series in it")
    days: dict[date, dict[str, list[Any]]] = {}
    for position, (stamp, value) in enumerate(zip(times, values, strict=True)):
        at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(UTC)
        if at.hour % 3 or at.minute or at.second:
            raise ValueError(f"GFZ gave a Kp interval that starts at {stamp}")
        day = days.setdefault(at.date(), {"kp": [None] * _INTERVALS, "status": [""] * _INTERVALS})
        measured = isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 9.0
        day["kp"][at.hour // 3] = float(value) if measured else None
        if position < len(statuses):
            day["status"][at.hour // 3] = str(statuses[position])
    return days


def fetch_kp(first: date, last: date) -> dict[date, dict[str, list[Any]]]:
    """The Kp record from ``first`` to ``last``, from GFZ.

    Raises:
        Exception: Whatever the egress gate or the service raises; the
            caller decides what a failed read means.
    """
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    document = SafeHTTPClient.get_json(
        KP_URL,
        params={
            "start": f"{first.isoformat()}T00:00:00Z",
            "end": f"{last.isoformat()}T23:59:59Z",
            "index": "Kp",
        },
        timeout=30.0,
    )
    return parse_gfz_json(document)


class Forecaster:
    """Issues tomorrow's forecast when it can be made, and resolves the ones that are over."""

    #: How long between two looks at what is owed, and how long its lease lasts.
    LOOK_EVERY = timedelta(minutes=10)
    LEASE = timedelta(minutes=15)

    def __init__(
        self,
        ledger: ForecastLedger,
        *,
        model: Model | None = None,
        fetch: Callable[[date, date], dict[date, dict[str, list[Any]]]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Build one over the ledger its forecasts are committed to.

        Args:
            ledger: Where forecasts and outcomes are recorded.
            model: The forecaster. Defaults to the one that ships.
            fetch: Reads the Kp record for a range of days. Defaults to
                :func:`fetch_kp`.
            clock: Returns the current UTC time; replaceable in tests.
        """
        self.ledger = ledger
        self.model = model if model is not None else load_model()
        self._fetch = fetch
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._looked_at: datetime | None = None

    @staticmethod
    def enabled() -> bool:
        """Whether this deployment makes forecasts (it does unless told not to)."""
        return os.environ.get(ENABLED_ENV, "").strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def forecast_id(target: date) -> str:
        """The ledger's name for the forecast of ``target``."""
        return f"{TARGET}:{target.isoformat()}"

    def owed(self, now: datetime) -> tuple[date | None, list[date]]:
        """The day a forecast can be issued for now, and the days whose forecasts are over."""
        state = self.ledger.state(TARGET)
        tomorrow = now.date() + timedelta(days=1)
        to_issue = (
            tomorrow
            if now.time() >= ISSUE_FROM and self.forecast_id(tomorrow) not in state
            else None
        )
        to_resolve = [
            date.fromisoformat(name.partition(":")[2])
            for name, entry in state.items()
            if entry["outcome"] is None
            and RESOLVE_AFTER <= now - datetime.fromisoformat(entry["window"][1]) <= RESOLVE_FOR
        ]
        return to_issue, sorted(to_resolve)

    def run(self, holder: str = "forecast") -> dict[str, int]:
        """Do what is owed now. Safe to call on every pass of the watch.

        Args:
            holder: Who asks, for the lease that keeps two watches of one
                store from forecasting twice.

        Returns:
            How many forecasts were ``issued`` and how many ``resolved``.
        """
        done = {"issued": 0, "resolved": 0}
        now = self._clock()
        if not self.enabled() or (
            self._looked_at is not None and now - self._looked_at < self.LOOK_EVERY
        ):
            return done
        self._looked_at = now
        to_issue, to_resolve = self.owed(now)
        if to_issue is None and not to_resolve:
            return done
        if not self.ledger.store.hold_lease(holder, now, self.LEASE, name="forecast"):
            return done
        first = min([*to_resolve, *([to_issue - timedelta(days=HISTORY_DAYS)] if to_issue else [])])
        try:
            record = (self._fetch or fetch_kp)(first, now.date())
        except Exception as exc:  # the record could not be read: nothing is made up
            logger.warning(
                "watch: the Kp record could not be read (%s): %s", type(exc).__name__, exc
            )
            return done
        for day in to_resolve:
            read = record.get(day)
            if read is None or any(value is None for value in read["kp"]):
                continue  # the day is not all published yet
            self.ledger.resolve(
                self.forecast_id(day),
                outcome=storm_day(read["kp"]),
                observed={"kp": read["kp"], "status": read["status"], "source": KP_URL},
                now=now,
            )
            done["resolved"] += 1
        if to_issue is not None:
            values = features({day: read["kp"] for day, read in record.items()}, to_issue)
            if values is None:
                logger.warning(
                    "watch: the Kp record is not complete to %02d:00 UT; no forecast for %s yet",
                    CUTOFF_HOUR, to_issue,
                )  # fmt: skip
            else:
                start = datetime.combine(to_issue, time(0), tzinfo=UTC)
                probability = self.model.probability(values)
                basis = {
                    "model": self.model.version,
                    "features": dict(zip(FEATURES, values, strict=True)),
                    "cutoff": datetime.combine(
                        to_issue - timedelta(days=1), time(CUTOFF_HOUR), tzinfo=UTC
                    ).isoformat(),
                    "source": KP_URL,
                }
                # Uniform conformal output: when the shipped artifact carries
                # a certificate, the issued forecast states its label set and
                # measured coverage, exactly as the anomaly path does.
                certificate = self.model.prediction_set(probability)
                if certificate is not None:
                    basis["conformal"] = certificate
                self.ledger.issue(
                    self.forecast_id(to_issue),
                    target=TARGET,
                    window=(start, start + timedelta(days=1)),
                    probability=probability,
                    references=self.model.references(values),
                    basis=basis,
                    now=now,
                )
                done["issued"] += 1
        return done
